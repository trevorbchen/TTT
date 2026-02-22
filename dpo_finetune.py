"""
Diffusion-DPO finetuning for inverse problems.

Trains LoRA across a dataset of images sharing the same forward operator A(x)
so the score model learns to internalize measurement guidance.  Uses the DPO
loss on the DDPM noise-prediction objective (Salesforce DiffusionDPO).

Pipeline:
  1. For each training image, generate candidate reconstructions via DPS
  2. Rank by measurement consistency, form winner/loser pairs
  3. Train LoRA with DPO loss on DDPM objective (random timestep noising)
  4. Save LoRA weights -- at inference, plain diffusion without guidance

Key insight: our model is a DDPM noise predictor wrapped in VPPrecond.
  F_x = UNet(c_in * x, c_noise)        # noise prediction
  D_x = x - sigma * F_x                # Tweedie estimate
We recover noise_pred = (x - D_x) / sigma and use MSE as ELBO proxy.

Reference: https://github.com/SalesforceAIResearch/DiffusionDPO
"""

import yaml
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import hydra
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from forward_operator import get_operator
from data import get_dataset
from model import get_model
from sampler import get_sampler
from lora import (apply_lora, apply_conditioned_lora, remove_lora,
                  get_lora_params, frozen_tweedie, save_lora)


# ---------------------------------------------------------------------------
# DDPM helpers
# ---------------------------------------------------------------------------

def sample_sigma(model, batch_size, device):
    """Sample noise levels from the VP training distribution."""
    precond = model.model  # VPPrecond
    eps = 1e-5
    t = torch.rand(batch_size, device=device) * (1.0 - eps) + eps
    return precond.sigma(t)


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_all_pairs(model, sampler, operator, images, measurements):
    """Generate one winner/loser pair for every image in the dataset.

    For each image, runs DPS twice with different seeds, picks the one
    with lower measurement loss as winner.

    Returns:
        winners: [N, C, H, W]
        losers:  [N, C, H, W]
        pair_measurements: [N, C_y, H_y, W_y] — measurement for each pair
    """
    all_winners, all_losers, all_y = [], [], []

    for img_idx in tqdm.trange(len(images), desc="Generating pairs"):
        y_i = measurements[img_idx: img_idx + 1]

        # two trajectories with different random seeds
        x_start_a = sampler.get_start(1, model)
        with torch.enable_grad():
            x_hat_a = sampler.sample(model, x_start_a, operator, y_i, verbose=False)
        loss_a = operator.loss(x_hat_a, y_i)

        x_start_b = sampler.get_start(1, model)
        with torch.enable_grad():
            x_hat_b = sampler.sample(model, x_start_b, operator, y_i, verbose=False)
        loss_b = operator.loss(x_hat_b, y_i)

        # lower loss = better reconstruction = winner
        if loss_a.item() <= loss_b.item():
            all_winners.append(x_hat_a)
            all_losers.append(x_hat_b)
        else:
            all_winners.append(x_hat_b)
            all_losers.append(x_hat_a)
        all_y.append(y_i)

    return torch.cat(all_winners), torch.cat(all_losers), torch.cat(all_y)


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def dpo_loss_batch(model, lora_modules, winners, losers, beta_dpo=5000.0,
                   store=None, y_batch=None):
    """DPO loss on a batch of winner/loser pairs at a random DDPM timestep."""
    device = winners.device
    B = winners.shape[0]

    sigma = sample_sigma(model, B, device)
    sigma_bc = sigma.view(-1, *([1] * (winners.ndim - 1)))
    noise = torch.randn_like(winners)

    x_w_noisy = winners + sigma_bc * noise
    x_l_noisy = losers + sigma_bc * noise

    # model predictions (LoRA active — store sees measurement)
    if store is not None and y_batch is not None:
        store.set(y_batch)
    model.requires_grad_(True)
    D_w = model.tweedie(x_w_noisy, sigma)
    D_l = model.tweedie(x_l_noisy, sigma)

    eps_w = (x_w_noisy - D_w) / sigma_bc
    eps_l = (x_l_noisy - D_l) / sigma_bc
    model_mse_w = (eps_w - noise).pow(2).mean(dim=[1, 2, 3])
    model_mse_l = (eps_l - noise).pow(2).mean(dim=[1, 2, 3])

    # reference predictions (LoRA zeroed)
    D_w_ref = frozen_tweedie(model, lora_modules, x_w_noisy, sigma)
    D_l_ref = frozen_tweedie(model, lora_modules, x_l_noisy, sigma)
    eps_w_ref = (x_w_noisy - D_w_ref) / sigma_bc
    eps_l_ref = (x_l_noisy - D_l_ref) / sigma_bc
    ref_mse_w = (eps_w_ref - noise).pow(2).mean(dim=[1, 2, 3])
    ref_mse_l = (eps_l_ref - noise).pow(2).mean(dim=[1, 2, 3])

    model_diff = model_mse_w - model_mse_l
    ref_diff = ref_mse_w - ref_mse_l
    inside_term = -0.5 * beta_dpo * (model_diff - ref_diff)

    loss = -F.logsigmoid(inside_term).mean()
    with torch.no_grad():
        acc = (inside_term > 0).float().mean()
    return loss, acc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="configs", config_name="default.yaml")
def main(args: DictConfig):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(f"cuda:{args.gpu}")

    print(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))

    # --- config ---
    cfg = OmegaConf.to_container(args.get("dpo", {}), resolve=True)
    lora_rank = cfg.get("lora_rank", 64)
    lora_alpha = cfg.get("lora_alpha", 1.0)
    y_channels = cfg.get("y_channels", 3)  # 0 = unconditioned LoRA
    target_modules = cfg.get("target_modules", "all")
    lr = cfg.get("lr", 1e-4)
    num_epochs = cfg.get("num_epochs", 10)
    beta_dpo = cfg.get("beta_dpo", 5000.0)
    grad_clip = cfg.get("grad_clip", 1.0)
    batch_size = cfg.get("train_batch_size", 4)

    # --- data ---
    dataset = get_dataset(**args.data)
    num_images = len(dataset)
    images = dataset.get_data(num_images, 0)

    # --- operator & measurement ---
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    y = operator.measure(images)

    # --- sampler & model ---
    sampler = get_sampler(**args.sampler,
                          mcmc_sampler_config=task_group.get("mcmc_sampler_config", None))
    model = get_model(**args.model)

    # --- output ---
    root = Path(args.save_dir) / args.name
    root.mkdir(parents=True, exist_ok=True)
    with open(str(root / "config.yaml"), "w") as f:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), f)

    # --- generate pairs (one-time, before LoRA) ---
    print(f"Generating 2 candidates per image ({num_images} images)...")
    winners, losers, pair_y = generate_all_pairs(
        model, sampler, operator, images, y)
    num_pairs = len(winners)
    print(f"Total pairs: {num_pairs}")

    # --- setup LoRA ---
    if y_channels > 0:
        lora_modules, store = apply_conditioned_lora(
            model, rank=lora_rank, alpha=lora_alpha, y_channels=y_channels,
            target_modules=target_modules)
    else:
        lora_modules = apply_lora(model, rank=lora_rank, alpha=lora_alpha)
        store = None
    lora_params = get_lora_params(lora_modules)
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.0)

    print(f"Training LoRA ({sum(p.numel() for p in lora_params)} params) "
          f"for {num_epochs} epochs, batch_size={batch_size}")

    # --- training ---
    for epoch in range(num_epochs):
        order = np.random.permutation(num_pairs)
        epoch_loss, epoch_acc, num_batches = 0.0, 0.0, 0

        pbar = tqdm.tqdm(range(0, num_pairs, batch_size),
                         desc=f"Epoch {epoch+1}/{num_epochs}")
        for start in pbar:
            idx = order[start: start + batch_size]
            w_batch = winners[idx]
            l_batch = losers[idx]
            y_batch = pair_y[idx]

            optimizer.zero_grad()
            loss, acc = dpo_loss_batch(model, lora_modules, w_batch, l_batch,
                                       beta_dpo=beta_dpo,
                                       store=store, y_batch=y_batch)
            loss.backward()
            model.requires_grad_(False)
            torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.2f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}, avg acc: {avg_acc:.2f}")

        save_lora(lora_modules, str(root / f"lora_epoch{epoch+1}.pt"),
                  metadata={"epoch": epoch+1, "avg_loss": avg_loss})

    # save final
    save_lora(lora_modules, str(root / "lora_final.pt"),
              metadata={"epochs": num_epochs,
                        "operator": task_group.operator.name,
                        "lora_rank": lora_rank, "lora_alpha": lora_alpha})

    remove_lora(model)
    print(f"\nLoRA saved to {root / 'lora_final.pt'}")


if __name__ == "__main__":
    main()
