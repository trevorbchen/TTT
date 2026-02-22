"""
GRPO (Group Relative Policy Optimization) finetuning for inverse problems.

Trains LoRA across a dataset of images sharing the same forward operator A(x)
so the score model learns to internalize measurement guidance.  Uses
advantage-weighted DDPM MSE loss over groups of candidate reconstructions.

Pipeline:
  1. For each training image, generate G candidate reconstructions via DPS
  2. Compute rewards (measurement consistency), derive group-relative advantages
  3. Train LoRA with advantage-weighted DDPM loss across the dataset
  4. Save LoRA weights -- at inference, plain diffusion without guidance

The advantage-weighted DDPM loss:
  L = (1/G) * sum_i  A_i * MSE(noise_pred_i, noise_i)
where A_i = (R_i - mean(R)) / std(R) is the group-relative advantage.

References:
  - DDPO: https://github.com/kvablack/ddpo-pytorch
  - DeepSeek GRPO: https://arxiv.org/abs/2402.03300
"""

import yaml
import torch
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
    precond = model.model
    eps = 1e-5
    t = torch.rand(batch_size, device=device) * (1.0 - eps) + eps
    return precond.sigma(t)


# ---------------------------------------------------------------------------
# Candidate generation + advantage computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_all_candidates(model, sampler, operator, images, measurements,
                            num_candidates=6):
    """Generate candidates + advantages for every image in the dataset.

    For each image, runs DPS num_candidates times, computes rewards
    (negative measurement loss), and normalizes advantages within each group.

    Returns:
        candidates: [N_total, C, H, W]  (num_images * num_candidates)
        advantages: [N_total]
        cand_measurements: [N_total, C_y, H_y, W_y] — measurement for each candidate
    """
    all_candidates, all_advantages, all_y = [], [], []

    for img_idx in tqdm.trange(len(images), desc="Generating candidates"):
        y_i = measurements[img_idx: img_idx + 1]
        cands, rewards = [], []

        for _ in range(num_candidates):
            x_start = sampler.get_start(1, model)
            with torch.enable_grad():
                x_hat = sampler.sample(model, x_start, operator, y_i, verbose=False)
            loss = operator.loss(x_hat, y_i)
            cands.append(x_hat)
            rewards.append(-loss)  # higher = better

        cands = torch.cat(cands, dim=0)    # [G, C, H, W]
        rewards = torch.cat(rewards, dim=0)  # [G]

        # group-relative advantages
        std = rewards.std()
        if std < 1e-8:
            advs = torch.zeros_like(rewards)
        else:
            advs = (rewards - rewards.mean()) / std

        all_candidates.append(cands)
        all_advantages.append(advs)
        all_y.append(y_i.expand(num_candidates, -1, -1, -1))

    return torch.cat(all_candidates), torch.cat(all_advantages), torch.cat(all_y)


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def grpo_loss_batch(model, lora_modules, candidates, advantages,
                    lambda_kl=0.0, store=None, y_batch=None):
    """Advantage-weighted DDPM MSE loss on a batch."""
    device = candidates.device
    B = candidates.shape[0]

    sigma = sample_sigma(model, B, device)
    sigma_bc = sigma.view(-1, *([1] * (candidates.ndim - 1)))
    noise = torch.randn_like(candidates)

    x_noisy = candidates + sigma_bc * noise

    # model predictions (LoRA active — store sees measurement)
    if store is not None and y_batch is not None:
        store.set(y_batch)
    model.requires_grad_(True)
    D_x = model.tweedie(x_noisy, sigma)

    noise_pred = (x_noisy - D_x) / sigma_bc
    mse_per_sample = (noise_pred - noise).pow(2).mean(dim=[1, 2, 3])

    # advantage-weighted loss
    policy_loss = (advantages * mse_per_sample).mean()

    # optional KL regularization
    kl_loss = torch.tensor(0.0, device=device)
    if lambda_kl > 0:
        D_x_ref = frozen_tweedie(model, lora_modules, x_noisy, sigma)
        noise_pred_ref = (x_noisy - D_x_ref) / sigma_bc
        kl_loss = (noise_pred - noise_pred_ref.detach()).pow(2).mean()

    loss = policy_loss + lambda_kl * kl_loss
    return loss


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
    cfg = OmegaConf.to_container(args.get("grpo", {}), resolve=True)
    lora_rank = cfg.get("lora_rank", 64)
    lora_alpha = cfg.get("lora_alpha", 1.0)
    y_channels = cfg.get("y_channels", 3)  # 0 = unconditioned LoRA
    target_modules = cfg.get("target_modules", "all")
    lr = cfg.get("lr", 1e-4)
    num_epochs = cfg.get("num_epochs", 10)
    num_candidates = cfg.get("num_candidates", 6)
    lambda_kl = cfg.get("lambda_kl", 0.0)
    grad_clip = cfg.get("grad_clip", 1.0)
    batch_size = cfg.get("train_batch_size", 8)

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

    # --- generate candidates + advantages (one-time, before LoRA) ---
    print(f"Generating {num_candidates} candidates per image ({num_images} images)...")
    candidates, advantages, cand_y = generate_all_candidates(
        model, sampler, operator, images, y, num_candidates=num_candidates)
    num_samples = len(candidates)
    print(f"Total samples: {num_samples}, "
          f"positive advantage: {(advantages > 0).sum()}/{num_samples}")

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
        order = np.random.permutation(num_samples)
        epoch_loss, num_batches = 0.0, 0

        pbar = tqdm.tqdm(range(0, num_samples, batch_size),
                         desc=f"Epoch {epoch+1}/{num_epochs}")
        for start in pbar:
            idx = order[start: start + batch_size]
            c_batch = candidates[idx]
            a_batch = advantages[idx]
            y_batch = cand_y[idx]

            optimizer.zero_grad()
            loss = grpo_loss_batch(model, lora_modules, c_batch, a_batch,
                                   lambda_kl=lambda_kl,
                                   store=store, y_batch=y_batch)
            loss.backward()
            model.requires_grad_(False)
            torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

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
