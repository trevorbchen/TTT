"""
DRaFT-style direct finetuning for DPS.

Trains LoRA across a dataset of images sharing the same forward operator A(x)
so the score model learns to internalize measurement guidance:
    score_lora(x_t) ~ nabla log p(x_t) + nabla log p(y|x_t)

After training, the LoRA-adapted model can reconstruct from measurements
WITHOUT running DPS guidance gradients at each step -- plain diffusion
sampling suffices.

Approach: For each training image, run DPS prefix steps (no grad) to get
near-clean x_t, then backprop the reward signal R = -||y - A(tweedie(x_t))||^2
through the final Tweedie step into the LoRA weights (ReFL, K=1).
"""

import yaml
import torch
import numpy as np
import tqdm
import hydra
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from forward_operator import get_operator
from data import get_dataset
from model import get_model
from sampler import get_sampler
from cores.scheduler import get_diffusion_scheduler
from lora import (apply_lora, apply_conditioned_lora, remove_lora,
                  get_lora_params, frozen_tweedie, save_lora)


# ---------------------------------------------------------------------------
# DPS sampling prefix (no-grad) -- replicates sampler.py DPS.sample logic
# ---------------------------------------------------------------------------

def dps_sample_prefix(model, scheduler, guidance_scale, x_start, operator,
                      measurement, num_steps):
    """Run the first num_steps of DPS (Euler PF-ODE with guidance).

    Returns:
        (xt, sigma_boundary, st_boundary) at the step after the prefix.
    """
    sigma_steps = scheduler.sigma_steps
    total_steps = len(sigma_steps) - 1
    assert num_steps <= total_steps

    xt = x_start
    for step in range(num_steps):
        sigma = sigma_steps[step]
        sigma_next = sigma_steps[step + 1]
        t = scheduler.get_sigma_inv(sigma)
        t_next = scheduler.get_sigma_inv(sigma_next)
        dt = t_next - t
        st = scheduler.get_scaling(t)
        dst = scheduler.get_scaling_derivative(t)
        dsigma = scheduler.get_sigma_derivative(t)

        model.requires_grad_(True)
        xt_in = xt.detach().requires_grad_(True)
        x0hat = model.tweedie(xt_in / st, sigma)
        loss_per_sample = operator.loss(x0hat, measurement)
        grad_xt = torch.autograd.grad(loss_per_sample.sum(), xt_in)[0]
        model.requires_grad_(False)

        with torch.no_grad():
            norm_factor = loss_per_sample.sqrt().view(-1, *([1] * (grad_xt.ndim - 1)))
            norm_factor = norm_factor.clamp(min=1e-8)
            normalized_grad = grad_xt / norm_factor

        with torch.no_grad():
            score = (x0hat.detach() - xt / st) / sigma ** 2
            deriv = dst / st * xt - st * dsigma * sigma * score
            xt_next = xt + dt * deriv
            xt = xt_next - guidance_scale * normalized_grad
            if torch.isnan(xt).any():
                break

    sigma_boundary = sigma_steps[num_steps]
    t_boundary = scheduler.get_sigma_inv(sigma_boundary)
    st_boundary = scheduler.get_scaling(t_boundary)
    return xt.detach(), sigma_boundary, st_boundary


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
    cfg = OmegaConf.to_container(args.get("ttt", {}), resolve=True)
    lora_rank = cfg.get("lora_rank", 64)
    lora_alpha = cfg.get("lora_alpha", 1.0)
    y_channels = cfg.get("y_channels", 3)  # 0 = unconditioned LoRA
    target_modules = cfg.get("target_modules", "all")
    lr = cfg.get("lr", 1e-4)
    num_epochs = cfg.get("num_epochs", 5)
    lambda_kl = cfg.get("lambda_kl", 0.01)
    grad_clip = cfg.get("grad_clip", 1.0)
    grad_accum = cfg.get("grad_accum", 1)  # accumulate over N images before stepping
    K = cfg.get("K", 1)

    # --- data ---
    dataset = get_dataset(**args.data)
    num_images = len(dataset)
    images = dataset.get_data(num_images, 0)

    # --- operator & measurement ---
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    y = operator.measure(images)

    # --- sampler ---
    sampler = get_sampler(**args.sampler,
                          mcmc_sampler_config=task_group.get("mcmc_sampler_config", None))

    # --- model ---
    model = get_model(**args.model)

    # --- output ---
    root = Path(args.save_dir) / args.name
    root.mkdir(parents=True, exist_ok=True)
    with open(str(root / "config.yaml"), "w") as f:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), f)

    # --- setup LoRA (once, across entire dataset) ---
    scheduler = sampler.scheduler
    guidance_scale = sampler.guidance_scale
    sigma_steps = scheduler.sigma_steps
    total_dps_steps = len(sigma_steps) - 1
    prefix_steps = max(total_dps_steps - K, 0)

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
          f"across {num_images} images for {num_epochs} epochs, "
          f"grad_accum={grad_accum}")

    # --- training ---
    step_losses = []  # per optimizer-step loss (averaged over accum window)
    epoch_avg_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        order = np.random.permutation(num_images)
        pbar = tqdm.tqdm(order, desc=f"Epoch {epoch+1}/{num_epochs}")

        optimizer.zero_grad()
        accum_loss = 0.0

        for i, img_idx in enumerate(pbar):
            gt_i = images[img_idx: img_idx + 1]
            y_i = y[img_idx: img_idx + 1]

            # fresh noise
            x_T = sampler.get_start(1, model)

            # DPS prefix (no grad)
            if prefix_steps > 0:
                xt, sigma_b, st_b = dps_sample_prefix(
                    model, scheduler, guidance_scale, x_T, operator,
                    y_i, prefix_steps)
            else:
                xt = x_T.detach()
                sigma_b = sigma_steps[0]
                t_b = scheduler.get_sigma_inv(sigma_b)
                st_b = scheduler.get_scaling(t_b)

            # gradient-tracked Tweedie (through LoRA)
            if store is not None:
                store.set(y_i)
            model.requires_grad_(True)
            x0_hat = model.tweedie(xt / st_b, sigma_b)

            # reward loss
            reward_loss = operator.loss(x0_hat, y_i).mean()

            # KL regularisation
            kl_loss = torch.tensor(0.0, device=xt.device)
            if lambda_kl > 0:
                x0_frozen = frozen_tweedie(model, lora_modules, xt / st_b, sigma_b)
                kl_loss = ((x0_hat - x0_frozen) ** 2).mean()

            total_loss = (reward_loss + lambda_kl * kl_loss) / grad_accum
            total_loss.backward()
            model.requires_grad_(False)

            loss_val = total_loss.item() * grad_accum  # un-scaled for logging
            accum_loss += loss_val
            epoch_loss += loss_val
            pbar.set_postfix(loss=f"{loss_val:.4f}")

            # step every grad_accum images (or at end of epoch)
            if (i + 1) % grad_accum == 0 or (i + 1) == num_images:
                torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                n_in_window = ((i % grad_accum) + 1) if (i + 1) == num_images else grad_accum
                step_losses.append(accum_loss / n_in_window)
                accum_loss = 0.0

        avg_loss = epoch_loss / num_images
        epoch_avg_losses.append(avg_loss)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # save checkpoint each epoch
        save_lora(lora_modules, str(root / f"lora_epoch{epoch+1}.pt"),
                  metadata={"epoch": epoch+1, "avg_loss": avg_loss,
                            "operator": task_group.operator.name})

    # save final
    save_lora(lora_modules, str(root / "lora_final.pt"),
              metadata={"epochs": num_epochs,
                        "operator": task_group.operator.name,
                        "lora_rank": lora_rank, "lora_alpha": lora_alpha})

    # --- plot loss curve ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(step_losses, alpha=0.3, label="per step")
    if len(step_losses) > 10:
        window = max(len(step_losses) // 20, 5)
        smoothed = np.convolve(step_losses, np.ones(window)/window, mode='valid')
        ax1.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
                 label=f"smooth (w={window})")
    ax1.set_xlabel("Optimizer step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Per-step loss")
    ax1.legend()

    ax2.plot(range(1, num_epochs + 1), epoch_avg_losses, "o-")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Avg loss")
    ax2.set_title("Per-epoch avg loss")

    plt.tight_layout()
    plt.savefig(str(root / "loss_curve.png"), dpi=150)
    plt.close()

    remove_lora(model)
    print(f"\nLoRA saved to {root / 'lora_final.pt'}")
    print(f"Loss curve saved to {root / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
