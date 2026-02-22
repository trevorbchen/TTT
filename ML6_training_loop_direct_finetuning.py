"""
Hyperparameter sweep for measurement-conditioned LoRA direct finetuning.

Searches over learning rates and gradient accumulation (effective batch size)
on a subset of the data, logs per-step losses, and saves loss curves for
each configuration. Designed for multi-GPU machines (e.g., 4x A100).

Usage:
  python ML6_training_loop_direct_finetuning.py \
      +data=demo-ffhq +model=ffhq256ddpm +task=gaussian_blur \
      +sampler=edm_dps task_group=pixel \
      sampler.scheduler_config.num_steps=200 \
      name=hp_sweep save_dir=./results gpu=0
"""

import json
import yaml
import torch
import numpy as np
import tqdm
import hydra
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf, DictConfig

from forward_operator import get_operator
from data import get_dataset
from model import get_model
from sampler import get_sampler
from lora import (apply_conditioned_lora, remove_lora, get_lora_params,
                  frozen_tweedie, save_lora)


# ---------------------------------------------------------------------------
# DPS sampling prefix (same as direct_finetune.py)
# ---------------------------------------------------------------------------

def dps_sample_prefix(model, scheduler, guidance_scale, x_start, operator,
                      measurement, num_steps):
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
# Single training run
# ---------------------------------------------------------------------------

def train_one_config(model, images, y, operator, sampler, *,
                     lr, grad_accum, lora_rank, lora_alpha, y_channels,
                     target_modules, num_epochs, lambda_kl, grad_clip, K,
                     seed, save_path=None):
    """Train conditioned LoRA with one hyperparameter config.

    Returns:
        dict with step_losses, epoch_losses, final lora_modules
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    scheduler = sampler.scheduler
    guidance_scale = sampler.guidance_scale
    sigma_steps = scheduler.sigma_steps
    total_dps_steps = len(sigma_steps) - 1
    prefix_steps = max(total_dps_steps - K, 0)
    num_images = len(images)

    lora_modules, store = apply_conditioned_lora(
        model, rank=lora_rank, alpha=lora_alpha, y_channels=y_channels,
        target_modules=target_modules)
    lora_params = get_lora_params(lora_modules)
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.0)

    step_losses = []
    epoch_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        order = np.random.permutation(num_images)
        optimizer.zero_grad()
        accum_loss = 0.0

        pbar = tqdm.tqdm(order, desc=f"  lr={lr}, accum={grad_accum} | Ep {epoch+1}/{num_epochs}",
                         leave=False)

        for i, img_idx in enumerate(pbar):
            y_i = y[img_idx: img_idx + 1]
            x_T = sampler.get_start(1, model)

            if prefix_steps > 0:
                xt, sigma_b, st_b = dps_sample_prefix(
                    model, scheduler, guidance_scale, x_T, operator,
                    y_i, prefix_steps)
            else:
                xt = x_T.detach()
                sigma_b = sigma_steps[0]
                t_b = scheduler.get_sigma_inv(sigma_b)
                st_b = scheduler.get_scaling(t_b)

            store.set(y_i)
            model.requires_grad_(True)
            x0_hat = model.tweedie(xt / st_b, sigma_b)

            reward_loss = operator.loss(x0_hat, y_i).mean()

            kl_loss = torch.tensor(0.0, device=xt.device)
            if lambda_kl > 0:
                x0_frozen = frozen_tweedie(model, lora_modules, xt / st_b, sigma_b)
                kl_loss = ((x0_hat - x0_frozen) ** 2).mean()

            total_loss = (reward_loss + lambda_kl * kl_loss) / grad_accum
            total_loss.backward()
            model.requires_grad_(False)

            loss_val = total_loss.item() * grad_accum
            accum_loss += loss_val
            epoch_loss += loss_val
            pbar.set_postfix(loss=f"{loss_val:.1f}")

            if (i + 1) % grad_accum == 0 or (i + 1) == num_images:
                torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                n_in_window = ((i % grad_accum) + 1) if (i + 1) == num_images else grad_accum
                step_losses.append(accum_loss / n_in_window)
                accum_loss = 0.0

        avg_loss = epoch_loss / num_images
        epoch_losses.append(avg_loss)

    # compute weight norms
    up_norm = sum(m.lora_up.weight.data.norm().item() for m in lora_modules)
    down_norm = sum(m.lora_down.weight.data.norm().item() for m in lora_modules)

    # optionally save LoRA before cleanup
    if save_path is not None:
        save_lora(lora_modules, save_path,
                  metadata={"lr": lr, "grad_accum": grad_accum,
                            "num_epochs": num_epochs,
                            "final_loss": epoch_losses[-1]})

    # clean up
    remove_lora(model)

    return {
        "step_losses": step_losses,
        "epoch_losses": epoch_losses,
        "up_norm": up_norm,
        "down_norm": down_norm,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="configs", config_name="default.yaml")
def main(args: DictConfig):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(f"cuda:{args.gpu}")

    print(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))

    # --- sweep config ---
    cfg = OmegaConf.to_container(args.get("ttt", {}), resolve=True)
    lora_alpha = cfg.get("lora_alpha", 1.0)
    y_channels = cfg.get("y_channels", 3)
    target_modules = cfg.get("target_modules", "all")
    lambda_kl = cfg.get("lambda_kl", 0.0)
    grad_clip = cfg.get("grad_clip", 1.0)
    K = cfg.get("K", 1)
    num_epochs = cfg.get("num_epochs", 5)

    # hyperparameter grid
    lr_list = cfg.get("lr_list", [1e-3, 1e-2, 1e-1, 1.0])
    accum_list = cfg.get("accum_list", [1, 5, 10])
    rank_list = cfg.get("rank_list", [4, 16, 64])

    # full training config
    full_data_config = cfg.get("full_data", None)  # e.g., {root: dataset/test-ffhq, end_id: 100}
    full_num_epochs = cfg.get("full_num_epochs", 30)

    # --- data (use subset — 10-20% of available) ---
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

    # --- run sweep ---
    configs = list(itertools.product(lr_list, accum_list, rank_list))
    all_results = {}

    print(f"\n{'='*60}")
    print(f"Hyperparameter sweep: {len(configs)} configs")
    print(f"  LRs: {lr_list}")
    print(f"  Grad accum: {accum_list}")
    print(f"  Ranks: {rank_list}")
    print(f"  Data: {num_images} images, {num_epochs} epochs each")
    print(f"  DPS steps: {len(sampler.scheduler.sigma_steps)-1}, K={K}")
    print(f"{'='*60}\n")

    for idx, (lr, accum, rank) in enumerate(configs):
        tag = f"lr{lr}_accum{accum}_r{rank}"
        print(f"\n[{idx+1}/{len(configs)}] {tag}")

        result = train_one_config(
            model, images, y, operator, sampler,
            lr=lr, grad_accum=accum, lora_rank=rank,
            lora_alpha=lora_alpha, y_channels=y_channels,
            target_modules=target_modules,
            num_epochs=num_epochs, lambda_kl=lambda_kl,
            grad_clip=grad_clip, K=K, seed=args.seed)

        all_results[tag] = result
        print(f"  Final epoch loss: {result['epoch_losses'][-1]:.1f}, "
              f"up_norm: {result['up_norm']:.4f}")

    # --- plot all loss curves ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Per-step loss (all configs)
    ax = axes[0]
    for tag, res in all_results.items():
        ax.plot(res["step_losses"], alpha=0.7, label=tag)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss")
    ax.set_title("Per-step loss")
    ax.legend(fontsize=7)
    ax.set_yscale("log")

    # 2. Per-epoch loss (all configs)
    ax = axes[1]
    for tag, res in all_results.items():
        ax.plot(range(1, num_epochs + 1), res["epoch_losses"], "o-", label=tag)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg loss")
    ax.set_title("Per-epoch avg loss")
    ax.legend(fontsize=7)
    ax.set_yscale("log")

    # 3. Final loss vs config (bar chart)
    ax = axes[2]
    tags = list(all_results.keys())
    final_losses = [all_results[t]["epoch_losses"][-1] for t in tags]
    colors = plt.cm.viridis(np.linspace(0, 1, len(tags)))
    bars = ax.bar(range(len(tags)), final_losses, color=colors)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Final epoch loss")
    ax.set_title("Final loss by config")
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(str(root / "sweep_results.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- summary table ---
    print(f"\n{'='*70}")
    print(f"{'Config':<25} {'Final Loss':>12} {'Up Norm':>10} {'Down Norm':>10}")
    print(f"{'-'*70}")
    ranked = sorted(all_results.items(), key=lambda x: x[1]["epoch_losses"][-1])
    for tag, res in ranked:
        print(f"{tag:<25} {res['epoch_losses'][-1]:>12.1f} "
              f"{res['up_norm']:>10.4f} {res['down_norm']:>10.4f}")
    print(f"{'-'*70}")
    best_tag = ranked[0][0]
    print(f"Best config: {best_tag} (loss={ranked[0][1]['epoch_losses'][-1]:.1f})")
    print(f"\nResults saved to {root}")
    print(f"  - Sweep plot: {root / 'sweep_results.png'}")

    # --- save raw results ---
    summary = {}
    for tag, res in all_results.items():
        summary[tag] = {
            "epoch_losses": res["epoch_losses"],
            "step_losses": res["step_losses"],
            "up_norm": res["up_norm"],
            "down_norm": res["down_norm"],
            "final_loss": res["epoch_losses"][-1],
        }
    with open(str(root / "sweep_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ===================================================================
    # Phase 2: Full-dataset training with best config
    # ===================================================================
    parts = best_tag.split("_")
    best_lr = float(parts[0].replace("lr", ""))
    best_accum = int(parts[1].replace("accum", ""))
    best_rank = int(parts[2].replace("r", ""))

    # load full dataset if configured, otherwise reuse sweep data
    if full_data_config is not None:
        full_dataset = get_dataset(**full_data_config)
        full_num_images = len(full_dataset)
        full_images = full_dataset.get_data(full_num_images, 0)
        full_y = operator.measure(full_images)
    else:
        full_num_images = num_images
        full_images = images
        full_y = y

    print(f"\n{'='*60}")
    print(f"Phase 2: Full training with best config")
    print(f"  Config: lr={best_lr}, grad_accum={best_accum}, rank={best_rank}")
    print(f"  Data: {full_num_images} images, {full_num_epochs} epochs")
    print(f"{'='*60}\n")

    full_save_path = str(root / "lora_final.pt")
    full_result = train_one_config(
        model, full_images, full_y, operator, sampler,
        lr=best_lr, grad_accum=best_accum, lora_rank=best_rank,
        lora_alpha=lora_alpha, y_channels=y_channels,
        target_modules=target_modules,
        num_epochs=full_num_epochs, lambda_kl=lambda_kl,
        grad_clip=grad_clip, K=K, seed=args.seed,
        save_path=full_save_path)

    # plot full training loss curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(full_result["step_losses"], alpha=0.3, label="per step")
    sl = full_result["step_losses"]
    if len(sl) > 10:
        window = max(len(sl) // 20, 5)
        smoothed = np.convolve(sl, np.ones(window)/window, mode='valid')
        ax1.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
                 label=f"smooth (w={window})")
    ax1.set_xlabel("Optimizer step")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Full training: lr={best_lr}, accum={best_accum}, r={best_rank}")
    ax1.legend()
    ax1.set_yscale("log")

    ax2.plot(range(1, full_num_epochs + 1), full_result["epoch_losses"], "o-")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Avg loss")
    ax2.set_title("Per-epoch avg loss")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(str(root / "full_training_loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFull training complete!")
    print(f"  Final loss: {full_result['epoch_losses'][-1]:.1f}")
    print(f"  LoRA saved: {full_save_path}")
    print(f"  Loss curve: {root / 'full_training_loss.png'}")


if __name__ == "__main__":
    main()
