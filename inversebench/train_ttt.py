"""
Unified TTT-LoRA training script for InverseBench.

Trains a LoRA adapter on a pretrained diffusion model using one of three
methods (DRaFT-direct, DPO, GRPO) so that plain unconditional diffusion
at inference time produces reconstructions for the chosen inverse problem.

Splits the dataset into train/eval sets, trains on train_pct%, then
evaluates on the held-out remainder comparing LoRA-adapted vs plain diffusion.

All output (log file, loss curves, checkpoints, eval results) is saved to
one directory so you can periodically check progress.

Stability features:
  - LR warmup (linear) + cosine decay
  - EMA of LoRA weights (saved as final checkpoint)
  - NaN / loss-spike detection with automatic rollback
  - Gradient norm logging and plotting
  - Early stopping (patience-based) with best checkpoint tracking
  - Weight decay regularization

Usage:
    python train_ttt.py \
        problem=inv-scatter pretrain=inv-scatter \
        +ttt.method=direct +ttt.train_pct=80 \
        +ttt.lr=1e-3 +ttt.lora_rank=64 +ttt.num_epochs=10
"""

import json
import pickle
import math
import time
import yaml
import torch
import torch.nn.functional as TF
import numpy as np
import tqdm
import hydra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

from utils.helper import open_url
from utils.scheduler import Scheduler
from algo.lora import (
    apply_conditioned_lora, remove_lora,
    get_lora_params, frozen_tweedie, save_lora, load_conditioned_lora,
)


# ---------------------------------------------------------------------------
# Logger that writes to both stdout and a file
# ---------------------------------------------------------------------------

class Logger:
    """Tees all print output to a log file and keeps structured training state."""

    def __init__(self, root):
        self.root = Path(root)
        self.log_path = self.root / "train.log"
        self.progress_path = self.root / "progress.json"
        self.log_file = open(str(self.log_path), "w")
        self.start_time = time.time()

        self.state = {
            "status": "initializing",
            "start_time": datetime.now().isoformat(),
            "step_losses": [],
            "epoch_losses": [],
            "epoch_times": [],
            "grad_norms": [],
            "current_epoch": 0,
            "total_epochs": 0,
            "elapsed_sec": 0,
        }

    def log(self, msg):
        ts = f"[{timedelta(seconds=int(time.time() - self.start_time))}]"
        line = f"{ts} {msg}"
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def update_progress(self, **kwargs):
        self.state["elapsed_sec"] = int(time.time() - self.start_time)
        self.state.update(kwargs)
        with open(str(self.progress_path), "w") as f:
            json.dump(self.state, f, indent=2)

    def close(self):
        self.log_file.close()


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup for `warmup_steps`, then cosine decay to `min_lr_ratio * base_lr`."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, 1)
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            factor = self.step_count / max(self.warmup_steps, 1)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)
            factor = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * factor

    @property
    def current_lr(self):
        return self.optimizer.param_groups[0]['lr']


# ---------------------------------------------------------------------------
# EMA of LoRA weights
# ---------------------------------------------------------------------------

class EMAWeights:
    """Exponential moving average of model parameters.

    Call update() after each optimizer step.
    Call apply() before saving / evaluation to use EMA weights.
    Call restore() to switch back to training weights.
    """

    def __init__(self, params, decay=0.995):
        self.decay = decay
        self.params = list(params)
        self.shadow = [p.data.clone() for p in self.params]
        self.enabled = decay > 0 and decay < 1.0
        self.backup = None

    def update(self):
        if not self.enabled:
            return
        for shadow, param in zip(self.shadow, self.params):
            shadow.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self):
        """Replace model params with EMA params. Call restore() to undo."""
        if not self.enabled:
            return
        self.backup = [p.data.clone() for p in self.params]
        for shadow, param in zip(self.shadow, self.params):
            param.data.copy_(shadow)

    def restore(self):
        """Restore original (non-EMA) params."""
        if not self.enabled or self.backup is None:
            return
        for backup, param in zip(self.backup, self.params):
            param.data.copy_(backup)


# ---------------------------------------------------------------------------
# Stability monitor: NaN detection, loss spike rollback, early stopping
# ---------------------------------------------------------------------------

class StabilityMonitor:
    """Tracks training health and prevents mode collapse.

    Features:
    - NaN/Inf detection
    - Loss spike detection (> threshold x rolling average)
    - Best checkpoint tracking with rollback
    - Early stopping (patience epochs without improvement)
    - Gradient norm logging
    """

    def __init__(self, lora_params, patience=3, spike_threshold=5.0):
        self.lora_params = list(lora_params)
        self.patience = patience
        self.spike_threshold = spike_threshold
        self.grad_norms = []
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_state = None
        self._recent_losses = []

    def compute_grad_norm(self):
        """Compute and log the total gradient norm (before clipping)."""
        total_norm = 0.0
        for p in self.lora_params:
            if p.grad is not None:
                total_norm += p.grad.data.float().norm().item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
        return total_norm

    def is_loss_ok(self, loss_val):
        """Returns False if loss is NaN, Inf, or a spike."""
        if math.isnan(loss_val) or math.isinf(loss_val):
            return False
        if len(self._recent_losses) >= 10:
            rolling_avg = np.mean(self._recent_losses[-50:])
            if rolling_avg > 0 and loss_val > self.spike_threshold * rolling_avg:
                return False
        self._recent_losses.append(loss_val)
        if len(self._recent_losses) > 100:
            self._recent_losses = self._recent_losses[-50:]
        return True

    def save_best(self, epoch_avg_loss):
        """Track best epoch loss. Returns True if this was a new best."""
        if epoch_avg_loss < self.best_loss:
            self.best_loss = epoch_avg_loss
            self.best_state = [p.data.clone() for p in self.lora_params]
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False

    def should_stop(self):
        """Returns True if no improvement for `patience` epochs."""
        return self.patience > 0 and self.epochs_without_improvement >= self.patience

    def rollback_to_best(self):
        """Restore parameters to best checkpoint."""
        if self.best_state is not None:
            for p, s in zip(self.lora_params, self.best_state):
                p.data.copy_(s)
            return True
        return False


# ---------------------------------------------------------------------------
# Loss curve plotting (called every epoch so you can check live)
# ---------------------------------------------------------------------------

def save_loss_curves(step_losses, epoch_avg_losses, grad_norms, root):
    """Save loss curve plot + raw data. Called every epoch."""
    with open(str(root / "losses.json"), "w") as f:
        json.dump({
            "step_losses": step_losses,
            "epoch_avg_losses": epoch_avg_losses,
            "grad_norms": grad_norms,
        }, f)

    if not epoch_avg_losses:
        return

    n_plots = 3 if grad_norms else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 2:
        axes = list(axes)

    # Per-step loss
    ax = axes[0]
    ax.plot(step_losses, alpha=0.3, linewidth=0.5, label="per step")
    if len(step_losses) > 10:
        window = max(len(step_losses) // 20, 5)
        smoothed = np.convolve(step_losses, np.ones(window)/window, mode='valid')
        ax.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
                linewidth=1.5, label=f"smoothed (w={window})")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss")
    ax.set_title("Per-step loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-epoch loss
    ax = axes[1]
    epochs = list(range(1, len(epoch_avg_losses) + 1))
    ax.plot(epochs, epoch_avg_losses, "o-", linewidth=2, markersize=6)
    for e, l in zip(epochs, epoch_avg_losses):
        ax.annotate(f"{l:.4f}", (e, l), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg loss")
    ax.set_title("Per-epoch avg loss")
    ax.grid(True, alpha=0.3)

    # Gradient norms
    if grad_norms and n_plots == 3:
        ax = axes[2]
        ax.plot(grad_norms, alpha=0.4, linewidth=0.5, label="per step")
        if len(grad_norms) > 10:
            window = max(len(grad_norms) // 20, 5)
            smoothed = np.convolve(grad_norms, np.ones(window)/window,
                                   mode='valid')
            ax.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
                    linewidth=1.5, label=f"smoothed (w={window})")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Gradient norm")
        ax.set_title("Gradient norms (pre-clip)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(root / "loss_curve.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Sigma sampling for DPO/GRPO (log-uniform, standard EDM)
# ---------------------------------------------------------------------------

def sample_sigma(net, batch_size, device):
    """Sample noise levels log-uniformly between sigma_min and sigma_max."""
    sigma_min = net.sigma_min
    sigma_max = net.sigma_max
    log_sigma = torch.rand(batch_size, device=device) * (
        math.log(sigma_max) - math.log(sigma_min)
    ) + math.log(sigma_min)
    return log_sigma.exp()


# ---------------------------------------------------------------------------
# DPS prefix (no-grad) — adapted for InverseBench scheduler API
# ---------------------------------------------------------------------------

def dps_prefix(net, scheduler, forward_op, x, observation, num_steps,
               guidance_scale=1.0):
    """Run first num_steps of DPS (ODE mode, no LoRA gradients).

    Uses torch.enable_grad() so this works even when called from a
    @torch.no_grad() context (e.g., DPO/GRPO candidate generation).
    """
    for i in range(num_steps):
        x_cur = x.detach().requires_grad_(True)
        sigma = scheduler.sigma_steps[i]
        scaling = scheduler.scaling_steps[i]
        factor = scheduler.factor_steps[i]
        scaling_factor = scheduler.scaling_factor[i]

        with torch.enable_grad():
            denoised = net(x_cur / scaling,
                           torch.as_tensor(sigma).to(x_cur.device))
            gradient, loss_scale = forward_op.gradient(
                denoised, observation, return_loss=True)
            ll_grad = torch.autograd.grad(denoised, x_cur, gradient)[0]
        ll_grad = ll_grad * 0.5 / torch.sqrt(loss_scale)

        with torch.no_grad():
            score = (denoised - x_cur / scaling) / sigma ** 2 / scaling
            x = x_cur * scaling_factor + factor * score * 0.5
            x = x - ll_grad * guidance_scale
            if torch.isnan(x).any():
                break

    return x.detach()


# ---------------------------------------------------------------------------
# Plain diffusion sampling (no guidance) — used for evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def plain_diffusion_sample(net, scheduler, num_samples, device):
    """Run unconditional reverse diffusion (Euler ODE)."""
    x = torch.randn(num_samples, net.img_channels, net.img_resolution,
                     net.img_resolution, device=device) * scheduler.sigma_max
    for i in range(scheduler.num_steps):
        sigma = scheduler.sigma_steps[i]
        scaling = scheduler.scaling_steps[i]
        factor = scheduler.factor_steps[i]
        scaling_factor = scheduler.scaling_factor[i]
        denoised = net(x / scaling, torch.as_tensor(sigma).to(device))
        score = (denoised - x / scaling) / sigma ** 2 / scaling
        x = x * scaling_factor + factor * score * 0.5
    return x


# ---------------------------------------------------------------------------
# DPO pair generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_dpo_pairs(net, scheduler, forward_op, images, measurements,
                       guidance_scale=1.0, prefix_steps=None):
    if prefix_steps is None:
        prefix_steps = scheduler.num_steps
    device = next(net.parameters()).device
    winners, losers, pair_y = [], [], []

    for idx in tqdm.trange(len(images), desc="Generating DPO pairs"):
        y_i = measurements[idx:idx+1].to(device)
        results = []
        for _ in range(2):
            x = torch.randn(1, net.img_channels, net.img_resolution,
                             net.img_resolution, device=device) * scheduler.sigma_max
            x = dps_prefix(net, scheduler, forward_op, x, y_i,
                           prefix_steps, guidance_scale)
            loss = forward_op.loss(x, y_i)
            results.append((x, loss.item()))
        if results[0][1] <= results[1][1]:
            winners.append(results[0][0]); losers.append(results[1][0])
        else:
            winners.append(results[1][0]); losers.append(results[0][0])
        pair_y.append(y_i)

    return torch.cat(winners), torch.cat(losers), torch.cat(pair_y)


# ---------------------------------------------------------------------------
# GRPO candidate generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_grpo_candidates(net, scheduler, forward_op, images, measurements,
                             num_candidates=6, guidance_scale=1.0,
                             prefix_steps=None):
    if prefix_steps is None:
        prefix_steps = scheduler.num_steps
    device = next(net.parameters()).device
    all_cands, all_advs, all_y = [], [], []

    for idx in tqdm.trange(len(images), desc="Generating GRPO candidates"):
        y_i = measurements[idx:idx+1].to(device)
        cands, rewards = [], []
        for _ in range(num_candidates):
            x = torch.randn(1, net.img_channels, net.img_resolution,
                             net.img_resolution, device=device) * scheduler.sigma_max
            x = dps_prefix(net, scheduler, forward_op, x, y_i,
                           prefix_steps, guidance_scale)
            loss = forward_op.loss(x, y_i)
            cands.append(x); rewards.append(-loss)
        cands = torch.cat(cands, dim=0)
        rewards = torch.cat(rewards, dim=0)
        std = rewards.std()
        advs = torch.zeros_like(rewards) if std < 1e-8 else (rewards - rewards.mean()) / std
        all_cands.append(cands); all_advs.append(advs)
        all_y.append(y_i.expand(num_candidates, -1, -1, -1))

    return torch.cat(all_cands), torch.cat(all_advs), torch.cat(all_y)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dpo_loss_batch(net, lora_modules, winners, losers, beta_dpo=5000.0,
                   store=None, y_batch=None):
    device = winners.device
    B = winners.shape[0]
    sigma = sample_sigma(net, B, device)
    sigma_bc = sigma.view(-1, *([1] * (winners.ndim - 1)))
    noise = torch.randn_like(winners)
    x_w_noisy = winners + sigma_bc * noise
    x_l_noisy = losers + sigma_bc * noise

    if store is not None and y_batch is not None:
        store.set(y_batch)
    net.requires_grad_(True)
    D_w = net(x_w_noisy, sigma)
    D_l = net(x_l_noisy, sigma)

    eps_w = (x_w_noisy - D_w) / sigma_bc
    eps_l = (x_l_noisy - D_l) / sigma_bc
    dims = list(range(1, winners.ndim))
    model_mse_w = (eps_w - noise).pow(2).mean(dim=dims)
    model_mse_l = (eps_l - noise).pow(2).mean(dim=dims)

    D_w_ref = frozen_tweedie(net, lora_modules, x_w_noisy, sigma)
    D_l_ref = frozen_tweedie(net, lora_modules, x_l_noisy, sigma)
    eps_w_ref = (x_w_noisy - D_w_ref) / sigma_bc
    eps_l_ref = (x_l_noisy - D_l_ref) / sigma_bc
    ref_mse_w = (eps_w_ref - noise).pow(2).mean(dim=dims)
    ref_mse_l = (eps_l_ref - noise).pow(2).mean(dim=dims)

    inside_term = -0.5 * beta_dpo * ((model_mse_w - model_mse_l) - (ref_mse_w - ref_mse_l))
    loss = -TF.logsigmoid(inside_term).mean()
    with torch.no_grad():
        acc = (inside_term > 0).float().mean()
    return loss, acc


def grpo_loss_batch(net, lora_modules, candidates, advantages,
                    lambda_kl=0.0, adv_clip=5.0, store=None, y_batch=None):
    device = candidates.device
    B = candidates.shape[0]
    sigma = sample_sigma(net, B, device)
    sigma_bc = sigma.view(-1, *([1] * (candidates.ndim - 1)))
    noise = torch.randn_like(candidates)
    x_noisy = candidates + sigma_bc * noise

    if store is not None and y_batch is not None:
        store.set(y_batch)
    net.requires_grad_(True)
    D_x = net(x_noisy, sigma)

    noise_pred = (x_noisy - D_x) / sigma_bc
    dims = list(range(1, candidates.ndim))
    mse_per_sample = (noise_pred - noise).pow(2).mean(dim=dims)
    clipped_advs = advantages.clamp(-adv_clip, adv_clip)
    policy_loss = (clipped_advs * mse_per_sample).mean()

    kl_loss = torch.tensor(0.0, device=device)
    if lambda_kl > 0:
        D_x_ref = frozen_tweedie(net, lora_modules, x_noisy, sigma)
        noise_pred_ref = (x_noisy - D_x_ref) / sigma_bc
        kl_loss = (noise_pred - noise_pred_ref.detach()).pow(2).mean()
    return policy_loss + lambda_kl * kl_loss


# ---------------------------------------------------------------------------
# Held-out evaluation: LoRA vs plain diffusion
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_held_out(net, scheduler, forward_op, eval_images,
                      eval_measurements, lora_modules, store, root, logger):
    device = eval_images.device
    n_eval = len(eval_images)
    logger.log(f"{'='*60}")
    logger.log(f"Evaluating on {n_eval} held-out samples...")
    logger.log(f"{'='*60}")

    lora_losses, plain_losses = [], []
    lora_recons, plain_recons = [], []

    # LoRA ON
    t0 = time.time()
    for idx in tqdm.trange(n_eval, desc="Eval (LoRA)"):
        y_i = eval_measurements[idx:idx+1]
        if store is not None:
            store.set(y_i)
        recon = plain_diffusion_sample(net, scheduler, 1, device)
        loss = forward_op.loss(recon, y_i).item()
        lora_losses.append(loss)
        lora_recons.append(recon.cpu())
    logger.log(f"  LoRA eval done in {time.time()-t0:.0f}s")

    # LoRA OFF (baseline)
    saved_scalings = [m.scaling for m in lora_modules]
    for m in lora_modules:
        m.scaling = 0.0

    t0 = time.time()
    for idx in tqdm.trange(n_eval, desc="Eval (plain)"):
        y_i = eval_measurements[idx:idx+1]
        if store is not None:
            store.set(y_i)
        recon = plain_diffusion_sample(net, scheduler, 1, device)
        loss = forward_op.loss(recon, y_i).item()
        plain_losses.append(loss)
        plain_recons.append(recon.cpu())
    logger.log(f"  Plain eval done in {time.time()-t0:.0f}s")

    for m, s in zip(lora_modules, saved_scalings):
        m.scaling = s
    if store is not None:
        store.clear()

    # Relative L2
    lora_rel_l2, plain_rel_l2 = [], []
    for idx in range(n_eval):
        gt = eval_images[idx:idx+1].cpu()
        gt_norm = gt.flatten().norm()
        if gt_norm > 1e-8:
            lora_rel_l2.append((lora_recons[idx] - gt).flatten().norm().item() / gt_norm.item())
            plain_rel_l2.append((plain_recons[idx] - gt).flatten().norm().item() / gt_norm.item())

    # Build results
    results = {
        "n_eval": n_eval,
        "lora_measurement_loss": {"mean": float(np.mean(lora_losses)), "std": float(np.std(lora_losses)),
                                  "per_sample": [float(x) for x in lora_losses]},
        "plain_measurement_loss": {"mean": float(np.mean(plain_losses)), "std": float(np.std(plain_losses)),
                                   "per_sample": [float(x) for x in plain_losses]},
    }
    if lora_rel_l2:
        results["lora_relative_l2"] = {"mean": float(np.mean(lora_rel_l2)), "std": float(np.std(lora_rel_l2))}
        results["plain_relative_l2"] = {"mean": float(np.mean(plain_rel_l2)), "std": float(np.std(plain_rel_l2))}

    lm, pm = results['lora_measurement_loss']['mean'], results['plain_measurement_loss']['mean']
    logger.log(f"")
    logger.log(f"--- Held-out Results ({n_eval} samples) ---")
    logger.log(f"  Measurement loss (LoRA):  {lm:.6f} +/- {results['lora_measurement_loss']['std']:.6f}")
    logger.log(f"  Measurement loss (plain): {pm:.6f} +/- {results['plain_measurement_loss']['std']:.6f}")
    if pm > 1e-10:
        logger.log(f"  Improvement: {(pm - lm) / pm * 100:+.1f}%")
    if lora_rel_l2:
        logger.log(f"  Relative L2 (LoRA):  {results['lora_relative_l2']['mean']:.6f} +/- {results['lora_relative_l2']['std']:.6f}")
        logger.log(f"  Relative L2 (plain): {results['plain_relative_l2']['mean']:.6f} +/- {results['plain_relative_l2']['std']:.6f}")

    # Save
    with open(str(root / "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save({
        "lora_recons": torch.cat(lora_recons),
        "plain_recons": torch.cat(plain_recons),
        "eval_images": eval_images.cpu(),
        "eval_measurements": eval_measurements.cpu(),
        "lora_losses": lora_losses,
        "plain_losses": plain_losses,
    }, str(root / "eval_data.pt"))
    logger.log(f"  Saved eval_results.json + eval_data.pt")

    # Comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(lora_losses, bins=20, alpha=0.6, label="LoRA")
    axes[0].hist(plain_losses, bins=20, alpha=0.6, label="Plain")
    axes[0].set_xlabel("Measurement loss"); axes[0].set_ylabel("Count")
    axes[0].set_title("Measurement loss distribution"); axes[0].legend()
    if lora_rel_l2:
        axes[1].hist(lora_rel_l2, bins=20, alpha=0.6, label="LoRA")
        axes[1].hist(plain_rel_l2, bins=20, alpha=0.6, label="Plain")
        axes[1].set_xlabel("Relative L2"); axes[1].set_ylabel("Count")
        axes[1].set_title("Relative L2 distribution"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(str(root / "eval_comparison.png"), dpi=150)
    plt.close()

    return results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_subset(dataset, indices, forward_op, device):
    images, measurements = [], []
    for i in tqdm.tqdm(indices, desc="Loading data"):
        sample = dataset[int(i)]
        target = sample['target']
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target.copy())
        target = target.float().to(device)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        images.append(target)
        obs = forward_op({'target': target.unsqueeze(0)})
        measurements.append(obs)
    return torch.stack(images), torch.cat(measurements)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.tf32:
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # --- TTT config ---
    ttt = OmegaConf.to_container(config.get("ttt", {}), resolve=True)
    method = ttt.get("method", "direct")
    train_pct = ttt.get("train_pct", 80)
    lora_rank = ttt.get("lora_rank", 64)
    lora_alpha = ttt.get("lora_alpha", 1.0)
    y_channels = ttt.get("y_channels", 0)
    target_modules = ttt.get("target_modules", "all")
    lr = ttt.get("lr", 1e-3)
    grad_clip = ttt.get("grad_clip", 1.0)
    num_epochs = ttt.get("num_epochs", 10)
    save_dir = ttt.get("save_dir", "exps/ttt")
    sched_cfg = ttt.get("diffusion_scheduler_config",
                        {"num_steps": 200, "schedule": "vp",
                         "timestep": "vp", "scaling": "vp"})
    K = ttt.get("K", 1)
    lambda_kl = ttt.get("lambda_kl", 0.01)
    grad_accum = ttt.get("grad_accum", 1)
    guidance_scale = ttt.get("guidance_scale", 1.0)
    beta_dpo = ttt.get("beta_dpo", 5000.0)
    dpo_batch_size = ttt.get("dpo_batch_size", 4)
    num_candidates = ttt.get("num_candidates", 6)
    adv_clip = ttt.get("adv_clip", 5.0)
    grpo_lambda_kl = ttt.get("grpo_lambda_kl", 0.0)
    grpo_batch_size = ttt.get("grpo_batch_size", 8)

    # Stability config
    warmup_steps = ttt.get("warmup_steps", 50)
    weight_decay = ttt.get("weight_decay", 1e-4)
    ema_decay = ttt.get("ema_decay", 0.995)
    loss_spike_threshold = ttt.get("loss_spike_threshold", 5.0)
    patience = ttt.get("patience", 3)
    min_lr_ratio = ttt.get("min_lr_ratio", 0.1)

    # --- Output directory ---
    problem_name = config.problem.get("name", "unknown")
    root = Path(save_dir) / f"{problem_name}_{method}_{train_pct}pct"
    root.mkdir(parents=True, exist_ok=True)

    # --- Logger ---
    logger = Logger(root)
    logger.log(f"TTT-LoRA training")
    logger.log(f"  method={method}, train_pct={train_pct}%, rank={lora_rank}, "
               f"epochs={num_epochs}, lr={lr}")
    logger.log(f"  stability: warmup={warmup_steps}, wd={weight_decay}, "
               f"ema={ema_decay}, spike_thresh={loss_spike_threshold}, "
               f"patience={patience}")
    logger.log(f"  output: {root}")
    logger.log(f"  device: {device}")
    logger.update_progress(status="loading", total_epochs=num_epochs)

    # Save config
    with open(str(root / "config.yaml"), "w") as f:
        yaml.safe_dump(OmegaConf.to_container(config, resolve=True), f,
                       default_flow_style=False)

    # --- 1. Load components ---
    logger.log("Loading forward operator...")
    forward_op = instantiate(config.problem.model, device=device)

    logger.log("Loading training dataset...")
    train_dataset = instantiate(config.pretrain.data)

    logger.log("Loading pretrained model...")
    ckpt_path = config.problem.prior
    try:
        with open_url(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
            net = ckpt['ema'].to(device)
    except Exception:
        net = instantiate(config.pretrain.model)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'ema' in ckpt.keys():
            net.load_state_dict(ckpt['ema'])
        else:
            net.load_state_dict(ckpt['net'])
        net = net.to(device)
    del ckpt
    net.eval()
    logger.log(f"  Model: {type(net).__name__} -> {type(net.model).__name__}")
    logger.log(f"  Resolution: {net.img_resolution}, Channels: {net.img_channels}")

    # --- 2. Train/eval split ---
    N = len(train_dataset)
    all_indices = np.random.permutation(N)
    n_train = max(1, int(N * train_pct / 100))
    n_eval = N - n_train
    train_indices = all_indices[:n_train]
    eval_indices = all_indices[n_train:]

    logger.log(f"Split: {N} total -> {n_train} train, {n_eval} eval")
    np.savez(str(root / "split_indices.npz"),
             train=train_indices, eval=eval_indices)

    logger.log("Loading train split...")
    images, measurements = load_subset(train_dataset, train_indices, forward_op, device)
    logger.log(f"  images={images.shape}, measurements={measurements.shape}")

    if n_eval > 0:
        logger.log("Loading eval split...")
        eval_images, eval_measurements = load_subset(
            train_dataset, eval_indices, forward_op, device)
        logger.log(f"  eval_images={eval_images.shape}")

    # --- 3. Scheduler ---
    scheduler = Scheduler(**sched_cfg)
    total_steps = scheduler.num_steps
    prefix_steps = max(total_steps - K, 0)

    # --- 4. Pre-LoRA generation for DPO/GRPO ---
    if method == "dpo":
        logger.log(f"Generating DPO pairs ({n_train} images x 2)...")
        logger.update_progress(status="generating_pairs")
        winners, losers, pair_y = generate_dpo_pairs(
            net, scheduler, forward_op, images, measurements,
            guidance_scale=guidance_scale, prefix_steps=prefix_steps)
        logger.log(f"  {len(winners)} pairs ready")
    elif method == "grpo":
        logger.log(f"Generating GRPO candidates ({n_train} images x {num_candidates})...")
        logger.update_progress(status="generating_candidates")
        candidates, advantages, cand_y = generate_grpo_candidates(
            net, scheduler, forward_op, images, measurements,
            num_candidates=num_candidates, guidance_scale=guidance_scale,
            prefix_steps=prefix_steps)
        logger.log(f"  {len(candidates)} candidates, "
                    f"{(advantages > 0).sum()} positive advantage")

    # --- 5. Apply LoRA ---
    lora_modules, store = apply_conditioned_lora(
        net, rank=lora_rank, alpha=lora_alpha, y_channels=y_channels,
        target_modules=target_modules)
    lora_params = get_lora_params(lora_modules)
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    n_params = sum(p.numel() for p in lora_params)
    logger.log(f"LoRA: {len(lora_modules)} modules, {n_params:,} params")

    # --- 5b. Stability objects ---
    if method == "direct":
        steps_per_epoch = math.ceil(n_train / grad_accum)
    elif method == "dpo":
        steps_per_epoch = math.ceil(len(winners) / dpo_batch_size)
    elif method == "grpo":
        steps_per_epoch = math.ceil(len(candidates) / grpo_batch_size)
    else:
        steps_per_epoch = n_train
    total_opt_steps = steps_per_epoch * num_epochs

    lr_scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=warmup_steps,
        total_steps=total_opt_steps, min_lr_ratio=min_lr_ratio)

    ema = EMAWeights(lora_params, decay=ema_decay)

    monitor = StabilityMonitor(
        lora_params, patience=patience,
        spike_threshold=loss_spike_threshold)

    logger.log(f"LR schedule: warmup {warmup_steps} steps, cosine decay over "
               f"{total_opt_steps} total steps, min ratio={min_lr_ratio}")
    if ema.enabled:
        logger.log(f"EMA: decay={ema_decay}")
    logger.log(f"Early stopping: patience={patience} epochs")

    # --- 6. Train ---
    step_losses = []
    epoch_avg_losses = []
    logger.update_progress(status="training")

    train_kwargs = dict(
        lr_scheduler=lr_scheduler, ema=ema, monitor=monitor,
        step_losses=step_losses, epoch_avg_losses=epoch_avg_losses)

    if method == "direct":
        _train_direct(net, scheduler, forward_op, optimizer, lora_modules,
                      lora_params, store, images, measurements, root, logger,
                      num_epochs=num_epochs, K=K, lambda_kl=lambda_kl,
                      grad_accum=grad_accum, grad_clip=grad_clip,
                      guidance_scale=guidance_scale,
                      **train_kwargs)
    elif method == "dpo":
        _train_dpo(net, optimizer, lora_modules, lora_params, store,
                   winners, losers, pair_y, root, logger,
                   num_epochs=num_epochs, beta_dpo=beta_dpo,
                   batch_size=dpo_batch_size, grad_clip=grad_clip,
                   **train_kwargs)
    elif method == "grpo":
        _train_grpo(net, optimizer, lora_modules, lora_params, store,
                    candidates, advantages, cand_y, root, logger,
                    num_epochs=num_epochs, lambda_kl=grpo_lambda_kl,
                    adv_clip=adv_clip, batch_size=grpo_batch_size,
                    grad_clip=grad_clip,
                    **train_kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    # --- 7. Save final (use EMA weights if enabled) ---
    ema.apply()
    save_lora(lora_modules, str(root / "lora_final.pt"),
              metadata={"method": method, "problem": problem_name,
                        "train_pct": train_pct, "epochs": num_epochs,
                        "lora_rank": lora_rank, "lora_alpha": lora_alpha,
                        "ema": ema.enabled})
    logger.log(f"Saved lora_final.pt" + (" (EMA weights)" if ema.enabled else ""))

    # --- 8. Evaluate (with EMA weights applied) ---
    if n_eval > 0:
        logger.update_progress(status="evaluating")
        evaluate_held_out(net, scheduler, forward_op,
                          eval_images, eval_measurements,
                          lora_modules, store, root, logger)

    ema.restore()
    remove_lora(net)
    logger.update_progress(status="done")
    logger.log(f"")
    logger.log(f"All done! Outputs in: {root}")
    logger.log(f"  train.log          - this log file")
    logger.log(f"  progress.json      - check training status anytime")
    logger.log(f"  losses.json        - raw loss numbers + grad norms")
    logger.log(f"  loss_curve.png     - loss + grad norm plots (updated every epoch)")
    logger.log(f"  eval_results.json  - LoRA vs plain comparison")
    logger.log(f"  eval_comparison.png- histogram of eval metrics")
    logger.log(f"  eval_data.pt       - all reconstructions")
    logger.log(f"  lora_final.pt      - final LoRA checkpoint (EMA)")
    logger.log(f"  lora_best.pt       - best epoch checkpoint")
    logger.log(f"  lora_epoch*.pt     - per-epoch checkpoints")
    logger.log(f"  split_indices.npz  - train/eval index split")
    logger.close()


# ---------------------------------------------------------------------------
# Direct training
# ---------------------------------------------------------------------------

def _train_direct(net, scheduler, forward_op, optimizer, lora_modules,
                  lora_params, store, images, measurements, root, logger, *,
                  num_epochs, K, lambda_kl, grad_accum, grad_clip,
                  guidance_scale, lr_scheduler, ema, monitor,
                  step_losses, epoch_avg_losses):

    total_steps = scheduler.num_steps
    prefix_steps = max(total_steps - K, 0)
    n = len(images)
    device = images.device

    for epoch in range(num_epochs):
        t_epoch = time.time()
        epoch_loss = 0.0
        epoch_steps = 0
        n_nan_skips = 0
        order = np.random.permutation(n)
        pbar = tqdm.tqdm(order, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()
        accum_loss = 0.0

        for i, img_idx in enumerate(pbar):
            y_i = measurements[img_idx:img_idx+1]
            x = torch.randn(1, net.img_channels, net.img_resolution,
                             net.img_resolution, device=device) * scheduler.sigma_max
            if prefix_steps > 0:
                x = dps_prefix(net, scheduler, forward_op, x, y_i,
                               prefix_steps, guidance_scale)

            sigma_b = scheduler.sigma_steps[prefix_steps]
            scaling_b = scheduler.scaling_steps[prefix_steps]

            if store is not None:
                store.set(y_i)
            net.requires_grad_(True)
            x0_hat = net(x / scaling_b, torch.as_tensor(sigma_b).to(device))

            reward_loss = forward_op.loss(x0_hat, y_i).mean()
            kl_loss = torch.tensor(0.0, device=device)
            if lambda_kl > 0:
                x0_frozen = frozen_tweedie(net, lora_modules, x / scaling_b,
                                           torch.as_tensor(sigma_b).to(device))
                kl_loss = ((x0_hat - x0_frozen) ** 2).mean()

            total_loss = (reward_loss + lambda_kl * kl_loss) / grad_accum
            loss_val = total_loss.item() * grad_accum

            # NaN / spike detection — skip bad steps
            if not monitor.is_loss_ok(loss_val):
                n_nan_skips += 1
                optimizer.zero_grad()
                net.requires_grad_(False)
                pbar.set_postfix(loss="SKIP", nan=n_nan_skips)
                if n_nan_skips > n * 0.1:
                    logger.log(f"  WARNING: >10% NaN/spike steps, rolling back")
                    monitor.rollback_to_best()
                    n_nan_skips = 0
                continue

            total_loss.backward()
            net.requires_grad_(False)

            accum_loss += loss_val
            epoch_loss += loss_val
            epoch_steps += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}",
                             lr=f"{lr_scheduler.current_lr:.2e}")

            if (i + 1) % grad_accum == 0 or (i + 1) == n:
                grad_norm = monitor.compute_grad_norm()
                torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
                optimizer.step()
                lr_scheduler.step()
                ema.update()
                optimizer.zero_grad()
                n_in_window = ((i % grad_accum) + 1) if (i + 1) == n else grad_accum
                step_losses.append(accum_loss / n_in_window)
                accum_loss = 0.0

        epoch_time = time.time() - t_epoch
        avg = epoch_loss / max(epoch_steps, 1)
        epoch_avg_losses.append(avg)

        is_best = monitor.save_best(avg)
        best_tag = " [BEST]" if is_best else ""
        recent_gnorms = monitor.grad_norms[-max(epoch_steps, 1):] if monitor.grad_norms else [0.0]
        logger.log(f"Epoch {epoch+1}/{num_epochs}: avg_loss={avg:.6f}, "
                    f"lr={lr_scheduler.current_lr:.2e}, "
                    f"grad_norm={np.mean(recent_gnorms):.2f}, "
                    f"time={epoch_time:.0f}s"
                    f"{best_tag}")
        if n_nan_skips > 0:
            logger.log(f"  ({n_nan_skips} NaN/spike skips this epoch)")

        logger.update_progress(
            current_epoch=epoch+1, epoch_losses=epoch_avg_losses,
            epoch_times=logger.state["epoch_times"] + [epoch_time],
            step_losses=step_losses, grad_norms=monitor.grad_norms)

        save_lora(lora_modules, str(root / f"lora_epoch{epoch+1}.pt"),
                  metadata={"epoch": epoch+1, "avg_loss": avg, "time": epoch_time})
        if is_best:
            save_lora(lora_modules, str(root / "lora_best.pt"),
                      metadata={"epoch": epoch+1, "avg_loss": avg})
        save_loss_curves(step_losses, epoch_avg_losses, monitor.grad_norms, root)

        if monitor.should_stop():
            logger.log(f"Early stopping: no improvement for {monitor.patience} epochs")
            logger.log(f"Rolling back to best (loss={monitor.best_loss:.6f})")
            monitor.rollback_to_best()
            break


# ---------------------------------------------------------------------------
# DPO training
# ---------------------------------------------------------------------------

def _train_dpo(net, optimizer, lora_modules, lora_params, store,
               winners, losers, pair_y, root, logger, *,
               num_epochs, beta_dpo, batch_size, grad_clip,
               lr_scheduler, ema, monitor, step_losses, epoch_avg_losses):

    num_pairs = len(winners)

    for epoch in range(num_epochs):
        t_epoch = time.time()
        order = np.random.permutation(num_pairs)
        epoch_loss, epoch_acc, num_batches = 0.0, 0.0, 0
        n_nan_skips = 0

        pbar = tqdm.tqdm(range(0, num_pairs, batch_size),
                         desc=f"Epoch {epoch+1}/{num_epochs}")
        for start in pbar:
            idx = order[start:start+batch_size]
            optimizer.zero_grad()
            loss, acc = dpo_loss_batch(net, lora_modules,
                                       winners[idx], losers[idx],
                                       beta_dpo=beta_dpo,
                                       store=store, y_batch=pair_y[idx])

            loss_val = loss.item()
            if not monitor.is_loss_ok(loss_val):
                n_nan_skips += 1
                optimizer.zero_grad()
                net.requires_grad_(False)
                pbar.set_postfix(loss="SKIP")
                continue

            loss.backward()
            net.requires_grad_(False)
            grad_norm = monitor.compute_grad_norm()
            torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
            optimizer.step()
            lr_scheduler.step()
            ema.update()

            epoch_loss += loss_val; epoch_acc += acc.item()
            num_batches += 1
            step_losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc.item():.2f}",
                             lr=f"{lr_scheduler.current_lr:.2e}")

        epoch_time = time.time() - t_epoch
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        epoch_avg_losses.append(avg_loss)

        is_best = monitor.save_best(avg_loss)
        best_tag = " [BEST]" if is_best else ""
        logger.log(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.6f}, "
                    f"acc={avg_acc:.2f}, lr={lr_scheduler.current_lr:.2e}, "
                    f"time={epoch_time:.0f}s{best_tag}")
        if n_nan_skips > 0:
            logger.log(f"  ({n_nan_skips} NaN/spike skips)")

        logger.update_progress(
            current_epoch=epoch+1, epoch_losses=epoch_avg_losses,
            epoch_times=logger.state["epoch_times"] + [epoch_time],
            step_losses=step_losses, grad_norms=monitor.grad_norms)

        save_lora(lora_modules, str(root / f"lora_epoch{epoch+1}.pt"),
                  metadata={"epoch": epoch+1, "avg_loss": avg_loss})
        if is_best:
            save_lora(lora_modules, str(root / "lora_best.pt"),
                      metadata={"epoch": epoch+1, "avg_loss": avg_loss})
        save_loss_curves(step_losses, epoch_avg_losses, monitor.grad_norms, root)

        if monitor.should_stop():
            logger.log(f"Early stopping: no improvement for {monitor.patience} epochs")
            monitor.rollback_to_best()
            break


# ---------------------------------------------------------------------------
# GRPO training
# ---------------------------------------------------------------------------

def _train_grpo(net, optimizer, lora_modules, lora_params, store,
                candidates, advantages, cand_y, root, logger, *,
                num_epochs, lambda_kl, adv_clip, batch_size, grad_clip,
                lr_scheduler, ema, monitor, step_losses, epoch_avg_losses):

    num_samples = len(candidates)

    for epoch in range(num_epochs):
        t_epoch = time.time()
        order = np.random.permutation(num_samples)
        epoch_loss, num_batches = 0.0, 0
        n_nan_skips = 0

        pbar = tqdm.tqdm(range(0, num_samples, batch_size),
                         desc=f"Epoch {epoch+1}/{num_epochs}")
        for start in pbar:
            idx = order[start:start+batch_size]
            optimizer.zero_grad()
            loss = grpo_loss_batch(net, lora_modules,
                                   candidates[idx], advantages[idx],
                                   lambda_kl=lambda_kl, adv_clip=adv_clip,
                                   store=store, y_batch=cand_y[idx])

            loss_val = loss.item()
            if not monitor.is_loss_ok(loss_val):
                n_nan_skips += 1
                optimizer.zero_grad()
                net.requires_grad_(False)
                pbar.set_postfix(loss="SKIP")
                continue

            loss.backward()
            net.requires_grad_(False)
            grad_norm = monitor.compute_grad_norm()
            torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
            optimizer.step()
            lr_scheduler.step()
            ema.update()

            epoch_loss += loss_val; num_batches += 1
            step_losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}",
                             lr=f"{lr_scheduler.current_lr:.2e}")

        epoch_time = time.time() - t_epoch
        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_avg_losses.append(avg_loss)

        is_best = monitor.save_best(avg_loss)
        best_tag = " [BEST]" if is_best else ""
        logger.log(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.6f}, "
                    f"lr={lr_scheduler.current_lr:.2e}, "
                    f"time={epoch_time:.0f}s{best_tag}")
        if n_nan_skips > 0:
            logger.log(f"  ({n_nan_skips} NaN/spike skips)")

        logger.update_progress(
            current_epoch=epoch+1, epoch_losses=epoch_avg_losses,
            epoch_times=logger.state["epoch_times"] + [epoch_time],
            step_losses=step_losses, grad_norms=monitor.grad_norms)

        save_lora(lora_modules, str(root / f"lora_epoch{epoch+1}.pt"),
                  metadata={"epoch": epoch+1, "avg_loss": avg_loss})
        if is_best:
            save_lora(lora_modules, str(root / "lora_best.pt"),
                      metadata={"epoch": epoch+1, "avg_loss": avg_loss})
        save_loss_curves(step_losses, epoch_avg_losses, monitor.grad_norms, root)

        if monitor.should_stop():
            logger.log(f"Early stopping: no improvement for {monitor.patience} epochs")
            monitor.rollback_to_best()
            break


if __name__ == "__main__":
    main()
