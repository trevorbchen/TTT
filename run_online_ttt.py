"""
Online amortized test-time training with LoRA and replay buffer.

LoRA is initialized once and persists across test instances.  For each
incoming measurement y_i the pipeline:
  1. Samples x_0 via DRaFT (plain diffusion, grad through last K steps)
  2. Computes the DRaFT reward  r = -||A(x_0) - y||^2
  3. Appends (x_0, y_i) to a persistent replay buffer
  4. Samples a mini-batch from the buffer, noises the stored x_0 to a
     random sigma, denoises, and computes a measurement-consistency loss
  5. Updates LoRA with (current DRaFT loss + buffer replay loss)

Later instances benefit from knowledge accumulated in earlier rounds
(the "amortization curve").

Usage:
  python run_online_ttt.py \
      data=test-imagenet model=imagenet256ddpm \
      sampler=edm_dps task=gaussian_blur \
      +ttt.lora_rank=4 +ttt.lr=1e-3 +ttt.num_draft_rounds=1 \
      name=online_ttt_blur
"""

import json
import random
import yaml
import torch
import numpy as np
import tqdm
import hydra
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torchvision.utils import save_image

from forward_operator import get_operator
from data import get_dataset
from model import get_model
from eval import get_eval_fn, Evaluator
from sampler import get_sampler
from lora import (apply_lora, remove_lora, get_lora_params, save_lora,
                  frozen_tweedie)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def plain_diffusion_sample(model, scheduler, device):
    """Plain PF-ODE Euler sampling (no guidance, no grad)."""
    in_shape = model.get_in_shape()
    x = torch.randn(1, *in_shape, device=device) * scheduler.get_prior_sigma()
    sigma_steps = scheduler.sigma_steps
    num_steps = len(sigma_steps) - 1

    for i in range(num_steps):
        sigma = sigma_steps[i]
        sigma_next = sigma_steps[i + 1]
        t = scheduler.get_sigma_inv(sigma)
        t_next = scheduler.get_sigma_inv(sigma_next)
        dt = t_next - t
        st = scheduler.get_scaling(t)
        dst = scheduler.get_scaling_derivative(t)
        dsigma = scheduler.get_sigma_derivative(t)

        x0hat = model.tweedie(x / st, sigma)
        score = (x0hat - x / st) / sigma ** 2
        deriv = dst / st * x - st * dsigma * sigma * score
        x = x + dt * deriv

    return x


def draft_k_sample(model, scheduler, device, draft_k=1):
    """Plain PF-ODE Euler sampling, differentiable through last draft_k steps.

    Returns x_0 whose grad graph connects to LoRA parameters through the
    last draft_k denoising steps.
    """
    in_shape = model.get_in_shape()
    x = torch.randn(1, *in_shape, device=device) * scheduler.get_prior_sigma()
    sigma_steps = scheduler.sigma_steps
    num_steps = len(sigma_steps) - 1
    grad_start = max(num_steps - draft_k, 0)

    for i in range(num_steps):
        sigma = sigma_steps[i]
        sigma_next = sigma_steps[i + 1]
        t = scheduler.get_sigma_inv(sigma)
        t_next = scheduler.get_sigma_inv(sigma_next)
        dt = t_next - t
        st = scheduler.get_scaling(t)
        dst = scheduler.get_scaling_derivative(t)
        dsigma = scheduler.get_sigma_derivative(t)

        if i < grad_start:
            with torch.no_grad():
                x0hat = model.tweedie(x / st, sigma)
                score = (x0hat - x / st) / sigma ** 2
                deriv = dst / st * x - st * dsigma * sigma * score
                x = x + dt * deriv
        else:
            if i == grad_start:
                x = x.detach()
            # grad flows through model.tweedie → LoRA params
            x0hat = model.tweedie(x / st, sigma)
            score = (x0hat - x / st) / sigma ** 2
            deriv = dst / st * x - st * dsigma * sigma * score
            x = x + dt * deriv

    return x


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple buffer that stores (x_0, y) pairs on CPU."""

    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.data = []  # list of (x_0, y) tensors on CPU

    def add(self, x0, y):
        self.data.append((x0.detach().cpu(), y.detach().cpu()))
        if len(self.data) > self.max_size:
            self.data.pop(0)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.data))
        indices = random.sample(range(len(self.data)), batch_size)
        x0s = torch.cat([self.data[i][0] for i in indices], dim=0)
        ys = torch.cat([self.data[i][1] for i in indices], dim=0)
        return x0s, ys

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------------------
# Buffer training step
# ---------------------------------------------------------------------------

def buffer_update_loss(model, scheduler, forward_op, buffer, batch_size,
                       device):
    """Compute measurement-consistency loss on noised buffer samples.

    For each (x_0, y) from the buffer:
      - Sample a random noise level sigma from the schedule
      - Noise x_0:  x_noisy = x_0 + sigma * noise  (in the unscaled space)
      - Denoise:    x_hat = model.tweedie(x_noisy, sigma)
      - Loss:       ||A(x_hat) - y||^2

    Returns scalar loss with grad through model (LoRA params).
    """
    if len(buffer) == 0:
        return torch.tensor(0.0, device=device)

    x0_batch, y_batch = buffer.sample(batch_size)
    x0_batch = x0_batch.to(device)
    y_batch = y_batch.to(device)

    # random noise level from schedule (exclude sigma=0)
    sigma_steps = scheduler.sigma_steps
    valid_sigmas = sigma_steps[sigma_steps > 0]
    idx = torch.randint(0, len(valid_sigmas), (1,)).item()
    sigma = valid_sigmas[idx]

    noise = torch.randn_like(x0_batch)
    x_noisy = x0_batch + sigma * noise  # unscaled space: x/st = x_0 + sigma*eps

    x_hat = model.tweedie(x_noisy, sigma)
    loss = forward_op.loss(x_hat, y_batch).mean()
    return loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm(x):
    """[-1,1] -> [0,1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)


def resize_y(y, target_shape):
    if y.shape != target_shape:
        return torch.nn.functional.interpolate(
            y, size=target_shape[-2:], mode='bilinear', align_corners=False)
    return y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="configs",
            config_name="default.yaml")
def main(args: DictConfig):
    # --- reproducibility ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(f"cuda:{args.gpu}")

    print(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))

    # --- TTT config ---
    ttt = OmegaConf.to_container(args.get("ttt", {}), resolve=True)
    lora_rank = ttt.get("lora_rank", 4)
    lora_alpha = ttt.get("lora_alpha", 1.0)
    target_modules = ttt.get("target_modules", "all")
    lr = ttt.get("lr", 1e-3)
    grad_clip = ttt.get("grad_clip", 1.0)
    draft_k = ttt.get("draft_k", 1)
    num_draft_rounds = ttt.get("num_draft_rounds", 1)
    buffer_batch_size = ttt.get("buffer_batch_size", 4)
    buffer_max_size = ttt.get("buffer_max_size", 10000)
    lambda_buffer = ttt.get("lambda_buffer", 1.0)
    skip_baseline = ttt.get("skip_baseline", False)

    # --- data ---
    dataset = get_dataset(**args.data)
    total_number = len(dataset)
    images = dataset.get_data(total_number, 0)

    # --- operator & measurement ---
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    y = operator.measure(images)

    # --- sampler (for DPS baseline & scheduler) ---
    sampler = get_sampler(
        **args.sampler,
        mcmc_sampler_config=task_group.get("mcmc_sampler_config", None))
    scheduler = sampler.scheduler

    # --- model ---
    model = get_model(**args.model)
    device = next(model.parameters()).device

    # --- evaluator ---
    eval_fn_list = [get_eval_fn(name) for name in args.eval_fn_list]
    evaluator = Evaluator(eval_fn_list)

    # --- output dirs ---
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    root = save_dir / args.name
    root.mkdir(exist_ok=True)
    (root / "samples").mkdir(exist_ok=True)
    (root / "comparisons").mkdir(exist_ok=True)

    with open(str(root / "config.yaml"), "w") as f:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), f)

    # =================================================================
    # Phase 0 (optional): DPS baseline
    # =================================================================
    if skip_baseline:
        all_baseline = None
    else:
        print(f"\n{'='*60}")
        print(f"DPS baseline ({total_number} images)")
        print(f"{'='*60}")
        all_baseline = []
        for img_idx in tqdm.trange(total_number, desc="DPS baseline"):
            y_i = y[img_idx: img_idx + 1]
            x_start = sampler.get_start(1, model)
            x_hat = sampler.sample(model, x_start, operator, y_i,
                                   verbose=False)
            all_baseline.append(x_hat)
        all_baseline = torch.cat(all_baseline, dim=0)

    # =================================================================
    # Phase 1: Online amortized TTT
    # =================================================================
    print(f"\n{'='*60}")
    print(f"Online TTT ({total_number} instances)")
    print(f"  rank={lora_rank}  alpha={lora_alpha}  lr={lr}")
    print(f"  draft_k={draft_k}  rounds={num_draft_rounds}")
    print(f"  buffer_batch={buffer_batch_size}  lambda_buf={lambda_buffer}")
    print(f"{'='*60}")

    # (1) Initialize LoRA and optimizer ONCE
    lora_modules = apply_lora(model, rank=lora_rank, alpha=lora_alpha,
                              target_modules=target_modules)
    lora_params = get_lora_params(lora_modules)
    optimizer = torch.optim.Adam(lora_params, lr=lr)

    # (2) Persistent replay buffer
    buffer_D = ReplayBuffer(max_size=buffer_max_size)

    all_recons = []
    all_metrics = []
    metric_names = None

    pbar = tqdm.trange(total_number, desc="Online TTT")
    for img_idx in pbar:
        gt_i = images[img_idx: img_idx + 1]
        y_i = y[img_idx: img_idx + 1]

        # --- DRaFT sampling rounds for current instance ---
        x_0 = None
        for rnd in range(num_draft_rounds):
            optimizer.zero_grad()

            # DRaFT sample: plain diffusion, grad through last draft_k steps
            x_0 = draft_k_sample(model, scheduler, device, draft_k=draft_k)

            # Current-instance DRaFT loss
            loss_current = operator.loss(x_0, y_i).mean()

            # Buffer replay loss
            loss_buffer = torch.tensor(0.0, device=device)
            if len(buffer_D) > 0:
                loss_buffer = buffer_update_loss(
                    model, scheduler, operator, buffer_D,
                    buffer_batch_size, device)

            total_loss = loss_current + lambda_buffer * loss_buffer
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
            optimizer.step()

        # Final clean sample with updated LoRA (no grad)
        with torch.no_grad():
            x_final = plain_diffusion_sample(model, scheduler, device)

        # (2) Append to buffer
        buffer_D.add(x_final, y_i)

        all_recons.append(x_final.detach())

        # --- Per-instance metrics ---
        with torch.no_grad():
            metrics_i = evaluator(gt_i, y_i, x_final)
            mc_i = operator.loss(x_final, y_i)
            metrics_i["meas_l2"] = mc_i.mean()

        row = {"image_idx": img_idx, "buffer_size": len(buffer_D)}
        if metric_names is None:
            metric_names = list(metrics_i.keys())
        for name in metric_names:
            row[f"{name}_ttt"] = metrics_i[name].item()

        if all_baseline is not None:
            baseline_i = all_baseline[img_idx: img_idx + 1]
            with torch.no_grad():
                metrics_b = evaluator(gt_i, y_i, baseline_i)
                mc_b = operator.loss(baseline_i, y_i)
                metrics_b["meas_l2"] = mc_b.mean()
            for name in metric_names:
                row[f"{name}_baseline"] = metrics_b[name].item()

        all_metrics.append(row)

        # progress bar
        psnr_val = row.get("psnr_ttt", 0)
        pbar.set_postfix(psnr=f"{psnr_val:.2f}", buf=len(buffer_D))

        # save sample + comparison
        save_image(norm(x_final),
                   str(root / "samples" / f"{img_idx:05d}_ttt.png"))
        y_resized = resize_y(y_i, gt_i.shape)
        if all_baseline is not None:
            grid = torch.cat([norm(gt_i), norm(y_resized),
                              norm(baseline_i), norm(x_final)], dim=0)
            save_image(grid,
                       str(root / "comparisons" / f"{img_idx:05d}.png"),
                       nrow=4, padding=2)
        else:
            grid = torch.cat([norm(gt_i), norm(y_resized),
                              norm(x_final)], dim=0)
            save_image(grid,
                       str(root / "comparisons" / f"{img_idx:05d}.png"),
                       nrow=3, padding=2)

    # =================================================================
    # Phase 2: Save LoRA & aggregate results
    # =================================================================
    all_recons = torch.cat(all_recons, dim=0)

    # Save final LoRA
    save_lora(lora_modules, str(root / "lora_final.pt"),
              metadata={"method": "online_ttt", "lora_rank": lora_rank,
                        "lora_alpha": lora_alpha, "lr": lr,
                        "draft_k": draft_k,
                        "num_draft_rounds": num_draft_rounds,
                        "buffer_batch_size": buffer_batch_size,
                        "num_instances_seen": total_number})

    remove_lora(model)

    # --- Print amortization curve ---
    ttt_vals = {n: np.array([m[f"{n}_ttt"] for m in all_metrics])
                for n in metric_names}

    if all_baseline is not None:
        base_vals = {n: np.array([m[f"{n}_baseline"] for m in all_metrics])
                     for n in metric_names}
        delta_vals = {n: ttt_vals[n] - base_vals[n] for n in metric_names}

        print(f"\n{'='*60}")
        print(f"Results: Online TTT vs DPS | {total_number} images")
        print(f"{'='*60}")
        print(f"{'metric':<8} {'DPS baseline':>18} {'Online TTT':>18} "
              f"{'delta':>18}")
        print(f"{'':<8} {'mean +/- std':>18} {'mean +/- std':>18} "
              f"{'mean +/- std':>18}")
        print(f"{'-'*62}")
        for n in metric_names:
            bm, bs = base_vals[n].mean(), base_vals[n].std()
            am, astd = ttt_vals[n].mean(), ttt_vals[n].std()
            dm, ds = delta_vals[n].mean(), delta_vals[n].std()
            print(f"{n:<8} {bm:>7.4f} +/- {bs:<6.4f} "
                  f"{am:>7.4f} +/- {astd:<6.4f} "
                  f"{dm:>+7.4f} +/- {ds:<6.4f}")
        print(f"{'-'*62}")

        # Amortization curve: first half vs second half
        half = total_number // 2
        if half > 0:
            print(f"\nAmortization curve (first {half} vs last {half}):")
            for n in metric_names:
                first = ttt_vals[n][:half].mean()
                second = ttt_vals[n][half:].mean()
                print(f"  {n}: first_half={first:.4f}  "
                      f"second_half={second:.4f}  "
                      f"delta={second - first:+.4f}")
    else:
        print(f"\n{'='*60}")
        print(f"Results: Online TTT | {total_number} images")
        print(f"{'='*60}")
        for n in metric_names:
            am, astd = ttt_vals[n].mean(), ttt_vals[n].std()
            print(f"  {n}: {am:.4f} +/- {astd:.4f}")

        half = total_number // 2
        if half > 0:
            print(f"\nAmortization curve (first {half} vs last {half}):")
            for n in metric_names:
                first = ttt_vals[n][:half].mean()
                second = ttt_vals[n][half:].mean()
                print(f"  {n}: first_half={first:.4f}  "
                      f"second_half={second:.4f}  "
                      f"delta={second - first:+.4f}")

    # --- Save full comparison grid ---
    y_resized = resize_y(y, images.shape)
    if all_baseline is not None:
        full_grid = torch.cat([norm(images), norm(y_resized),
                               norm(all_baseline), norm(all_recons)], dim=0)
    else:
        full_grid = torch.cat([norm(images), norm(y_resized),
                               norm(all_recons)], dim=0)
    save_image(full_grid, str(root / "full_comparison.png"),
               nrow=total_number, padding=2)

    # --- Save metrics JSON ---
    aggregate = {}
    for n in metric_names:
        agg = {
            "ttt_mean": float(ttt_vals[n].mean()),
            "ttt_std": float(ttt_vals[n].std()),
        }
        if all_baseline is not None:
            agg.update({
                "baseline_mean": float(base_vals[n].mean()),
                "baseline_std": float(base_vals[n].std()),
                "delta_mean": float(delta_vals[n].mean()),
            })
        aggregate[n] = agg

    output = {
        "method": "online_ttt",
        "num_images": total_number,
        "config": {
            "lora_rank": lora_rank, "lora_alpha": lora_alpha,
            "target_modules": target_modules, "lr": lr,
            "draft_k": draft_k, "num_draft_rounds": num_draft_rounds,
            "buffer_batch_size": buffer_batch_size,
            "lambda_buffer": lambda_buffer,
        },
        "aggregate": aggregate,
        "per_image": all_metrics,
    }
    with open(str(root / "metrics.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {root}")
    print(f"  - Comparisons:  {root / 'comparisons'}")
    print(f"  - Samples:      {root / 'samples'}")
    print(f"  - Metrics:      {root / 'metrics.json'}")
    print(f"  - LoRA:         {root / 'lora_final.pt'}")


if __name__ == "__main__":
    main()
