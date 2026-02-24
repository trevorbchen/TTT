"""
CBG vs DPS evaluation script for InverseBench.

Compares CBG (Classifier-Based Guidance) against DPS (Diffusion Posterior
Sampling) on test data.  Both use PF-ODE Euler with the same noise seeds,
same test samples, and same number of diffusion steps.

Reports per-method:
  - Mean L2 measurement error (||A(recon) - y||^2)
  - Std, 95% CI via Student-t
  - Timing

Usage:
    python eval_cbg.py problem=inv-scatter pretrain=inv-scatter \
        +eval.classifier_path=exps/cbg_tweedie_full/.../classifier_best.pt \
        +eval.guidance_scale=1.0 +eval.num_steps=200 +eval.num_test=100 \
        +eval.run_dps=True +eval.dps_guidance_scale=1.0 \
        +eval.out_dir=exps/eval_tweedie
"""

import json
import pickle
import time
import torch
import numpy as np
import tqdm
import hydra
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from scipy import stats

import sys
_repo_root = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from classifier import load_classifier
from utils.helper import open_url
from utils.scheduler import Scheduler


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------

def confidence_interval_95(values):
    """Compute mean, half-width of 95% CI, and std for a list of values."""
    arr = np.array(values)
    n = len(arr)
    mean = arr.mean()
    std = arr.std(ddof=1)
    if n < 2:
        return mean, 0.0, std
    half_width = stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n)
    return float(mean), float(half_width), float(std)


# ---------------------------------------------------------------------------
# CBG sampling (adapted from ttt_cbg.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def cbg_sample(net, classifier, forward_op, observation, scheduler,
               guidance_scale=1.0, device='cuda'):
    """Single-sample CBG reconstruction using PF-ODE Euler + classifier guidance."""
    obs = observation  # [1, ...]

    # Initial noise
    x = torch.randn(
        1, net.img_channels, net.img_resolution, net.img_resolution,
        device=device
    ) * scheduler.sigma_max

    for i in range(scheduler.num_steps):
        sigma = scheduler.sigma_steps[i]
        scaling = scheduler.scaling_steps[i]
        factor = scheduler.factor_steps[i]
        scaling_factor = scheduler.scaling_factor[i]

        # 1. Tweedie denoising (no grad)
        denoised = net(x / scaling, torch.as_tensor(sigma).to(device))

        # 2. Classifier gradient (grad only through small network)
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            pred_residual = classifier(
                x_in / scaling,
                torch.as_tensor(sigma).to(device),
                obs)
            loss_val = pred_residual.pow(2).flatten(1).sum(-1)
            grad_x = torch.autograd.grad(loss_val.sum(), x_in)[0]

        # 3. Per-sample normalization
        norm_factor = loss_val.sqrt().view(-1, *([1] * (grad_x.ndim - 1)))
        norm_factor = norm_factor.clamp(min=1e-8)
        normalized_grad = grad_x / norm_factor

        # 4. PF-ODE Euler step
        score = (denoised - x / scaling) / sigma ** 2 / scaling
        x = x * scaling_factor + factor * score * 0.5

        # Apply guidance
        x = x - guidance_scale * normalized_grad

        if torch.isnan(x).any():
            break

    return x


# ---------------------------------------------------------------------------
# DPS sampling (gradient through full diffusion model)
# ---------------------------------------------------------------------------

def dps_sample(net, forward_op, observation, scheduler,
               guidance_scale=1.0, device='cuda'):
    """Single-sample DPS reconstruction using PF-ODE Euler + full-model gradient."""
    obs = observation  # [1, ...]

    # Initial noise
    x = torch.randn(
        1, net.img_channels, net.img_resolution, net.img_resolution,
        device=device
    ) * scheduler.sigma_max

    for i in range(scheduler.num_steps):
        sigma = scheduler.sigma_steps[i]
        scaling = scheduler.scaling_steps[i]
        factor = scheduler.factor_steps[i]
        scaling_factor = scheduler.scaling_factor[i]

        sigma_t = torch.as_tensor(sigma).to(device)

        # 1. Tweedie with grad through full model
        x_in = x.detach().requires_grad_(True)
        net.requires_grad_(True)
        denoised = net(x_in / scaling, sigma_t)

        # 2. Measurement loss: ||A(denoised) - y||^2
        y_hat = forward_op({'target': denoised})
        residual = y_hat - obs
        if residual.is_complex():
            loss_val = torch.view_as_real(residual).pow(2).flatten(1).sum(-1)
        else:
            loss_val = residual.pow(2).flatten(1).sum(-1)

        # 3. Backprop through full diffusion model
        grad_x = torch.autograd.grad(loss_val.sum(), x_in)[0]
        net.requires_grad_(False)

        # 4. Per-sample normalization
        with torch.no_grad():
            norm_factor = loss_val.sqrt().view(-1, *([1] * (grad_x.ndim - 1)))
            norm_factor = norm_factor.clamp(min=1e-8)
            normalized_grad = grad_x / norm_factor

        # 5. PF-ODE Euler step (recompute denoised cleanly)
        with torch.no_grad():
            denoised_clean = net(x / scaling, sigma_t)
            score = (denoised_clean - x / scaling) / sigma ** 2 / scaling
            x = x * scaling_factor + factor * score * 0.5

            # Apply guidance
            x = x - guidance_scale * normalized_grad

            if torch.isnan(x).any():
                break

    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.tf32:
        torch.set_float32_matmul_precision("high")

    # --- Eval config ---
    ev = OmegaConf.to_container(config.get("eval", {}), resolve=True)
    classifier_path = ev.get("classifier_path")
    guidance_scale = ev.get("guidance_scale", 1.0)
    num_steps = ev.get("num_steps", 200)
    num_test = ev.get("num_test", 100)
    run_dps = ev.get("run_dps", True)
    dps_guidance_scale = ev.get("dps_guidance_scale", 1.0)
    out_dir = Path(ev.get("out_dir", "exps/eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    assert classifier_path is not None, \
        "Must provide +eval.classifier_path=... pointing to a trained classifier"

    print(f"=== CBG vs DPS Evaluation ===")
    print(f"  classifier: {classifier_path}")
    print(f"  guidance_scale: {guidance_scale}")
    print(f"  num_steps: {num_steps}")
    print(f"  num_test: {num_test}")
    print(f"  run_dps: {run_dps}")
    print(f"  dps_guidance_scale: {dps_guidance_scale}")
    print(f"  out_dir: {out_dir}")
    print()

    # --- Load components ---
    print("Loading forward operator...")
    forward_op = instantiate(config.problem.model, device=device)

    print("Loading test dataset...")
    test_dataset = instantiate(config.problem.data)

    print("Loading pretrained model...")
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
    net.requires_grad_(False)
    print(f"  Model: {type(net).__name__}, "
          f"res={net.img_resolution}, ch={net.img_channels}")

    print("Loading classifier...")
    classifier = load_classifier(classifier_path, device=device)
    classifier.eval()
    num_params = sum(p.numel() for p in classifier.parameters())
    print(f"  Classifier: {num_params/1e6:.2f}M params")

    # --- Build scheduler ---
    sched_cfg = {"num_steps": num_steps, "schedule": "vp",
                 "timestep": "vp", "scaling": "vp"}
    scheduler = Scheduler(**sched_cfg)

    # --- Select test samples (fixed seed for reproducibility) ---
    N = len(test_dataset)
    rng = np.random.RandomState(42)
    test_indices = rng.choice(N, size=min(num_test, N), replace=False)
    print(f"\nSelected {len(test_indices)} test samples from {N} total")

    # --- Load test data ---
    print("Loading test samples...")
    test_images = []
    test_measurements = []
    for i in tqdm.tqdm(test_indices, desc="Loading test data"):
        sample = test_dataset[int(i)]
        target = sample['target']
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target.copy())
        target = target.float().to(device)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        test_images.append(target)
        obs = forward_op({'target': target.unsqueeze(0)})
        test_measurements.append(obs)
    test_images = torch.stack(test_images)
    test_measurements = torch.cat(test_measurements)
    print(f"  images={test_images.shape}, measurements={test_measurements.shape}")

    results = {}

    # --- CBG evaluation ---
    print(f"\n{'='*60}")
    print(f"Running CBG sampling ({len(test_indices)} samples, {num_steps} steps)")
    print(f"{'='*60}")

    cbg_losses = []
    cbg_t0 = time.time()
    for idx in tqdm.trange(len(test_indices), desc="CBG"):
        y_i = test_measurements[idx:idx+1]

        # Seed for reproducibility (same noise for CBG and DPS)
        torch.manual_seed(42 + idx)
        recon = cbg_sample(net, classifier, forward_op, y_i, scheduler,
                           guidance_scale=guidance_scale, device=device)

        loss = forward_op.loss(recon, y_i).item()
        cbg_losses.append(loss)

    cbg_time = time.time() - cbg_t0
    cbg_mean, cbg_ci, cbg_std = confidence_interval_95(cbg_losses)
    results['cbg'] = {
        'mean': cbg_mean, 'std': cbg_std, 'ci95_half_width': cbg_ci,
        'per_sample': cbg_losses, 'total_time_sec': cbg_time,
        'time_per_sample_sec': cbg_time / len(test_indices),
        'guidance_scale': guidance_scale,
    }
    print(f"\nCBG: mean={cbg_mean:.6f} +/- {cbg_ci:.6f} "
          f"(std={cbg_std:.6f}, n={len(cbg_losses)})")
    print(f"  Time: {cbg_time:.1f}s total, "
          f"{cbg_time/len(test_indices):.2f}s/sample")

    # --- DPS evaluation ---
    if run_dps:
        print(f"\n{'='*60}")
        print(f"Running DPS sampling ({len(test_indices)} samples, {num_steps} steps)")
        print(f"{'='*60}")

        dps_losses = []
        dps_t0 = time.time()
        for idx in tqdm.trange(len(test_indices), desc="DPS"):
            y_i = test_measurements[idx:idx+1]

            # Same seed as CBG for fair comparison
            torch.manual_seed(42 + idx)
            recon = dps_sample(net, forward_op, y_i, scheduler,
                               guidance_scale=dps_guidance_scale, device=device)

            loss = forward_op.loss(recon, y_i).item()
            dps_losses.append(loss)

        dps_time = time.time() - dps_t0
        dps_mean, dps_ci, dps_std = confidence_interval_95(dps_losses)
        results['dps'] = {
            'mean': dps_mean, 'std': dps_std, 'ci95_half_width': dps_ci,
            'per_sample': dps_losses, 'total_time_sec': dps_time,
            'time_per_sample_sec': dps_time / len(test_indices),
            'guidance_scale': dps_guidance_scale,
        }
        print(f"\nDPS: mean={dps_mean:.6f} +/- {dps_ci:.6f} "
              f"(std={dps_std:.6f}, n={len(dps_losses)})")
        print(f"  Time: {dps_time:.1f}s total, "
              f"{dps_time/len(test_indices):.2f}s/sample")

    # --- Save results ---
    results['config'] = {
        'classifier_path': str(classifier_path),
        'num_steps': num_steps,
        'num_test': len(test_indices),
        'guidance_scale': guidance_scale,
        'dps_guidance_scale': dps_guidance_scale,
    }

    with open(str(out_dir / "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_dir / 'eval_results.json'}")

    # --- Comparison table ---
    print(f"\n{'='*60}")
    print(f"{'Method':<10} {'Mean L2':>12} {'Std':>12} {'95% CI':>18} {'Time/sample':>14}")
    print(f"{'-'*60}")
    print(f"{'CBG':<10} {cbg_mean:>12.6f} {cbg_std:>12.6f} "
          f"{'['+f'{cbg_mean-cbg_ci:.4f}, {cbg_mean+cbg_ci:.4f}'+']':>18} "
          f"{cbg_time/len(test_indices):>12.2f}s")
    if run_dps:
        print(f"{'DPS':<10} {dps_mean:>12.6f} {dps_std:>12.6f} "
              f"{'['+f'{dps_mean-dps_ci:.4f}, {dps_mean+dps_ci:.4f}'+']':>18} "
              f"{dps_time/len(test_indices):>12.2f}s")
        speedup = dps_time / max(cbg_time, 1e-6)
        print(f"\nCBG is {speedup:.1f}x faster than DPS per sample")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
