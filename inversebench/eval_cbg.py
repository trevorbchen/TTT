"""
CBG vs DPS vs Plain evaluation script for InverseBench.

Compares CBG (Classifier-Based Guidance), DPS (Diffusion Posterior Sampling),
and Plain (unconditional diffusion) on test data.  All methods use PF-ODE
Euler with the same noise seeds, same test samples, and same number of
diffusion steps.

Reports per-method:
  - Relative measurement error: ||A(recon) - y||_2 / ||y||_2 * 100 (%)
  - Std, 95% CI via Student-t
  - Timing

Optionally saves a reconstruction grid image (rows=samples, cols=methods).

Usage:
    python eval_cbg.py problem=inv-scatter pretrain=inv-scatter \
        +eval.classifier_path=exps/cbg_tweedie_full/.../classifier_best.pt \
        +eval.guidance_scale=1.0 +eval.num_steps=200 +eval.num_test=100 \
        +eval.run_dps=True +eval.dps_guidance_scale=1.0 \
        +eval.run_plain=True +eval.save_images=True +eval.num_vis=8 \
        +eval.out_dir=exps/eval_tweedie
"""

import json
import pickle
import time
import torch
import numpy as np
import tqdm
import hydra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from scipy import stats

import sys
_repo_root = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from classifier import load_classifier, GradientPredictor
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


def relative_measurement_error(forward_op, recon, y):
    """||A(recon) - y||_2 / ||y||_2 * 100  (percentage)."""
    with torch.no_grad():
        y_hat = forward_op({'target': recon})
        residual = y_hat - y
        if residual.is_complex():
            res_norm = torch.view_as_real(residual).pow(2).flatten(1).sum(-1).sqrt()
        else:
            res_norm = residual.pow(2).flatten(1).sum(-1).sqrt()
        if y.is_complex():
            y_norm = torch.view_as_real(y).pow(2).flatten(1).sum(-1).sqrt()
        else:
            y_norm = y.pow(2).flatten(1).sum(-1).sqrt()
        return (res_norm / y_norm.clamp(min=1e-8) * 100).item()


# ---------------------------------------------------------------------------
# CBG sampling (adapted from ttt_cbg.py)
# ---------------------------------------------------------------------------

def cbg_sample(net, classifier, forward_op, observation, scheduler,
               guidance_scale=1.0, device='cuda', target_mode='tweedie'):
    """Single-sample CBG reconstruction using PF-ODE Euler + classifier guidance.

    For forward mode: gradient flows through the diffusion model (like DPS)
    but uses the classifier as a differentiable surrogate for the forward
    operator A.  This gives meaningful gradients w.r.t. x_t.
    """
    obs = observation  # [1, ...]

    # Precompute flat y for forward-mode guidance
    if target_mode == "forward":
        if obs.is_complex():
            y_flat = torch.view_as_real(obs).flatten(1).float()
        else:
            y_flat = obs.flatten(1).float()

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

        if target_mode == "forward":
            # Gradient flows through diffusion model → classifier
            # (like DPS, but classifier replaces A)
            x_in = x.detach().requires_grad_(True)
            net.requires_grad_(True)
            denoised = net(x_in / scaling, sigma_t)

            pred = classifier(
                x_in / scaling, sigma_t, None,
                denoised=denoised)
            loss_val = (pred.flatten(1) - y_flat).pow(2).sum(-1)
            grad_x = torch.autograd.grad(loss_val.sum(), x_in)[0]

            net.requires_grad_(False)

            # Recompute denoised cleanly for the ODE step
            with torch.no_grad():
                denoised_clean = net(x / scaling, sigma_t)
        else:
            # Legacy mode: grad only through classifier (no diffusion model)
            with torch.no_grad():
                denoised = net(x / scaling, sigma_t)

            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                pred = classifier(
                    x_in / scaling, sigma_t, obs,
                    denoised=denoised)
                if getattr(classifier, 'scalar_output', False):
                    loss_val = pred.squeeze(-1)
                else:
                    loss_val = pred.pow(2).flatten(1).sum(-1)
                grad_x = torch.autograd.grad(loss_val.sum(), x_in)[0]

            denoised_clean = denoised

        # Per-sample normalization
        with torch.no_grad():
            norm_factor = loss_val.sqrt().view(-1, *([1] * (grad_x.ndim - 1)))
            norm_factor = norm_factor.clamp(min=1e-8)
            normalized_grad = grad_x / norm_factor

            # PF-ODE Euler step
            score = (denoised_clean - x / scaling) / sigma ** 2 / scaling
            x = x * scaling_factor + factor * score * 0.5

            # Apply guidance
            x = x - guidance_scale * normalized_grad

            if torch.isnan(x).any():
                break

    return x


# ---------------------------------------------------------------------------
# Gradient predictor sampling (no autograd at inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def grad_pred_sample(net, classifier, observation, scheduler,
                     guidance_scale=1.0, device='cuda'):
    """Single-sample reconstruction using a GradientPredictor.

    The classifier directly predicts the guidance gradient image, so no
    autograd is needed at inference — just forward passes through net and
    classifier.
    """
    obs = observation  # [1, ...]

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

        denoised = net(x / scaling, sigma_t)
        pred_grad = classifier(x / scaling, sigma_t, obs, denoised=denoised)

        # Normalize by gradient norm for stable guidance
        gnorm = pred_grad.flatten(1).norm(dim=-1).clamp(min=1e-8)
        normalized_grad = pred_grad / gnorm.view(-1, 1, 1, 1)

        # PF-ODE Euler step
        score = (denoised - x / scaling) / sigma ** 2 / scaling
        x = x * scaling_factor + factor * score * 0.5

        # Apply guidance
        x = x - guidance_scale * normalized_grad

        if torch.isnan(x).any():
            break

    return x


# ---------------------------------------------------------------------------
# Hybrid sampling: surrogate early, DPS late
# ---------------------------------------------------------------------------

def hybrid_sample(net, classifier, forward_op, observation, scheduler,
                  guidance_scale=1.0, device='cuda', target_mode='forward',
                  switch_sigma=12.0):
    """Hybrid reconstruction: surrogate for high-sigma steps, DPS for low-sigma.

    Uses the classifier as a cheap differentiable surrogate for A while sigma
    is large (gradients align well with DPS there).  Once sigma drops below
    ``switch_sigma``, switches to the true forward operator (DPS-style) for
    the remaining steps where fine-grained guidance matters.
    """
    obs = observation  # [1, ...]

    # Precompute flat y for surrogate guidance
    if obs.is_complex():
        y_flat = torch.view_as_real(obs).flatten(1).float()
    else:
        y_flat = obs.flatten(1).float()

    # Initial noise
    x = torch.randn(
        1, net.img_channels, net.img_resolution, net.img_resolution,
        device=device
    ) * scheduler.sigma_max

    switched = False
    for i in range(scheduler.num_steps):
        sigma = scheduler.sigma_steps[i]
        scaling = scheduler.scaling_steps[i]
        factor = scheduler.factor_steps[i]
        scaling_factor = scheduler.scaling_factor[i]
        sigma_t = torch.as_tensor(sigma).to(device)

        use_dps = (float(sigma) < switch_sigma)
        if use_dps and not switched:
            switched = True

        # Gradient through diffusion model in both cases
        x_in = x.detach().requires_grad_(True)
        net.requires_grad_(True)
        denoised = net(x_in / scaling, sigma_t)

        if use_dps:
            # DPS: true forward operator
            y_hat = forward_op({'target': denoised})
            residual = y_hat - obs
            if residual.is_complex():
                loss_val = torch.view_as_real(residual).pow(2).flatten(1).sum(-1)
            else:
                loss_val = residual.pow(2).flatten(1).sum(-1)
        else:
            # Surrogate: classifier replaces A
            pred = classifier(
                x_in / scaling, sigma_t, None,
                denoised=denoised)
            loss_val = (pred.flatten(1) - y_flat).pow(2).sum(-1)

        grad_x = torch.autograd.grad(loss_val.sum(), x_in)[0]
        net.requires_grad_(False)

        # Recompute denoised cleanly for ODE step
        with torch.no_grad():
            denoised_clean = net(x / scaling, sigma_t)

            norm_factor = loss_val.sqrt().view(-1, *([1] * (grad_x.ndim - 1)))
            norm_factor = norm_factor.clamp(min=1e-8)
            normalized_grad = grad_x / norm_factor

            # PF-ODE Euler step
            score = (denoised_clean - x / scaling) / sigma ** 2 / scaling
            x = x * scaling_factor + factor * score * 0.5
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
# Plain unconditional sampling (no guidance)
# ---------------------------------------------------------------------------

@torch.no_grad()
def plain_sample(net, scheduler, device='cuda'):
    """Unconditional PF-ODE Euler sample (no guidance of any kind)."""
    x = torch.randn(
        1, net.img_channels, net.img_resolution, net.img_resolution,
        device=device
    ) * scheduler.sigma_max

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
    run_plain = ev.get("run_plain", False)
    run_hybrid = ev.get("run_hybrid", False)
    hybrid_switch_sigma = ev.get("hybrid_switch_sigma", 12.0)
    save_images = ev.get("save_images", False)
    num_vis = ev.get("num_vis", 8)
    out_dir = Path(ev.get("out_dir", "exps/eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    assert classifier_path is not None, \
        "Must provide +eval.classifier_path=... pointing to a trained classifier"

    print(f"=== CBG vs DPS vs Plain Evaluation ===")
    print(f"  classifier: {classifier_path}")
    print(f"  guidance_scale: {guidance_scale}")
    print(f"  num_steps: {num_steps}")
    print(f"  num_test: {num_test}")
    print(f"  run_dps: {run_dps}")
    print(f"  dps_guidance_scale: {dps_guidance_scale}")
    print(f"  run_plain: {run_plain}")
    print(f"  run_hybrid: {run_hybrid} (switch_sigma={hybrid_switch_sigma})")
    print(f"  save_images: {save_images} (num_vis={num_vis})")
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

    # Detect architecture and target_mode from checkpoint metadata
    ckpt_meta = torch.load(classifier_path, map_location='cpu', weights_only=False)
    target_mode = ckpt_meta.get("target_mode", "tweedie")
    is_grad_pred = isinstance(classifier, GradientPredictor)
    print(f"  Target mode: {target_mode}")
    if is_grad_pred:
        print(f"  Architecture: GradientPredictor (no-autograd inference)")
    del ckpt_meta

    # --- Build scheduler ---
    sched_cfg = {"num_steps": num_steps, "schedule": "vp",
                 "timestep": "vp", "scaling": "vp"}
    scheduler = Scheduler(**sched_cfg)

    # --- Select test samples ---
    seed = ev.get("seed", 42)
    N = len(test_dataset)
    rng = np.random.RandomState(seed)
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
    # Collect first num_vis reconstructions for image grid
    vis_recons = {}  # method_name -> list of tensors

    # --- CBG evaluation ---
    print(f"\n{'='*60}")
    print(f"Running CBG sampling ({len(test_indices)} samples, {num_steps} steps)")
    print(f"{'='*60}")

    cbg_losses = []
    if save_images:
        vis_recons['CBG'] = []
    cbg_t0 = time.time()
    for idx in tqdm.trange(len(test_indices), desc="CBG"):
        y_i = test_measurements[idx:idx+1]

        # Seed for reproducibility (same noise for CBG and DPS)
        torch.manual_seed(42 + idx)
        if is_grad_pred:
            recon = grad_pred_sample(net, classifier, y_i, scheduler,
                                     guidance_scale=guidance_scale, device=device)
        else:
            recon = cbg_sample(net, classifier, forward_op, y_i, scheduler,
                               guidance_scale=guidance_scale, device=device,
                               target_mode=target_mode)

        err = relative_measurement_error(forward_op, recon, y_i)
        cbg_losses.append(err)
        if save_images and idx < num_vis:
            vis_recons['CBG'].append(recon.cpu())

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
        if save_images:
            vis_recons['DPS'] = []
        dps_t0 = time.time()
        for idx in tqdm.trange(len(test_indices), desc="DPS"):
            y_i = test_measurements[idx:idx+1]

            # Same seed as CBG for fair comparison
            torch.manual_seed(42 + idx)
            recon = dps_sample(net, forward_op, y_i, scheduler,
                               guidance_scale=dps_guidance_scale, device=device)

            err = relative_measurement_error(forward_op, recon, y_i)
            dps_losses.append(err)
            if save_images and idx < num_vis:
                vis_recons['DPS'].append(recon.cpu())

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

    # --- Plain (unconditional) evaluation ---
    if run_plain:
        print(f"\n{'='*60}")
        print(f"Running Plain sampling ({len(test_indices)} samples, {num_steps} steps)")
        print(f"{'='*60}")

        plain_losses = []
        if save_images:
            vis_recons['Plain'] = []
        plain_t0 = time.time()
        for idx in tqdm.trange(len(test_indices), desc="Plain"):
            y_i = test_measurements[idx:idx+1]

            # Same seed as CBG/DPS for fair comparison
            torch.manual_seed(42 + idx)
            recon = plain_sample(net, scheduler, device=device)

            err = relative_measurement_error(forward_op, recon, y_i)
            plain_losses.append(err)
            if save_images and idx < num_vis:
                vis_recons['Plain'].append(recon.cpu())

        plain_time = time.time() - plain_t0
        plain_mean, plain_ci, plain_std = confidence_interval_95(plain_losses)
        results['plain'] = {
            'mean': plain_mean, 'std': plain_std, 'ci95_half_width': plain_ci,
            'per_sample': plain_losses, 'total_time_sec': plain_time,
            'time_per_sample_sec': plain_time / len(test_indices),
        }
        print(f"\nPlain: mean={plain_mean:.6f} +/- {plain_ci:.6f} "
              f"(std={plain_std:.6f}, n={len(plain_losses)})")
        print(f"  Time: {plain_time:.1f}s total, "
              f"{plain_time/len(test_indices):.2f}s/sample")

    # --- Hybrid evaluation ---
    if run_hybrid:
        print(f"\n{'='*60}")
        print(f"Running Hybrid sampling ({len(test_indices)} samples, {num_steps} steps)")
        print(f"  Surrogate for sigma >= {hybrid_switch_sigma}, DPS for sigma < {hybrid_switch_sigma}")
        print(f"{'='*60}")

        hybrid_losses = []
        if save_images:
            vis_recons['Hybrid'] = []
        hybrid_t0 = time.time()
        for idx in tqdm.trange(len(test_indices), desc="Hybrid"):
            y_i = test_measurements[idx:idx+1]

            torch.manual_seed(42 + idx)
            recon = hybrid_sample(net, classifier, forward_op, y_i, scheduler,
                                  guidance_scale=guidance_scale, device=device,
                                  target_mode=target_mode,
                                  switch_sigma=hybrid_switch_sigma)

            err = relative_measurement_error(forward_op, recon, y_i)
            hybrid_losses.append(err)
            if save_images and idx < num_vis:
                vis_recons['Hybrid'].append(recon.cpu())

        hybrid_time = time.time() - hybrid_t0
        hybrid_mean, hybrid_ci, hybrid_std = confidence_interval_95(hybrid_losses)
        results['hybrid'] = {
            'mean': hybrid_mean, 'std': hybrid_std, 'ci95_half_width': hybrid_ci,
            'per_sample': hybrid_losses, 'total_time_sec': hybrid_time,
            'time_per_sample_sec': hybrid_time / len(test_indices),
            'guidance_scale': guidance_scale,
            'switch_sigma': hybrid_switch_sigma,
        }
        print(f"\nHybrid: mean={hybrid_mean:.6f} +/- {hybrid_ci:.6f} "
              f"(std={hybrid_std:.6f}, n={len(hybrid_losses)})")
        print(f"  Time: {hybrid_time:.1f}s total, "
              f"{hybrid_time/len(test_indices):.2f}s/sample")

    # --- Save reconstruction grid ---
    if save_images and vis_recons:
        n_vis = min(num_vis, len(test_indices))
        # Build column list: GT + each method
        cols = [("GT", [test_images[i:i+1].cpu() for i in range(n_vis)], None)]
        for method_name in ['CBG', 'Hybrid', 'DPS', 'Plain']:
            if method_name in vis_recons:
                method_losses = results.get(method_name.lower(), {}).get('per_sample', [])
                cols.append((method_name, vis_recons[method_name], method_losses[:n_vis]))

        ncols = len(cols)
        fig, axes = plt.subplots(n_vis, ncols,
                                 figsize=(3.5 * ncols, 3.5 * n_vis))
        if n_vis == 1:
            axes = axes[None, :]

        for i in range(n_vis):
            for j, (label, recons_list, losses) in enumerate(cols):
                ax = axes[i, j]
                img = recons_list[i].squeeze().numpy()
                ax.imshow(img, cmap='viridis')
                if losses is not None and i < len(losses):
                    ax.set_title(f"{label} ({losses[i]:.1f}%)", fontsize=10)
                else:
                    ax.set_title(f"{label} #{test_indices[i]}", fontsize=10)
                ax.axis('off')

        # Summary title
        parts = ["Rel. Meas. Error (%)"]
        parts.append(f"CBG={cbg_mean:.1f}%")
        if run_hybrid:
            parts.append(f"Hybrid={hybrid_mean:.1f}%")
        if run_dps:
            parts.append(f"DPS={dps_mean:.1f}%")
        if run_plain:
            parts.append(f"Plain={plain_mean:.1f}%")
        plt.suptitle(" | ".join(parts), fontsize=13, fontweight='bold')
        plt.tight_layout()
        img_path = out_dir / "reconstructions.png"
        plt.savefig(str(img_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved reconstruction grid to {img_path}")

    # --- Save results ---
    results['config'] = {
        'classifier_path': str(classifier_path),
        'num_steps': num_steps,
        'num_test': len(test_indices),
        'guidance_scale': guidance_scale,
        'dps_guidance_scale': dps_guidance_scale,
        'run_plain': run_plain,
        'run_hybrid': run_hybrid,
        'hybrid_switch_sigma': hybrid_switch_sigma,
        'save_images': save_images,
    }

    with open(str(out_dir / "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_dir / 'eval_results.json'}")

    # --- Comparison table ---
    print(f"\n{'='*60}")
    print(f"{'Method':<10} {'Err(%)':>10} {'Std':>10} {'95% CI':>18} {'Time/sample':>14}")
    print(f"{'-'*62}")
    print(f"{'CBG':<10} {cbg_mean:>10.2f} {cbg_std:>10.2f} "
          f"{'['+f'{cbg_mean-cbg_ci:.2f}, {cbg_mean+cbg_ci:.2f}'+']':>18} "
          f"{cbg_time/len(test_indices):>12.2f}s")
    if run_hybrid:
        print(f"{'Hybrid':<10} {hybrid_mean:>10.2f} {hybrid_std:>10.2f} "
              f"{'['+f'{hybrid_mean-hybrid_ci:.2f}, {hybrid_mean+hybrid_ci:.2f}'+']':>18} "
              f"{hybrid_time/len(test_indices):>12.2f}s")
    if run_dps:
        print(f"{'DPS':<10} {dps_mean:>10.2f} {dps_std:>10.2f} "
              f"{'['+f'{dps_mean-dps_ci:.2f}, {dps_mean+dps_ci:.2f}'+']':>18} "
              f"{dps_time/len(test_indices):>12.2f}s")
    if run_plain:
        print(f"{'Plain':<10} {plain_mean:>10.2f} {plain_std:>10.2f} "
              f"{'['+f'{plain_mean-plain_ci:.2f}, {plain_mean+plain_ci:.2f}'+']':>18} "
              f"{plain_time/len(test_indices):>12.2f}s")
    if run_dps:
        speedup = dps_time / max(cbg_time, 1e-6)
        print(f"\nCBG is {speedup:.1f}x faster than DPS per sample")
        if run_hybrid:
            speedup_h = dps_time / max(hybrid_time, 1e-6)
            print(f"Hybrid is {speedup_h:.1f}x faster than DPS per sample")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
