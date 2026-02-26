"""
Guidance scale tuner for CBG.

Loads the model once and sweeps guidance_scale values to find the optimum.
Much faster than running eval_cbg.py multiple times.

Usage:
    python tune_guidance.py problem=inv-scatter pretrain=inv-scatter \
        +eval.classifier_path=exps/cbg_tweedie_full/.../classifier_best.pt \
        +eval.num_test=20 +eval.num_steps=200 \
        +eval.scales="0.1,0.3,0.5,1.0,2.0,5.0,10.0"
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

import sys
_repo_root = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from classifier import load_classifier
from utils.helper import open_url
from utils.scheduler import Scheduler


@torch.no_grad()
def cbg_sample(net, classifier, forward_op, obs, scheduler,
               guidance_scale=1.0, device='cuda'):
    """CBG reconstruction — gradient only through small classifier."""
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

        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            pred = classifier(
                x_in / scaling,
                torch.as_tensor(sigma).to(device), obs)
            loss_val = pred.pow(2).flatten(1).sum(-1)
            grad_x = torch.autograd.grad(loss_val.sum(), x_in)[0]

        norm_factor = loss_val.sqrt().view(-1, *([1] * (grad_x.ndim - 1)))
        norm_factor = norm_factor.clamp(min=1e-8)
        normalized_grad = grad_x / norm_factor

        score = (denoised - x / scaling) / sigma ** 2 / scaling
        x = x * scaling_factor + factor * score * 0.5
        x = x - guidance_scale * normalized_grad

        if torch.isnan(x).any():
            break
    return x


def dps_sample(net, forward_op, obs, scheduler,
               guidance_scale=1.0, device='cuda'):
    """DPS reconstruction — gradient through full diffusion model."""
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

        x_in = x.detach().requires_grad_(True)
        net.requires_grad_(True)
        denoised = net(x_in / scaling, sigma_t)
        y_hat = forward_op({'target': denoised})
        residual = y_hat - obs
        if residual.is_complex():
            loss_val = torch.view_as_real(residual).pow(2).flatten(1).sum(-1)
        else:
            loss_val = residual.pow(2).flatten(1).sum(-1)
        grad_x = torch.autograd.grad(loss_val.sum(), x_in)[0]
        net.requires_grad_(False)

        with torch.no_grad():
            norm_factor = loss_val.sqrt().view(-1, *([1] * (grad_x.ndim - 1)))
            norm_factor = norm_factor.clamp(min=1e-8)
            normalized_grad = grad_x / norm_factor

            denoised_clean = net(x / scaling, sigma_t)
            score = (denoised_clean - x / scaling) / sigma ** 2 / scaling
            x = x * scaling_factor + factor * score * 0.5
            x = x - guidance_scale * normalized_grad

            if torch.isnan(x).any():
                break
    return x


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.tf32:
        torch.set_float32_matmul_precision("high")

    ev = OmegaConf.to_container(config.get("eval", {}), resolve=True)
    classifier_path = ev.get("classifier_path")
    num_steps = ev.get("num_steps", 200)
    num_test = ev.get("num_test", 20)
    run_dps = ev.get("run_dps", True)
    dps_guidance_scale = ev.get("dps_guidance_scale", 1.0)
    scales_str = ev.get("scales", "0.1,0.3,0.5,1.0,2.0,3.0,5.0,10.0,20.0")
    out_dir = Path(ev.get("out_dir", "exps/tune_guidance"))
    out_dir.mkdir(parents=True, exist_ok=True)

    scales = [float(s) for s in scales_str.split(",")]

    assert classifier_path, "Must provide +eval.classifier_path=..."

    print(f"=== Guidance Scale Tuner ===")
    print(f"  classifier: {classifier_path}")
    print(f"  scales: {scales}")
    print(f"  num_test: {num_test}, num_steps: {num_steps}")
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

    print("Loading classifier...")
    classifier = load_classifier(classifier_path, device=device)
    classifier.eval()
    print(f"  Classifier: {sum(p.numel() for p in classifier.parameters())/1e6:.1f}M params")

    scheduler = Scheduler(num_steps=num_steps, schedule="vp",
                          timestep="vp", scaling="vp")

    # --- Load test samples ---
    N = len(test_dataset)
    rng = np.random.RandomState(42)
    indices = rng.choice(N, size=min(num_test, N), replace=False)

    print(f"Loading {len(indices)} test samples...")
    measurements = []
    for i in tqdm.tqdm(indices, desc="Loading"):
        sample = test_dataset[int(i)]
        target = sample['target']
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target.copy())
        target = target.float().to(device)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        obs = forward_op({'target': target.unsqueeze(0)})
        measurements.append(obs)
    measurements = torch.cat(measurements)

    # --- DPS baseline (run once) ---
    dps_mean = None
    if run_dps:
        print(f"\n{'='*60}")
        print(f"Running DPS baseline (scale={dps_guidance_scale})...")
        dps_losses = []
        for idx in tqdm.trange(len(indices), desc="DPS"):
            torch.manual_seed(42 + idx)
            recon = dps_sample(net, forward_op, measurements[idx:idx+1],
                               scheduler, guidance_scale=dps_guidance_scale,
                               device=device)
            dps_losses.append(forward_op.loss(recon, measurements[idx:idx+1]).item())
        dps_mean = np.mean(dps_losses)
        dps_std = np.std(dps_losses)
        print(f"  DPS: mean={dps_mean:.6f} std={dps_std:.6f}")

    # --- Sweep guidance scales ---
    print(f"\n{'='*60}")
    print(f"Sweeping {len(scales)} guidance scales...")
    print(f"{'='*60}")

    results = {}
    best_scale = None
    best_mean = float('inf')

    for scale in scales:
        cbg_losses = []
        t0 = time.time()
        for idx in tqdm.trange(len(indices), desc=f"CBG scale={scale}"):
            torch.manual_seed(42 + idx)
            recon = cbg_sample(net, classifier, forward_op,
                               measurements[idx:idx+1], scheduler,
                               guidance_scale=scale, device=device)
            cbg_losses.append(forward_op.loss(recon, measurements[idx:idx+1]).item())
        elapsed = time.time() - t0
        mean_l2 = np.mean(cbg_losses)
        std_l2 = np.std(cbg_losses)

        results[str(scale)] = {
            'mean': float(mean_l2), 'std': float(std_l2),
            'per_sample': cbg_losses, 'time': elapsed,
        }

        marker = ""
        if mean_l2 < best_mean:
            best_mean = mean_l2
            best_scale = scale
            marker = " <-- BEST"

        dps_str = f"  (DPS={dps_mean:.4f})" if dps_mean else ""
        print(f"  scale={scale:6.1f} | L2={mean_l2:.6f} +/- {std_l2:.4f} | "
              f"{elapsed:.1f}s{dps_str}{marker}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scale':>8} {'Mean L2':>12} {'Std':>10}")
    print(f"{'-'*32}")
    for scale in scales:
        r = results[str(scale)]
        tag = " ***" if scale == best_scale else ""
        print(f"{scale:>8.1f} {r['mean']:>12.6f} {r['std']:>10.4f}{tag}")
    if dps_mean:
        print(f"{'DPS':>8} {dps_mean:>12.6f} {dps_std:>10.4f}")
    print(f"\nBest CBG: scale={best_scale}, L2={best_mean:.6f}")
    if dps_mean:
        ratio = best_mean / dps_mean
        print(f"DPS:      scale={dps_guidance_scale}, L2={dps_mean:.6f}")
        print(f"CBG/DPS ratio: {ratio:.2f}x")

    # --- Save ---
    summary = {
        'best_scale': best_scale, 'best_mean': best_mean,
        'dps_mean': dps_mean, 'scales': results,
        'config': {
            'classifier_path': str(classifier_path),
            'num_steps': num_steps, 'num_test': len(indices),
        },
    }
    with open(str(out_dir / "guidance_sweep.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir / 'guidance_sweep.json'}")


if __name__ == "__main__":
    main()
