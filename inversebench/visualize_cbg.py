"""
Visualize CBG vs DPS reconstructions side-by-side.

Generates a comparison grid: GT | CBG | DPS | Plain Diffusion
for a handful of test samples.

Usage:
    python visualize_cbg.py problem=inv-scatter pretrain=inv-scatter \
        +eval.classifier_path=exps/cbg_tweedie_full/.../classifier_best.pt \
        +eval.num_samples=8 +eval.out_path=comparison.png
"""

import pickle
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hydra
import tqdm
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
def plain_sample(net, scheduler, device):
    """Plain unconditional diffusion sample (no guidance)."""
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


def cbg_sample(net, classifier, forward_op, obs, scheduler,
               guidance_scale=1.0, device='cuda'):
    """CBG reconstruction — gradient only through small classifier."""
    x = torch.randn(
        1, net.img_channels, net.img_resolution, net.img_resolution,
        device=device
    ) * scheduler.sigma_max

    with torch.no_grad():
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
    guidance_scale = ev.get("guidance_scale", 1.0)
    dps_guidance_scale = ev.get("dps_guidance_scale", 1.0)
    num_steps = ev.get("num_steps", 200)
    num_samples = ev.get("num_samples", 8)
    out_path = ev.get("out_path", "comparison.png")
    run_dps = ev.get("run_dps", True)
    run_plain = ev.get("run_plain", True)

    assert classifier_path, "Must provide +eval.classifier_path=..."

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

    scheduler = Scheduler(num_steps=num_steps, schedule="vp",
                          timestep="vp", scaling="vp")

    # --- Pick test samples ---
    N = len(test_dataset)
    rng = np.random.RandomState(42)
    indices = rng.choice(N, size=min(num_samples, N), replace=False)

    # --- Load GT + measurements ---
    images, measurements = [], []
    for i in indices:
        sample = test_dataset[int(i)]
        target = sample['target']
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target.copy())
        target = target.float().to(device)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        images.append(target)
        obs = forward_op({'target': target.unsqueeze(0)})
        measurements.append(obs)
    images = torch.stack(images)
    measurements = torch.cat(measurements)

    # --- Generate reconstructions ---
    cbg_recons, dps_recons, plain_recons = [], [], []
    cbg_losses, dps_losses, plain_losses = [], [], []

    for i in tqdm.trange(num_samples, desc="Generating samples"):
        y_i = measurements[i:i+1]

        # CBG
        torch.manual_seed(42 + i)
        recon = cbg_sample(net, classifier, forward_op, y_i, scheduler,
                           guidance_scale=guidance_scale, device=device)
        cbg_recons.append(recon.cpu())
        cbg_losses.append(forward_op.loss(recon, y_i).item())

        # DPS
        if run_dps:
            torch.manual_seed(42 + i)
            recon = dps_sample(net, forward_op, y_i, scheduler,
                               guidance_scale=dps_guidance_scale, device=device)
            dps_recons.append(recon.cpu())
            dps_losses.append(forward_op.loss(recon, y_i).item())

        # Plain
        if run_plain:
            torch.manual_seed(42 + i)
            recon = plain_sample(net, scheduler, device)
            plain_recons.append(recon.cpu())
            plain_losses.append(forward_op.loss(recon, y_i).item())

    # --- Build column list ---
    cols = [("GT", images.cpu(), None)]
    cols.append(("CBG", torch.cat(cbg_recons), cbg_losses))
    if run_dps:
        cols.append(("DPS", torch.cat(dps_recons), dps_losses))
    if run_plain:
        cols.append(("Plain", torch.cat(plain_recons), plain_losses))

    ncols = len(cols)

    # --- Plot ---
    fig, axes = plt.subplots(num_samples, ncols,
                             figsize=(3.5 * ncols, 3.5 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    for i in range(num_samples):
        for j, (label, recons, losses) in enumerate(cols):
            ax = axes[i, j]
            img = recons[i].squeeze().numpy()
            ax.imshow(img, cmap='viridis')
            if losses is not None:
                ax.set_title(f"{label} (L2={losses[i]:.3f})", fontsize=10)
            else:
                ax.set_title(f"{label} #{indices[i]}", fontsize=10)
            ax.axis('off')

    # Summary title
    parts = [f"CBG avg={np.mean(cbg_losses):.4f}"]
    if run_dps:
        parts.append(f"DPS avg={np.mean(dps_losses):.4f}")
    if run_plain:
        parts.append(f"Plain avg={np.mean(plain_losses):.4f}")
    plt.suptitle(" | ".join(parts), fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {out_path}")
    print(f"CBG losses:   {[f'{x:.3f}' for x in cbg_losses]}")
    if run_dps:
        print(f"DPS losses:   {[f'{x:.3f}' for x in dps_losses]}")
    if run_plain:
        print(f"Plain losses: {[f'{x:.3f}' for x in plain_losses]}")


if __name__ == "__main__":
    main()
