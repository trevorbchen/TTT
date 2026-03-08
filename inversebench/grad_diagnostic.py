"""Compare CBG vs DPS gradient magnitudes at each diffusion step.

Runs both methods on the same test sample with the same seed, logging:
  - raw gradient norm ||∇_{x_t} loss||
  - loss value
  - normalized gradient norm
  - cosine similarity between CBG and DPS gradients

Usage:
    python grad_diagnostic.py problem=inv-scatter pretrain=inv-scatter \
        +eval.classifier_path=path/to/classifier_best.pt
"""

import pickle
import torch
import numpy as np
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


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig):
    device = torch.device('cuda')
    ev = OmegaConf.to_container(config.get("eval", {}), resolve=True)
    classifier_path = ev["classifier_path"]
    num_steps = ev.get("num_steps", 200)
    num_samples = ev.get("num_samples", 3)
    guidance_scale = ev.get("guidance_scale", 1.0)

    # Load components
    forward_op = instantiate(config.problem.model, device=device)
    test_dataset = instantiate(config.problem.data)

    ckpt_path = config.problem.prior
    try:
        with open_url(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
            net = ckpt['ema'].to(device)
    except Exception:
        net = instantiate(config.pretrain.model)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt.get('ema', ckpt.get('net', ckpt)))
        net = net.to(device)
    del ckpt
    net.eval()
    net.requires_grad_(False)

    classifier = load_classifier(classifier_path, device=device)
    classifier.eval()

    ckpt_meta = torch.load(classifier_path, map_location='cpu', weights_only=False)
    target_mode = ckpt_meta.get("target_mode", "tweedie")
    del ckpt_meta
    print(f"Target mode: {target_mode}")

    scheduler = Scheduler(num_steps=num_steps, schedule="vp",
                          timestep="vp", scaling="vp")

    F_cos = torch.nn.functional.cosine_similarity

    for sample_idx in range(num_samples):
        sample = test_dataset[sample_idx]
        target = sample['target']
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target.copy())
        target = target.float().to(device)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        obs = forward_op({'target': target.unsqueeze(0)})

        # Precompute y_flat for CBG forward mode
        if obs.is_complex():
            y_flat = torch.view_as_real(obs).flatten(1).float()
        else:
            y_flat = obs.flatten(1).float()

        # Same initial noise
        torch.manual_seed(42 + sample_idx)
        x_init = torch.randn(1, net.img_channels, net.img_resolution,
                              net.img_resolution, device=device) * scheduler.sigma_max

        # Run both in parallel, tracking gradients
        x_cbg = x_init.clone()
        x_dps = x_init.clone()

        print(f"\n{'='*100}")
        print(f"Sample {sample_idx}")
        print(f"{'='*100}")
        print(f"{'step':>5} {'sigma':>8} | {'CBG_grad':>10} {'CBG_loss':>10} {'CBG_norm':>10} | "
              f"{'DPS_grad':>10} {'DPS_loss':>10} {'DPS_norm':>10} | {'cos_sim':>8} {'grad_ratio':>10}")
        print("-" * 115)

        log_steps = list(range(0, 20)) + list(range(20, num_steps, 10))

        for i in range(num_steps):
            sigma = scheduler.sigma_steps[i]
            scaling = scheduler.scaling_steps[i]
            factor = scheduler.factor_steps[i]
            scaling_factor = scheduler.scaling_factor[i]
            sigma_t = torch.as_tensor(sigma).to(device)

            # --- CBG gradient ---
            x_in_cbg = x_cbg.detach().requires_grad_(True)
            net.requires_grad_(True)
            denoised_cbg = net(x_in_cbg / scaling, sigma_t)
            pred_cbg = classifier(denoised_cbg)
            loss_cbg = (pred_cbg.flatten(1) - y_flat).pow(2).sum(-1)
            grad_cbg = torch.autograd.grad(loss_cbg.sum(), x_in_cbg)[0]
            net.requires_grad_(False)

            # --- DPS gradient ---
            x_in_dps = x_dps.detach().requires_grad_(True)
            net.requires_grad_(True)
            denoised_dps = net(x_in_dps / scaling, sigma_t)
            y_hat_dps = forward_op({'target': denoised_dps})
            residual_dps = y_hat_dps - obs
            if residual_dps.is_complex():
                loss_dps = torch.view_as_real(residual_dps).pow(2).flatten(1).sum(-1)
            else:
                loss_dps = residual_dps.pow(2).flatten(1).sum(-1)
            grad_dps = torch.autograd.grad(loss_dps.sum(), x_in_dps)[0]
            net.requires_grad_(False)

            # --- Metrics ---
            grad_cbg_norm = grad_cbg.flatten(1).norm(dim=-1).item()
            grad_dps_norm = grad_dps.flatten(1).norm(dim=-1).item()
            loss_cbg_val = loss_cbg.item()
            loss_dps_val = loss_dps.item()

            # Normalized gradients (same normalization used in sampling)
            cbg_nf = max(loss_cbg_val ** 0.5, 1e-8)
            dps_nf = max(loss_dps_val ** 0.5, 1e-8)
            norm_cbg = grad_cbg_norm / cbg_nf
            norm_dps = grad_dps_norm / dps_nf

            cos = F_cos(grad_cbg.flatten(1), grad_dps.flatten(1), dim=-1).item()
            ratio = grad_cbg_norm / max(grad_dps_norm, 1e-8)

            if i in log_steps:
                print(f"{i:>5} {sigma:>8.3f} | {grad_cbg_norm:>10.4f} {loss_cbg_val:>10.4f} {norm_cbg:>10.4f} | "
                      f"{grad_dps_norm:>10.4f} {loss_dps_val:>10.4f} {norm_dps:>10.4f} | {cos:>8.4f} {ratio:>10.4f}")

            # --- Update both trajectories with their own gradients ---
            with torch.no_grad():
                # CBG update
                denoised_cbg_clean = net(x_cbg / scaling, sigma_t)
                score_cbg = (denoised_cbg_clean - x_cbg / scaling) / sigma ** 2 / scaling
                x_cbg = x_cbg * scaling_factor + factor * score_cbg * 0.5
                normalized_cbg = grad_cbg / max(cbg_nf, 1e-8)
                x_cbg = x_cbg - guidance_scale * normalized_cbg

                # DPS update
                denoised_dps_clean = net(x_dps / scaling, sigma_t)
                score_dps = (denoised_dps_clean - x_dps / scaling) / sigma ** 2 / scaling
                x_dps = x_dps * scaling_factor + factor * score_dps * 0.5
                normalized_dps = grad_dps / max(dps_nf, 1e-8)
                x_dps = x_dps - guidance_scale * normalized_dps

        # Final losses
        with torch.no_grad():
            cbg_final = forward_op.loss(x_cbg, obs).item()
            dps_final = forward_op.loss(x_dps, obs).item()
        print(f"\nFinal: CBG L2={cbg_final:.4f}, DPS L2={dps_final:.4f}")


if __name__ == "__main__":
    main()
