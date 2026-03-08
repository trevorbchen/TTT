"""
TTT-CBG evaluation plugin for InverseBench.

Loads a pre-trained MeasurementPredictor (CBG classifier) and runs
CBGDPS-style guided sampling: the guidance gradient flows through the
small classifier network (~10M params) instead of the full diffusion
model (~300M params).

Usage via InverseBench:
    python main.py problem=inv-scatter pretrain=inv-scatter algorithm=ttt_cbg
"""

import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from .base import Algo
from utils.scheduler import Scheduler

# classifier.py lives in the repo root (copied there by setup script)
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from classifier import load_classifier, GradientPredictor


class TTTCBG(Algo):
    """Inference algorithm using a pre-trained CBG classifier.

    Runs reverse diffusion (PF-ODE Euler) with classifier-based guidance.
    For forward mode: gradient flows through the diffusion model (like DPS)
    but uses the classifier as a differentiable surrogate for the forward
    operator A.
    For legacy (tweedie/direct) mode: gradient only through the classifier.
    """

    def __init__(self, net, forward_op, classifier_path,
                 diffusion_scheduler_config, guidance_scale=1.0, sde=False):
        super().__init__(net, forward_op)
        device = next(net.parameters()).device
        self.classifier = load_classifier(classifier_path, device=device)
        self.classifier.eval()
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.guidance_scale = guidance_scale
        self.sde = sde

        # Load target_mode from checkpoint metadata
        ckpt_meta = torch.load(classifier_path, map_location='cpu', weights_only=False)
        self.target_mode = ckpt_meta.get("target_mode", "tweedie")
        self.is_grad_pred = isinstance(self.classifier, GradientPredictor)
        del ckpt_meta

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device

        if num_samples > 1:
            obs = observation.repeat(
                num_samples, *([1] * (observation.ndim - 1)))
        else:
            obs = observation

        # Precompute flat y for forward-mode guidance
        if not self.is_grad_pred and self.target_mode == "forward":
            if obs.is_complex():
                y_flat = torch.view_as_real(obs).flatten(1).float()
            else:
                y_flat = obs.flatten(1).float()

        # Initial noise
        x = torch.randn(
            num_samples, self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device
        ) * self.scheduler.sigma_max

        pbar = tqdm(range(self.scheduler.num_steps), desc="TTT-CBG sampling")
        for i in pbar:
            sigma = self.scheduler.sigma_steps[i]
            scaling = self.scheduler.scaling_steps[i]
            factor = self.scheduler.factor_steps[i]
            scaling_factor = self.scheduler.scaling_factor[i]
            sigma_t = torch.as_tensor(sigma).to(device)

            if self.is_grad_pred:
                # GradientPredictor: no autograd needed
                with torch.no_grad():
                    denoised = self.net(x / scaling, sigma_t)
                    pred_grad = self.classifier(
                        x / scaling, sigma_t, obs, denoised=denoised)
                    gnorm = pred_grad.flatten(1).norm(dim=-1).clamp(min=1e-8)
                    normalized_grad = pred_grad / gnorm.view(-1, 1, 1, 1)
                    denoised_clean = denoised
            elif self.target_mode == "forward":
                # Gradient flows through diffusion model → classifier
                # (like DPS, but classifier replaces A)
                x_in = x.detach().requires_grad_(True)
                self.net.requires_grad_(True)
                denoised = self.net(x_in / scaling, sigma_t)

                pred = self.classifier(denoised)
                loss_per_sample = (pred.flatten(1) - y_flat).pow(2).sum(-1)
                grad_x = torch.autograd.grad(loss_per_sample.sum(), x_in)[0]

                self.net.requires_grad_(False)

                # Recompute denoised cleanly for the ODE step
                with torch.no_grad():
                    denoised_clean = self.net(x / scaling, sigma_t)

                with torch.no_grad():
                    norm_factor = loss_per_sample.sqrt()
                    norm_factor = norm_factor.view(
                        -1, *([1] * (grad_x.ndim - 1)))
                    norm_factor = norm_factor.clamp(min=1e-8)
                    normalized_grad = grad_x / norm_factor
            else:
                # Legacy mode: grad only through classifier
                with torch.no_grad():
                    denoised = self.net(x / scaling, sigma_t)

                x_in = x.detach().requires_grad_(True)
                pred = self.classifier(
                    x_in / scaling, sigma_t, obs,
                    denoised=denoised)
                if getattr(self.classifier, 'scalar_output', False):
                    loss_per_sample = pred.squeeze(-1)
                else:
                    loss_per_sample = pred.pow(2).flatten(1).sum(-1)
                grad_x = torch.autograd.grad(loss_per_sample.sum(), x_in)[0]

                denoised_clean = denoised

                with torch.no_grad():
                    norm_factor = loss_per_sample.sqrt()
                    norm_factor = norm_factor.view(
                        -1, *([1] * (grad_x.ndim - 1)))
                    norm_factor = norm_factor.clamp(min=1e-8)
                    normalized_grad = grad_x / norm_factor

            # PF-ODE Euler step
            with torch.no_grad():
                score = (denoised_clean - x / scaling) / sigma ** 2 / scaling

                if self.sde:
                    epsilon = torch.randn_like(x)
                    x = (x * scaling_factor + factor * score
                         + np.sqrt(factor) * epsilon)
                else:
                    x = x * scaling_factor + factor * score * 0.5

                # Apply guidance
                x = x - self.guidance_scale * normalized_grad

                # NaN guard
                if torch.isnan(x).any():
                    break

        return x
