"""
DPS (Diffusion Posterior Sampling) baseline for InverseBench.

Gradient flows through the full diffusion model (~300M params) at each step.
This is the standard DPS approach (Chung et al., 2023): at each diffusion step,
compute Tweedie estimate, apply forward operator, compute measurement loss,
and backprop through the full model to get the guidance gradient.

Much slower per-step than CBG but requires no pre-training.

Usage via InverseBench:
    python main.py problem=inv-scatter pretrain=inv-scatter algorithm=dps
"""

import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from .base import Algo
from utils.scheduler import Scheduler


class DPS(Algo):
    """DPS inference: gradient through full diffusion model.

    At each reverse-diffusion step:
      1. Enable gradients on net, compute Tweedie denoised estimate
      2. Apply forward operator, compute measurement loss ||A(denoised) - y||^2
      3. Backprop through full diffusion model to get grad w.r.t. x
      4. Disable gradients on net
      5. Per-sample normalization + PF-ODE Euler step + guidance
    """

    def __init__(self, net, forward_op, diffusion_scheduler_config,
                 guidance_scale=1.0, sde=False):
        super().__init__(net, forward_op)
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.guidance_scale = guidance_scale
        self.sde = sde

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device

        if num_samples > 1:
            obs = observation.repeat(
                num_samples, *([1] * (observation.ndim - 1)))
        else:
            obs = observation

        # Initial noise
        x = torch.randn(
            num_samples, self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device
        ) * self.scheduler.sigma_max

        pbar = tqdm(range(self.scheduler.num_steps), desc="DPS sampling")
        for i in pbar:
            sigma = self.scheduler.sigma_steps[i]
            scaling = self.scheduler.scaling_steps[i]
            factor = self.scheduler.factor_steps[i]
            scaling_factor = self.scheduler.scaling_factor[i]

            sigma_t = torch.as_tensor(sigma).to(device)

            # 1. Tweedie with grad through full model
            x_in = x.detach().requires_grad_(True)
            self.net.requires_grad_(True)
            denoised = self.net(x_in / scaling, sigma_t)

            # 2. Measurement loss: ||A(denoised) - y||^2
            y_hat = self.forward_op({'target': denoised})
            residual = y_hat - obs
            if residual.is_complex():
                # Complex observations: use real-valued squared norm
                loss_per_sample = torch.view_as_real(residual).pow(2).flatten(1).sum(-1)
            else:
                loss_per_sample = residual.pow(2).flatten(1).sum(-1)

            # 3. Backprop through full diffusion model
            grad_x = torch.autograd.grad(loss_per_sample.sum(), x_in)[0]
            self.net.requires_grad_(False)

            # 4. Per-sample normalization
            with torch.no_grad():
                norm_factor = loss_per_sample.sqrt()
                norm_factor = norm_factor.view(
                    -1, *([1] * (grad_x.ndim - 1)))
                norm_factor = norm_factor.clamp(min=1e-8)
                normalized_grad = grad_x / norm_factor

            # 5. PF-ODE Euler step (recompute denoised without grad for clean step)
            with torch.no_grad():
                denoised_clean = self.net(x / scaling, sigma_t)
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
