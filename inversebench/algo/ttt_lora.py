"""
TTT-LoRA evaluation plugin for InverseBench.

Loads a pre-trained LoRA checkpoint, then runs plain unconditional diffusion
(no measurement guidance) to produce reconstructions. The LoRA adapter has
already internalized the measurement-to-reconstruction mapping during training.

Usage via InverseBench:
    python main.py problem=inv-scatter pretrain=inv-scatter algorithm=ttt_lora
"""

import torch
import numpy as np
from tqdm import tqdm

from .base import Algo
from .lora import load_conditioned_lora
from utils.scheduler import Scheduler


class TTTLoRA(Algo):
    """Inference algorithm using a pre-trained TTT-LoRA adapter.

    Runs unconditional reverse diffusion (ODE or SDE) with the LoRA-adapted
    score model.  If the LoRA was trained with y-conditioning (y_channels > 0),
    the observation is fed through the MeasurementStore before sampling.
    """

    def __init__(self, net, forward_op, lora_path,
                 diffusion_scheduler_config, sde=False):
        super().__init__(net, forward_op)
        self.lora_modules, self.store = load_conditioned_lora(net, lora_path)
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde

        # Freeze LoRA params at inference
        for m in self.lora_modules:
            for p in m.parameters():
                p.requires_grad = False

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device

        # Set measurement for y-conditioned LoRA
        if self.store is not None:
            if num_samples > 1:
                obs = observation.repeat(num_samples, *([1] * (observation.ndim - 1)))
            else:
                obs = observation
            self.store.set(obs)

        # Initial noise
        x = torch.randn(
            num_samples, self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device
        ) * self.scheduler.sigma_max

        # Reverse diffusion (Euler)
        pbar = tqdm(range(self.scheduler.num_steps),
                    desc="TTT-LoRA sampling")
        for i in pbar:
            sigma = self.scheduler.sigma_steps[i]
            scaling = self.scheduler.scaling_steps[i]
            factor = self.scheduler.factor_steps[i]
            scaling_factor = self.scheduler.scaling_factor[i]

            with torch.no_grad():
                denoised = self.net(
                    x / scaling,
                    torch.as_tensor(sigma).to(device))

            score = (denoised - x / scaling) / sigma ** 2 / scaling

            if self.sde:
                epsilon = torch.randn_like(x)
                x = x * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x = x * scaling_factor + factor * score * 0.5

        if self.store is not None:
            self.store.clear()

        return x
