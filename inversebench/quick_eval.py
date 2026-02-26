"""Quick eval: generate a few samples to check if conditioned LoRA is working.

Usage:
    python quick_eval.py problem=inv-scatter pretrain=inv-scatter \
        +ttt.lora_path=exps/ttt/.../lora_final.pt \
        +ttt.num_eval=8 +ttt.eval_out=quick_eval.png
"""
import os
import hydra
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_original_cwd
from utils.helper import open_url
from algo.lora import load_conditioned_lora
from utils.scheduler import Scheduler


def plain_diffusion_sample(net, scheduler, n, device, store=None, y=None):
    if store is not None and y is not None:
        store.set(y)
    C = net.img_channels
    H = W = net.img_resolution
    x = torch.randn(n, C, H, W, device=device) * scheduler.sigma_steps[0]
    with torch.no_grad():
        for i in range(scheduler.num_steps):
            sigma = scheduler.sigma_steps[i]
            scaling = scheduler.scaling_steps[i]
            factor = scheduler.factor_steps[i]
            denoised = net(x / scaling, sigma)
            score = (denoised - x / scaling) / sigma**2 / scaling
            x = x * scheduler.scaling_factor[i] + factor * score * 0.5
    if store is not None:
        store.clear()
    return x


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig):
    device = torch.device('cuda')
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    ttt = OmegaConf.to_container(config.get("ttt", {}), resolve=True)
    lora_path = ttt.get("lora_path")
    num_samples = ttt.get("num_eval", 8)
    out_path = ttt.get("eval_out", "quick_eval.png")
    sched_cfg = {"num_steps": 50, "schedule": "vp", "timestep": "vp", "scaling": "vp"}

    # Load model (same as train_ttt.py)
    print("Loading model...")
    forward_op = instantiate(config.problem.model, device=device)
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
    scheduler = Scheduler(**sched_cfg)

    # Load dataset
    print("Loading dataset...")
    dataset = instantiate(config.pretrain.data)
    N = len(dataset)

    # Pick random eval indices
    rng = np.random.RandomState(123)
    indices = rng.choice(N, size=num_samples, replace=False)

    # Load ground truth + measurements
    images, measurements = [], []
    for i in indices:
        sample = dataset[int(i)]
        target = torch.from_numpy(sample['target'].copy()).float().to(device)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        images.append(target)
        obs = forward_op({'target': target.unsqueeze(0)})
        measurements.append(obs)
    images = torch.stack(images)  # [N, C, H, W]
    measurements = torch.cat(measurements)  # [N, ...]

    # Load LoRA
    print(f"Loading LoRA from {lora_path}...")
    lora_modules, store = load_conditioned_lora(net, lora_path)

    # Generate LoRA samples
    print("Generating LoRA samples...")
    lora_recons = []
    for i in range(num_samples):
        y_i = measurements[i:i+1]
        recon = plain_diffusion_sample(net, scheduler, 1, device, store=store, y=y_i)
        lora_recons.append(recon.cpu())
    lora_recons = torch.cat(lora_recons)

    # Generate plain (LoRA off) samples
    print("Generating plain samples...")
    for m in lora_modules:
        m.scaling = 0.0
    plain_recons = []
    for i in range(num_samples):
        y_i = measurements[i:i+1]
        recon = plain_diffusion_sample(net, scheduler, 1, device, store=store, y=y_i)
        plain_recons.append(recon.cpu())
    plain_recons = torch.cat(plain_recons)

    # Compute measurement losses
    lora_losses, plain_losses = [], []
    for i in range(num_samples):
        y_i = measurements[i:i+1]
        lora_losses.append(forward_op.loss(lora_recons[i:i+1].to(device), y_i).item())
        plain_losses.append(forward_op.loss(plain_recons[i:i+1].to(device), y_i).item())

    # Plot grid: rows = samples, cols = [GT, LoRA, Plain]
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    for i in range(num_samples):
        gt = images[i].cpu().squeeze()
        lora_img = lora_recons[i].squeeze()
        plain_img = plain_recons[i].squeeze()

        axes[i, 0].imshow(gt.numpy(), cmap='viridis')
        axes[i, 0].set_title(f"GT #{indices[i]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(lora_img.numpy(), cmap='viridis')
        axes[i, 1].set_title(f"LoRA (loss={lora_losses[i]:.3f})")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(plain_img.numpy(), cmap='viridis')
        axes[i, 2].set_title(f"Plain (loss={plain_losses[i]:.3f})")
        axes[i, 2].axis('off')

    plt.suptitle(f"LoRA avg={np.mean(lora_losses):.4f}, Plain avg={np.mean(plain_losses):.4f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")
    print(f"LoRA losses: {[f'{x:.3f}' for x in lora_losses]}")
    print(f"Plain losses: {[f'{x:.3f}' for x in plain_losses]}")


if __name__ == "__main__":
    main()
