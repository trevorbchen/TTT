"""
CBG (Classifier-Based Guidance) training script for InverseBench.

Trains a MeasurementPredictor network on a pretrained diffusion model
using the InverseBench data/model/operator interfaces.

The training loop mirrors cbg_train.py but uses InverseBench conventions:
  - Data loaded via `instantiate(config.pretrain.data)`
  - Diffusion model loaded from `config.problem.prior`
  - Forward operator via `instantiate(config.problem.model)`
  - Sigma sampled log-uniformly (matching InverseBench convention)

Outputs: classifier_best.pt, classifier_final.pt, progress.json,
         losses.json, loss_curve.png, eval_results.json

Usage:
    python train_cbg.py \
        problem=inv-scatter pretrain=inv-scatter \
        +cbg.train_pct=10 +cbg.lr=1e-4 +cbg.num_sigma_steps=200
"""

import json
import math
import pickle
import time
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import hydra
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

import sys
# classifier.py lives in the repo root (same dir as this script when deployed)
_repo_root = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from classifier import MeasurementPredictor, TransformerCBG, ForwardSurrogate, FNOSurrogate, UNetSurrogate, GradientPredictor, save_classifier
from utils.helper import open_url


# ---------------------------------------------------------------------------
# Logger (same pattern as train_ttt.py)
# ---------------------------------------------------------------------------

class Logger:
    """Tees all print output to a log file and keeps structured training state."""

    def __init__(self, root):
        self.root = Path(root)
        self.log_path = self.root / "train.log"
        self.progress_path = self.root / "progress.json"
        self.log_file = open(str(self.log_path), "w")
        self.start_time = time.time()

        self.state = {
            "status": "initializing",
            "start_time": datetime.now().isoformat(),
            "pass": 0,
            "step": 0,
            "total_steps": 0,
            "train_loss": None,
            "val_loss": None,
            "best_val_loss": None,
            "elapsed_sec": 0,
        }

    def log(self, msg):
        ts = f"[{timedelta(seconds=int(time.time() - self.start_time))}]"
        line = f"{ts} {msg}"
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def update_progress(self, **kwargs):
        self.state["elapsed_sec"] = int(time.time() - self.start_time)
        self.state.update(kwargs)
        with open(str(self.progress_path), "w") as f:
            json.dump(self.state, f, indent=2)

    def close(self):
        self.log_file.close()


# ---------------------------------------------------------------------------
# Sigma sampling (log-uniform, matching InverseBench convention)
# ---------------------------------------------------------------------------

def sample_sigma(net, batch_size, device):
    """Sample noise levels log-uniformly between sigma_min and sigma_max."""
    sigma_min = net.sigma_min if net.sigma_min > 0 else 0.01
    sigma_max = net.sigma_max if net.sigma_max < float('inf') else 100.0
    log_sigma = torch.rand(batch_size, device=device) * (
        math.log(sigma_max) - math.log(sigma_min)
    ) + math.log(sigma_min)
    return log_sigma.exp()


def get_vp_sigma_steps(num_steps, beta_max=20, beta_min=0.1, epsilon=1e-5):
    """Compute VP scheduler sigma levels (matching InverseBench VPScheduler).

    Returns num_steps sigma values from sigma_max (high noise) to sigma_min.
    """
    time_steps = torch.linspace(1, epsilon, num_steps + 1)
    t = time_steps[:-1]  # exclude final zero-sigma step
    a = beta_max - beta_min
    b = beta_min
    beta_int = ((a * t + b) ** 2 - b ** 2) / (2 * a)
    alpha = torch.exp(-beta_int)
    sigma = torch.sqrt(1.0 / alpha - 1.0)
    return sigma


# ---------------------------------------------------------------------------
# Data loading (same pattern as train_ttt.py)
# ---------------------------------------------------------------------------

def load_subset(dataset, indices, forward_op, device):
    """Load a subset of images and compute measurements."""
    images, measurements = [], []
    for i in tqdm.tqdm(indices, desc="Loading data"):
        sample = dataset[int(i)]
        target = sample['target']
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target.copy())
        target = target.float().to(device)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        images.append(target)
        obs = forward_op({'target': target.unsqueeze(0)})
        measurements.append(obs)
    return torch.stack(images), torch.cat(measurements)


# ---------------------------------------------------------------------------
# On-the-fly diffusion prior sampling
# ---------------------------------------------------------------------------

class DiffusionPriorBuffer:
    """Buffer of images sampled from diffusion prior + their measurements."""

    def __init__(self, net, forward_op, device,
                 buffer_size=128, gen_batch_size=8, num_steps=50):
        self.net = net
        self.forward_op = forward_op
        self.device = device
        self.buffer_size = buffer_size
        self.gen_batch_size = gen_batch_size
        self.num_steps = num_steps

        from utils.scheduler import Scheduler
        self.scheduler = Scheduler(
            num_steps=num_steps, schedule="vp",
            timestep="vp", scaling="vp")

        self.images = None       # [buffer_size, C, H, W]
        self.measurements = None # [buffer_size, ...]
        self._cursor = 0        # next unused index
        self._perm = None       # shuffled index order

    @torch.no_grad()
    def _sample_batch(self, batch_size):
        """PF-ODE Euler sampling (same as plain_sample in eval_cbg.py)."""
        x = torch.randn(batch_size, self.net.img_channels,
                        self.net.img_resolution, self.net.img_resolution,
                        device=self.device) * self.scheduler.sigma_max
        for i in range(self.num_steps):
            sigma = self.scheduler.sigma_steps[i]
            scaling = self.scheduler.scaling_steps[i]
            factor = self.scheduler.factor_steps[i]
            scaling_factor = self.scheduler.scaling_factor[i]
            denoised = self.net(
                x / scaling, torch.as_tensor(sigma).to(self.device))
            score = (denoised - x / scaling) / sigma ** 2 / scaling
            x = x * scaling_factor + factor * score * 0.5
        return x

    def refresh(self, logger=None):
        """Regenerate entire buffer."""
        t0 = time.time()
        all_imgs, all_meas = [], []
        remaining = self.buffer_size
        while remaining > 0:
            bs = min(remaining, self.gen_batch_size)
            imgs = self._sample_batch(bs)
            # forward_op may not support batching — apply per-sample
            for j in range(bs):
                meas_j = self.forward_op({'target': imgs[j:j+1]})
                all_meas.append(meas_j)
            all_imgs.append(imgs)
            remaining -= bs
        self.images = torch.cat(all_imgs)[:self.buffer_size]
        self.measurements = torch.cat(all_meas)[:self.buffer_size]
        self._perm = torch.randperm(self.buffer_size)
        self._cursor = 0
        if logger:
            logger.log(f"  [GenBuffer] Refreshed {self.buffer_size} samples "
                       f"in {time.time()-t0:.1f}s")

    def sample(self, logger=None):
        """Next unused (x0, y) pair. Auto-refreshes when buffer is exhausted."""
        if self._cursor >= self.buffer_size:
            self.refresh(logger)
        idx = self._perm[self._cursor].item()
        self._cursor += 1
        return self.images[idx:idx+1], self.measurements[idx:idx+1]


# ---------------------------------------------------------------------------
# Loss curve plotting
# ---------------------------------------------------------------------------

def save_loss_curves(step_losses, epoch_train, epoch_val, grad_norms, root):
    """Save loss curve plot + raw data."""
    with open(str(root / "losses.json"), "w") as f:
        json.dump({
            "step_losses": step_losses,
            "epoch_train_losses": epoch_train,
            "epoch_val_losses": epoch_val,
            "grad_norms": grad_norms,
        }, f)

    if not epoch_train:
        return

    n_plots = 3 if grad_norms else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))

    ax = axes[0]
    ax.plot(step_losses, alpha=0.3, linewidth=0.5, label="per step")
    if len(step_losses) > 10:
        window = max(len(step_losses) // 20, 5)
        smoothed = np.convolve(step_losses, np.ones(window)/window,
                               mode='valid')
        ax.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
                linewidth=1.5, label=f"smoothed (w={window})")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Loss")
    ax.set_title("Per-step loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    epochs = list(range(1, len(epoch_train) + 1))
    ax.plot(epochs, epoch_train, "o-", label="train")
    ax.plot(epochs, epoch_val, "s-", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg loss")
    ax.set_title("Per-epoch loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if grad_norms and n_plots == 3:
        ax = axes[2]
        ax.plot(grad_norms, alpha=0.4, linewidth=0.5, label="per step")
        if len(grad_norms) > 10:
            window = max(len(grad_norms) // 20, 5)
            smoothed = np.convolve(grad_norms, np.ones(window)/window,
                                   mode='valid')
            ax.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
                    linewidth=1.5, label=f"smoothed (w={window})")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Gradient norm")
        ax.set_title("Gradient norms (pre-clip)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(root / "loss_curve.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_held_out(classifier, net, forward_op, eval_images,
                      eval_measurements, root, logger, batch_size=4,
                      target_mode="tweedie", normalize_target=True):
    """Evaluate classifier quality on held-out data."""
    device = eval_images.device
    n_eval = len(eval_images)
    logger.log(f"{'='*60}")
    logger.log(f"Evaluating on {n_eval} held-out samples (target_mode={target_mode})...")
    logger.log(f"{'='*60}")

    classifier.eval()
    total_pred_loss = 0.0
    total_target_loss = 0.0
    num_batches = 0

    for start in tqdm.trange(0, n_eval, batch_size, desc="Eval"):
        x0 = eval_images[start:start+batch_size]
        y_batch = eval_measurements[start:start+batch_size]
        B = x0.shape[0]

        sigma = sample_sigma(net, B, device)
        sigma_bc = sigma.view(-1, 1, 1, 1)
        eps = torch.randn_like(x0)
        x_noisy = x0 + sigma_bc * eps

        if target_mode in ("tweedie", "forward"):
            denoised = net(x_noisy, sigma)
            # forward_op may not batch — apply per-sample
            y_hat = torch.cat([forward_op({'target': denoised[j:j+1]})
                               for j in range(B)])
        else:  # direct
            denoised = None
            y_hat = torch.cat([forward_op({'target': x_noisy[j:j+1]})
                               for j in range(B)])
        if target_mode == "forward":
            raw_target = y_hat
        else:
            raw_target = y_hat - y_batch
        has_meas_dec = hasattr(classifier, 'measurement_decoder') and classifier.measurement_decoder is not None
        if has_meas_dec or isinstance(classifier, (ForwardSurrogate, FNOSurrogate, UNetSurrogate)):
            if raw_target.is_complex():
                target = torch.view_as_real(raw_target).flatten(1).float()
            else:
                target = raw_target.flatten(1).float()
        else:
            target = raw_target
        if normalize_target:
            tnorm = target.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            target = target / tnorm
        pred = classifier(x_noisy, sigma,
                          None if target_mode == "forward" else y_batch,
                          denoised=denoised if target_mode in ("tweedie", "forward") else None)

        total_pred_loss += (pred - target).pow(2).flatten(1).sum(-1).mean().item()
        total_target_loss += target.norm().item()
        num_batches += 1

    avg_pred_loss = total_pred_loss / max(num_batches, 1)
    avg_target_mag = total_target_loss / max(num_batches, 1)

    results = {
        "n_eval": n_eval,
        "prediction_mse": avg_pred_loss,
        "avg_target_magnitude": avg_target_mag,
    }

    logger.log(f"  Prediction MSE: {avg_pred_loss:.6f}")
    logger.log(f"  Avg target magnitude: {avg_target_mag:.4f}")

    with open(str(root / "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.tf32:
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # --- CBG config ---
    cbg = OmegaConf.to_container(config.get("cbg", {}), resolve=True)
    base_channels = cbg.get("base_channels", 64)
    channel_mult  = cbg.get("channel_mult", [1, 2, 4, 4])
    emb_dim       = cbg.get("emb_dim", 256)
    attn_heads    = cbg.get("attn_heads", 4)
    lr            = cbg.get("lr", 1e-4)
    weight_decay  = cbg.get("weight_decay", 1e-4)
    batch_size    = cbg.get("batch_size", 8)
    grad_clip     = cbg.get("grad_clip", 10.0)
    train_pct     = cbg.get("train_pct", 10)
    val_fraction  = cbg.get("val_fraction", 0.1)
    save_dir      = cbg.get("save_dir", "exps/cbg")
    target_mode   = cbg.get("target_mode", "tweedie")  # "tweedie" or "direct"
    normalize_target = cbg.get("normalize_target", True)
    decoder_hidden = cbg.get("decoder_hidden", 2048)
    num_res_blocks = cbg.get("num_res_blocks", 1)
    num_tokens     = cbg.get("num_tokens", 0)
    # Sequential sigma training (TTT-style): single pass, all sigmas per sample
    num_sigma_steps  = cbg.get("num_sigma_steps", 200)
    sigma_batch_size = cbg.get("sigma_batch_size", 8)
    num_passes       = cbg.get("num_passes", 1)
    val_every_steps  = cbg.get("val_every_steps", 500)
    save_every_steps = cbg.get("save_every_steps", 2000)
    snr_gamma        = cbg.get("snr_gamma", 5.0)  # Min-SNR-γ weighting (0=off)
    save_final       = cbg.get("save_final", True)  # save classifier_final.pt
    # Transformer CBG config
    arch             = cbg.get("arch", "unet")  # "unet" or "transformer"
    loss_type        = cbg.get("loss_type", "mse")  # "mse" or "cosine"
    cosine_eps       = cbg.get("cosine_eps", 1e-6)
    t_embed_dim      = cbg.get("embed_dim", 256)    # transformer embed dim
    t_num_layers     = cbg.get("num_layers", 4)
    t_num_heads      = cbg.get("num_heads", 4)
    t_ffn_mult       = cbg.get("ffn_mult", 2)
    t_dropout        = cbg.get("dropout", 0.0)
    use_tweedie_input = cbg.get("use_tweedie_input", False)
    t_scalar_output   = cbg.get("scalar_output", False)
    t_input_norm      = cbg.get("input_norm", True)
    t_query_output    = cbg.get("query_output", False)
    sigma_min_curriculum = cbg.get("sigma_min_curriculum", 0.0)
    sigma_max_curriculum = cbg.get("sigma_max_curriculum", 1000.0)
    # FNO architecture config
    fno_width     = cbg.get("fno_width", 64)
    fno_modes     = cbg.get("fno_modes", 16)
    fno_mlp_ratio = cbg.get("fno_mlp_ratio", 2)
    # Jacobian matching loss
    jac_loss      = cbg.get("jac_loss", False)
    jac_lambda    = cbg.get("jac_lambda", 1.0)
    grad_normalize_target = cbg.get("grad_normalize_target", False)
    # On-the-fly diffusion prior sampling
    gen_samples          = cbg.get("gen_samples", False)
    gen_buffer_size      = cbg.get("gen_buffer_size", 128)
    gen_batch_size       = cbg.get("gen_batch_size", 8)
    gen_steps            = cbg.get("gen_steps", 50)
    gen_mix_ratio        = cbg.get("gen_mix_ratio", 1.0)
    assert target_mode in ("tweedie", "direct", "forward"), \
        f"Unknown target_mode={target_mode!r}, expected 'tweedie', 'direct', or 'forward'"
    assert arch in ("unet", "transformer", "surrogate", "fno", "unet_surrogate", "grad_predictor"), \
        f"Unknown arch={arch!r}, expected 'unet', 'transformer', 'surrogate', 'fno', 'unet_surrogate', or 'grad_predictor'"
    assert loss_type in ("mse", "cosine"), \
        f"Unknown loss_type={loss_type!r}, expected 'mse' or 'cosine'"

    # --- Output directory ---
    problem_name = config.problem.get("name", "unknown")
    snr_tag = f"_snrg{snr_gamma}" if snr_gamma > 0 else "_nosnr"
    norm_tag = "_norm" if normalize_target else "_nonorm"
    pass_tag = f"_p{num_passes}" if num_passes > 1 else ""
    arch_tag = f"_{arch}" if arch != "unet" else ""
    loss_tag = f"_{loss_type}" if loss_type != "mse" else ""
    jac_tag = f"_jac{jac_lambda}" if jac_loss else ""
    gen_tag = f"_gen{gen_buffer_size}" if gen_samples else ""
    root = Path(save_dir) / f"{problem_name}_cbg_{target_mode}_{train_pct}pct_lr{lr}_ch{base_channels}{arch_tag}{loss_tag}{snr_tag}{norm_tag}{pass_tag}{jac_tag}{gen_tag}"
    root.mkdir(parents=True, exist_ok=True)

    # --- Logger ---
    logger = Logger(root)
    logger.log(f"CBG Training for InverseBench (sequential sigma, TTT-style)")
    logger.log(f"  num_sigma_steps={num_sigma_steps}, "
               f"sigma_batch_size={sigma_batch_size}, "
               f"num_passes={num_passes}")
    logger.log(f"  arch={arch}, loss_type={loss_type}, "
               f"target_mode={target_mode}, normalize_target={normalize_target}, "
               f"base_channels={base_channels}, decoder_hidden={decoder_hidden}, "
               f"num_res_blocks={num_res_blocks}, num_tokens={num_tokens}, "
               f"lr={lr}, train_pct={train_pct}")
    if arch == "transformer":
        logger.log(f"  transformer: embed_dim={t_embed_dim}, num_layers={t_num_layers}, "
                   f"num_heads={t_num_heads}, ffn_mult={t_ffn_mult}, dropout={t_dropout}, "
                   f"use_tweedie_input={use_tweedie_input}, scalar_output={t_scalar_output}, "
                   f"input_norm={t_input_norm}, query_output={t_query_output}")
    if arch == "fno":
        logger.log(f"  fno: width={fno_width}, modes={fno_modes}, num_layers={t_num_layers}, "
                   f"dropout={t_dropout}, mlp_ratio={fno_mlp_ratio}")
    if jac_loss:
        logger.log(f"  jacobian matching: enabled, lambda={jac_lambda}")
    if sigma_min_curriculum > 0 or sigma_max_curriculum < 999:
        logger.log(f"  sigma curriculum: [{sigma_min_curriculum}, {sigma_max_curriculum}]")
    logger.log(f"  snr_gamma={snr_gamma} ({'enabled' if snr_gamma > 0 else 'disabled'})")
    if gen_samples:
        logger.log(f"  gen: buffer={gen_buffer_size}, steps={gen_steps}, mix={gen_mix_ratio}")
    logger.log(f"  output: {root}")
    logger.log(f"  device: {device}")
    logger.update_progress(status="loading")

    # Save config
    with open(str(root / "config.yaml"), "w") as f:
        yaml.safe_dump(OmegaConf.to_container(config, resolve=True), f,
                       default_flow_style=False)

    # --- 1. Load components ---
    logger.log("Loading forward operator...")
    forward_op = instantiate(config.problem.model, device=device)

    logger.log("Loading training dataset...")
    train_dataset = instantiate(config.pretrain.data)

    logger.log("Loading pretrained model...")
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
    logger.log(f"  Model: {type(net).__name__} -> {type(net.model).__name__}")
    logger.log(f"  Resolution: {net.img_resolution}, Channels: {net.img_channels}")

    # --- 1b. Diffusion prior buffer (optional) ---
    gen_buffer = None
    if gen_samples:
        logger.log(f"Setting up diffusion prior buffer: size={gen_buffer_size}, "
                   f"steps={gen_steps}, batch={gen_batch_size}, mix={gen_mix_ratio}")
        gen_buffer = DiffusionPriorBuffer(
            net=net, forward_op=forward_op, device=device,
            buffer_size=gen_buffer_size, gen_batch_size=gen_batch_size,
            num_steps=gen_steps)
        gen_buffer.refresh(logger)

    # --- 2. Train/eval split ---
    N = len(train_dataset)
    all_indices = np.random.permutation(N)
    n_total = max(1, int(N * train_pct / 100))
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    train_indices = all_indices[:n_train]
    eval_indices = all_indices[n_train:n_total]

    logger.log(f"Split: {N} total -> using {n_total} ({train_pct}%) "
               f"-> {n_train} train, {n_val} eval")
    np.savez(str(root / "split_indices.npz"),
             train=train_indices, eval=eval_indices)

    logger.log("Loading train split...")
    images, measurements = load_subset(train_dataset, train_indices,
                                       forward_op, device)
    logger.log(f"  images={images.shape}, measurements={measurements.shape}")

    if n_val > 0:
        logger.log("Loading eval split...")
        eval_images, eval_measurements = load_subset(
            train_dataset, eval_indices, forward_op, device)
        logger.log(f"  eval_images={eval_images.shape}")
    else:
        eval_images = eval_measurements = None

    # --- 3. Infer operator output shape ---
    y_sample = measurements[:1]
    is_image_obs = (y_sample.ndim == 4 and not y_sample.is_complex())
    if is_image_obs:
        # Image-like observations: directly use channels and spatial dims
        obs_shape = None
        y_channels = y_sample.shape[1]
        out_channels = y_sample.shape[1]
        out_size = (y_sample.shape[2], y_sample.shape[3])
        logger.log(f"Operator output (image): channels={out_channels}, "
                   f"size={out_size}")
    else:
        # Non-image observations (e.g. complex scattering data):
        # use MeasurementEncoder to project to spatial features
        obs_shape = tuple(y_sample.shape[1:])
        y_channels = cbg.get("y_channels", 4)
        out_channels = y_channels
        out_size = (net.img_resolution, net.img_resolution)
        # Compute flat measurement dim for decoder (complex -> view_as_real doubles)
        if y_sample.is_complex():
            meas_flat_dim = y_sample[0].numel() * 2
        else:
            meas_flat_dim = y_sample[0].numel()
        logger.log(f"Operator output (non-image): obs_shape={obs_shape}, "
                   f"complex={y_sample.is_complex()}, meas_flat_dim={meas_flat_dim}")
        logger.log(f"  Using MeasurementEncoder -> y_channels={y_channels}, "
                   f"out_size={out_size}")

    # --- 4. Build classifier ---
    if arch == "unet_surrogate":
        assert not is_image_obs, "UNetSurrogate requires non-image observations"
        classifier = UNetSurrogate(
            img_resolution=net.img_resolution,
            img_channels=net.img_channels,
            obs_shape=obs_shape,
            meas_flat_dim=meas_flat_dim,
            base_channels=base_channels,
            channel_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            attn_heads=attn_heads,
            dropout=t_dropout,
            mlp_ratio=fno_mlp_ratio,
        ).to(device)
        num_params = sum(p.numel() for p in classifier.parameters())
        logger.log(f"UNetSurrogate: {num_params/1e6:.2f}M parameters")
        logger.log(f"  UNet: ch={base_channels}, mult={channel_mult}, "
                   f"res_blocks={num_res_blocks}, dropout={t_dropout}")
    elif arch == "grad_predictor":
        assert not is_image_obs, "GradientPredictor requires non-image observations"
        classifier = GradientPredictor(
            img_resolution=net.img_resolution,
            img_channels=net.img_channels,
            obs_shape=obs_shape,
            y_channels=y_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            emb_dim=emb_dim,
            attn_heads=attn_heads,
            num_res_blocks=num_res_blocks,
            dropout=t_dropout,
        ).to(device)
        num_params = sum(p.numel() for p in classifier.parameters())
        logger.log(f"GradientPredictor: {num_params/1e6:.2f}M parameters")
        logger.log(f"  UNet: ch={base_channels}, mult={channel_mult}, "
                   f"res_blocks={num_res_blocks}, emb_dim={emb_dim}, dropout={t_dropout}")
        if grad_normalize_target:
            logger.log(f"  Gradient target normalization: ENABLED")
    elif arch == "fno":
        assert not is_image_obs, "FNOSurrogate requires non-image observations"
        classifier = FNOSurrogate(
            img_resolution=net.img_resolution,
            img_channels=net.img_channels,
            obs_shape=obs_shape,
            meas_flat_dim=meas_flat_dim,
            width=fno_width,
            modes=fno_modes,
            num_layers=t_num_layers,
            dropout=t_dropout,
            mlp_ratio=fno_mlp_ratio,
        ).to(device)
        num_params = sum(p.numel() for p in classifier.parameters())
        logger.log(f"FNOSurrogate: {num_params/1e6:.2f}M parameters")
        logger.log(f"  FNO: width={fno_width}, modes={fno_modes}, "
                   f"layers={t_num_layers}, mlp_ratio={fno_mlp_ratio}")
    elif arch == "surrogate":
        assert not is_image_obs, "ForwardSurrogate requires non-image observations"
        classifier = ForwardSurrogate(
            img_resolution=net.img_resolution,
            img_channels=net.img_channels,
            obs_shape=obs_shape,
            meas_flat_dim=meas_flat_dim,
            embed_dim=t_embed_dim,
            num_layers=t_num_layers,
            num_heads=t_num_heads,
            ffn_mult=t_ffn_mult,
            dropout=t_dropout,
        ).to(device)
        num_params = sum(p.numel() for p in classifier.parameters())
        logger.log(f"ForwardSurrogate: {num_params/1e6:.2f}M parameters")
        logger.log(f"  Simple ViT: image -> A(image), no sigma/x_t/y")
    elif arch == "transformer":
        assert not is_image_obs, "TransformerCBG requires non-image observations"
        classifier = TransformerCBG(
            img_resolution=net.img_resolution,
            img_channels=net.img_channels,
            obs_shape=obs_shape,
            meas_flat_dim=meas_flat_dim,
            embed_dim=t_embed_dim,
            num_layers=t_num_layers,
            num_heads=t_num_heads,
            ffn_mult=t_ffn_mult,
            dropout=t_dropout,
            use_tweedie_input=use_tweedie_input,
            scalar_output=t_scalar_output,
            input_norm=t_input_norm,
            query_output=t_query_output,
        ).to(device)
        num_params = sum(p.numel() for p in classifier.parameters())
        logger.log(f"TransformerCBG: {num_params/1e6:.2f}M parameters")
        if use_tweedie_input:
            logger.log(f"  Tweedie input: ENABLED (denoised image fed as extra tokens)")
        if t_scalar_output:
            logger.log(f"  Scalar output: ENABLED (predicting ||residual||²)")
    else:
        classifier = MeasurementPredictor(
            in_channels=net.img_channels,
            y_channels=y_channels,
            out_channels=out_channels,
            out_size=out_size,
            base_channels=base_channels,
            emb_dim=emb_dim,
            channel_mult=channel_mult,
            attn_heads=attn_heads,
            obs_shape=obs_shape,
            img_resolution=net.img_resolution,
            meas_flat_dim=meas_flat_dim if not is_image_obs else None,
            decoder_hidden=decoder_hidden,
            num_res_blocks=num_res_blocks,
            num_tokens=num_tokens,
            enc_spatial_size=32,
        ).to(device)
        num_params = sum(p.numel() for p in classifier.parameters())
        dec_params = sum(p.numel() for p in classifier.measurement_decoder.parameters()) \
            if classifier.measurement_decoder is not None else 0
        logger.log(f"MeasurementPredictor: {num_params/1e6:.2f}M parameters "
                   f"(decoder: {dec_params/1e6:.2f}M)")

    # --- 5. Optimizer ---
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    # --- 6. Train ---
    step_losses = []
    epoch_train_losses = []
    epoch_val_losses = []
    grad_norms = []
    best_val_loss = float('inf')
    logger.update_progress(status="training")

    # =================================================================
    # Continuous sigma training:
    #   Each step picks one training sample and samples sigma_batch_size
    #   random sigmas from log-uniform distribution (same as validation).
    #   num_passes controls how many times each sample is seen (each
    #   time with fresh random sigmas and noise).
    # =================================================================
    total_steps = n_train * num_passes

    # Step-based LR warmup + cosine decay
    warmup_lr_steps = max(int(total_steps * 0.05), 1)
    if warmup_lr_steps > 0 and total_steps > warmup_lr_steps:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_lr_steps)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_lr_steps, 1),
            eta_min=lr * 0.01)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup_sched, cosine_sched],
            milestones=[warmup_lr_steps])
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 10), eta_min=lr * 0.01)

    logger.log(f"Continuous sigma training: {num_passes} seeds/sample, "
               f"sigma_batch_size={sigma_batch_size}, total={total_steps}")
    logger.log(f"  sigma: log-uniform in [{net.sigma_min}, {net.sigma_max}]")
    if snr_gamma > 0:
        logger.log(f"  Min-SNR-γ={snr_gamma}")
    logger.log(f"  LR warmup: {warmup_lr_steps} steps, cosine to {total_steps}")
    logger.update_progress(total_steps=total_steps)

    global_step = 0
    log_interval = max(total_steps // 100, 50)
    running_loss = 0.0
    running_rel = 0.0
    running_count = 0

    # Build sample indices: each sample repeated num_passes times, all shuffled
    all_indices = list(range(n_train)) * num_passes
    np.random.shuffle(all_indices)

    classifier.train()
    t_pass = time.time()
    is_grad_pred = isinstance(classifier, GradientPredictor)

    pbar = tqdm.tqdm(all_indices, desc="Training")

    for i in pbar:
                # Data source: generated buffer vs LMDB
                use_generated = (gen_buffer is not None and
                                 torch.rand(1).item() < gen_mix_ratio)
                if use_generated:
                    x0, y_single = gen_buffer.sample(logger)
                else:
                    x0 = images[i:i+1]
                    y_single = measurements[i:i+1]
                B = sigma_batch_size
                sigma = sample_sigma(net, B, device)

                # Repeat sample for each sigma in batch
                x0_rep = x0.expand(B, -1, -1, -1)
                y_rep = y_single.expand(B, *[-1] * (y_single.ndim - 1))
                sigma_bc = sigma.view(-1, 1, 1, 1)
                eps = torch.randn(B, *x0.shape[1:], device=device)
                x_noisy = x0_rep + sigma_bc * eps

                if is_grad_pred:
                    # --- GradientPredictor: target is true gradient ---
                    with torch.no_grad():
                        denoised = net(x_noisy, sigma)
                    # Compute true gradient: ∇_denoised ||A(denoised) - y||²
                    denoised_g = denoised.detach().requires_grad_(True)
                    # forward_op doesn't batch — apply per-sample
                    y_hat_list = [forward_op({'target': denoised_g[j:j+1]})
                                  for j in range(B)]
                    y_hat = torch.cat(y_hat_list)
                    residual = y_hat - y_rep
                    if residual.is_complex():
                        loss_fwd = torch.view_as_real(residual).pow(2).flatten(1).sum(-1)
                    else:
                        loss_fwd = residual.pow(2).flatten(1).sum(-1)
                    grad_true = torch.autograd.grad(
                        loss_fwd.sum(), denoised_g, create_graph=False)[0].detach()
                    # [B, img_channels, H, W]
                    target = grad_true
                    if grad_normalize_target:
                        gnorm = target.flatten(1).norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        target = target / gnorm.view(-1, 1, 1, 1)

                    pred = classifier(x_noisy, sigma, y_rep, denoised=denoised)
                    # MSE loss on gradient images
                    per_sample_loss = (pred - target).pow(2).flatten(1).sum(-1)  # [B]

                else:
                    with torch.no_grad():
                        if target_mode == "tweedie":
                            denoised = net(x_noisy, sigma)
                            y_hat = forward_op({'target': denoised})
                        elif target_mode == "forward":
                            denoised = net(x_noisy, sigma)
                            y_hat = forward_op({'target': denoised})
                        else:  # direct
                            denoised = None
                            y_hat = forward_op({'target': x_noisy})
                        # Target: residual (tweedie/direct) or forward prediction
                        if target_mode == "forward":
                            raw_target = y_hat  # predict A(denoised) directly
                        else:
                            raw_target = y_hat - y_rep  # predict residual
                        # Flatten to measurement space
                        if hasattr(classifier, 'measurement_decoder') and classifier.measurement_decoder is not None:
                            if raw_target.is_complex():
                                target = torch.view_as_real(raw_target).flatten(1).float()
                            else:
                                target = raw_target.flatten(1).float()
                        elif isinstance(classifier, (TransformerCBG, ForwardSurrogate, FNOSurrogate, UNetSurrogate)):
                            if raw_target.is_complex():
                                target = torch.view_as_real(raw_target).flatten(1).float()
                            else:
                                target = raw_target.flatten(1).float()
                        else:
                            target = raw_target
                        # Scalar target: ||residual||² (1 dim)
                        if t_scalar_output:
                            target = target.pow(2).flatten(1).sum(-1, keepdim=True)  # [B, 1]
                        elif normalize_target:
                            tnorm = target.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                            target = target / tnorm

                    pred = classifier(x_noisy, sigma,
                                      None if target_mode == "forward" else y_rep,
                                      denoised=denoised if (use_tweedie_input or target_mode == "forward") else None)
                if is_grad_pred:
                    pass  # per_sample_loss already set above
                elif t_scalar_output:
                    # Scalar output: always MSE loss
                    per_sample_loss = (pred - target).pow(2).squeeze(-1)  # [B]
                elif loss_type == "cosine":
                    # Cosine similarity loss; mask near-zero targets
                    target_norm = target.norm(dim=-1)
                    valid = target_norm > cosine_eps
                    cos_sim = F.cosine_similarity(pred, target, dim=-1)  # [B]
                    per_sample_loss = 1.0 - cos_sim  # [B]
                    per_sample_loss = torch.where(valid, per_sample_loss,
                                                  torch.zeros_like(per_sample_loss))
                else:
                    per_sample_loss = (pred - target).pow(2).flatten(1).sum(-1)  # [B]
                # Min-SNR-γ weighting (computed on-the-fly for sampled sigmas)
                if snr_gamma > 0:
                    snr = 1.0 / sigma.pow(2)
                    w = torch.clamp(snr, max=snr_gamma)
                    w = w / w.mean()  # normalize so mean weight = 1
                else:
                    w = torch.ones(B, device=device)
                forward_loss = (w * per_sample_loss).mean()

                # --- Jacobian matching loss ---
                jac_match_loss = torch.tensor(0.0, device=device)
                jac_cos_sim_val = 0.0
                if jac_loss and not is_grad_pred and target_mode == "forward" and denoised is not None:
                    # True operator gradient (forward_op doesn't batch)
                    denoised_jac = denoised.detach().requires_grad_(True)
                    y_hat_true = torch.cat([forward_op({'target': denoised_jac[j:j+1]})
                                            for j in range(B)])
                    if y_hat_true.is_complex():
                        y_hat_true_real = torch.view_as_real(y_hat_true)
                    else:
                        y_hat_true_real = y_hat_true
                    y_flat_for_jac = target.detach()  # [B, meas_flat_dim]
                    if y_rep.is_complex():
                        y_rep_flat = torch.view_as_real(y_rep).flatten(1).float()
                    else:
                        y_rep_flat = y_rep.flatten(1).float()
                    loss_true = (y_hat_true_real.flatten(1).float() - y_rep_flat).pow(2).sum(-1)
                    grad_true = torch.autograd.grad(
                        loss_true.sum(), denoised_jac,
                        create_graph=False)[0].detach()

                    # Surrogate gradient (create_graph=True to backprop through cos_sim)
                    denoised_jac2 = denoised.detach().requires_grad_(True)
                    pred_surr = classifier(denoised_jac2, sigma, None,
                                           denoised=denoised_jac2)
                    loss_surr = (pred_surr - y_rep_flat).pow(2).sum(-1)
                    grad_surr = torch.autograd.grad(
                        loss_surr.sum(), denoised_jac2,
                        create_graph=True)[0]

                    # Cosine similarity loss
                    cos_sim_jac = F.cosine_similarity(
                        grad_surr.flatten(1), grad_true.flatten(1), dim=-1)
                    valid_jac = grad_true.flatten(1).norm(dim=-1) > 1e-6
                    jac_match_loss = torch.where(
                        valid_jac, 1.0 - cos_sim_jac,
                        torch.zeros_like(cos_sim_jac)).mean()
                    jac_cos_sim_val = cos_sim_jac[valid_jac].mean().item() if valid_jac.any() else 0.0

                loss = forward_loss + jac_lambda * jac_match_loss

                optimizer.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(
                    classifier.parameters(), grad_clip).item()
                grad_norms.append(gn)
                optimizer.step()
                lr_scheduler.step()

                loss_val = loss.item()
                step_losses.append(loss_val)
                with torch.no_grad():
                    target_energy = target.pow(2).flatten(1).sum(-1).mean().item()
                    rel_err = loss_val / max(target_energy, 1e-10)
                running_loss += loss_val
                running_rel += rel_err
                running_count += 1
                global_step += 1

                pbar.set_postfix(
                    step=f"{global_step}/{total_steps}",
                    loss=f"{loss_val:.4f}",
                    rel=f"{rel_err:.3f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}")

                # Periodic logging
                if global_step % log_interval == 0 and running_count > 0:
                    avg_loss = running_loss / running_count
                    avg_rel = running_rel / running_count
                    avg_gn = np.mean(grad_norms[-running_count:])
                    jac_str = f" | jac_cos={jac_cos_sim_val:.3f}" if jac_loss else ""
                    logger.log(
                        f"Step {global_step:6d}/{total_steps} | "
                        f"loss={avg_loss:.6f} | rel_err={avg_rel:.3f} | "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                        f"gnorm={avg_gn:.4f}{jac_str}")
                    running_loss = 0.0
                    running_rel = 0.0
                    running_count = 0

                # Periodic validation
                if (global_step % val_every_steps == 0
                        and eval_images is not None):
                    classifier.eval()
                    val_loss = 0.0
                    val_batches = 0
                    if is_grad_pred:
                        # GradientPredictor validation: gradient cosine similarity
                        for v_st in range(0, n_val, batch_size):
                            x0v = eval_images[v_st:v_st+batch_size]
                            y_bv = eval_measurements[v_st:v_st+batch_size]
                            Bv = x0v.shape[0]
                            sv = sample_sigma(net, Bv, device)
                            sv_bc = sv.view(-1, 1, 1, 1)
                            epsv = torch.randn_like(x0v)
                            x_nv = x0v + sv_bc * epsv
                            with torch.no_grad():
                                dv = net(x_nv, sv)
                            # True gradient (needs autograd)
                            dv_g = dv.detach().requires_grad_(True)
                            y_hv = torch.cat([forward_op({'target': dv_g[j:j+1]})
                                              for j in range(Bv)])
                            res_v = y_hv - y_bv
                            if res_v.is_complex():
                                loss_v = torch.view_as_real(res_v).pow(2).flatten(1).sum(-1)
                            else:
                                loss_v = res_v.pow(2).flatten(1).sum(-1)
                            grad_true_v = torch.autograd.grad(
                                loss_v.sum(), dv_g, create_graph=False)[0].detach()
                            with torch.no_grad():
                                pv = classifier(x_nv, sv, y_bv, denoised=dv)
                                cos_v = F.cosine_similarity(
                                    pv.flatten(1), grad_true_v.flatten(1), dim=-1)
                                valid_v = grad_true_v.flatten(1).norm(dim=-1) > 1e-6
                                per_v = torch.where(valid_v, 1.0 - cos_v,
                                                    torch.zeros_like(cos_v))
                                val_loss += per_v.mean().item()
                            val_batches += 1
                    else:
                        with torch.no_grad():
                            for v_st in range(0, n_val, batch_size):
                                x0v = eval_images[v_st:v_st+batch_size]
                                y_bv = eval_measurements[v_st:v_st+batch_size]
                                Bv = x0v.shape[0]
                                sv = sample_sigma(net, Bv, device)
                                sv_bc = sv.view(-1, 1, 1, 1)
                                epsv = torch.randn_like(x0v)
                                x_nv = x0v + sv_bc * epsv
                                if target_mode in ("tweedie", "forward"):
                                    dv = net(x_nv, sv)
                                    # forward_op may not batch — apply per-sample
                                    y_hv = torch.cat([forward_op({'target': dv[j:j+1]})
                                                      for j in range(Bv)])
                                else:
                                    dv = None
                                    y_hv = torch.cat([forward_op({'target': x_nv[j:j+1]})
                                                      for j in range(Bv)])
                                if target_mode == "forward":
                                    raw_tv = y_hv
                                else:
                                    raw_tv = y_hv - y_bv
                                if hasattr(classifier, 'measurement_decoder') and classifier.measurement_decoder is not None:
                                    if raw_tv.is_complex():
                                        tv = torch.view_as_real(raw_tv).flatten(1).float()
                                    else:
                                        tv = raw_tv.flatten(1).float()
                                elif isinstance(classifier, (TransformerCBG, ForwardSurrogate, FNOSurrogate, UNetSurrogate)):
                                    if raw_tv.is_complex():
                                        tv = torch.view_as_real(raw_tv).flatten(1).float()
                                    else:
                                        tv = raw_tv.flatten(1).float()
                                else:
                                    tv = raw_tv
                                if t_scalar_output:
                                    tv = tv.pow(2).flatten(1).sum(-1, keepdim=True)
                                elif normalize_target:
                                    tn = tv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                                    tv = tv / tn
                                pv = classifier(x_nv, sv,
                                                None if target_mode == "forward" else y_bv,
                                                denoised=dv if (use_tweedie_input or target_mode == "forward") else None)
                                if t_scalar_output:
                                    val_loss += (pv - tv).pow(2).squeeze(-1).mean().item()
                                elif loss_type == "cosine":
                                    tv_norm = tv.norm(dim=-1)
                                    valid_v = tv_norm > cosine_eps
                                    cos_v = F.cosine_similarity(pv, tv, dim=-1)
                                    per_v = 1.0 - cos_v
                                    per_v = torch.where(valid_v, per_v,
                                                        torch.zeros_like(per_v))
                                    val_loss += per_v.mean().item()
                                else:
                                    val_loss += (pv - tv).pow(2).flatten(1).sum(-1).mean().item()
                                val_batches += 1
                    avg_val = val_loss / max(val_batches, 1)
                    epoch_val_losses.append(avg_val)
                    avg_recent = np.mean(
                        step_losses[max(0, len(step_losses) - val_every_steps):])
                    epoch_train_losses.append(avg_recent)

                    # Gen-domain validation: evaluate on fresh diffusion samples
                    gen_val_str = ""
                    if gen_buffer is not None:
                        gen_val_loss = 0.0
                        gen_val_batches = 0
                        with torch.no_grad():
                            for _ in range(max(1, n_val // gen_buffer.buffer_size)):
                                gen_buffer.refresh()
                                for gi in range(gen_buffer.buffer_size):
                                    x0g, y_bg = gen_buffer.images[gi:gi+1], gen_buffer.measurements[gi:gi+1]
                                    svg = sample_sigma(net, 1, device)
                                    svg_bc = svg.view(-1, 1, 1, 1)
                                    epsg = torch.randn_like(x0g)
                                    x_ng = x0g + svg_bc * epsg
                                    dg = net(x_ng, svg)
                                    y_hg = forward_op({'target': dg})
                                    if y_hg.is_complex():
                                        tg = torch.view_as_real(y_hg).flatten(1).float()
                                    else:
                                        tg = y_hg.flatten(1).float()
                                    pg = classifier(x_ng, svg, None, denoised=dg)
                                    gen_val_loss += (pg - tg).pow(2).flatten(1).sum(-1).mean().item()
                                    gen_val_batches += 1
                        avg_gen_val = gen_val_loss / max(gen_val_batches, 1)
                        gen_val_str = f" | gen_val={avg_gen_val:.6f}"
                        gen_buffer.refresh()  # restore buffer for training

                    logger.log(f"  VAL @ step {global_step}: "
                               f"train_avg={avg_recent:.6f} | "
                               f"val={avg_val:.6f}{gen_val_str}")

                    is_best = avg_val < best_val_loss
                    if is_best:
                        best_val_loss = avg_val
                        save_classifier(
                            classifier,
                            str(root / "classifier_best.pt"),
                            metadata={"step": global_step,
                                      "val_loss": avg_val,
                                      "target_mode": target_mode})
                        logger.log(f"  -> New best val_loss={avg_val:.6f}")

                    classifier.train()
                    save_loss_curves(step_losses, epoch_train_losses,
                                     epoch_val_losses, grad_norms, root)

                # Periodic checkpoint
                if global_step % save_every_steps == 0:
                    save_classifier(
                        classifier,
                        str(root / f"classifier_step{global_step}.pt"),
                        metadata={"step": global_step,
                                  "target_mode": target_mode})

                # Progress update
                logger.update_progress(
                    status="training",
                    step=global_step, train_loss=loss_val,
                    best_val_loss=best_val_loss)

    pbar.close()
    total_time = time.time() - t_pass
    logger.log(f"Training complete in {total_time:.0f}s "
               f"({global_step} total steps)")

    # Final loss curve
    save_loss_curves(step_losses, epoch_train_losses,
                     epoch_val_losses, grad_norms, root)

    # --- 7. Save final ---
    final_meta = {
        "problem": problem_name,
        "arch": arch,
        "loss_type": loss_type,
        "train_pct": train_pct,
        "target_mode": target_mode,
        "normalize_target": normalize_target,
        "snr_gamma": snr_gamma,
        "best_val_loss": best_val_loss,
        "num_passes": num_passes,
        "num_sigma_steps": num_sigma_steps,
        "sigma_batch_size": sigma_batch_size,
        "total_steps": total_steps,
        "grad_normalize_target": grad_normalize_target,
    }
    if epoch_train_losses:
        final_meta["final_train_loss"] = epoch_train_losses[-1]
    if epoch_val_losses:
        final_meta["final_val_loss"] = epoch_val_losses[-1]
    if save_final:
        save_classifier(classifier, str(root / "classifier_final.pt"),
                        metadata=final_meta)
        logger.log(f"Saved classifier_final.pt")
    else:
        # Save just the metadata (no weights) for sweep analysis
        with open(str(root / "final_meta.json"), "w") as f:
            json.dump(final_meta, f, indent=2)
        logger.log(f"Skipped classifier_final.pt (save_final=false)")

    # --- 8. Evaluate ---
    if eval_images is not None:
        logger.update_progress(status="evaluating")
        evaluate_held_out(classifier, net, forward_op,
                          eval_images, eval_measurements,
                          root, logger, batch_size=batch_size,
                          target_mode=target_mode,
                          normalize_target=normalize_target)

    logger.update_progress(status="done")
    logger.log(f"")
    logger.log(f"All done! Outputs in: {root}")
    logger.log(f"  train.log           - this log file")
    logger.log(f"  progress.json       - check training status anytime")
    logger.log(f"  losses.json         - raw loss numbers + grad norms")
    logger.log(f"  loss_curve.png      - loss + grad norm plots")
    logger.log(f"  eval_results.json   - held-out evaluation")
    logger.log(f"  classifier_final.pt - final checkpoint")
    logger.log(f"  classifier_best.pt  - best epoch checkpoint")
    logger.log(f"  split_indices.npz   - train/eval index split")
    logger.close()


if __name__ == "__main__":
    main()
