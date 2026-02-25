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
        +cbg.train_pct=10 +cbg.lr=1e-4 +cbg.num_epochs=50
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

from classifier import MeasurementPredictor, save_classifier
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
            "epoch": 0,
            "total_epochs": 0,
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
    ax.set_ylabel("MSE loss")
    ax.set_title("Per-step loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    epochs = list(range(1, len(epoch_train) + 1))
    ax.plot(epochs, epoch_train, "o-", label="train")
    ax.plot(epochs, epoch_val, "s-", label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg MSE loss")
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

        if target_mode == "tweedie":
            denoised = net(x_noisy, sigma)
            y_hat = forward_op({'target': denoised})
        else:  # direct
            y_hat = forward_op({'target': x_noisy})
        residual = y_hat - y_batch
        if classifier.measurement_decoder is not None:
            if residual.is_complex():
                target = torch.view_as_real(residual).flatten(1).float()
            else:
                target = residual.flatten(1).float()
        else:
            target = residual
        if normalize_target:
            tnorm = target.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            target = target / tnorm
        pred = classifier(x_noisy, sigma, y_batch)

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
    num_epochs    = cbg.get("num_epochs", 50)
    grad_clip     = cbg.get("grad_clip", 10.0)
    train_pct     = cbg.get("train_pct", 10)
    val_fraction  = cbg.get("val_fraction", 0.1)
    save_every    = cbg.get("save_every", 10)
    save_dir      = cbg.get("save_dir", "exps/cbg")
    target_mode   = cbg.get("target_mode", "tweedie")  # "tweedie" or "direct"
    normalize_target = cbg.get("normalize_target", True)
    assert target_mode in ("tweedie", "direct"), \
        f"Unknown target_mode={target_mode!r}, expected 'tweedie' or 'direct'"

    # --- Output directory ---
    problem_name = config.problem.get("name", "unknown")
    root = Path(save_dir) / f"{problem_name}_cbg_{target_mode}_{train_pct}pct_lr{lr}_ch{base_channels}"
    root.mkdir(parents=True, exist_ok=True)

    # --- Logger ---
    logger = Logger(root)
    logger.log(f"CBG Training for InverseBench")
    logger.log(f"  target_mode={target_mode}, normalize_target={normalize_target}, "
               f"base_channels={base_channels}, "
               f"lr={lr}, epochs={num_epochs}, train_pct={train_pct}")
    logger.log(f"  output: {root}")
    logger.log(f"  device: {device}")
    logger.update_progress(status="loading", total_epochs=num_epochs)

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
    ).to(device)

    num_params = sum(p.numel() for p in classifier.parameters())
    logger.log(f"MeasurementPredictor: {num_params/1e6:.2f}M parameters")

    # --- 5. Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_epochs, 10), eta_min=lr * 0.01)

    # --- 6. Train ---
    step_losses = []
    epoch_train_losses = []
    epoch_val_losses = []
    grad_norms = []
    best_val_loss = float('inf')
    logger.update_progress(status="training")

    for epoch in range(1, num_epochs + 1):
        t_epoch = time.time()
        classifier.train()
        order = np.random.permutation(n_train)
        epoch_loss = 0.0
        epoch_target_energy = 0.0
        num_batches = 0

        pbar = tqdm.tqdm(range(0, n_train, batch_size),
                         desc=f"Epoch {epoch}/{num_epochs}")

        for start in pbar:
            idx = order[start:start+batch_size]
            x0 = images[idx]
            y_batch = measurements[idx]
            B = x0.shape[0]

            sigma = sample_sigma(net, B, device)
            sigma_bc = sigma.view(-1, 1, 1, 1)
            eps = torch.randn_like(x0)
            x_noisy = x0 + sigma_bc * eps

            with torch.no_grad():
                if target_mode == "tweedie":
                    denoised = net(x_noisy, sigma)
                    y_hat = forward_op({'target': denoised})
                else:  # direct
                    y_hat = forward_op({'target': x_noisy})
                residual = y_hat - y_batch
                if classifier.measurement_decoder is not None:
                    # Non-image obs: flatten residual to real for measurement-space loss
                    if residual.is_complex():
                        target = torch.view_as_real(residual).flatten(1).float()
                    else:
                        target = residual.flatten(1).float()
                else:
                    target = residual
                # Per-sample normalization: classifier learns direction, not magnitude
                if normalize_target:
                    tnorm = target.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    target = target / tnorm

            pred = classifier(x_noisy, sigma, y_batch)
            # Per-sample sum loss (not MSE mean) — prevents gradient vanishing
            # when measurement dim D is large (MSE divides by D, shrinking grads)
            loss = (pred - target).pow(2).flatten(1).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                classifier.parameters(), grad_clip).item()
            grad_norms.append(gn)
            optimizer.step()

            loss_val = loss.item()
            # Track relative error: ||pred - target||² / ||target||²
            with torch.no_grad():
                target_energy = target.pow(2).flatten(1).sum(-1).mean().item()
                rel_err = loss_val / max(target_energy, 1e-10)
            step_losses.append(loss_val)
            epoch_loss += loss_val
            epoch_target_energy += target_energy
            num_batches += 1
            pbar.set_postfix(loss=f"{loss_val:.6f}", rel=f"{rel_err:.3f}")

        avg_train_loss = epoch_loss / max(num_batches, 1)
        epoch_train_losses.append(avg_train_loss)

        # Validation
        classifier.eval()
        val_loss = 0.0
        val_batches = 0
        if eval_images is not None:
            with torch.no_grad():
                for start in range(0, n_val, batch_size):
                    x0 = eval_images[start:start+batch_size]
                    y_batch = eval_measurements[start:start+batch_size]
                    B = x0.shape[0]

                    sigma = sample_sigma(net, B, device)
                    sigma_bc = sigma.view(-1, 1, 1, 1)
                    eps = torch.randn_like(x0)
                    x_noisy = x0 + sigma_bc * eps

                    if target_mode == "tweedie":
                        denoised = net(x_noisy, sigma)
                        y_hat = forward_op({'target': denoised})
                    else:  # direct
                        y_hat = forward_op({'target': x_noisy})
                    residual = y_hat - y_batch
                    if classifier.measurement_decoder is not None:
                        if residual.is_complex():
                            target = torch.view_as_real(residual).flatten(1).float()
                        else:
                            target = residual.flatten(1).float()
                    else:
                        target = residual
                    if normalize_target:
                        tnorm = target.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        target = target / tnorm
                    pred = classifier(x_noisy, sigma, y_batch)
                    val_loss += (pred - target).pow(2).flatten(1).sum(-1).mean().item()
                    val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1) if val_batches > 0 \
            else avg_train_loss
        epoch_val_losses.append(avg_val_loss)

        lr_scheduler.step()
        epoch_time = time.time() - t_epoch
        avg_gnorm = np.mean(grad_norms[-num_batches:]) if num_batches > 0 else 0

        avg_target_energy = epoch_target_energy / max(num_batches, 1)
        avg_rel_err = avg_train_loss / max(avg_target_energy, 1e-10)
        logger.log(f"Epoch {epoch:3d}/{num_epochs} | "
                   f"train={avg_train_loss:.6f} | val={avg_val_loss:.6f} | "
                   f"rel_err={avg_rel_err:.3f} | "
                   f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                   f"gnorm={avg_gnorm:.4f} | time={epoch_time:.0f}s")

        # Save best
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            save_classifier(classifier, str(root / "classifier_best.pt"),
                            metadata={"epoch": epoch,
                                      "val_loss": avg_val_loss,
                                      "target_mode": target_mode})
            logger.log(f"  -> New best val_loss={avg_val_loss:.6f}")

        # Periodic checkpoint
        if epoch % save_every == 0:
            save_classifier(classifier, str(root / f"classifier_epoch{epoch}.pt"),
                            metadata={"epoch": epoch,
                                      "train_loss": avg_train_loss,
                                      "val_loss": avg_val_loss,
                                      "target_mode": target_mode})

        # Progress + loss curves
        logger.update_progress(
            status="training", epoch=epoch,
            train_loss=avg_train_loss, val_loss=avg_val_loss,
            best_val_loss=best_val_loss)
        save_loss_curves(step_losses, epoch_train_losses, epoch_val_losses,
                         grad_norms, root)

    # --- 7. Save final ---
    save_classifier(classifier, str(root / "classifier_final.pt"),
                    metadata={"epochs": num_epochs,
                              "problem": problem_name,
                              "train_pct": train_pct,
                              "target_mode": target_mode,
                              "normalize_target": normalize_target,
                              "best_val_loss": best_val_loss,
                              "final_train_loss": epoch_train_losses[-1],
                              "final_val_loss": epoch_val_losses[-1]})
    logger.log(f"Saved classifier_final.pt")

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
