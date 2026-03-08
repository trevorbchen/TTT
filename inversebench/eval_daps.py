"""
DAPS evaluation: compare DAPS with true operator vs DAPS with surrogate.

Uses InverseBench's own algo/daps.py (same code as the paper's results)
with the paper's tuned hyperparameters for inverse scattering (Table 12).

DAPS runs Langevin dynamics at x0 level, so the surrogate gradient is just
nabla_{x0} ||surrogate(x0) - y||^2 — NO diffusion model backprop needed.

Usage:
    python eval_daps.py problem=inv-scatter pretrain=inv-scatter \
        +eval.classifier_path=exps/.../classifier_best.pt \
        +eval.num_test=100 +eval.out_dir=exps/eval_daps
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

from classifier import load_classifier
from utils.helper import open_url
from algo.daps import DAPS, LangevinDynamics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def confidence_interval_95(values):
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
# Surrogate operator (drop-in replacement for forward_op in DAPS)
# ---------------------------------------------------------------------------

class SurrogateOperator:
    """Wraps ForwardSurrogate as a drop-in for InverseBench forward_op.

    LangevinDynamics calls operator.gradient(x, measurement) — this routes
    the gradient through the ~3M param surrogate ViT instead of the true A.
    """

    def __init__(self, classifier, device='cuda'):
        self.classifier = classifier
        self.device = device

    def _flat_y(self, y):
        if y.is_complex():
            return torch.view_as_real(y).flatten(1).float()
        return y.flatten(1).float()

    def gradient(self, x, y, return_loss=False):
        """nabla_x ||surrogate(x) - y||^2"""
        x_in = x.clone().detach().requires_grad_(True)
        pred = self.classifier(x_in)
        y_flat = self._flat_y(y)
        loss_val = (pred.flatten(1) - y_flat).pow(2).sum(-1)
        grad_x = torch.autograd.grad(loss_val.sum(), x_in)[0]
        if return_loss:
            return grad_x, loss_val
        return grad_x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config.tf32:
        torch.set_float32_matmul_precision("high")

    # --- Config ---
    ev = OmegaConf.to_container(config.get("eval", {}), resolve=True)
    classifier_path = ev.get("classifier_path")
    num_test = ev.get("num_test", 100)
    out_dir = Path(ev.get("out_dir", "exps/eval_daps"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # DAPS hyperparams — defaults from paper Table 12 (inv-scatter, 360 recv)
    num_annealing_steps = ev.get("num_annealing_steps", 200)
    diffusion_substeps = ev.get("diffusion_substeps", 10)
    sigma_max = ev.get("sigma_max", 100.0)
    sigma_min = ev.get("sigma_min", 0.1)

    # Langevin dynamics — tuned for inv-scatter (Table 12)
    num_lgvd_steps = ev.get("num_lgvd_steps", 50)
    lgvd_lr = ev.get("lgvd_lr", 4e-5)
    tau = ev.get("tau", 1e-4)
    lr_min_ratio = ev.get("lr_min_ratio", 1.0)

    run_true = ev.get("run_true", True)
    run_surrogate = ev.get("run_surrogate", True)
    save_images = ev.get("save_images", False)
    num_vis = ev.get("num_vis", 8)

    assert classifier_path, "Must provide +eval.classifier_path=..."

    print("=== DAPS Evaluation (InverseBench algo/daps.py) ===")
    print(f"  classifier:          {classifier_path}")
    print(f"  num_annealing_steps: {num_annealing_steps}")
    print(f"  diffusion_substeps:  {diffusion_substeps}")
    print(f"  sigma_max/min:       {sigma_max}/{sigma_min}")
    print(f"  num_lgvd_steps:      {num_lgvd_steps}")
    print(f"  lgvd_lr:             {lgvd_lr}")
    print(f"  tau:                 {tau}")
    print(f"  lr_min_ratio:        {lr_min_ratio}")
    print(f"  num_test:            {num_test}")
    print(f"  run_true: {run_true}, run_surrogate: {run_surrogate}")
    print()

    # --- Load components ---
    print("Loading forward operator...")
    forward_op = instantiate(config.problem.model, device=device)

    print("Loading test dataset...")
    test_dataset = instantiate(config.problem.data)

    print("Loading pretrained diffusion model...")
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

    print("Loading classifier (surrogate)...")
    classifier = load_classifier(classifier_path, device=device)
    classifier.eval()
    num_params = sum(p.numel() for p in classifier.parameters())
    print(f"  Surrogate: {num_params/1e6:.2f}M params")

    # --- Build DAPS configs (paper Table 12 defaults for inv-scatter) ---
    annealing_config = {
        'num_steps': num_annealing_steps,
        'sigma_max': sigma_max,
        'sigma_min': sigma_min,
        'schedule': 'linear',
        'timestep': 'poly-7',
    }
    diffusion_config = {
        'num_steps': diffusion_substeps,
        'sigma_min': 0.01,
        'schedule': 'linear',
        'timestep': 'poly-7',
    }
    lgvd_config = {
        'num_steps': num_lgvd_steps,
        'lr': lgvd_lr,
        'tau': tau,
        'lr_min_ratio': lr_min_ratio,
    }

    print(f"\nDAPS config (paper Table 12 defaults for inv-scatter):")
    print(f"  Annealing: {num_annealing_steps} levels, "
          f"sigma [{sigma_min}, {sigma_max}]")
    print(f"  Diffusion: {diffusion_substeps} PF-ODE steps per level")
    print(f"  Langevin:  {num_lgvd_steps} steps, "
          f"lr={lgvd_lr}, tau={tau}, lr_min_ratio={lr_min_ratio}")

    # --- Select test samples ---
    N = len(test_dataset)
    rng = np.random.RandomState(42)
    test_indices = rng.choice(N, size=min(num_test, N), replace=False)
    print(f"\nSelected {len(test_indices)} test samples")

    # --- Load test data ---
    print("Loading test samples...")
    test_images, test_measurements = [], []
    for i in tqdm.tqdm(test_indices, desc="Loading"):
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

    results = {}
    vis_recons = {}

    # --- DAPS with true operator ---
    if run_true:
        print(f"\n{'='*60}")
        print(f"DAPS with TRUE operator ({len(test_indices)} samples)")
        print(f"{'='*60}")

        daps_true = DAPS(net, forward_op, annealing_config,
                         diffusion_config, lgvd_config)

        true_errs = []
        if save_images:
            vis_recons['DAPS-True'] = []
        t0 = time.time()
        for idx in tqdm.trange(len(test_indices), desc="DAPS-True"):
            y_i = test_measurements[idx:idx+1]
            torch.manual_seed(42 + idx)
            recon = daps_true.inference(y_i, verbose=False)
            err = relative_measurement_error(forward_op, recon, y_i)
            true_errs.append(err)
            if save_images and idx < num_vis:
                vis_recons['DAPS-True'].append(recon.detach().cpu())

        true_time = time.time() - t0
        true_mean, true_ci, true_std = confidence_interval_95(true_errs)
        results['daps_true'] = {
            'mean': true_mean, 'std': true_std, 'ci95_half_width': true_ci,
            'per_sample': true_errs, 'total_time_sec': true_time,
            'time_per_sample_sec': true_time / len(test_indices),
        }
        print(f"\nDAPS-True: mean={true_mean:.2f}% +/- {true_ci:.2f} "
              f"(std={true_std:.2f})")
        print(f"  Time: {true_time:.1f}s total, "
              f"{true_time/len(test_indices):.2f}s/sample")

    # --- DAPS with surrogate ---
    if run_surrogate:
        print(f"\n{'='*60}")
        print(f"DAPS with SURROGATE ({len(test_indices)} samples)")
        print(f"{'='*60}")

        surrogate_op = SurrogateOperator(classifier, device=device)
        daps_surr = DAPS(net, surrogate_op, annealing_config,
                         diffusion_config, lgvd_config)

        surr_errs = []
        if save_images:
            vis_recons['DAPS-Surr'] = []
        t0 = time.time()
        for idx in tqdm.trange(len(test_indices), desc="DAPS-Surr"):
            y_i = test_measurements[idx:idx+1]
            torch.manual_seed(42 + idx)
            recon = daps_surr.inference(y_i, verbose=False)
            err = relative_measurement_error(forward_op, recon, y_i)
            surr_errs.append(err)
            if save_images and idx < num_vis:
                vis_recons['DAPS-Surr'].append(recon.detach().cpu())

        surr_time = time.time() - t0
        surr_mean, surr_ci, surr_std = confidence_interval_95(surr_errs)
        results['daps_surrogate'] = {
            'mean': surr_mean, 'std': surr_std, 'ci95_half_width': surr_ci,
            'per_sample': surr_errs, 'total_time_sec': surr_time,
            'time_per_sample_sec': surr_time / len(test_indices),
        }
        print(f"\nDAPS-Surr: mean={surr_mean:.2f}% +/- {surr_ci:.2f} "
              f"(std={surr_std:.2f})")
        print(f"  Time: {surr_time:.1f}s total, "
              f"{surr_time/len(test_indices):.2f}s/sample")

    # --- Save results ---
    results['config'] = {
        'classifier_path': str(classifier_path),
        'num_test': len(test_indices),
        'num_annealing_steps': num_annealing_steps,
        'diffusion_substeps': diffusion_substeps,
        'sigma_max': sigma_max, 'sigma_min': sigma_min,
        'num_lgvd_steps': num_lgvd_steps,
        'lgvd_lr': lgvd_lr, 'tau': tau,
        'lr_min_ratio': lr_min_ratio,
    }

    with open(str(out_dir / "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_dir / 'eval_results.json'}")

    # --- Comparison table ---
    print(f"\n{'='*60}")
    print(f"{'Method':<15} {'Err(%)':>10} {'Std':>10} "
          f"{'95% CI':>18} {'Time/sample':>14}")
    print(f"{'-'*67}")
    if run_true:
        print(f"{'DAPS-True':<15} {true_mean:>10.2f} {true_std:>10.2f} "
              f"{'['+f'{true_mean-true_ci:.2f}, {true_mean+true_ci:.2f}'+']':>18} "
              f"{true_time/len(test_indices):>12.2f}s")
    if run_surrogate:
        print(f"{'DAPS-Surr':<15} {surr_mean:>10.2f} {surr_std:>10.2f} "
              f"{'['+f'{surr_mean-surr_ci:.2f}, {surr_mean+surr_ci:.2f}'+']':>18} "
              f"{surr_time/len(test_indices):>12.2f}s")
    if run_true and run_surrogate:
        speedup = true_time / max(surr_time, 1e-6)
        print(f"\nSurrogate is {speedup:.1f}x "
              f"{'faster' if speedup > 1 else 'slower'} "
              f"than true operator")
    print(f"  Paper reference (DAPS, 360 recv): 1.03% +/- 0.25")
    print(f"{'='*60}")

    # --- Save image grid ---
    if save_images and vis_recons:
        n_vis = min(num_vis, len(test_indices))
        cols = [("GT", [test_images[i:i+1].cpu() for i in range(n_vis)],
                 None)]
        for method_name in ['DAPS-True', 'DAPS-Surr']:
            if method_name in vis_recons:
                key = ('daps_true' if method_name == 'DAPS-True'
                       else 'daps_surrogate')
                losses = results.get(key, {}).get('per_sample', [])
                cols.append((method_name, vis_recons[method_name],
                             losses[:n_vis]))

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
                    ax.set_title(f"{label} ({losses[i]:.1f}%)",
                                 fontsize=10)
                else:
                    ax.set_title(f"{label} #{test_indices[i]}",
                                 fontsize=10)
                ax.axis('off')

        parts = ["Rel. Meas. Error (%)"]
        if run_true:
            parts.append(f"True={true_mean:.1f}%")
        if run_surrogate:
            parts.append(f"Surr={surr_mean:.1f}%")
        plt.suptitle(" | ".join(parts), fontsize=13, fontweight='bold')
        plt.tight_layout()
        img_path = out_dir / "reconstructions.png"
        plt.savefig(str(img_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved reconstruction grid to {img_path}")


if __name__ == "__main__":
    main()
