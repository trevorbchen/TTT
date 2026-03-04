"""
DAPS evaluation: compare DAPS with true operator vs DAPS with surrogate.

Uses the existing DAPS infrastructure (sampler.py, cores/mcmc.py,
cores/scheduler.py) with thin wrappers around InverseBench components.

DAPS runs MCMC at x0 level, so the surrogate gradient is just
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
_project_root = str(Path(__file__).resolve().parent.parent)
for p in [_repo_root, _project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from classifier import load_classifier
from utils.helper import open_url
from sampler import DAPS
import torch.nn as nn


# ---------------------------------------------------------------------------
# Confidence interval
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


# ---------------------------------------------------------------------------
# Model wrapper: InverseBench EDMPrecond → DAPS model interface
# ---------------------------------------------------------------------------

class ModelWrapper(nn.Module):
    """Wraps InverseBench's EDMPrecond with the interface expected by
    DAPS / DiffusionPFODE: .tweedie(), .score(), .get_in_shape().

    With EDM scheduler (scaling=1), DiffusionPFODE passes xt directly
    to model.tweedie(xt, sigma).  InverseBench's net expects input in
    the form (x_0 + sigma*eps), which matches the EDM convention.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def tweedie(self, x, sigma):
        return self.net(x, torch.as_tensor(sigma).to(x.device))

    def score(self, x, sigma):
        d = self.tweedie(x, sigma)
        return (d - x) / sigma ** 2

    def get_in_shape(self):
        return (self.net.img_channels, self.net.img_resolution,
                self.net.img_resolution)


# ---------------------------------------------------------------------------
# Operator wrappers (match MCMCSampler's .gradient / .loss interface)
# ---------------------------------------------------------------------------

class SurrogateOperator:
    """Wraps ForwardSurrogate as a drop-in operator for MCMC.

    gradient(x, y) only flows through the ~3M param surrogate ViT,
    NOT through the forward operator A or the diffusion model.
    """

    def __init__(self, classifier, obs_is_complex=False):
        self.classifier = classifier
        self.obs_is_complex = obs_is_complex

    def _flat_y(self, y):
        if y.is_complex():
            return torch.view_as_real(y).flatten(1).float()
        return y.flatten(1).float()

    def loss(self, x, y):
        """||surrogate(x) - y||^2, per sample. [B]"""
        with torch.no_grad():
            pred = self.classifier(x, None, None, denoised=x)
            y_flat = self._flat_y(y)
            return (pred.flatten(1) - y_flat).pow(2).sum(-1)

    def gradient(self, x, y, return_loss=False):
        """nabla_x ||surrogate(x) - y||^2"""
        x_in = x.clone().detach().requires_grad_(True)
        pred = self.classifier(x_in, None, None, denoised=x_in)
        y_flat = self._flat_y(y)
        loss_val = (pred.flatten(1) - y_flat).pow(2).sum(-1)
        grad_x = torch.autograd.grad(loss_val.sum(), x_in)[0]
        if return_loss:
            return grad_x, loss_val
        return grad_x


class TrueOperatorWrapper:
    """Wraps InverseBench forward_op with gradient/loss interface for MCMC."""

    def __init__(self, forward_op):
        self.forward_op = forward_op

    def loss(self, x, y):
        """||A(x) - y||^2, per sample. [B]"""
        return self.forward_op.loss(x, y)

    def gradient(self, x, y, return_loss=False):
        """nabla_x ||A(x) - y||^2"""
        x_in = x.clone().detach().requires_grad_(True)
        loss_val = self.forward_op.loss(x_in, y)
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

    # DAPS hyperparams (defaults from configs/sampler/edm_daps.yaml)
    num_annealing_steps = ev.get("num_annealing_steps", 200)
    diffusion_substeps = ev.get("diffusion_substeps", 5)
    sigma_max = ev.get("sigma_max", 100.0)
    sigma_min = ev.get("sigma_min", 0.1)

    # MCMC hyperparams (defaults from configs/task/*.yaml pixel configs)
    num_mcmc_steps = ev.get("num_mcmc_steps", 100)
    mcmc_lr = ev.get("mcmc_lr", 5e-5)
    tau = ev.get("tau", 0.01)
    lr_min_ratio = ev.get("lr_min_ratio", 0.01)
    mc_algo = ev.get("mc_algo", "langevin")
    prior_solver = ev.get("prior_solver", "gaussian")

    run_true = ev.get("run_true", True)
    run_surrogate = ev.get("run_surrogate", True)
    save_images = ev.get("save_images", False)
    num_vis = ev.get("num_vis", 8)

    assert classifier_path, "Must provide +eval.classifier_path=..."

    print("=== DAPS Evaluation (using existing DAPS infrastructure) ===")
    print(f"  classifier:          {classifier_path}")
    print(f"  num_annealing_steps: {num_annealing_steps}")
    print(f"  diffusion_substeps:  {diffusion_substeps}")
    print(f"  sigma_max/min:       {sigma_max}/{sigma_min}")
    print(f"  num_mcmc_steps:      {num_mcmc_steps}")
    print(f"  mcmc_lr:             {mcmc_lr}")
    print(f"  tau:                 {tau}")
    print(f"  mc_algo:             {mc_algo}")
    print(f"  prior_solver:        {prior_solver}")
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

    # --- Wrap model and operators ---
    model = ModelWrapper(net)
    surrogate_op = SurrogateOperator(classifier)
    true_op = TrueOperatorWrapper(forward_op)

    # --- Build DAPS sampler using existing infrastructure ---
    annealing_scheduler_config = {
        'name': 'edm',
        'num_steps': num_annealing_steps,
        'sigma_max': sigma_max,
        'sigma_min': sigma_min,
        'timestep': 'poly-7',
    }
    diffusion_scheduler_config = {
        'name': 'edm',
        'num_steps': diffusion_substeps,
        'sigma_min': 0.01,
        'timestep': 'poly-7',
    }
    mcmc_sampler_config = {
        'num_steps': num_mcmc_steps,
        'lr': mcmc_lr,
        'tau': tau,
        'lr_min_ratio': lr_min_ratio,
        'mc_algo': mc_algo,
        'prior_solver': prior_solver,
    }

    daps = DAPS(annealing_scheduler_config, diffusion_scheduler_config,
                mcmc_sampler_config)

    print(f"\nDAPS sampler built:")
    print(f"  Annealing: {num_annealing_steps} levels, "
          f"sigma [{sigma_min}, {sigma_max}]")
    print(f"  Diffusion: {diffusion_substeps} PF-ODE steps per level")
    print(f"  MCMC: {num_mcmc_steps} {mc_algo} steps, "
          f"lr={mcmc_lr}, tau={tau}")

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

        true_losses = []
        if save_images:
            vis_recons['DAPS-True'] = []
        t0 = time.time()
        for idx in tqdm.trange(len(test_indices), desc="DAPS-True"):
            y_i = test_measurements[idx:idx+1]
            torch.manual_seed(42 + idx)
            x_start = daps.get_start(1, model)
            recon = daps.sample(model, x_start, true_op, y_i)
            with torch.no_grad():
                loss = forward_op.loss(recon, y_i).item()
            true_losses.append(loss)
            if save_images and idx < num_vis:
                vis_recons['DAPS-True'].append(recon.detach().cpu())

        true_time = time.time() - t0
        true_mean, true_ci, true_std = confidence_interval_95(true_losses)
        results['daps_true'] = {
            'mean': true_mean, 'std': true_std, 'ci95_half_width': true_ci,
            'per_sample': true_losses, 'total_time_sec': true_time,
            'time_per_sample_sec': true_time / len(test_indices),
        }
        print(f"\nDAPS-True: mean={true_mean:.6f} +/- {true_ci:.6f} "
              f"(std={true_std:.6f})")
        print(f"  Time: {true_time:.1f}s total, "
              f"{true_time/len(test_indices):.2f}s/sample")

    # --- DAPS with surrogate ---
    if run_surrogate:
        print(f"\n{'='*60}")
        print(f"DAPS with SURROGATE ({len(test_indices)} samples)")
        print(f"{'='*60}")

        surr_losses = []
        if save_images:
            vis_recons['DAPS-Surr'] = []
        t0 = time.time()
        for idx in tqdm.trange(len(test_indices), desc="DAPS-Surr"):
            y_i = test_measurements[idx:idx+1]
            torch.manual_seed(42 + idx)
            x_start = daps.get_start(1, model)
            recon = daps.sample(model, x_start, surrogate_op, y_i)
            with torch.no_grad():
                loss = forward_op.loss(recon, y_i).item()
            surr_losses.append(loss)
            if save_images and idx < num_vis:
                vis_recons['DAPS-Surr'].append(recon.detach().cpu())

        surr_time = time.time() - t0
        surr_mean, surr_ci, surr_std = confidence_interval_95(surr_losses)
        results['daps_surrogate'] = {
            'mean': surr_mean, 'std': surr_std, 'ci95_half_width': surr_ci,
            'per_sample': surr_losses, 'total_time_sec': surr_time,
            'time_per_sample_sec': surr_time / len(test_indices),
        }
        print(f"\nDAPS-Surr: mean={surr_mean:.6f} +/- {surr_ci:.6f} "
              f"(std={surr_std:.6f})")
        print(f"  Time: {surr_time:.1f}s total, "
              f"{surr_time/len(test_indices):.2f}s/sample")

    # --- Save results ---
    results['config'] = {
        'classifier_path': str(classifier_path),
        'num_test': len(test_indices),
        'num_annealing_steps': num_annealing_steps,
        'diffusion_substeps': diffusion_substeps,
        'sigma_max': sigma_max, 'sigma_min': sigma_min,
        'num_mcmc_steps': num_mcmc_steps,
        'mcmc_lr': mcmc_lr, 'tau': tau,
        'lr_min_ratio': lr_min_ratio,
        'mc_algo': mc_algo, 'prior_solver': prior_solver,
    }

    with open(str(out_dir / "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_dir / 'eval_results.json'}")

    # --- Comparison table ---
    print(f"\n{'='*60}")
    print(f"{'Method':<15} {'Mean L2':>12} {'Std':>12} "
          f"{'95% CI':>18} {'Time/sample':>14}")
    print(f"{'-'*71}")
    if run_true:
        print(f"{'DAPS-True':<15} {true_mean:>12.6f} {true_std:>12.6f} "
              f"{'['+f'{true_mean-true_ci:.4f}, {true_mean+true_ci:.4f}'+']':>18} "
              f"{true_time/len(test_indices):>12.2f}s")
    if run_surrogate:
        print(f"{'DAPS-Surr':<15} {surr_mean:>12.6f} {surr_std:>12.6f} "
              f"{'['+f'{surr_mean-surr_ci:.4f}, {surr_mean+surr_ci:.4f}'+']':>18} "
              f"{surr_time/len(test_indices):>12.2f}s")
    if run_true and run_surrogate:
        speedup = true_time / max(surr_time, 1e-6)
        print(f"\nSurrogate is {speedup:.1f}x "
              f"{'faster' if speedup > 1 else 'slower'} "
              f"than true operator")
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
                    ax.set_title(f"{label} (L2={losses[i]:.3f})",
                                 fontsize=10)
                else:
                    ax.set_title(f"{label} #{test_indices[i]}",
                                 fontsize=10)
                ax.axis('off')

        parts = []
        if run_true:
            parts.append(f"True={true_mean:.4f}")
        if run_surrogate:
            parts.append(f"Surr={surr_mean:.4f}")
        plt.suptitle(" | ".join(parts), fontsize=13, fontweight='bold')
        plt.tight_layout()
        img_path = out_dir / "reconstructions.png"
        plt.savefig(str(img_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved reconstruction grid to {img_path}")


if __name__ == "__main__":
    main()
