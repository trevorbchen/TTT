"""
Sweep CBG→DPS hybrid switch point.

Runs the hybrid sampler at different switch percentages:
  0% CBG = pure DPS (switch immediately)
  50% CBG = surrogate for first half, DPS for second half
  100% CBG = pure CBG (never switch)

Reports RME at each switch point to find the optimal tradeoff.

Usage:
    python sweep_hybrid.py problem=inv-scatter pretrain=inv-scatter \
        +eval.classifier_path=exps/.../classifier_best.pt \
        +eval.num_test=50 +eval.out_dir=exps/sweep_hybrid
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

from classifier import load_classifier, UNetSurrogate, FNOSurrogate, ForwardSurrogate
from utils.helper import open_url
from utils.scheduler import Scheduler


def confidence_interval_95(values):
    arr = np.array(values)
    n = len(arr)
    mean = arr.mean()
    std = arr.std(ddof=1)
    if n < 2:
        return mean, 0.0, std
    t_val = stats.t.ppf(0.975, n - 1)
    ci = t_val * std / np.sqrt(n)
    return mean, ci, std


def hybrid_sample(net, classifier, forward_op, observation, scheduler,
                  guidance_scale=1.0, device='cuda', switch_sigma=0.0):
    """Hybrid: surrogate for sigma >= switch_sigma, DPS for sigma < switch_sigma."""
    obs = observation
    if obs.is_complex():
        y_flat = torch.view_as_real(obs).flatten(1).float()
    else:
        y_flat = obs.flatten(1).float()

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

        use_dps = (float(sigma) < switch_sigma)

        x_in = x.detach().requires_grad_(True)
        net.requires_grad_(True)
        denoised = net(x_in / scaling, sigma_t)

        if use_dps:
            y_hat = forward_op({'target': denoised})
            residual = y_hat - obs
            if residual.is_complex():
                loss_val = torch.view_as_real(residual).pow(2).flatten(1).sum(-1)
            else:
                loss_val = residual.pow(2).flatten(1).sum(-1)
        else:
            pred = classifier(denoised)
            loss_val = (pred.flatten(1) - y_flat).pow(2).sum(-1)

        grad_x = torch.autograd.grad(loss_val.sum(), x_in)[0]
        net.requires_grad_(False)

        with torch.no_grad():
            denoised_clean = net(x / scaling, sigma_t)
            norm_factor = loss_val.sqrt().view(-1, *([1] * (grad_x.ndim - 1)))
            norm_factor = norm_factor.clamp(min=1e-8)
            normalized_grad = grad_x / norm_factor

            score = (denoised_clean - x / scaling) / sigma ** 2 / scaling
            x = x * scaling_factor + factor * score * 0.5
            x = x - guidance_scale * normalized_grad

            if torch.isnan(x).any():
                break

    return x


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ev = OmegaConf.to_container(config.get("eval", {}), resolve=True) or {}

    classifier_path = ev.get("classifier_path", "")
    guidance_scale = ev.get("guidance_scale", 1.0)
    num_steps = ev.get("num_steps", 200)
    num_test = ev.get("num_test", 50)
    out_dir = ev.get("out_dir", "exps/sweep_hybrid")

    # Switch percentages to sweep
    cbg_pcts = ev.get("cbg_pcts", [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    assert classifier_path, "Must provide +eval.classifier_path=..."

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    print("=== Hybrid CBG→DPS Switch Point Sweep ===")
    print(f"  classifier: {classifier_path}")
    print(f"  guidance_scale: {guidance_scale}")
    print(f"  num_steps: {num_steps}")
    print(f"  num_test: {num_test}")
    print(f"  cbg_pcts: {cbg_pcts}")
    print()

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
    num_params = sum(p.numel() for p in classifier.parameters())
    print(f"  Classifier: {num_params/1e6:.2f}M params")

    # --- Build scheduler ---
    scheduler = Scheduler(num_steps=num_steps, schedule="vp",
                          timestep="vp", scaling="vp")

    # --- Compute switch_sigma for each percentage ---
    sigmas = [float(scheduler.sigma_steps[i]) for i in range(num_steps)]
    switch_configs = []
    for pct in cbg_pcts:
        if pct == 0:
            # 0% CBG = pure DPS: switch_sigma = very high (always DPS)
            sw = float('inf')
        elif pct >= 100:
            # 100% CBG = never switch: switch_sigma = 0
            sw = 0.0
        else:
            # Switch after pct% of steps
            step_idx = int(pct / 100.0 * (num_steps - 1))
            sw = sigmas[step_idx]
        switch_configs.append((pct, sw))
        print(f"  {pct:3d}% CBG → switch_sigma = {sw:.4f}")

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

    # --- Sweep ---
    all_results = {}
    for pct, sw in switch_configs:
        label = f"{pct}% CBG"
        print(f"\n{'='*60}")
        print(f"Running: {label} (switch_sigma={sw:.4f})")
        print(f"{'='*60}")

        errs = []
        t0 = time.time()
        for j in tqdm.trange(len(test_indices), desc=label):
            y_j = test_measurements[j:j+1]
            gt_j = test_images[j:j+1]

            with torch.no_grad():
                recon = hybrid_sample(
                    net, classifier, forward_op, y_j, scheduler,
                    guidance_scale=guidance_scale, device=device,
                    switch_sigma=sw)

            # RME
            y_hat = forward_op({'target': recon})
            residual = y_hat - y_j
            if residual.is_complex():
                err_num = torch.view_as_real(residual).pow(2).flatten(1).sum(-1).sqrt()
                err_den = torch.view_as_real(y_j).pow(2).flatten(1).sum(-1).sqrt()
            else:
                err_num = residual.pow(2).flatten(1).sum(-1).sqrt()
                err_den = y_j.pow(2).flatten(1).sum(-1).sqrt()
            rme = (err_num / err_den.clamp(min=1e-8) * 100).item()
            errs.append(rme)

        elapsed = time.time() - t0
        mean, ci, std = confidence_interval_95(errs)
        all_results[pct] = {
            'mean': mean, 'std': std, 'ci95': ci,
            'time_per_sample': elapsed / len(test_indices),
            'switch_sigma': sw,
            'per_sample': errs,
        }
        print(f"  {label}: RME={mean:.2f}% +/- {ci:.2f} "
              f"(std={std:.2f}), {elapsed/len(test_indices):.1f}s/sample")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"{'CBG %':>8} {'switch_σ':>10} {'RME (%)':>10} {'Std':>8} {'95% CI':>18} {'Time/s':>10}")
    print(f"{'-'*60}")
    for pct, sw in switch_configs:
        r = all_results[pct]
        ci_str = f"[{r['mean']-r['ci95']:.2f}, {r['mean']+r['ci95']:.2f}]"
        print(f"{pct:>7}% {sw:>10.4f} {r['mean']:>10.2f} {r['std']:>8.2f} {ci_str:>18} {r['time_per_sample']:>9.1f}s")

    # --- Save ---
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'per_sample'}
                    for k, v in all_results.items()}
    with open(root / "sweep_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    # --- Plot ---
    pcts = sorted(all_results.keys())
    means = [all_results[p]['mean'] for p in pcts]
    stds = [all_results[p]['std'] for p in pcts]
    times = [all_results[p]['time_per_sample'] for p in pcts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.errorbar(pcts, means, yerr=stds, marker='o', capsize=4)
    ax1.set_xlabel('% of steps using CBG (surrogate)')
    ax1.set_ylabel('RME (%)')
    ax1.set_title('Hybrid CBG→DPS: RME vs Switch Point')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(pcts)

    ax2.plot(pcts, times, marker='s', color='tab:orange')
    ax2.set_xlabel('% of steps using CBG (surrogate)')
    ax2.set_ylabel('Time per sample (s)')
    ax2.set_title('Hybrid CBG→DPS: Speed vs Switch Point')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(pcts)

    plt.tight_layout()
    plt.savefig(str(root / "sweep_plot.png"), dpi=150)
    plt.close()
    print(f"\nSaved plot to {root / 'sweep_plot.png'}")
    print(f"Saved results to {root / 'sweep_results.json'}")


if __name__ == "__main__":
    main()
