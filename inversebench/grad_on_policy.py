"""On-policy gradient diagnostic: follow DPS trajectory, compare surrogate grads.

At each diffusion step along the DPS (true A) trajectory, computes:
  - DPS gradient (true forward operator)
  - CBG gradient (surrogate classifier)
  - Cosine similarity, magnitude ratio, per-pixel error

This isolates surrogate accuracy from trajectory divergence — we're always
evaluating the surrogate at the exact same x_t that DPS would see.

Usage:
    python grad_on_policy.py problem=inv-scatter pretrain=inv-scatter \
        +eval.classifier_path=path/to/classifier_best.pt \
        +eval.num_test=5 +eval.guidance_scale=280.0
"""

import json
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
    num_test = ev.get("num_test", 5)
    guidance_scale = ev.get("guidance_scale", 280.0)

    # Load components
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
        net.load_state_dict(ckpt.get('ema', ckpt.get('net', ckpt)))
        net = net.to(device)
    del ckpt
    net.eval()
    net.requires_grad_(False)

    print("Loading classifier...")
    classifier = load_classifier(classifier_path, device=device)
    classifier.eval()

    scheduler = Scheduler(num_steps=num_steps, schedule="vp",
                          timestep="vp", scaling="vp")

    F_cos = torch.nn.functional.cosine_similarity

    # Collect per-step stats across samples for averaging
    all_stats = []  # list of dicts per (sample, step)

    for sample_idx in range(num_test):
        sample = test_dataset[sample_idx]
        target = sample['target']
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target.copy())
        target = target.float().to(device)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        obs = forward_op({'target': target.unsqueeze(0)})

        # Precompute y_flat for surrogate
        if obs.is_complex():
            y_flat = torch.view_as_real(obs).flatten(1).float()
        else:
            y_flat = obs.flatten(1).float()

        # Initial noise (same seed as eval_cbg.py)
        torch.manual_seed(42 + sample_idx)
        x = torch.randn(1, net.img_channels, net.img_resolution,
                         net.img_resolution, device=device) * scheduler.sigma_max

        print(f"\n{'='*120}")
        print(f"Sample {sample_idx}")
        print(f"{'='*120}")
        print(f"{'step':>5} {'sigma':>8} | {'cos_sim':>8} {'mag_ratio':>10} | "
              f"{'dps_gnorm':>10} {'cbg_gnorm':>10} | "
              f"{'dps_loss':>10} {'cbg_loss':>10} | "
              f"{'norm_dps':>10} {'norm_cbg':>10} | {'grad_mse':>10}")
        print("-" * 130)

        for i in range(num_steps):
            sigma = scheduler.sigma_steps[i]
            scaling = scheduler.scaling_steps[i]
            factor = scheduler.factor_steps[i]
            scaling_factor = scheduler.scaling_factor[i]
            sigma_t = torch.as_tensor(sigma).to(device)

            # --- DPS gradient (true A) ---
            x_in = x.detach().requires_grad_(True)
            net.requires_grad_(True)
            denoised = net(x_in / scaling, sigma_t)

            y_hat = forward_op({'target': denoised})
            residual = y_hat - obs
            if residual.is_complex():
                loss_dps = torch.view_as_real(residual).pow(2).flatten(1).sum(-1)
            else:
                loss_dps = residual.pow(2).flatten(1).sum(-1)
            grad_dps = torch.autograd.grad(loss_dps.sum(), x_in, retain_graph=True)[0]

            # --- CBG gradient (surrogate) --- same x_in, same denoised
            pred_cbg = classifier(x_in / scaling, sigma_t, None, denoised=denoised)
            loss_cbg = (pred_cbg.flatten(1) - y_flat).pow(2).sum(-1)
            grad_cbg = torch.autograd.grad(loss_cbg.sum(), x_in)[0]

            net.requires_grad_(False)

            # --- Metrics ---
            with torch.no_grad():
                gd = grad_dps.flatten(1)
                gc = grad_cbg.flatten(1)

                cos = F_cos(gd, gc, dim=-1).item()
                dps_norm = gd.norm(dim=-1).item()
                cbg_norm = gc.norm(dim=-1).item()
                ratio = cbg_norm / max(dps_norm, 1e-12)

                loss_dps_val = loss_dps.item()
                loss_cbg_val = loss_cbg.item()

                # Normalized gradient norms (same normalization used in sampling)
                dps_nf = max(loss_dps_val ** 0.5, 1e-8)
                cbg_nf = max(loss_cbg_val ** 0.5, 1e-8)
                norm_dps = dps_norm / dps_nf
                norm_cbg = cbg_norm / cbg_nf

                # MSE between normalized gradients (what actually gets applied)
                norm_grad_dps = gd / max(dps_nf, 1e-8)
                norm_grad_cbg = gc / max(cbg_nf, 1e-8)
                grad_mse = (norm_grad_dps - norm_grad_cbg).pow(2).mean().item()

            step_data = {
                'sample': sample_idx, 'step': i, 'sigma': float(sigma),
                'cos_sim': cos, 'mag_ratio': ratio,
                'dps_gnorm': dps_norm, 'cbg_gnorm': cbg_norm,
                'dps_loss': loss_dps_val, 'cbg_loss': loss_cbg_val,
                'norm_dps': norm_dps, 'norm_cbg': norm_cbg,
                'grad_mse': grad_mse,
            }
            all_stats.append(step_data)

            # Print every step for first 20, then every 10
            if i < 20 or i % 10 == 0 or i == num_steps - 1:
                print(f"{i:>5} {sigma:>8.3f} | {cos:>8.4f} {ratio:>10.4f} | "
                      f"{dps_norm:>10.4f} {cbg_norm:>10.4f} | "
                      f"{loss_dps_val:>10.2f} {loss_cbg_val:>10.2f} | "
                      f"{norm_dps:>10.4f} {norm_cbg:>10.4f} | {grad_mse:>10.6f}")

            # --- Update x using DPS trajectory only ---
            with torch.no_grad():
                denoised_clean = net(x / scaling, sigma_t)
                score = (denoised_clean - x / scaling) / sigma ** 2 / scaling
                x = x * scaling_factor + factor * score * 0.5

                # Apply DPS guidance
                normalized_dps = grad_dps / max(dps_nf, 1e-8)
                x = x - guidance_scale * normalized_dps

                if torch.isnan(x).any():
                    print(f"  NaN at step {i}, stopping")
                    break

    # --- Aggregate stats by step (average across samples) ---
    print(f"\n\n{'='*120}")
    print(f"AVERAGED ACROSS {num_test} SAMPLES")
    print(f"{'='*120}")

    # Group by step
    steps_data = {}
    for s in all_stats:
        step = s['step']
        if step not in steps_data:
            steps_data[step] = []
        steps_data[step].append(s)

    print(f"{'step':>5} {'sigma':>8} | {'cos_sim':>8} {'cos_std':>8} | "
          f"{'mag_ratio':>10} | {'grad_mse':>10} | {'dps_loss':>10} {'cbg_loss':>10}")
    print("-" * 100)

    summary_rows = []
    for step in sorted(steps_data.keys()):
        entries = steps_data[step]
        sigma = entries[0]['sigma']

        cos_vals = [e['cos_sim'] for e in entries]
        ratio_vals = [e['mag_ratio'] for e in entries]
        mse_vals = [e['grad_mse'] for e in entries]
        dps_loss_vals = [e['dps_loss'] for e in entries]
        cbg_loss_vals = [e['cbg_loss'] for e in entries]

        row = {
            'step': step, 'sigma': sigma,
            'cos_mean': np.mean(cos_vals), 'cos_std': np.std(cos_vals),
            'ratio_mean': np.mean(ratio_vals),
            'grad_mse_mean': np.mean(mse_vals),
            'dps_loss_mean': np.mean(dps_loss_vals),
            'cbg_loss_mean': np.mean(cbg_loss_vals),
        }
        summary_rows.append(row)

        if step < 20 or step % 10 == 0 or step == num_steps - 1:
            print(f"{step:>5} {sigma:>8.3f} | {row['cos_mean']:>8.4f} {row['cos_std']:>8.4f} | "
                  f"{row['ratio_mean']:>10.4f} | {row['grad_mse_mean']:>10.6f} | "
                  f"{row['dps_loss_mean']:>10.2f} {row['cbg_loss_mean']:>10.2f}")

    # --- Sigma band summary ---
    print(f"\n{'='*80}")
    print("SIGMA BAND SUMMARY")
    print(f"{'='*80}")

    bands = [
        ("High  (σ>60)", lambda s: s > 60),
        ("Mid   (20<σ≤60)", lambda s: 20 < s <= 60),
        ("Low   (5<σ≤20)", lambda s: 5 < s <= 20),
        ("VLow  (σ≤5)", lambda s: s <= 5),
    ]
    print(f"{'Band':<20} {'cos_sim':>8} {'mag_ratio':>10} {'grad_mse':>10} {'n_steps':>8}")
    print("-" * 60)
    for band_name, band_fn in bands:
        band_rows = [r for r in summary_rows if band_fn(r['sigma'])]
        if not band_rows:
            continue
        cos_m = np.mean([r['cos_mean'] for r in band_rows])
        rat_m = np.mean([r['ratio_mean'] for r in band_rows])
        mse_m = np.mean([r['grad_mse_mean'] for r in band_rows])
        print(f"{band_name:<20} {cos_m:>8.4f} {rat_m:>10.4f} {mse_m:>10.6f} {len(band_rows):>8}")

    # Save raw data
    out_path = Path("exps/grad_on_policy_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w") as f:
        json.dump({
            'per_step': all_stats,
            'summary': summary_rows,
            'config': {
                'num_steps': num_steps,
                'num_test': num_test,
                'guidance_scale': guidance_scale,
                'classifier_path': str(classifier_path),
            }
        }, f, indent=2)
    print(f"\nSaved raw data to {out_path}")


if __name__ == "__main__":
    main()
