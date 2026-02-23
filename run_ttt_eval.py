"""
Evaluation script for LoRA-finetuned diffusion models.

Compares DPS baseline (guided, no LoRA) against a LoRA-adapted model
running plain diffusion (no guidance) on a test dataset.

The LoRA model has learned to internalize measurement guidance for a
specific forward operator A(x), so plain unguided sampling produces
reconstructions comparable to (or better than) DPS.

Usage:
  # 1. Train LoRA with one of the finetuning scripts:
  python dpo_finetune.py data=... model=... sampler=edm_dps ...

  # 2. Evaluate:
  python run_ttt_eval.py \
      data=test-imagenet model=imagenet256ddpm \
      sampler=edm_dps task=gaussian_blur \
      +eval.lora_path=results/dpo_blur/lora_final.pt \
      name=eval_dpo_blur
"""

import json
import yaml
import torch
import numpy as np
import tqdm
import hydra
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torchvision.utils import save_image

from forward_operator import get_operator
from data import get_dataset
from model import get_model
from eval import get_eval_fn, Evaluator
from sampler import get_sampler
from lora import load_lora, load_conditioned_lora, remove_lora


# ---------------------------------------------------------------------------
# Plain diffusion sampling (no guidance)
# ---------------------------------------------------------------------------

@torch.no_grad()
def plain_diffusion_sample(model, scheduler, x_start, verbose=False,
                           store=None, y=None):
    """Plain PF-ODE Euler sampling without measurement guidance.

    After LoRA finetuning, the model's score already accounts for
    the measurement operator, so no guidance is needed.

    For conditioned LoRA, set store/y so the LoRA modules see the
    measurement at every denoising step.

    Uses the same scheduler/sigma schedule as DPS for a fair comparison.
    """
    if store is not None and y is not None:
        store.set(y)

    sigma_steps = scheduler.sigma_steps
    num_steps = len(sigma_steps) - 1
    pbar = tqdm.trange(num_steps, desc="Plain diffusion") if verbose else range(num_steps)
    xt = x_start

    for step in pbar:
        sigma = sigma_steps[step]
        sigma_next = sigma_steps[step + 1]
        t = scheduler.get_sigma_inv(sigma)
        t_next = scheduler.get_sigma_inv(sigma_next)
        dt = t_next - t
        st = scheduler.get_scaling(t)
        dst = scheduler.get_scaling_derivative(t)
        dsigma = scheduler.get_sigma_derivative(t)

        x0hat = model.tweedie(xt / st, sigma)
        score = (x0hat - xt / st) / sigma ** 2
        deriv = dst / st * xt - st * dsigma * sigma * score
        xt = xt + dt * deriv

    if store is not None:
        store.clear()

    return xt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm(x):
    """[-1,1] -> [0,1]"""
    return (x * 0.5 + 0.5).clamp(0, 1)


def resize_y(y, target_shape):
    """Resize measurement to match image shape for visualization."""
    if y.shape != target_shape:
        return torch.nn.functional.interpolate(
            y, size=target_shape[-2:], mode='bilinear', align_corners=False,
        )
    return y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_path="configs", config_name="default.yaml")
def main(args: DictConfig):
    # --- reproducibility ---
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(f"cuda:{args.gpu}")

    print(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))

    # --- eval config ---
    eval_cfg = OmegaConf.to_container(args.get("eval", {}), resolve=True)
    lora_path = eval_cfg.get("lora_path", None)
    if lora_path is None:
        raise ValueError("Must provide +eval.lora_path=<path_to_lora.pt>")
    skip_baseline = eval_cfg.get("skip_baseline", False)

    # --- data ---
    dataset = get_dataset(**args.data)
    total_number = len(dataset)
    images = dataset.get_data(total_number, 0)

    # --- operator & measurement ---
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    y = operator.measure(images)

    # --- sampler (DPS for baseline) ---
    sampler = get_sampler(**args.sampler,
                          mcmc_sampler_config=task_group.get("mcmc_sampler_config", None))

    # --- model ---
    model = get_model(**args.model)

    # --- evaluator ---
    eval_fn_list = [get_eval_fn(name) for name in args.eval_fn_list]
    evaluator = Evaluator(eval_fn_list)

    # --- output dirs ---
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    root = save_dir / args.name
    root.mkdir(exist_ok=True)
    (root / "samples").mkdir(exist_ok=True)
    (root / "comparisons").mkdir(exist_ok=True)

    with open(str(root / "config.yaml"), "w") as f:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), f)

    # ===================================================================
    # Phase 1: DPS baseline (guided, no LoRA)
    # ===================================================================
    if skip_baseline:
        print(f"\n{'='*60}")
        print(f"Phase 1: SKIPPED (skip_baseline=True)")
        print(f"{'='*60}")
        all_baseline = None
    else:
        print(f"\n{'='*60}")
        print(f"Phase 1: DPS baseline ({total_number} images)")
        print(f"{'='*60}")

        all_baseline = []
        for img_idx in tqdm.trange(total_number, desc="DPS baseline"):
            gt_i = images[img_idx: img_idx + 1]
            y_i = y[img_idx: img_idx + 1]

            x_start = sampler.get_start(1, model)
            x_hat = sampler.sample(model, x_start, operator, y_i, verbose=False)
            all_baseline.append(x_hat)

        all_baseline = torch.cat(all_baseline, dim=0)

    # ===================================================================
    # Phase 2: LoRA model plain diffusion (unguided)
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"Phase 2: LoRA plain diffusion ({total_number} images)")
    print(f"Loading LoRA from: {lora_path}")
    print(f"{'='*60}")

    # Detect conditioned vs unconditioned LoRA from checkpoint
    checkpoint_meta = torch.load(lora_path, map_location="cpu")
    is_conditioned = checkpoint_meta.get("conditioned", False)

    if is_conditioned:
        lora_modules, store = load_conditioned_lora(model, lora_path)
    else:
        lora_modules = load_lora(model, lora_path)
        store = None

    scheduler = sampler.scheduler

    # Reset RNG for fair comparison (same starting noise)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    all_lora = []
    for img_idx in tqdm.trange(total_number, desc="LoRA plain diffusion"):
        y_i = y[img_idx: img_idx + 1]
        x_start = sampler.get_start(1, model)
        x_hat = plain_diffusion_sample(model, scheduler, x_start, verbose=False,
                                       store=store, y=y_i)
        all_lora.append(x_hat)

    all_lora = torch.cat(all_lora, dim=0)
    remove_lora(model)

    # ===================================================================
    # Phase 3: Compute metrics and save results
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"Phase 3: Computing metrics")
    print(f"{'='*60}")

    all_metrics = []
    metric_names = None

    for img_idx in range(total_number):
        gt_i = images[img_idx: img_idx + 1]
        y_i = y[img_idx: img_idx + 1]
        lora_i = all_lora[img_idx: img_idx + 1]

        with torch.no_grad():
            metrics_lora = evaluator(gt_i, y_i, lora_i)
            mc_lora = operator.loss(lora_i, y_i)
            metrics_lora["meas_l2"] = mc_lora.mean()

            if all_baseline is not None:
                baseline_i = all_baseline[img_idx: img_idx + 1]
                metrics_baseline = evaluator(gt_i, y_i, baseline_i)
                mc_baseline = operator.loss(baseline_i, y_i)
                metrics_baseline["meas_l2"] = mc_baseline.mean()

        img_metrics = {"image_idx": img_idx}
        if metric_names is None:
            metric_names = list(metrics_lora.keys())

        for name in metric_names:
            a = metrics_lora[name].item()
            img_metrics[f"{name}_lora"] = a
            if all_baseline is not None:
                b = metrics_baseline[name].item()
                img_metrics[f"{name}_baseline"] = b
                img_metrics[f"{name}_delta"] = a - b
        all_metrics.append(img_metrics)

        # save individual samples
        save_image(norm(lora_i),
                   str(root / "samples" / f"{img_idx:05d}_lora.png"))
        if all_baseline is not None:
            save_image(norm(baseline_i),
                       str(root / "samples" / f"{img_idx:05d}_baseline.png"))

        # comparison grid
        y_resized = resize_y(y_i, gt_i.shape)
        if all_baseline is not None:
            grid = torch.cat([
                norm(gt_i), norm(y_resized), norm(baseline_i), norm(lora_i),
            ], dim=0)
            save_image(grid, str(root / "comparisons" / f"{img_idx:05d}.png"),
                       nrow=4, padding=2)
        else:
            grid = torch.cat([
                norm(gt_i), norm(y_resized), norm(lora_i),
            ], dim=0)
            save_image(grid, str(root / "comparisons" / f"{img_idx:05d}.png"),
                       nrow=3, padding=2)

    # --- aggregate statistics ---
    lora_vals = {n: np.array([m[f"{n}_lora"] for m in all_metrics]) for n in metric_names}

    if all_baseline is not None:
        baseline_vals = {n: np.array([m[f"{n}_baseline"] for m in all_metrics]) for n in metric_names}
        delta_vals = {n: lora_vals[n] - baseline_vals[n] for n in metric_names}

        print(f"\n{'='*60}")
        print(f"Results: LoRA vs DPS baseline | {total_number} images")
        print(f"{'='*60}")
        print(f"{'metric':<8} {'DPS baseline':>18} {'LoRA (no guide)':>18} {'delta':>18}")
        print(f"{'':<8} {'mean +/- std':>18} {'mean +/- std':>18} {'mean +/- std':>18}")
        print(f"{'-'*62}")
        for n in metric_names:
            bm, bs = baseline_vals[n].mean(), baseline_vals[n].std()
            am, astd = lora_vals[n].mean(), lora_vals[n].std()
            dm, ds = delta_vals[n].mean(), delta_vals[n].std()
            print(f"{n:<8} {bm:>7.4f} +/- {bs:<6.4f} {am:>7.4f} +/- {astd:<6.4f} {dm:>+7.4f} +/- {ds:<6.4f}")
        print(f"{'-'*62}")

        for n in metric_names:
            improved = (delta_vals[n] > 0).sum() if n in ("psnr", "ssim") else (delta_vals[n] < 0).sum()
            print(f"  {n}: {improved}/{total_number} images improved")
    else:
        print(f"\n{'='*60}")
        print(f"Results: LoRA only (baseline skipped) | {total_number} images")
        print(f"{'='*60}")
        print(f"{'metric':<8} {'LoRA (no guide)':>18}")
        print(f"{'':<8} {'mean +/- std':>18}")
        print(f"{'-'*30}")
        for n in metric_names:
            am, astd = lora_vals[n].mean(), lora_vals[n].std()
            print(f"{n:<8} {am:>7.4f} +/- {astd:<6.4f}")
        print(f"{'-'*30}")

    # --- save full comparison grid ---
    y_resized = resize_y(y, images.shape)
    if all_baseline is not None:
        full_grid = torch.cat([
            norm(images), norm(y_resized), norm(all_baseline), norm(all_lora),
        ], dim=0)
    else:
        full_grid = torch.cat([
            norm(images), norm(y_resized), norm(all_lora),
        ], dim=0)
    save_image(full_grid,
               str(root / "full_comparison.png"),
               nrow=total_number, padding=2)

    # --- save metrics to JSON ---
    aggregate = {}
    for n in metric_names:
        agg = {
            "lora_mean": float(lora_vals[n].mean()),
            "lora_std": float(lora_vals[n].std()),
        }
        if all_baseline is not None:
            agg.update({
                "baseline_mean": float(baseline_vals[n].mean()),
                "baseline_std": float(baseline_vals[n].std()),
                "delta_mean": float(delta_vals[n].mean()),
                "delta_std": float(delta_vals[n].std()),
                "delta_min": float(delta_vals[n].min()),
                "delta_max": float(delta_vals[n].max()),
            })
        aggregate[n] = agg

    output = {
        "lora_path": str(lora_path),
        "num_images": total_number,
        "skip_baseline": skip_baseline,
        "aggregate": aggregate,
        "per_image": all_metrics,
    }
    with open(str(root / "metrics.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {root}")
    print(f"  - Per-image comparisons: {root / 'comparisons'}")
    print(f"  - Individual samples:    {root / 'samples'}")
    print(f"  - Full grid:             {root / 'full_comparison.png'}")
    print(f"  - Metrics:               {root / 'metrics.json'}")


if __name__ == "__main__":
    main()
