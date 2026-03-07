"""Diagnostic: does CBG predict the right thing during sampling?

At each timestep during the sampling trajectory, compare:
  1. CBG prediction: classifier(x_t / scaling, sigma, obs)
  2. Ground truth: normalize(view_as_real(A(denoised) - y).flatten())
     where denoised = net(x_t / scaling, sigma)

Also checks: classifier(x_t, sigma, obs) [training-style, no /scaling]
to detect train/eval input mismatch.
"""
import torch, pickle, json, sys
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classifier import load_classifier
from utils.helper import open_url
from utils.scheduler import Scheduler


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig):
    device = torch.device("cuda")
    ev = OmegaConf.to_container(config.get("eval", {}), resolve=True)
    num_steps = ev.get("num_steps", 200)
    classifier_path = ev["classifier_path"]

    forward_op = instantiate(config.problem.model, device=device)
    test_dataset = instantiate(config.problem.data)

    ckpt_path = config.problem.prior
    try:
        with open_url(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
            net = ckpt["ema"].to(device)
    except Exception:
        net = instantiate(config.pretrain.model)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt.get("ema", ckpt.get("net", ckpt)))
        net = net.to(device)
    del ckpt
    net.eval()
    net.requires_grad_(False)

    classifier = load_classifier(classifier_path, device=device)
    classifier.eval()

    scheduler = Scheduler(num_steps=num_steps)

    # Load one test sample
    sample = test_dataset[0]
    target_img = sample["target"]
    if isinstance(target_img, np.ndarray):
        target_img = torch.from_numpy(target_img.copy())
    gt_image = target_img.float().to(device).unsqueeze(0)
    obs = forward_op({"target": gt_image})

    print("gt_image:", gt_image.shape, "obs:", obs.shape, "complex=", obs.is_complex())
    print("num_steps=", num_steps)
    print()

    # Run sampling trajectory
    torch.manual_seed(42)
    x = torch.randn(1, net.img_channels, net.img_resolution, net.img_resolution,
                     device=device) * scheduler.sigma_max

    results = []

    header = "{:>4} {:>10} {:>8} {:>8} {:>10} {:>10} {:>10} {:>8} {:>10}".format(
        "step", "sigma", "scaling", "cos_sim", "pred_norm", "gt_norm",
        "l2_err", "cos_raw", "raw_note")
    print(header)
    print("-" * 100)

    for i in range(num_steps):
        sigma = scheduler.sigma_steps[i]
        scaling = scheduler.scaling_steps[i]
        factor = scheduler.factor_steps[i]
        scaling_factor = scheduler.scaling_factor[i]
        sigma_t = torch.as_tensor(sigma).to(device)

        with torch.no_grad():
            # --- Eval-style: x / scaling (how cbg_sample calls it) ---
            x_scaled = x / scaling
            pred_eval = classifier(x_scaled, sigma_t, obs)

            # --- Ground truth target (same computation as training) ---
            denoised = net(x_scaled, sigma_t)
            y_hat = forward_op({"target": denoised})
            residual = y_hat - obs
            if residual.is_complex():
                gt_flat = torch.view_as_real(residual).flatten(1).float()
            else:
                gt_flat = residual.flatten(1).float()
            gt_norm_val = gt_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            gt_normalized = gt_flat / gt_norm_val

            # --- Training-style: x directly (how train_cbg.py calls it) ---
            pred_raw = classifier(x, sigma_t, obs)

            # --- Metrics ---
            pred_eval_flat = pred_eval.flatten(1)
            pred_raw_flat = pred_raw.flatten(1)
            gt_flat_n = gt_normalized.flatten(1)

            cos_eval = torch.nn.functional.cosine_similarity(
                pred_eval_flat, gt_flat_n, dim=-1).item()
            cos_raw = torch.nn.functional.cosine_similarity(
                pred_raw_flat, gt_flat_n, dim=-1).item()

            pred_eval_norm = pred_eval_flat.norm().item()
            gt_target_norm = gt_flat_n.norm().item()

            l2_err = (pred_eval_flat - gt_flat_n).pow(2).sum().sqrt().item()

        row = {
            "step": i, "sigma": float(sigma), "scaling": float(scaling),
            "cos_eval": cos_eval, "cos_raw": cos_raw,
            "pred_norm": pred_eval_norm, "gt_norm": gt_target_norm,
            "l2_err": l2_err,
            "gt_residual_mag": float(gt_norm_val.item()),
        }
        results.append(row)

        if i % 10 == 0 or i == num_steps - 1:
            raw_note = "BETTER" if abs(cos_raw) > abs(cos_eval) + 0.05 else ""
            line = "{:>4} {:>10.4f} {:>8.4f} {:>8.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>8.4f} {:>10}".format(
                i, sigma, scaling, cos_eval, pred_eval_norm, gt_target_norm,
                l2_err, cos_raw, raw_note)
            print(line)

        # PF-ODE Euler step (no guidance, just denoise)
        with torch.no_grad():
            score = (denoised - x_scaled) / sigma ** 2 / scaling
            x = x * scaling_factor + factor * score * 0.5

    # Summary stats
    cos_evals = [r["cos_eval"] for r in results]
    cos_raws = [r["cos_raw"] for r in results]
    print()
    print("=" * 60)
    print("SUMMARY (over {} steps):".format(num_steps))
    print("  Eval-style (x/scaling) cosine sim: mean={:.4f}, min={:.4f}, max={:.4f}".format(
        np.mean(cos_evals), np.min(cos_evals), np.max(cos_evals)))
    print("  Train-style (x raw)    cosine sim: mean={:.4f}, min={:.4f}, max={:.4f}".format(
        np.mean(cos_raws), np.min(cos_raws), np.max(cos_raws)))
    print("  Scaling mismatch: eval uses x/scaling, training uses x directly")
    if np.mean(cos_raws) > np.mean(cos_evals) + 0.05:
        print("  *** TRAIN-STYLE INPUT IS BETTER -- possible train/eval mismatch! ***")

    out_path = Path(ev.get("out_dir", "exps/eval")) / "cbg_diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to", out_path)


if __name__ == "__main__":
    main()
