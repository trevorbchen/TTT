"""Diagnostic 2: Does CBG predict correctly on TRAINING-distribution inputs?

For a test image x_0, at each sigma level:
  1. Create training-style input: x_noisy = x_0 + sigma * eps
  2. Compute ground truth: normalize(A(net(x_noisy, sigma)) - y)
  3. Compute CBG prediction: classifier(x_noisy, sigma, y)
  4. Compare cosine similarity

Also compares to the SAMPLING trajectory (from diagnostic 1).
This tells us: does the model work on training data but not sampling data?

Additionally tests with DIFFERENT x_0 images to check if model truly uses x_t.
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


def compute_gt_and_pred(classifier, net, forward_op, x_input, sigma_t, obs, device):
    """Compute ground truth target and classifier prediction."""
    with torch.no_grad():
        denoised = net(x_input, sigma_t)
        pred = classifier(x_input, sigma_t, obs, denoised=denoised)
        y_hat = forward_op({"target": denoised})
        residual = y_hat - obs
        if residual.is_complex():
            gt_flat = torch.view_as_real(residual).flatten(1).float()
        else:
            gt_flat = residual.flatten(1).float()
        gt_norm_val = gt_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        gt_normalized = gt_flat / gt_norm_val

        pred_flat = pred.flatten(1)
        cos = torch.nn.functional.cosine_similarity(
            pred_flat, gt_normalized, dim=-1).item()
        pred_norm = pred_flat.norm().item()
        gt_mag = gt_norm_val.item()

    return cos, pred_norm, gt_mag


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig):
    device = torch.device("cuda")
    ev = OmegaConf.to_container(config.get("eval", {}), resolve=True)
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

    # Load a few test images
    n_images = 5
    images = []
    observations = []
    for idx in range(n_images):
        sample = test_dataset[idx]
        t = sample["target"]
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t.copy())
        img = t.float().to(device).unsqueeze(0)
        images.append(img)
        observations.append(forward_op({"target": img}))

    # Test sigma levels (subset of training sigmas)
    sigma_values = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    print("=" * 90)
    print("PART 1: Training-distribution inputs (x_noisy = x_0 + sigma * eps)")
    print("  Does the model predict correctly when given inputs like it saw during training?")
    print("=" * 90)
    print()
    print("{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}".format(
        "sigma", "img0", "img1", "img2", "img3", "img4", "mean"))
    print("-" * 70)

    training_results = {}
    for sigma_val in sigma_values:
        sigma_t = torch.tensor(sigma_val, device=device)
        cos_vals = []
        for k in range(n_images):
            x0 = images[k]
            obs = observations[k]
            # Training-style: x_noisy = x_0 + sigma * eps
            torch.manual_seed(42 + k)
            eps = torch.randn_like(x0)
            x_noisy = x0 + sigma_val * eps
            cos, pn, gm = compute_gt_and_pred(
                classifier, net, forward_op, x_noisy, sigma_t, obs, device)
            cos_vals.append(cos)

        mean_cos = np.mean(cos_vals)
        training_results[sigma_val] = {"per_image": cos_vals, "mean": mean_cos}
        print("{:>8.2f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>10.4f}".format(
            sigma_val, *cos_vals, mean_cos))

    print()
    print("=" * 90)
    print("PART 2: Does the model actually use x_t? (same sigma, different x_t)")
    print("  Feed WRONG image's x_noisy but correct y. If model uses x_t, cos should drop.")
    print("=" * 90)
    print()

    # For image 0's observation, feed x_noisy from image 0 vs image 1,2,3,4
    obs_0 = observations[0]
    test_sigmas = [50.0, 5.0, 0.5, 0.05]
    print("{:>8} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "sigma", "correct_x0", "wrong_x1", "wrong_x2", "wrong_x3", "wrong_x4"))
    print("-" * 72)

    for sigma_val in test_sigmas:
        sigma_t = torch.tensor(sigma_val, device=device)
        cos_vals = []
        for k in range(n_images):
            x0_k = images[k]
            torch.manual_seed(42)
            eps = torch.randn_like(x0_k)
            x_noisy = x0_k + sigma_val * eps
            # Always use obs from image 0, but x_noisy from image k
            cos, _, _ = compute_gt_and_pred(
                classifier, net, forward_op, x_noisy, sigma_t, obs_0, device)
            cos_vals.append(cos)

        print("{:>8.2f} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            sigma_val, *cos_vals))

    print()
    print("  If correct_x0 >> wrong_x1..x4, model uses x_t.")
    print("  If all similar, model ignores x_t (only uses y + sigma).")

    print()
    print("=" * 90)
    print("PART 3: Random noise input (sampling trajectory start)")
    print("  Feed pure noise (like step 0 of sampling) to see if model handles it")
    print("=" * 90)
    print()
    print("{:>8} {:>12} {:>12} {:>12}".format(
        "sigma", "training_x", "random_x", "diff"))
    print("-" * 50)

    for sigma_val in sigma_values:
        sigma_t = torch.tensor(sigma_val, device=device)
        x0 = images[0]
        obs = observations[0]

        # Training-style
        torch.manual_seed(42)
        eps = torch.randn_like(x0)
        x_train = x0 + sigma_val * eps
        cos_train, _, _ = compute_gt_and_pred(
            classifier, net, forward_op, x_train, sigma_t, obs, device)

        # Pure random noise (like sampling start, scaled to similar magnitude)
        torch.manual_seed(99)
        x_random = torch.randn_like(x0) * sigma_val
        cos_random, _, _ = compute_gt_and_pred(
            classifier, net, forward_op, x_random, sigma_t, obs, device)

        diff = cos_train - cos_random
        print("{:>8.2f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            sigma_val, cos_train, cos_random, diff))

    print()
    print("  If training_x >> random_x: model learned for training dist but not sampling dist")
    print("  If both similar: model doesn't distinguish (ignores x_t content)")

    print()
    print("=" * 90)
    print("PART 4: Prediction consistency check")
    print("  Same input, same sigma — does prediction change with different y?")
    print("=" * 90)
    print()

    sigma_val = 5.0
    sigma_t = torch.tensor(sigma_val, device=device)
    x0 = images[0]
    torch.manual_seed(42)
    eps = torch.randn_like(x0)
    x_noisy = x0 + sigma_val * eps

    print("{:>8} {:>12}".format("obs_src", "cos_w_gt0"))
    print("-" * 24)
    # Ground truth for image 0
    with torch.no_grad():
        denoised = net(x_noisy, sigma_t)
        y_hat = forward_op({"target": denoised})
        residual = y_hat - observations[0]
        if residual.is_complex():
            gt0 = torch.view_as_real(residual).flatten(1).float()
        else:
            gt0 = residual.flatten(1).float()
        gt0 = gt0 / gt0.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    for k in range(n_images):
        with torch.no_grad():
            pred = classifier(x_noisy, sigma_t, observations[k], denoised=denoised)
            pred_flat = pred.flatten(1)
            cos = torch.nn.functional.cosine_similarity(pred_flat, gt0, dim=-1).item()
        label = "correct" if k == 0 else "wrong_y{}".format(k)
        print("{:>8} {:>12.4f}".format(label, cos))

    print()
    print("  If correct >> wrong: model uses y to specialize prediction (good)")
    print("  If all similar: model ignores y too (very bad)")

    # Save all results
    out_path = Path(ev.get("out_dir", "exps/eval")) / "cbg_diagnostic2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(training_results, f, indent=2, default=str)
    print()
    print("Saved to", out_path)


if __name__ == "__main__":
    main()
