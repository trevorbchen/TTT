"""Sensitivity diagnostic for forward-mode CBG classifier (no y input).

Tests whether the model uses each of its inputs:
  1. Swap denoised (Tweedie x_0 estimate) — does prediction change?
  2. Swap sigma — does prediction change?
  3. Accuracy: cos_sim(pred, ground_truth) at each sigma
     Ground truth = A(denoised), not residual
  4. Ablation: zero out each input

Also supports legacy residual mode (auto-detected from checkpoint).
"""
import torch, pickle, sys, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classifier import load_classifier
from utils.helper import open_url

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig):
    device = torch.device("cuda")
    ev = config.get("eval", {})
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

    # Detect target mode from checkpoint
    ckpt_meta = torch.load(classifier_path, map_location="cpu", weights_only=False)
    target_mode = ckpt_meta.get("target_mode", "tweedie")
    del ckpt_meta
    is_forward = target_mode == "forward"
    print(f"Target mode: {target_mode} ({'no y input' if is_forward else 'y input'})")

    # Load test samples
    n_img = 5
    images, obs_list = [], []
    for idx in range(n_img):
        sample = test_dataset[idx]
        t = sample["target"]
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t.copy())
        img = t.float().to(device).unsqueeze(0)
        images.append(img)
        obs_list.append(forward_op({"target": img}))

    F_cos = torch.nn.functional.cosine_similarity
    sigmas = [50.0, 20.0, 5.0, 1.0, 0.5, 0.1]

    def get_denoised(x0, sigma_val, seed=42):
        sigma_t = torch.tensor(sigma_val, device=device)
        torch.manual_seed(seed)
        eps = torch.randn_like(x0)
        x_noisy = x0 + sigma_val * eps
        with torch.no_grad():
            denoised = net(x_noisy, sigma_t)
        return x_noisy, denoised

    def call_classifier(x_t, sigma_t, y, denoised):
        """Call classifier with appropriate y handling for target mode."""
        if is_forward:
            return classifier(x_t, sigma_t, None, denoised=denoised)
        else:
            return classifier(x_t, sigma_t, y, denoised=denoised)

    # ===== TEST 1: Swap denoised =====
    print("=" * 80)
    print("TEST 1: Swap DENOISED — same (x_t, sigma), different Tweedie x_0")
    print("  pred(denoised_from_img_k) vs pred(denoised_from_img_0)")
    print("  If ~1.0: model ignores denoised input")
    print("=" * 80)
    header = "{:>8}".format("sigma")
    for k in range(n_img):
        header += " {:>10}".format("den_img{}".format(k))
    print(header)
    print("-" * (8 + 11 * n_img))

    x0_ref = images[0]
    obs_ref = obs_list[0]
    for sigma_val in sigmas:
        sigma_t = torch.tensor(sigma_val, device=device)
        x_noisy_ref, den_ref = get_denoised(x0_ref, sigma_val)

        with torch.no_grad():
            pred_ref = call_classifier(x_noisy_ref, sigma_t, obs_ref, den_ref).flatten(1)

        row = "{:>8.1f}".format(sigma_val)
        for k in range(n_img):
            _, den_k = get_denoised(images[k], sigma_val)
            with torch.no_grad():
                pred_k = call_classifier(x_noisy_ref, sigma_t, obs_ref, den_k).flatten(1)
            cos = F_cos(pred_ref, pred_k, dim=-1).item()
            row += " {:>10.4f}".format(cos)
        print(row)

    # ===== TEST 2: Swap sigma =====
    print()
    print("=" * 80)
    print("TEST 2: Swap SIGMA — same (x_t, denoised), different sigma value")
    print("  pred(sigma_k) vs pred(sigma_ref=5.0)")
    print("  If ~1.0: model ignores sigma")
    print("=" * 80)
    sigma_ref_val = 5.0
    sigma_t_ref = torch.tensor(sigma_ref_val, device=device)
    x_noisy_ref, den_ref = get_denoised(x0_ref, sigma_ref_val)

    with torch.no_grad():
        pred_ref = call_classifier(x_noisy_ref, sigma_t_ref, obs_ref, den_ref).flatten(1)

    print("{:>8} {:>10} {:>12} {:>12}".format("sigma", "cos_w_ref", "pred_norm", "ref_norm"))
    print("-" * 45)
    for sigma_val in [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]:
        sigma_t = torch.tensor(sigma_val, device=device)
        with torch.no_grad():
            pred_k = call_classifier(x_noisy_ref, sigma_t, obs_ref, den_ref).flatten(1)
        cos = F_cos(pred_ref, pred_k, dim=-1).item()
        print("{:>8.2f} {:>10.4f} {:>12.4f} {:>12.4f}".format(
            sigma_val, cos, pred_k.norm().item(), pred_ref.norm().item()))

    # ===== TEST 3: Accuracy — cos_sim with ground truth =====
    print()
    print("=" * 80)
    if is_forward:
        print("TEST 3: Accuracy — cos_sim(pred, A(denoised))")
        print("  Ground truth = A(denoised) (forward mode)")
    else:
        print("TEST 3: Accuracy — cos_sim(pred, A(denoised) - y)")
        print("  Ground truth = A(denoised) - y (residual mode)")
    print("=" * 80)
    header = "{:>8} {:>10} {:>10}".format("sigma", "pred_norm", "cos_w_gt")
    for k in range(1, n_img):
        header += " {:>10}".format("img{}".format(k))
    print(header)
    print("-" * (30 + 11 * (n_img - 1)))

    for sigma_val in sigmas:
        sigma_t = torch.tensor(sigma_val, device=device)
        row = "{:>8.1f}".format(sigma_val)

        for k in range(n_img):
            x_noisy_k, den_k = get_denoised(images[k], sigma_val)
            with torch.no_grad():
                pred_k = call_classifier(x_noisy_k, sigma_t, obs_list[k], den_k).flatten(1)
                y_hat = forward_op({"target": den_k})
                if is_forward:
                    # Ground truth = A(denoised)
                    if y_hat.is_complex():
                        gt = torch.view_as_real(y_hat).flatten(1).float()
                    else:
                        gt = y_hat.flatten(1).float()
                else:
                    # Ground truth = A(denoised) - y
                    residual = y_hat - obs_list[k]
                    if residual.is_complex():
                        gt = torch.view_as_real(residual).flatten(1).float()
                    else:
                        gt = residual.flatten(1).float()
                cos_gt = F_cos(pred_k, gt, dim=-1).item()
                # Also compute MSE
                mse = (pred_k - gt).pow(2).sum(-1).item()

            if k == 0:
                row += " {:>10.4f} {:>10.4f}".format(pred_k.norm().item(), cos_gt)
            else:
                row += " {:>10.4f}".format(cos_gt)
        print(row)

    # ===== TEST 4: Ablation =====
    print()
    print("=" * 80)
    print("TEST 4: Ablation — zero out each input, measure prediction change")
    print("  cos(pred_normal, pred_ablated)")
    print("=" * 80)
    if is_forward:
        print("{:>8} {:>14} {:>14}".format("sigma", "zero_denoised", "zero_x_t"))
        print("-" * 40)
    else:
        print("{:>8} {:>14} {:>14} {:>14}".format(
            "sigma", "zero_denoised", "zero_y", "zero_x_t"))
        print("-" * 55)

    for sigma_val in sigmas:
        sigma_t = torch.tensor(sigma_val, device=device)
        x_noisy_ref, den_ref = get_denoised(x0_ref, sigma_val)

        with torch.no_grad():
            pred_normal = call_classifier(x_noisy_ref, sigma_t, obs_ref, den_ref).flatten(1)

            # Zero denoised
            den_zero = torch.zeros_like(den_ref)
            pred_no_den = call_classifier(x_noisy_ref, sigma_t, obs_ref, den_zero).flatten(1)

            # Zero x_t
            x_zero = torch.zeros_like(x_noisy_ref)
            pred_no_xt = call_classifier(x_zero, sigma_t, obs_ref, den_ref).flatten(1)

        cos_den = F_cos(pred_normal, pred_no_den, dim=-1).item()
        cos_xt = F_cos(pred_normal, pred_no_xt, dim=-1).item()

        if is_forward:
            print("{:>8.1f} {:>14.4f} {:>14.4f}".format(sigma_val, cos_den, cos_xt))
        else:
            # Zero y
            with torch.no_grad():
                y_zero = torch.zeros_like(obs_ref)
                pred_no_y = classifier(x_noisy_ref, sigma_t, y_zero, denoised=den_ref).flatten(1)
            cos_y = F_cos(pred_normal, pred_no_y, dim=-1).item()
            print("{:>8.1f} {:>14.4f} {:>14.4f} {:>14.4f}".format(
                sigma_val, cos_den, cos_y, cos_xt))

    print()
    print("  cos ~ 1.0: removing that input barely changes output (not used)")
    print("  cos ~ 0.0 or negative: removing it changes output a lot (used)")


if __name__ == "__main__":
    main()
