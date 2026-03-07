"""Compute val_jac_cos for a checkpoint that was trained without jac_loss."""
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classifier import load_classifier
from utils.helper import open_url


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig):
    device = torch.device("cuda")
    ev = config.get("eval", {})
    classifier_path = ev["classifier_path"]
    n_eval = int(ev.get("n_eval", 800))
    batch_size = int(ev.get("batch_size", 16))
    n_seeds = int(ev.get("n_seeds", 3))

    # Load forward op
    forward_op = instantiate(config.problem.model, device=device)

    # Load dataset
    dataset = instantiate(config.problem.data)

    # Load diffusion model
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

    # Load classifier
    classifier = load_classifier(classifier_path, device=device)
    classifier.eval()
    print(f"Loaded: {classifier_path}")

    # Load eval data
    rng = np.random.RandomState(42)
    indices = rng.choice(len(dataset), size=min(n_eval, len(dataset)), replace=False)
    images, measurements = [], []
    for i in indices:
        sample = dataset[int(i)]
        t = sample["target"]
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t.copy())
        img = t.float().to(device)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        images.append(img)
        measurements.append(forward_op({"target": img.unsqueeze(0)}))
    eval_images = torch.stack(images)
    eval_measurements = torch.cat(measurements)
    print(f"Eval set: {eval_images.shape}")

    # sigma sampler matching train_cbg
    sigma_min, sigma_max = 0.002, 80.0
    def sample_sigma(B):
        log_s = torch.empty(B, device=device).uniform_(np.log(sigma_min), np.log(sigma_max))
        return log_s.exp()

    # Compute val_jac_cos over multiple seeds for stability
    jac_cos_all = []
    for seed in range(n_seeds):
        torch.manual_seed(seed * 1000)
        jac_cos_sum = 0.0
        n_batches = 0
        for v_st in range(0, len(eval_images), batch_size):
            x0v = eval_images[v_st:v_st+batch_size]
            y_bv = eval_measurements[v_st:v_st+batch_size]
            Bv = x0v.shape[0]
            sv = sample_sigma(Bv)
            sv_bc = sv.view(-1, 1, 1, 1)
            epsv = torch.randn_like(x0v)
            x_nv = x0v + sv_bc * epsv

            with torch.no_grad():
                dv = net(x_nv, sv)

            # True gradient: ∇_denoised ||A(denoised) - y||²
            dv_jac = dv.detach().requires_grad_(True)
            y_hv_jac = torch.cat([forward_op({"target": dv_jac[j:j+1]}) for j in range(Bv)])
            res_jac = y_hv_jac - y_bv
            if res_jac.is_complex():
                loss_jac = torch.view_as_real(res_jac).pow(2).flatten(1).sum(-1)
            else:
                loss_jac = res_jac.pow(2).flatten(1).sum(-1)
            grad_true = torch.autograd.grad(loss_jac.sum(), dv_jac, create_graph=False)[0].detach()

            # Surrogate gradient: ∇_denoised ||f_θ(denoised) - y||²
            pv_jac = dv.detach().requires_grad_(True)
            pred_jac = classifier(x_nv, sv, None, denoised=pv_jac)
            if pred_jac.is_complex():
                pred_flat = torch.view_as_real(pred_jac).flatten(1).float()
            else:
                pred_flat = pred_jac.flatten(1).float()
            if y_bv.is_complex():
                y_flat_v = torch.view_as_real(y_bv).flatten(1).float()
            else:
                y_flat_v = y_bv.flatten(1).float()
            surr_loss = (pred_flat - y_flat_v).pow(2).sum(-1)
            grad_surr = torch.autograd.grad(surr_loss.sum(), pv_jac, create_graph=False)[0].detach()

            with torch.no_grad():
                cos_jac = F.cosine_similarity(grad_surr.flatten(1), grad_true.flatten(1), dim=-1)
                valid = grad_true.flatten(1).norm(dim=-1) > 1e-6
                cos_jac = torch.where(valid, cos_jac, torch.zeros_like(cos_jac))
                jac_cos_sum += cos_jac.mean().item()
            n_batches += 1

        seed_jac = jac_cos_sum / max(n_batches, 1)
        jac_cos_all.append(seed_jac)
        print(f"  seed={seed}: val_jac_cos={seed_jac:.4f}")

    mean_jac = np.mean(jac_cos_all)
    print(f"\nval_jac_cos (mean over {n_seeds} seeds): {mean_jac:.4f}")
    print(f"val_jac_loss = {1 - mean_jac:.4f}")


if __name__ == "__main__":
    main()
