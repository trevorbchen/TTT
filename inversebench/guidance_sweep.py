"""Quick guidance scale sweep using eval_cbg machinery."""
import sys, json, time, torch
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from pathlib import Path
import pickle
import numpy as np

from classifier import load_classifier
from eval_cbg import cbg_sample
from utils.scheduler import Scheduler

try:
    from torch_utils.misc import open_url
except ImportError:
    open_url = open


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(config: DictConfig):
    device = torch.device("cuda")
    ev = OmegaConf.to_container(config.get("eval", {}), resolve=True)
    classifier_path = ev["classifier_path"]
    num_steps = ev.get("num_steps", 200)
    num_test = ev.get("num_test", 20)

    # Load forward op
    forward_op = instantiate(config.problem.model, device=device)

    # Load test data
    test_dataset = instantiate(config.problem.data)

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

    # Scheduler
    scheduler = Scheduler(num_steps=num_steps)

    # Load test samples
    N = min(num_test, len(test_dataset))
    indices = list(range(N))
    images, observations = [], []
    for idx in indices:
        sample = test_dataset[idx]
        img = torch.from_numpy(sample["target"]).unsqueeze(0).float().to(device)
        obs = forward_op({"target": img})
        images.append(img)
        observations.append(obs)
    print(f"Loaded {N} test samples")

    scales = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    results = {}

    for scale in scales:
        l2s = []
        t0 = time.time()
        for i in range(N):
            x_recon = cbg_sample(net, classifier, forward_op, observations[i],
                                 scheduler, guidance_scale=scale, device=device)
            l2 = (x_recon - images[i]).pow(2).sum().sqrt().item()
            l2s.append(l2)
        elapsed = time.time() - t0
        mean_l2 = sum(l2s) / len(l2s)
        results[str(scale)] = {"mean_l2": mean_l2, "all_l2": l2s}
        print(f"scale={scale:<8}  mean_L2={mean_l2:.4f}  ({elapsed:.0f}s)")

    out_path = Path(ev.get("out_dir", "exps/eval")) / "guidance_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
