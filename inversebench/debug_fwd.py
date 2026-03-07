"""Debug: compare per-sample vs batched forward_op on denoised images."""
import sys, os, torch, math
import numpy as np
sys.path.insert(0, os.getcwd())

import hydra
from hydra.utils import instantiate

with hydra.initialize(config_path="configs", version_base="1.3"):
    cfg = hydra.compose(config_name="config", overrides=["problem=inv-scatter", "pretrain=inv-scatter"])

device = torch.device("cuda")
forward_op = instantiate(cfg.problem.model, device=device)

# Load a few val images
dataset = instantiate(cfg.pretrain.data)
imgs = []
for i in range(4):
    img = dataset[8000+i]['target']
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    imgs.append(img.unsqueeze(0))
imgs = torch.cat(imgs).float().to(device)

# Per-sample forward_op
per_sample = []
for i in range(4):
    m = forward_op({'target': imgs[i:i+1]})
    per_sample.append(m)
    print(f"  per-sample[{i}] shape={m.shape}")
per_sample = torch.cat(per_sample)

# Batched forward_op
batched = forward_op({'target': imgs})
print(f"batched shape={batched.shape}")
print(f"per_sample shape={per_sample.shape}")

# Compare
for i in range(min(4, batched.shape[0])):
    diff = (batched[i] - per_sample[i]).abs().max().item()
    print(f"  max diff[{i}] = {diff:.6f}")

if batched.shape[0] != per_sample.shape[0]:
    print(f"SHAPE MISMATCH: batched={batched.shape[0]} vs per_sample={per_sample.shape[0]}")
