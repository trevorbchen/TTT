"""Save a few LMDB images and generated images side by side for comparison."""
import sys, os, torch
sys.path.insert(0, os.getcwd())

import hydra
import numpy as np
import torchvision.utils as vutils
from utils.scheduler import Scheduler
from hydra.utils import instantiate

with hydra.initialize(config_path="configs", version_base="1.3"):
    cfg = hydra.compose(config_name="config", overrides=["problem=inv-scatter", "pretrain=inv-scatter"])

device = torch.device("cuda")

# 1) Load LMDB images
dataset = instantiate(cfg.pretrain.data)
lmdb_imgs = []
for i in range(8):
    img = dataset[i]['target']
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    lmdb_imgs.append(img.unsqueeze(0))
lmdb_imgs = torch.cat(lmdb_imgs).float().to(device)
print(f"LMDB stats: min={lmdb_imgs.min():.3f}, max={lmdb_imgs.max():.3f}, mean={lmdb_imgs.mean():.3f}, std={lmdb_imgs.std():.3f}, shape={lmdb_imgs.shape}")

# 2) Generate images from diffusion prior
net = instantiate(cfg.pretrain.model).to(device)
ckpt_path = cfg.problem.prior
ckpt = torch.load(ckpt_path, map_location=device)
if 'ema' in ckpt:
    net.load_state_dict(ckpt['ema'])
elif 'net' in ckpt:
    net.load_state_dict(ckpt['net'])
else:
    net.load_state_dict(ckpt)
net.eval()
print(f"Model: {net.__class__.__name__}, res={net.img_resolution}, ch={net.img_channels}")

scheduler = Scheduler(num_steps=200, schedule="vp", timestep="vp", scaling="vp")

with torch.no_grad():
    x = torch.randn(8, net.img_channels, net.img_resolution, net.img_resolution, device=device) * scheduler.sigma_max
    for i in range(200):
        sigma = scheduler.sigma_steps[i]
        scaling = scheduler.scaling_steps[i]
        factor = scheduler.factor_steps[i]
        scaling_factor = scheduler.scaling_factor[i]
        denoised = net(x / scaling, torch.as_tensor(sigma).to(device))
        score = (denoised - x / scaling) / sigma ** 2 / scaling
        x = x * scaling_factor + factor * score * 0.5
    gen_imgs = x

print(f"Gen  stats: min={gen_imgs.min():.3f}, max={gen_imgs.max():.3f}, mean={gen_imgs.mean():.3f}, std={gen_imgs.std():.3f}, shape={gen_imgs.shape}")

# 3) Save
os.makedirs("comparison_imgs", exist_ok=True)

def normalize(t):
    t = t.cpu().float()
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    return t

vutils.save_image(normalize(lmdb_imgs), "comparison_imgs/lmdb_grid.png", nrow=4, padding=2)
vutils.save_image(normalize(gen_imgs), "comparison_imgs/gen200_grid.png", nrow=4, padding=2)

for i in range(8):
    vutils.save_image(normalize(lmdb_imgs[i:i+1]), f"comparison_imgs/lmdb_{i}.png")
    vutils.save_image(normalize(gen_imgs[i:i+1]), f"comparison_imgs/gen200_{i}.png")

print("Saved to comparison_imgs/")
