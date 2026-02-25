"""Debug: check classifier prediction and gradient magnitudes during CBG inference."""
import torch
import numpy as np
import sys
sys.path.insert(0, ".")
from classifier import load_classifier
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from utils.scheduler import Scheduler

initialize_config_dir(config_dir="configs", version_base="1.3")
config = compose(config_name="config", overrides=["problem=inv-scatter", "pretrain=inv-scatter"])

device = torch.device("cuda")
forward_op = instantiate(config.problem.model, device=device)
dataset = instantiate(config.problem.data)
net = instantiate(config.pretrain.model)
ckpt = torch.load(config.problem.prior, map_location=device, weights_only=False)
net.load_state_dict(ckpt.get("ema", ckpt.get("net")))
net = net.to(device).eval()
net.requires_grad_(False)

clf_path = "exps/cbg_tweedie_full/inverse-scatter-linear_cbg_tweedie_80pct_lr0.003_ch64/classifier_best.pt"
clf = load_classifier(clf_path, device=device)
clf.eval()

scheduler = Scheduler(num_steps=200, schedule="vp", timestep="vp", scaling="vp")

# One test sample
s = dataset[0]
t = torch.from_numpy(s["target"].copy()).float().to(device).unsqueeze(0).unsqueeze(0)
y = forward_op({"target": t})

torch.manual_seed(42)
x = torch.randn(1, net.img_channels, net.img_resolution, net.img_resolution, device=device) * scheduler.sigma_max

header = f"{'Step':>5} {'sigma':>10} {'scaling':>10} {'|pred|':>12} {'loss':>12} {'|grad_x|':>12} {'|norm_grad|':>12} {'|ode_step|':>12} {'guid/ode':>10}"
print(header)
print("-" * len(header))

for i in range(scheduler.num_steps):
    sigma = scheduler.sigma_steps[i]
    scaling = scheduler.scaling_steps[i]
    factor = scheduler.factor_steps[i]
    scaling_factor = scheduler.scaling_factor[i]

    with torch.no_grad():
        denoised = net(x / scaling, torch.as_tensor(sigma).to(device))

    x_in = x.detach().requires_grad_(True)
    pred = clf(x_in / scaling, torch.as_tensor(sigma).to(device), y)
    loss_val = pred.pow(2).flatten(1).sum(-1)
    grad_x = torch.autograd.grad(loss_val.sum(), x_in)[0]

    with torch.no_grad():
        norm_factor = loss_val.sqrt().clamp(min=1e-8)
        normalized_grad = grad_x / norm_factor.view(-1, 1, 1, 1)

        score = (denoised - x / scaling) / sigma ** 2 / scaling
        ode_step = factor * score * 0.5

        guid_norm = normalized_grad.norm().item()
        ode_norm = ode_step.norm().item()
        ratio = guid_norm / max(ode_norm, 1e-10)

        if i in [0, 1, 5, 10, 25, 50, 100, 150, 175, 190, 195, 199] or i % 50 == 0:
            print(f"{i:5d} {float(sigma):10.4f} {float(scaling):10.6f} "
                  f"{pred.abs().mean().item():12.6f} {loss_val.item():12.4f} "
                  f"{grad_x.norm().item():12.4f} {guid_norm:12.4f} "
                  f"{ode_norm:12.4f} {ratio:10.4f}")

        x = x * scaling_factor + ode_step
        x = x - 1.0 * normalized_grad

# Weight magnitudes
print("\n--- Key weight magnitudes ---")
for name, p in clf.named_parameters():
    if "out_conv" in name or "measurement_decoder" in name or "bot_" in name:
        print(f"  {name:50s} |w|_mean={p.abs().mean().item():.8f}  |w|_max={p.abs().max().item():.6f}")

# Final reconstruction quality
loss_final = forward_op.loss(x, y).item()
print(f"\nFinal reconstruction L2 = {loss_final:.4f}")
print(f"(Plain diffusion baseline ~ 0.62, DPS ~ 0.09)")
