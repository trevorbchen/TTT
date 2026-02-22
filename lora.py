"""
LoRA (Low-Rank Adaptation) for diffusion model UNet layers.

Provides utilities to inject, remove, save, load, and manage LoRA adapters
on the UNet.  Supports both attention-only and ResBlock injection, with
optional measurement-conditioning (y-conditioned LoRA).

Target layers controlled by ``target_modules`` parameter:
  - ``"attention"`` — qkv + proj_out in AttentionBlocks only (8 modules)
  - ``"resblock"``  — in_conv + out_conv in ResBlocks only (60 modules)
  - ``"all"``       — both attention and ResBlock layers (68 modules)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from model.ddpm.unet import AttentionBlock, ResBlock


class LoRAConv1d(nn.Module):
    """Wraps a frozen Conv1d(in, out, 1) with a low-rank adapter."""

    def __init__(self, original_conv, rank=4, alpha=1.0):
        super().__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels

        self.lora_down = nn.Conv1d(in_channels, rank, 1, bias=False)
        self.lora_up = nn.Conv1d(rank, out_channels, 1, bias=False)

        # kaiming init for down, zero init for up (LoRA starts at identity)
        nn.init.kaiming_uniform_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

        # freeze the original conv
        for p in self.original_conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original_conv(x) + self.scaling * self.lora_up(self.lora_down(x))


class MeasurementStore:
    """Shared container for the current measurement y.

    Set once before each model forward; all conditioned LoRA modules read
    from the same store instance.
    """

    def __init__(self):
        self.measurement = None  # [B, C_y, H, W]

    def set(self, y):
        self.measurement = y

    def clear(self):
        self.measurement = None


class LoRAConv1dConditioned(nn.Module):
    """Wraps a frozen Conv1d(in, out, 1) with a measurement-conditioned low-rank adapter.

    Unlike LoRAConv1d, lora_down takes concat(x, y_resized) so the adapter
    can learn the y-specific correction directly.  Cannot be merged with
    base weights (input dim mismatch).
    """

    def __init__(self, original_conv, rank=4, alpha=1.0, y_channels=3, store=None):
        super().__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.y_channels = y_channels
        self.store = store  # shared MeasurementStore reference

        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels

        self.lora_down = nn.Conv1d(in_channels + y_channels, rank, 1, bias=False)
        self.lora_up = nn.Conv1d(rank, out_channels, 1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

        for p in self.original_conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.original_conv(x)
        y_2d = self.store.measurement if self.store is not None else None
        if y_2d is not None:
            B, C, L = x.shape
            H = W = int(L ** 0.5)
            y_resized = F.interpolate(y_2d, (H, W), mode='bilinear',
                                      align_corners=False)
            y_flat = y_resized.flatten(2)        # [B, y_channels, L]
            xy = torch.cat([x, y_flat], dim=1)   # [B, C + y_channels, L]
            out = out + self.scaling * self.lora_up(self.lora_down(xy))
        return out


class LoRAConv2dConditioned(nn.Module):
    """Wraps a frozen Conv2d with a measurement-conditioned low-rank adapter.

    Uses 1x1 convolutions for the low-rank branch (channel mixing only).
    The original conv handles spatial mixing; LoRA corrects channel-wise
    based on both activations and the measurement y.
    """

    def __init__(self, original_conv, rank=4, alpha=1.0, y_channels=3, store=None):
        super().__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.y_channels = y_channels
        self.store = store

        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels

        # 1x1 conv for low-rank bottleneck (channel mixing, no spatial)
        self.lora_down = nn.Conv2d(in_channels + y_channels, rank, 1, bias=False)
        self.lora_up = nn.Conv2d(rank, out_channels, 1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

        for p in self.original_conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.original_conv(x)
        y_2d = self.store.measurement if self.store is not None else None
        if y_2d is not None:
            B, C, H, W = x.shape
            y_resized = F.interpolate(y_2d, (H, W), mode='bilinear',
                                      align_corners=False)
            xy = torch.cat([x, y_resized], dim=1)  # [B, C + y_channels, H, W]
            out = out + self.scaling * self.lora_up(self.lora_down(xy))
        return out


def apply_lora(model, rank=4, alpha=1.0):
    """Inject LoRA adapters into all AttentionBlock qkv and proj_out layers.

    Walks ``model.model.model`` (the UNet inside VPPrecond) and replaces
    Conv1d projections with LoRAConv1d wrappers.

    Returns:
        list[LoRAConv1d]: All newly created LoRA modules.
    """
    unet = model.model.model  # DDPM -> VPPrecond -> UNet
    lora_modules = []

    for module in unet.modules():
        if isinstance(module, AttentionBlock):
            for attr_name in ("qkv", "proj_out"):
                orig_conv = getattr(module, attr_name)
                lora_conv = LoRAConv1d(orig_conv, rank=rank, alpha=alpha)
                lora_conv = lora_conv.to(next(orig_conv.parameters()).device)
                setattr(module, attr_name, lora_conv)
                lora_modules.append(lora_conv)

    return lora_modules


def apply_conditioned_lora(model, rank=4, alpha=1.0, y_channels=3,
                           target_modules="all"):
    """Inject measurement-conditioned LoRA adapters into UNet layers.

    Args:
        model: DDPM diffusion model.
        rank: LoRA rank (bottleneck dimension).
        alpha: LoRA scaling numerator (scaling = alpha / rank).
        y_channels: Number of measurement channels (e.g. 3 for RGB).
        target_modules: Which layers to wrap:
            ``"attention"`` — AttentionBlock qkv + proj_out (Conv1d)
            ``"resblock"``  — ResBlock in_conv + out_conv (Conv2d)
            ``"all"``       — both attention and ResBlock layers

    Returns:
        (list[LoRA modules], MeasurementStore)
    """
    unet = model.model.model
    store = MeasurementStore()
    lora_modules = []
    do_attn = target_modules in ("attention", "all")
    do_res = target_modules in ("resblock", "all")

    for module in unet.modules():
        if do_attn and isinstance(module, AttentionBlock):
            for attr_name in ("qkv", "proj_out"):
                orig_conv = getattr(module, attr_name)
                lora_conv = LoRAConv1dConditioned(
                    orig_conv, rank=rank, alpha=alpha,
                    y_channels=y_channels, store=store)
                lora_conv = lora_conv.to(next(orig_conv.parameters()).device)
                setattr(module, attr_name, lora_conv)
                lora_modules.append(lora_conv)

        if do_res and isinstance(module, ResBlock):
            # in_layers[-1] is the main Conv2d
            orig_in = module.in_layers[-1]
            if isinstance(orig_in, nn.Conv2d):
                lora_in = LoRAConv2dConditioned(
                    orig_in, rank=rank, alpha=alpha,
                    y_channels=y_channels, store=store)
                lora_in = lora_in.to(next(orig_in.parameters()).device)
                module.in_layers[-1] = lora_in
                lora_modules.append(lora_in)

            # out_layers[-1] is the zero-init Conv2d
            orig_out = module.out_layers[-1]
            if isinstance(orig_out, nn.Conv2d):
                lora_out = LoRAConv2dConditioned(
                    orig_out, rank=rank, alpha=alpha,
                    y_channels=y_channels, store=store)
                lora_out = lora_out.to(next(orig_out.parameters()).device)
                module.out_layers[-1] = lora_out
                lora_modules.append(lora_out)

    return lora_modules, store


def remove_lora(model):
    """Restore original conv layers, undoing apply_lora or apply_conditioned_lora."""
    unet = model.model.model
    for module in unet.modules():
        if isinstance(module, AttentionBlock):
            for attr_name in ("qkv", "proj_out"):
                wrapper = getattr(module, attr_name)
                if isinstance(wrapper, (LoRAConv1d, LoRAConv1dConditioned)):
                    setattr(module, attr_name, wrapper.original_conv)

        if isinstance(module, ResBlock):
            wrapper_in = module.in_layers[-1]
            if isinstance(wrapper_in, LoRAConv2dConditioned):
                module.in_layers[-1] = wrapper_in.original_conv
            wrapper_out = module.out_layers[-1]
            if isinstance(wrapper_out, LoRAConv2dConditioned):
                module.out_layers[-1] = wrapper_out.original_conv


def get_lora_params(lora_modules):
    """Return a flat list of all trainable LoRA parameters."""
    params = []
    for m in lora_modules:
        params.extend(m.lora_down.parameters())
        params.extend(m.lora_up.parameters())
    return params


def frozen_tweedie(model, lora_modules, x, sigma):
    """Evaluate Tweedie with LoRA scaling temporarily zeroed.

    This gives the frozen (pre-LoRA) model prediction without needing
    a separate model copy.
    """
    saved = []
    for m in lora_modules:
        saved.append(m.scaling)
        m.scaling = 0.0

    with torch.no_grad():
        out = model.tweedie(x, sigma)

    for m, s in zip(lora_modules, saved):
        m.scaling = s

    return out


def save_lora(lora_modules, path, metadata=None):
    """Save trained LoRA weights to disk.

    Works for all LoRA module types (Conv1d, Conv2d, conditioned or not).

    Args:
        lora_modules: list of LoRA modules from apply_lora or apply_conditioned_lora.
        path: file path to save to (.pt).
        metadata: optional dict of training info (operator, rank, etc.)
    """
    state = OrderedDict()
    module_types = []
    for i, m in enumerate(lora_modules):
        state[f"{i}.lora_down.weight"] = m.lora_down.weight.data.cpu()
        state[f"{i}.lora_up.weight"] = m.lora_up.weight.data.cpu()
        module_types.append(type(m).__name__)

    m0 = lora_modules[0]
    conditioned = isinstance(m0, (LoRAConv1dConditioned, LoRAConv2dConditioned))

    # infer target_modules from the module types present
    has_conv1d = any(isinstance(m, (LoRAConv1d, LoRAConv1dConditioned))
                     for m in lora_modules)
    has_conv2d = any(isinstance(m, LoRAConv2dConditioned) for m in lora_modules)
    if has_conv1d and has_conv2d:
        target_modules = "all"
    elif has_conv2d:
        target_modules = "resblock"
    else:
        target_modules = "attention"

    checkpoint = {
        "lora_state": state,
        "rank": m0.rank,
        "alpha": m0.alpha,
        "num_modules": len(lora_modules),
        "conditioned": conditioned,
        "target_modules": target_modules,
        "module_types": module_types,
    }
    if conditioned:
        checkpoint["y_channels"] = m0.y_channels
    if metadata is not None:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, path)


def load_lora(model, path):
    """Load saved LoRA weights and inject them into the model.

    Calls apply_lora with the saved rank/alpha, then loads the trained
    weights into the newly created LoRA modules.

    Args:
        model: DDPM diffusion model (same architecture used during training).
        path: path to saved .pt file from save_lora.

    Returns:
        list[LoRAConv1d]: LoRA modules with loaded weights.
    """
    checkpoint = torch.load(path, map_location="cpu")
    rank = checkpoint["rank"]
    alpha = checkpoint["alpha"]
    state = checkpoint["lora_state"]

    lora_modules = apply_lora(model, rank=rank, alpha=alpha)

    for i, m in enumerate(lora_modules):
        m.lora_down.weight.data.copy_(state[f"{i}.lora_down.weight"])
        m.lora_up.weight.data.copy_(state[f"{i}.lora_up.weight"])

    return lora_modules


def load_conditioned_lora(model, path):
    """Load saved conditioned LoRA weights and inject them into the model.

    Calls apply_conditioned_lora with the saved rank/alpha/y_channels/target_modules,
    then loads the trained weights.

    Args:
        model: DDPM diffusion model (same architecture used during training).
        path: path to saved .pt file from save_lora.

    Returns:
        (list[LoRA modules], MeasurementStore)
    """
    checkpoint = torch.load(path, map_location="cpu")
    rank = checkpoint["rank"]
    alpha = checkpoint["alpha"]
    y_channels = checkpoint.get("y_channels", 3)
    target_modules = checkpoint.get("target_modules", "attention")
    state = checkpoint["lora_state"]

    lora_modules, store = apply_conditioned_lora(
        model, rank=rank, alpha=alpha, y_channels=y_channels,
        target_modules=target_modules)

    for i, m in enumerate(lora_modules):
        m.lora_down.weight.data.copy_(state[f"{i}.lora_down.weight"])
        m.lora_up.weight.data.copy_(state[f"{i}.lora_up.weight"])

    return lora_modules, store


def merge_lora(model):
    """Fold LoRA weights into the original Conv1d and remove wrappers.

    After merging, the model runs at full speed with no LoRA overhead.
    This is irreversible -- the original weights are modified in place.
    """
    unet = model.model.model
    for module in unet.modules():
        if isinstance(module, AttentionBlock):
            for attr_name in ("qkv", "proj_out"):
                wrapper = getattr(module, attr_name)
                if isinstance(wrapper, LoRAConv1d):
                    # W_merged = W_original + scaling * up @ down
                    with torch.no_grad():
                        merged = (wrapper.scaling
                                  * wrapper.lora_up.weight.data
                                  @ wrapper.lora_down.weight.data)
                        wrapper.original_conv.weight.data.add_(merged)
                    setattr(module, attr_name, wrapper.original_conv)
