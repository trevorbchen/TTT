"""
LoRA (Low-Rank Adaptation) for InverseBench diffusion model UNet layers.

Supports both:
  - ADM-style UNets (UNetModel from models.ddpm) via Conv1d wrappers
  - DhariwalUNet / SongUNet (from models.unets) via custom Conv2d wrappers

Target layers controlled by ``target_modules`` parameter:
  - ``"attention"`` — qkv + proj in attention-bearing UNetBlocks
  - ``"resblock"``  — conv0 + conv1 in all UNetBlocks
  - ``"all"``       — both attention and resblock layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from models.modules import UNetBlock, Conv2d as CustomConv2d


# ---------------------------------------------------------------------------
# Measurement store
# ---------------------------------------------------------------------------

class MeasurementEncoder(nn.Module):
    """Projects arbitrary-shape measurements to spatial feature maps.

    Maps measurements of any shape (e.g. [B, 20, 360] complex scattering data)
    to [B, y_channels, latent_res, latent_res] spatial features that can be
    concatenated with UNet activations in conditioned LoRA layers.

    Outputs at a small latent resolution (default 8x8) to keep parameter count
    manageable. Each LoRA layer upscales via F.interpolate to its own resolution.

    For complex inputs, real and imaginary parts are stacked as channels.
    """

    def __init__(self, obs_shape, y_channels, img_resolution, latent_res=8):
        """
        Args:
            obs_shape: Shape of a single observation *without* batch dim,
                       e.g. (20, 360) or (1, 128, 128).
            y_channels: Number of output spatial channels.
            img_resolution: Spatial resolution of UNet (e.g. 128). Stored for
                            reference but encoder outputs at latent_res.
            latent_res: Spatial resolution of encoder output (default 8).
                        Kept small to avoid huge parameter counts.
        """
        super().__init__()
        self.obs_shape = obs_shape
        self.y_channels = y_channels
        self.img_resolution = img_resolution
        self.latent_res = latent_res

        # Flatten obs to a vector; double channels for complex inputs
        obs_numel = 1
        for s in obs_shape:
            obs_numel *= s
        self.obs_numel = obs_numel
        self.is_complex = False  # set at runtime on first forward

        # MLP: flatten → hidden → y_channels * latent_res^2
        spatial_out = y_channels * latent_res * latent_res
        # Input dim determined on first forward (depends on complex or not)
        self._built = False
        self._y_channels = y_channels
        self._spatial_out = spatial_out
        self._latent_res = latent_res

    def _build(self, in_dim, device):
        hidden = min(in_dim, 512)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self._spatial_out),
        ).to(device)
        self._built = True

    def forward(self, y):
        """
        Args:
            y: [B, ...] measurement tensor (real or complex).
        Returns:
            [B, y_channels, latent_res, latent_res] spatial features.
        """
        B = y.shape[0]
        if y.is_complex():
            y = torch.view_as_real(y).flatten(1).float()  # [B, 2*numel]
        else:
            y = y.flatten(1).float()

        if not self._built:
            self._build(y.shape[1], y.device)

        out = self.proj(y)  # [B, y_channels * latent_res^2]
        return out.view(B, self._y_channels, self._latent_res, self._latent_res)


class MeasurementStore:
    """Shared container for the current measurement y.

    Set once before each model forward; all conditioned LoRA modules read
    from the same store instance. If an encoder is attached, raw measurements
    are automatically projected to spatial features.
    """

    def __init__(self, encoder=None):
        self.measurement = None  # [B, C_y, H, W] after encoding
        self.encoder = encoder

    def set(self, y):
        if self.encoder is not None:
            self.measurement = self.encoder(y)
        else:
            self.measurement = y

    def clear(self):
        self.measurement = None


# ---------------------------------------------------------------------------
# LoRA wrappers for ADM UNet (Conv1d layers in AttentionBlock)
# ---------------------------------------------------------------------------

class LoRAConv1dConditioned(nn.Module):
    """Wraps a frozen Conv1d(in, out, 1) with a measurement-conditioned low-rank adapter."""

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

        self.lora_down = nn.Conv1d(in_channels + y_channels, rank, 1, bias=False)
        self.lora_up = nn.Conv1d(rank, out_channels, 1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

        for p in self.original_conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.original_conv(x)
        if self.y_channels > 0:
            B, C, L = x.shape
            H = W = int(L ** 0.5)
            y_2d = self.store.measurement if self.store is not None and self.store.measurement is not None else None
            if y_2d is not None:
                y_resized = F.interpolate(y_2d, (H, W), mode='bilinear',
                                          align_corners=False)
            else:
                y_resized = x.new_zeros(B, self.y_channels, H, W)
            y_flat = y_resized.flatten(2)        # [B, y_channels, L]
            xy = torch.cat([x, y_flat], dim=1)   # [B, C + y_channels, L]
            out = out + self.scaling * self.lora_up(self.lora_down(xy))
        else:
            out = out + self.scaling * self.lora_up(self.lora_down(x))
        return out


class LoRAConv2dConditioned(nn.Module):
    """Wraps a frozen nn.Conv2d with a measurement-conditioned low-rank adapter.

    Uses 1x1 convolutions for the low-rank branch.
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

        self.lora_down = nn.Conv2d(in_channels + y_channels, rank, 1, bias=False)
        self.lora_up = nn.Conv2d(rank, out_channels, 1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

        for p in self.original_conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.original_conv(x)
        if self.y_channels > 0:
            B, C, H, W = x.shape
            y_2d = self.store.measurement if self.store is not None and self.store.measurement is not None else None
            if y_2d is not None:
                y_resized = F.interpolate(y_2d, (H, W), mode='bilinear',
                                          align_corners=False)
            else:
                y_resized = x.new_zeros(B, self.y_channels, H, W)
            xy = torch.cat([x, y_resized], dim=1)
            out = out + self.scaling * self.lora_up(self.lora_down(xy))
        else:
            out = out + self.scaling * self.lora_up(self.lora_down(x))
        return out


# ---------------------------------------------------------------------------
# LoRA wrapper for DhariwalUNet / SongUNet (custom Conv2d from modules.py)
# ---------------------------------------------------------------------------

class LoRACustomConv2dConditioned(nn.Module):
    """Wraps a frozen models.modules.Conv2d with a measurement-conditioned low-rank adapter.

    The original Conv2d stores weight as nn.Parameter and calls F.conv2d directly
    (with optional up/down resampling). We call original.forward(x) to preserve
    all resampling logic, then add a low-rank correction via standard nn.Conv2d(1x1).
    """

    def __init__(self, original_conv, rank=4, alpha=1.0, y_channels=0, store=None):
        super().__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.y_channels = y_channels
        self.store = store

        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels

        # 1x1 low-rank branch (standard nn.Conv2d, no resampling)
        lora_in = in_channels + y_channels if y_channels > 0 else in_channels
        self.lora_down = nn.Conv2d(lora_in, rank, 1, bias=False)
        self.lora_up = nn.Conv2d(rank, out_channels, 1, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

        for p in self.original_conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.original_conv(x)  # preserves up/down resampling
        if self.y_channels > 0:
            B, C, H, W = x.shape
            y_2d = self.store.measurement if self.store is not None and self.store.measurement is not None else None
            if y_2d is not None:
                y_resized = F.interpolate(y_2d, (H, W),
                                          mode='bilinear', align_corners=False)
            else:
                y_resized = x.new_zeros(B, self.y_channels, H, W)
            xy = torch.cat([x, y_resized], dim=1)
            correction = self.scaling * self.lora_up(self.lora_down(xy))
        else:
            correction = self.scaling * self.lora_up(self.lora_down(x))
        # Handle spatial mismatch when original conv resamples (up/down)
        if correction.shape[-2:] != out.shape[-2:]:
            correction = F.interpolate(correction, out.shape[-2:],
                                       mode='bilinear', align_corners=False)
        out = out + correction
        return out


# ---------------------------------------------------------------------------
# Apply / remove / query LoRA
# ---------------------------------------------------------------------------

def _get_unet(net):
    """Extract the raw UNet from the preconditioner wrapper.

    InverseBench: net is a precond (VPPrecond/EDMPrecond/...) with net.model = UNet.
    Our TTT:      model.model.model = UNet (DDPM -> VPPrecond -> UNet).

    Returns (unet, unet_type) where unet_type is 'ADM', 'DhariwalUNet', or 'SongUNet'.
    """
    # InverseBench style: net.model is the UNet directly
    unet = net.model
    unet_type = type(unet).__name__
    if unet_type in ('DhariwalUNet', 'SongUNet'):
        return unet, unet_type
    # ADM (from models.ddpm)
    if unet_type == 'UNetModel':
        return unet, 'ADM'
    # TTT-style: model.model is VPPrecond, model.model.model is UNet
    if hasattr(unet, 'model'):
        inner = unet.model
        inner_type = type(inner).__name__
        if inner_type == 'UNetModel':
            return inner, 'ADM'
    raise ValueError(f"Unknown UNet type: {unet_type}")


def apply_conditioned_lora(net, rank=4, alpha=1.0, y_channels=0,
                           target_modules="all", obs_shape=None,
                           img_resolution=None, latent_res=8):
    """Inject measurement-conditioned LoRA adapters into UNet layers.

    Auto-detects UNet type (ADM vs DhariwalUNet/SongUNet).

    Args:
        net: Preconditioned diffusion model (e.g. VPPrecond, EDMPrecond).
        rank: LoRA rank.
        alpha: LoRA scaling numerator.
        y_channels: Number of measurement channels (0 = unconditioned).
        target_modules: "attention", "resblock", or "all".
        obs_shape: Shape of a single observation (no batch dim).
                   Required when y_channels > 0 to build the encoder.
        img_resolution: Spatial resolution of the UNet (e.g. 128).
                        Required when y_channels > 0.
        latent_res: Spatial resolution of encoder output (default 8).
                    Kept small to avoid huge parameter counts.

    Returns:
        (list[LoRA modules], MeasurementStore or None)
    """
    unet, unet_type = _get_unet(net)
    if y_channels > 0:
        assert obs_shape is not None, "obs_shape required when y_channels > 0"
        if img_resolution is None:
            img_resolution = net.img_resolution
        encoder = MeasurementEncoder(obs_shape, y_channels, img_resolution,
                                     latent_res=latent_res)
        store = MeasurementStore(encoder=encoder)
    else:
        store = None
    lora_modules = []
    do_attn = target_modules in ("attention", "all")
    do_res = target_modules in ("resblock", "all")

    if unet_type in ('DhariwalUNet', 'SongUNet'):
        # Walk all UNetBlocks
        for module in unet.modules():
            if not isinstance(module, UNetBlock):
                continue

            # Resblock convs: conv0 (kernel=3) and conv1 (kernel=3)
            if do_res:
                for attr_name in ('conv0', 'conv1'):
                    orig = getattr(module, attr_name)
                    if isinstance(orig, CustomConv2d):
                        wrapper = LoRACustomConv2dConditioned(
                            orig, rank=rank, alpha=alpha,
                            y_channels=y_channels, store=store)
                        wrapper = wrapper.to(next(orig.parameters()).device)
                        setattr(module, attr_name, wrapper)
                        lora_modules.append(wrapper)

            # Attention convs: qkv (kernel=1) and proj (kernel=1)
            if do_attn and module.num_heads > 0:
                for attr_name in ('qkv', 'proj'):
                    orig = getattr(module, attr_name)
                    if isinstance(orig, CustomConv2d):
                        wrapper = LoRACustomConv2dConditioned(
                            orig, rank=rank, alpha=alpha,
                            y_channels=y_channels, store=store)
                        wrapper = wrapper.to(next(orig.parameters()).device)
                        setattr(module, attr_name, wrapper)
                        lora_modules.append(wrapper)

    elif unet_type == 'ADM':
        # Import ADM-specific blocks
        from models.ddpm.unet import AttentionBlock, ResBlock
        for module in unet.modules():
            if do_attn and isinstance(module, AttentionBlock):
                for attr_name in ("qkv", "proj_out"):
                    orig = getattr(module, attr_name)
                    wrapper = LoRAConv1dConditioned(
                        orig, rank=rank, alpha=alpha,
                        y_channels=y_channels, store=store)
                    wrapper = wrapper.to(next(orig.parameters()).device)
                    setattr(module, attr_name, wrapper)
                    lora_modules.append(wrapper)

            if do_res and isinstance(module, ResBlock):
                orig_in = module.in_layers[-1]
                if isinstance(orig_in, nn.Conv2d):
                    wrapper = LoRAConv2dConditioned(
                        orig_in, rank=rank, alpha=alpha,
                        y_channels=y_channels, store=store)
                    wrapper = wrapper.to(next(orig_in.parameters()).device)
                    module.in_layers[-1] = wrapper
                    lora_modules.append(wrapper)

                orig_out = module.out_layers[-1]
                if isinstance(orig_out, nn.Conv2d):
                    wrapper = LoRAConv2dConditioned(
                        orig_out, rank=rank, alpha=alpha,
                        y_channels=y_channels, store=store)
                    wrapper = wrapper.to(next(orig_out.parameters()).device)
                    module.out_layers[-1] = wrapper
                    lora_modules.append(wrapper)
    else:
        raise ValueError(f"Unsupported UNet type: {unet_type}")

    return lora_modules, store


def remove_lora(net):
    """Restore original conv layers, undoing apply_conditioned_lora."""
    unet, unet_type = _get_unet(net)

    if unet_type in ('DhariwalUNet', 'SongUNet'):
        for module in unet.modules():
            if not isinstance(module, UNetBlock):
                continue
            for attr_name in ('conv0', 'conv1', 'qkv', 'proj'):
                if hasattr(module, attr_name):
                    wrapper = getattr(module, attr_name)
                    if isinstance(wrapper, LoRACustomConv2dConditioned):
                        setattr(module, attr_name, wrapper.original_conv)

    elif unet_type == 'ADM':
        from models.ddpm.unet import AttentionBlock, ResBlock
        for module in unet.modules():
            if isinstance(module, AttentionBlock):
                for attr_name in ("qkv", "proj_out"):
                    wrapper = getattr(module, attr_name)
                    if isinstance(wrapper, LoRAConv1dConditioned):
                        setattr(module, attr_name, wrapper.original_conv)
            if isinstance(module, ResBlock):
                for idx in [-1]:
                    for layers in [module.in_layers, module.out_layers]:
                        w = layers[idx]
                        if isinstance(w, LoRAConv2dConditioned):
                            layers[idx] = w.original_conv


def get_lora_params(lora_modules, store=None):
    """Return a flat list of all trainable LoRA parameters.

    Includes measurement encoder parameters if store has one.
    """
    params = []
    for m in lora_modules:
        params.extend(m.lora_down.parameters())
        params.extend(m.lora_up.parameters())
    if store is not None and store.encoder is not None:
        params.extend(store.encoder.parameters())
    return params


def frozen_tweedie(net, lora_modules, x, sigma):
    """Evaluate denoiser with LoRA scaling temporarily zeroed."""
    saved = []
    for m in lora_modules:
        saved.append(m.scaling)
        m.scaling = 0.0

    with torch.no_grad():
        out = net(x, sigma)

    for m, s in zip(lora_modules, saved):
        m.scaling = s

    return out


def save_lora(lora_modules, path, metadata=None, store=None):
    """Save trained LoRA weights (and measurement encoder if present) to disk."""
    state = OrderedDict()
    module_types = []
    for i, m in enumerate(lora_modules):
        state[f"{i}.lora_down.weight"] = m.lora_down.weight.data.cpu()
        state[f"{i}.lora_up.weight"] = m.lora_up.weight.data.cpu()
        module_types.append(type(m).__name__)

    m0 = lora_modules[0]

    has_custom = any(isinstance(m, LoRACustomConv2dConditioned) for m in lora_modules)
    has_conv1d = any(isinstance(m, LoRAConv1dConditioned) for m in lora_modules)
    has_conv2d = any(isinstance(m, LoRAConv2dConditioned) for m in lora_modules)

    if has_custom:
        # DhariwalUNet/SongUNet path — infer from kernel size
        has_attn_k = any(isinstance(m, LoRACustomConv2dConditioned) and
                         m.original_conv.weight is not None and
                         m.original_conv.weight.shape[-1] == 1
                         for m in lora_modules)
        has_res_k = any(isinstance(m, LoRACustomConv2dConditioned) and
                        m.original_conv.weight is not None and
                        m.original_conv.weight.shape[-1] == 3
                        for m in lora_modules)
        if has_attn_k and has_res_k:
            target_modules = "all"
        elif has_res_k:
            target_modules = "resblock"
        else:
            target_modules = "attention"
    else:
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
        "target_modules": target_modules,
        "module_types": module_types,
        "y_channels": m0.y_channels,
    }
    # Save measurement encoder if present
    if store is not None and store.encoder is not None:
        enc = store.encoder
        checkpoint["encoder_state"] = enc.state_dict()
        checkpoint["obs_shape"] = list(enc.obs_shape)
        checkpoint["img_resolution"] = enc.img_resolution
        checkpoint["latent_res"] = enc.latent_res
    if metadata is not None:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, path)


def load_conditioned_lora(net, path):
    """Load saved LoRA weights (and measurement encoder if present).

    Returns:
        (list[LoRA modules], MeasurementStore or None)
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    rank = checkpoint["rank"]
    alpha = checkpoint["alpha"]
    y_channels = checkpoint.get("y_channels", 0)
    target_modules = checkpoint.get("target_modules", "all")
    state = checkpoint["lora_state"]
    obs_shape = checkpoint.get("obs_shape", None)
    img_resolution = checkpoint.get("img_resolution", None)
    latent_res = checkpoint.get("latent_res", 8)
    if obs_shape is not None:
        obs_shape = tuple(obs_shape)

    lora_modules, store = apply_conditioned_lora(
        net, rank=rank, alpha=alpha, y_channels=y_channels,
        target_modules=target_modules, obs_shape=obs_shape,
        img_resolution=img_resolution, latent_res=latent_res)

    for i, m in enumerate(lora_modules):
        m.lora_down.weight.data.copy_(state[f"{i}.lora_down.weight"])
        m.lora_up.weight.data.copy_(state[f"{i}.lora_up.weight"])

    # Restore encoder weights if present
    if store is not None and store.encoder is not None and "encoder_state" in checkpoint:
        # Trigger lazy build by doing a dummy forward
        dummy = torch.zeros(1, *obs_shape)
        store.encoder(dummy)
        store.encoder.load_state_dict(checkpoint["encoder_state"])

    return lora_modules, store
