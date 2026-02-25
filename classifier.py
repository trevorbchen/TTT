"""
Classifier-Based Guidance (CBG) network for inverse problems.

MeasurementPredictor: small UNet-style encoder-decoder following the ADM
(Dhariwal & Nichol 2021) architecture conventions:

  - Pre-activation ResBlocks: GroupNorm -> SiLU -> Conv (x2), with
    residual skip connections and zero-initialized second conv
  - FiLM sigma conditioning (scale_shift_norm) between the two convolutions
  - Self-attention at the bottleneck (16x16)
  - log(sigma) sinusoidal embedding for stable multi-scale conditioning
  - Zero-initialized output head (network starts predicting zero residual)

Takes (x_t, sigma, y) and predicts the measurement residual:

    M_phi(x_t, sigma, y)  ~  A(Tweedie(x_t, sigma)) - y

At inference the guidance loss is ||M_phi(x_t, sigma, y)||^2, whose gradient
w.r.t. x_t flows only through this small network (~10M params), NOT through
the ~300M-param diffusion model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Measurement encoder (for non-image observations)
# ---------------------------------------------------------------------------

class MeasurementEncoder(nn.Module):
    """Projects arbitrary-shape measurements to spatial feature maps.

    Maps measurements of any shape (e.g. [B, 20, 360] complex scattering data)
    to [B, y_channels, H, W] spatial features that can be concatenated with
    x_t in the classifier's input conv.

    For complex inputs, real and imaginary parts are stacked as channels.
    Builds lazily on first forward (input dim depends on complex vs real).
    """

    def __init__(self, obs_shape: Tuple[int, ...], y_channels: int,
                 img_resolution: int):
        super().__init__()
        self.obs_shape = obs_shape
        self.y_channels = y_channels
        self.img_resolution = img_resolution
        self._spatial_out = y_channels * img_resolution * img_resolution
        self._built = False
        self._in_dim = None
        # placeholder so state_dict is non-empty before build
        self.proj = None

    def _build(self, in_dim, device):
        hidden = min(in_dim, self._spatial_out, 2048)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self._spatial_out),
        ).to(device)
        self._in_dim = in_dim
        self._built = True

    def forward(self, y):
        B = y.shape[0]
        if y.is_complex():
            y = torch.view_as_real(y).flatten(1).float()
        else:
            y = y.flatten(1).float()

        if not self._built:
            self._build(y.shape[1], y.device)

        out = self.proj(y)
        return out.view(B, self.y_channels, self.img_resolution,
                        self.img_resolution)


# ---------------------------------------------------------------------------
# Measurement decoder (spatial -> measurement space)
# ---------------------------------------------------------------------------

class MeasurementDecoder(nn.Module):
    """Maps spatial UNet features back to measurement space.

    Applied *before* the UNet's out_conv so it receives base_channels
    (e.g. 64) rather than out_channels (e.g. 1), avoiding a severe
    information bottleneck.

    Architecture: AdaptiveAvgPool2d → Flatten → [Linear → LN → GELU] x2 → Linear
    With base_channels=64, pool_size=8, hidden=2048:
        pool_dim=4096 → 2048 → 2048 → meas_flat_dim
    """

    def __init__(self, in_channels: int, meas_flat_dim: int,
                 pool_size: int = 8, hidden: int = 2048):
        super().__init__()
        self.meas_flat_dim = meas_flat_dim
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        pool_dim = in_channels * pool_size * pool_size
        hidden = min(pool_dim, hidden)
        self.proj = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(pool_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, meas_flat_dim),
        )
        # Kaiming init for hidden layers, small init for output layer
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init on output layer so decoder starts near zero
        out_layer = self.proj[-1]
        nn.init.normal_(out_layer.weight, std=0.01)

    def forward(self, h):
        return self.proj(self.pool(h))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embeddings (matches model/ddpm/nn.py)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


# ---------------------------------------------------------------------------
# Building blocks (following ADM / model/ddpm/unet.py conventions)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Pre-activation residual block with FiLM sigma conditioning.

    Structure (matches ADM ResBlock with use_scale_shift_norm=True):
        in_layers:  GroupNorm(in_ch) -> SiLU -> Conv(in_ch -> out_ch)
        FiLM:       emb -> SiLU -> Linear -> (scale, shift)
        out_layers: GroupNorm(out_ch) -> FiLM(scale, shift) -> SiLU -> Conv
        skip:       Identity or 1x1 Conv when channels change
        output:     skip(x) + out_layers(in_layers(x))

    The second conv is zero-initialized so the block starts as identity.
    """

    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        # First half: Norm -> Act -> Conv (changes channels)
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # Embedding projection -> scale & shift (FiLM)
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * out_ch),
        )

        # Second half: Norm -> FiLM -> Act -> zero_Conv
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1))

        # Skip connection
        if in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, emb):
        # First conv
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # FiLM conditioning (scale_shift_norm)
        emb_out = self.emb_proj(emb)
        while emb_out.ndim < h.ndim:
            emb_out = emb_out[..., None]
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        # Second conv (zero-initialized -> block starts as identity)
        h = self.act2(h)
        h = self.conv2(h)

        return self.skip(x) + h


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with pre-norm and zero-initialized output
    projection (matches ADM AttentionBlock).

    Applied at the bottleneck (16x16) for global spatial reasoning.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)                    # [B, C, HW]

        h = self.norm(x_flat)
        qkv = self.qkv(h)                               # [B, 3C, HW]
        q, k, v = qkv.reshape(b, 3, self.num_heads, self.head_dim, -1).unbind(1)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q * scale, k)
        attn = attn.softmax(dim=-1)
        h = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        h = h.reshape(b, c, -1)

        h = self.proj(h)                                 # zero-init -> starts as identity
        return (x_flat + h).reshape(b, c, *spatial)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class MeasurementPredictor(nn.Module):
    """
    Small UNet that predicts the measurement residual:

        M_phi(x_t, sigma, y)  ~  A(Tweedie(x_t, sigma)) - y

    Architecture (following ADM conventions)
    ----------------------------------------
    * y is bilinearly resized to match x_t and concatenated channel-wise
    * log(sigma) is embedded via sinusoidal encoding + MLP, injected via FiLM
    * Encoder: 4 levels, each with a ResBlock + stride-2 Downsample
    * Bottleneck: ResBlock + SelfAttention + ResBlock at 16x16
    * Decoder: mirrors encoder with skip connections + ResBlocks
    * Output: GroupNorm -> SiLU -> zero_init 1x1 Conv, then resize to out_size

    Channel progression with default channel_mult=(1, 2, 4, 4):
        Encoder: [C, 2C, 4C, 4C]  (C = base_channels = 64 -> [64, 128, 256, 256])
        Bottleneck: 4C
        Decoder mirrors encoder
    """

    def __init__(self, in_channels=3, y_channels=3, out_channels=3,
                 out_size=256, base_channels=64, emb_dim=256,
                 channel_mult=(1, 2, 4, 4), attn_heads=4,
                 obs_shape: Optional[Tuple[int, ...]] = None,
                 img_resolution: Optional[int] = None,
                 meas_flat_dim: Optional[int] = None,
                 decoder_hidden: int = 2048):
        """
        Args:
            obs_shape: Shape of a single observation *without* batch dim,
                       e.g. (20, 360) for complex scatter data. When provided,
                       a MeasurementEncoder is built to project arbitrary
                       observations into [B, y_channels, H, W] spatial features.
                       When None, y is assumed to already be 4D image-like.
            img_resolution: Spatial resolution for the encoder output. Required
                            when obs_shape is provided. Defaults to x_t resolution.
            meas_flat_dim: Flat dimension of real-valued measurements. When
                           provided with obs_shape, a MeasurementDecoder maps
                           UNet output back to measurement space so the loss
                           is computed there (not in the inflated spatial space).
            decoder_hidden: Hidden dim for MeasurementDecoder (default 2048).
        """
        super().__init__()
        self.in_channels = in_channels
        self.y_channels = y_channels
        self.out_channels = out_channels
        self.out_size = (out_size, out_size) if isinstance(out_size, int) else tuple(out_size)
        self.base_channels = base_channels
        self.emb_dim = emb_dim
        self.channel_mult = list(channel_mult)
        self.obs_shape = obs_shape
        self.meas_flat_dim = meas_flat_dim
        self.decoder_hidden = decoder_hidden

        # Build measurement encoder for non-image observations
        if obs_shape is not None:
            if img_resolution is None:
                img_resolution = self.out_size[0]
            self.measurement_encoder = MeasurementEncoder(
                obs_shape, y_channels, img_resolution)
        else:
            self.measurement_encoder = None

        # Build measurement decoder to project spatial output to measurement space
        # Uses base_channels (not out_channels) because it's applied before out_conv
        if obs_shape is not None and meas_flat_dim is not None:
            self.measurement_decoder = MeasurementDecoder(
                base_channels, meas_flat_dim, hidden=decoder_hidden)
        else:
            self.measurement_decoder = None

        C = base_channels
        ch_list = [C * m for m in channel_mult]  # e.g. [64, 128, 256, 256]
        num_levels = len(channel_mult)
        cat_ch = in_channels + y_channels  # input channels (default 6)
        bot_ch = ch_list[-1]  # bottleneck channels

        # -- sigma embedding (log-scale for better multi-scale coverage) -------
        self.sigma_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # -- input convolution -------------------------------------------------
        self.input_conv = nn.Conv2d(cat_ch, C, 3, padding=1)

        # -- encoder -----------------------------------------------------------
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = C
        for i, cur_ch in enumerate(ch_list):
            self.enc_blocks.append(ResBlock(prev_ch, cur_ch, emb_dim))
            self.downsamples.append(nn.Conv2d(cur_ch, cur_ch, 3, stride=2, padding=1))
            prev_ch = cur_ch

        # -- bottleneck (ResBlock + Attention + ResBlock) ----------------------
        self.bot_res1 = ResBlock(bot_ch, bot_ch, emb_dim)
        self.bot_attn = SelfAttention(bot_ch, num_heads=attn_heads)
        self.bot_res2 = ResBlock(bot_ch, bot_ch, emb_dim)

        # -- decoder -----------------------------------------------------------
        # Decoder level i receives: upsample(prev) concatenated with skip_i
        # and outputs the channel count for the next (higher-res) level.
        self.upsample_layers = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        # Iterate from deepest to shallowest
        dec_prev_ch = bot_ch
        for i in reversed(range(num_levels)):
            skip_ch = ch_list[i]
            # Output channels: ch_list of one level higher, or C for the top
            if i > 0:
                dec_out_ch = ch_list[i - 1]
            else:
                dec_out_ch = C
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.dec_blocks.append(ResBlock(dec_prev_ch + skip_ch, dec_out_ch, emb_dim))
            dec_prev_ch = dec_out_ch

        # -- output head (zero-init so the network starts predicting zeros) ----
        self.out_norm = nn.GroupNorm(min(32, C), C)
        self.out_act = nn.SiLU()
        self.out_conv = zero_module(nn.Conv2d(C, out_channels, 1))

    # ------------------------------------------------------------------
    def forward(self, x_t, sigma, y):
        """
        Args:
            x_t:   [B, C_in, 256, 256]  noisy image (pre-scaled by 1/s(t))
            sigma: [B] or scalar         noise level
            y:     [B, C_y, H_y, W_y]   measurement

        Returns:
            [B, out_channels, out_H, out_W]  predicted residual
        """
        B = x_t.shape[0]

        # --- sigma embedding (log-scale) ---
        if not isinstance(sigma, torch.Tensor):
            sigma_t = torch.tensor([sigma], dtype=torch.float32,
                                   device=x_t.device).expand(B)
        else:
            sigma_t = sigma.float().view(-1)
            if sigma_t.numel() == 1:
                sigma_t = sigma_t.expand(B)

        emb = timestep_embedding(sigma_t.log(), self.emb_dim)
        emb = self.sigma_mlp(emb)                        # [B, emb_dim]

        # --- encode / resize y to match x_t spatial dims ---
        if self.measurement_encoder is not None:
            # Non-image observation: project to spatial features
            y_in = self.measurement_encoder(y)
            if y_in.shape[-2:] != x_t.shape[-2:]:
                y_in = F.interpolate(y_in, size=x_t.shape[-2:],
                                     mode='bilinear', align_corners=False)
        elif y.ndim == 4 and y.shape[-2:] != x_t.shape[-2:]:
            y_in = F.interpolate(y, size=x_t.shape[-2:],
                                 mode='bilinear', align_corners=False)
        else:
            y_in = y

        # --- input conv ---
        h = self.input_conv(torch.cat([x_t, y_in], dim=1))  # [B, C, 256, 256]

        # --- encoder ---
        skips = []
        for enc, down in zip(self.enc_blocks, self.downsamples):
            h = enc(h, emb)
            skips.append(h)                               # save for decoder
            h = down(h)

        # --- bottleneck ---
        h = self.bot_res1(h, emb)
        h = self.bot_attn(h)
        h = self.bot_res2(h, emb)

        # --- decoder ---
        for up, dec in zip(self.upsample_layers, self.dec_blocks):
            h = up(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = dec(h, emb)

        # --- output ---
        h = self.out_norm(h)
        h = self.out_act(h)

        # --- decode to measurement space if decoder exists ---
        # Applied BEFORE out_conv so decoder gets base_channels features (e.g. 64)
        # instead of out_channels (e.g. 1) — avoids information bottleneck
        if self.measurement_decoder is not None:
            return self.measurement_decoder(h)  # [B, meas_flat_dim]

        h = self.out_conv(h)                              # [B, out_ch, 256, 256]

        # --- resize to target output size ---
        out_h, out_w = self.out_size
        if h.shape[-2:] != (out_h, out_w):
            h = F.interpolate(h, size=(out_h, out_w),
                              mode='bilinear', align_corners=False)
        return h


# ---------------------------------------------------------------------------
# Save / load utilities
# ---------------------------------------------------------------------------

def save_classifier(classifier, path, metadata=None):
    """Save classifier state_dict + architecture config + optional metadata."""
    config = {
        'in_channels':   classifier.in_channels,
        'y_channels':    classifier.y_channels,
        'out_channels':  classifier.out_channels,
        'out_size':      list(classifier.out_size),
        'base_channels': classifier.base_channels,
        'emb_dim':       classifier.emb_dim,
        'channel_mult':  classifier.channel_mult,
    }
    if classifier.obs_shape is not None:
        config['obs_shape'] = list(classifier.obs_shape)
        config['img_resolution'] = classifier.measurement_encoder.img_resolution
    if classifier.meas_flat_dim is not None:
        config['meas_flat_dim'] = classifier.meas_flat_dim
        config['decoder_hidden'] = classifier.decoder_hidden
    checkpoint = {
        'state_dict': classifier.state_dict(),
        'config': config,
    }
    # Save encoder build info so load_classifier can rebuild before load_state_dict
    if (classifier.measurement_encoder is not None
            and classifier.measurement_encoder._built):
        checkpoint['meas_enc_in_dim'] = classifier.measurement_encoder._in_dim
    if metadata:
        checkpoint.update(metadata)
    torch.save(checkpoint, path)


def load_classifier(path, device='cuda'):
    """Reconstruct MeasurementPredictor from a saved checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint['config'])
    # Convert obs_shape back to tuple if present
    if 'obs_shape' in config:
        config['obs_shape'] = tuple(config['obs_shape'])
    classifier = MeasurementPredictor(**config).to(device)
    # Rebuild lazy measurement encoder before loading state_dict
    if 'meas_enc_in_dim' in checkpoint and classifier.measurement_encoder is not None:
        classifier.measurement_encoder._build(
            checkpoint['meas_enc_in_dim'], device)
    classifier.load_state_dict(checkpoint['state_dict'])
    classifier.eval()
    return classifier
