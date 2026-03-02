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
from einops import rearrange


# ---------------------------------------------------------------------------
# TransformerCBG components (ViT-style pure transformer)
# ---------------------------------------------------------------------------

class QKNormAttention(nn.Module):
    """Multi-head self-attention with QK normalization and pre-LayerNorm.

    QK-norm (per-head L2 normalization of Q and K before dot product)
    prevents attention logit growth and stabilizes training at any depth.
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

        # Xavier init for QKV, small init for output projection
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        """x: [B, N, D] -> [B, N, D]"""
        B, N, D = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, H, N, Dh]

        # QK normalization: L2-normalize per head
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Scaled dot-product attention (scale by sqrt(head_dim) after norm)
        scale = self.head_dim ** 0.5  # learnable-temperature equivalent
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [B, H, N, Dh]

        out = out.transpose(1, 2).reshape(B, N, D)
        return x + self.out_proj(out)


class GEGLU_FFN(nn.Module):
    """Pre-norm feed-forward with GEGLU activation.

    GEGLU: splits the projection into two halves, applies GELU to one
    and multiplies element-wise. No dead neurons (unlike ReLU).
    """

    def __init__(self, dim, mult=2):
        super().__init__()
        hidden = int(dim * mult)
        self.norm = nn.LayerNorm(dim)
        self.w_gate = nn.Linear(dim, hidden * 2)  # gate + value
        self.w_out = nn.Linear(hidden, dim)

        nn.init.xavier_uniform_(self.w_gate.weight)
        nn.init.zeros_(self.w_gate.bias)
        nn.init.normal_(self.w_out.weight, std=0.02)
        nn.init.zeros_(self.w_out.bias)

    def forward(self, x):
        h = self.norm(x)
        gate, val = self.w_gate(h).chunk(2, dim=-1)
        h = F.gelu(gate) * val
        return x + self.w_out(h)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: QKNormAttention + GEGLU FFN."""

    def __init__(self, dim, num_heads=4, ffn_mult=2):
        super().__init__()
        self.attn = QKNormAttention(dim, num_heads)
        self.ffn = GEGLU_FFN(dim, ffn_mult)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x


class TransformerCBG(nn.Module):
    """Pure transformer CBG architecture (~2-3M params).

    Tokenizes image (ViT-style 16x16 patches), measurements (per-receiver),
    and sigma (sinusoidal embedding). Concatenates all tokens, applies
    N transformer blocks, then decodes measurement tokens to output.

    Same forward interface as MeasurementPredictor: forward(x_t, sigma, y)
    returns [B, meas_flat_dim] predictions.
    """

    def __init__(self, img_resolution=256, img_channels=1,
                 obs_shape=(20, 360), meas_flat_dim=14400,
                 embed_dim=256, num_layers=4, num_heads=4, ffn_mult=2,
                 patch_size=16):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.obs_shape = obs_shape
        self.meas_flat_dim = meas_flat_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_mult = ffn_mult
        self.patch_size = patch_size

        # --- Image tokenization (ViT-style) ---
        num_patches = (img_resolution // patch_size) ** 2  # e.g. 256
        patch_dim = img_channels * patch_size * patch_size  # e.g. 256
        self.num_patches = num_patches
        self.patch_proj = nn.Linear(patch_dim, embed_dim)
        self.patch_pos_emb = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

        # --- Measurement tokenization (per-receiver) ---
        # obs_shape = (num_receivers, samples_per_receiver), e.g. (20, 360)
        self.num_receivers = obs_shape[0]
        receiver_dim = 1
        for s in obs_shape[1:]:
            receiver_dim *= s
        self.receiver_real_dim = receiver_dim * 2  # complex -> view_as_real doubles
        self.meas_proj = nn.Linear(self.receiver_real_dim, embed_dim)
        self.meas_pos_emb = nn.Parameter(torch.randn(1, self.num_receivers, embed_dim) * 0.02)

        # --- Sigma token ---
        self.sigma_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_mult)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # --- Output head: decode measurement token positions ---
        self.out_proj = nn.Linear(embed_dim, self.receiver_real_dim)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_t, sigma, y):
        """
        Args:
            x_t:   [B, C, H, W] noisy image
            sigma: [B] or scalar noise level
            y:     [B, *obs_shape] measurements (complex)

        Returns:
            [B, meas_flat_dim] predicted measurement residual
        """
        B = x_t.shape[0]
        device = x_t.device

        # --- 1. Image tokens (ViT-style patching) ---
        # [B, C, H, W] -> [B, num_patches, patch_dim]
        patches = rearrange(x_t, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                           p1=self.patch_size, p2=self.patch_size)
        img_tokens = self.patch_proj(patches) + self.patch_pos_emb  # [B, 256, D]

        # --- 2. Measurement tokens (per-receiver) ---
        if y.is_complex():
            y_real = torch.view_as_real(y).float()  # [B, 20, 360, 2]
        else:
            y_real = y.float()
        y_flat = y_real.reshape(B, self.num_receivers, -1)  # [B, 20, 720]
        meas_tokens = self.meas_proj(y_flat) + self.meas_pos_emb  # [B, 20, D]

        # --- 3. Sigma token ---
        if not isinstance(sigma, torch.Tensor):
            sigma_t = torch.tensor([sigma], dtype=torch.float32,
                                   device=device).expand(B)
        else:
            sigma_t = sigma.float().view(-1)
            if sigma_t.numel() == 1:
                sigma_t = sigma_t.expand(B)
        sigma_emb = timestep_embedding(sigma_t.log(), self.embed_dim)
        sigma_token = self.sigma_mlp(sigma_emb).unsqueeze(1)  # [B, 1, D]

        # --- 4. Concatenate: [sigma | image | measurement] ---
        tokens = torch.cat([sigma_token, img_tokens, meas_tokens], dim=1)
        # total tokens: 1 + num_patches + num_receivers = 277

        # --- 5. Transformer ---
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)

        # --- 6. Extract measurement tokens and decode ---
        meas_start = 1 + self.num_patches  # after sigma + image tokens
        meas_out = tokens[:, meas_start:meas_start + self.num_receivers]  # [B, 20, D]
        decoded = self.out_proj(meas_out)  # [B, 20, 720]
        return decoded.reshape(B, -1)  # [B, 14400]


# ---------------------------------------------------------------------------
# Measurement encoder (for non-image observations)
# ---------------------------------------------------------------------------

class MeasurementEncoder(nn.Module):
    """Projects arbitrary-shape measurements to spatial feature maps.

    Maps measurements of any shape (e.g. [B, 20, 360] complex scattering data)
    to [B, y_channels, H, W] spatial features that can be concatenated with
    x_t in the classifier's input conv.

    Projects to a small spatial size (default 32x32) then lets the caller
    bilinearly upsample. This avoids a massive Linear layer when
    img_resolution is large (e.g. 256x256 = 262k output dim → 537M params).

    For complex inputs, real and imaginary parts are stacked as channels.
    Builds lazily on first forward (input dim depends on complex vs real).
    """

    def __init__(self, obs_shape: Tuple[int, ...], y_channels: int,
                 img_resolution: int, enc_spatial_size: Optional[int] = None):
        super().__init__()
        self.obs_shape = obs_shape
        self.y_channels = y_channels
        self.img_resolution = img_resolution
        # Internal projection spatial size. None → img_resolution for backwards
        # compat with old checkpoints. New models should pass 32 explicitly.
        self.enc_spatial_size = enc_spatial_size if enc_spatial_size is not None else img_resolution
        self._spatial_out = y_channels * self.enc_spatial_size * self.enc_spatial_size
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
        return out.view(B, self.y_channels, self.enc_spatial_size,
                        self.enc_spatial_size)


# ---------------------------------------------------------------------------
# Measurement decoder (spatial -> measurement space)
# ---------------------------------------------------------------------------

class MeasurementDecoder(nn.Module):
    """Maps spatial UNet features back to measurement space.

    Applied *before* the UNet's out_conv so it receives base_channels
    (e.g. 64) rather than out_channels (e.g. 1), avoiding a severe
    information bottleneck.

    Architecture: AdaptiveAvgPool2d → Flatten → PreNorm-ResidualMLP → Linear
    Pre-norm residual (transformer-style): h = h + Linear(LeakyReLU(LN(h)))
    - LayerNorm inside the branch stabilizes activations (prevents spikes)
    - Skip connection on the main path preserves gradient flow
    - LeakyReLU(0.2) has no dead zone for negative inputs
    """

    def __init__(self, in_channels: int, meas_flat_dim: int,
                 pool_size: int = 8, hidden: int = 512):
        super().__init__()
        self.meas_flat_dim = meas_flat_dim
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        pool_dim = in_channels * pool_size * pool_size
        hidden = min(pool_dim, hidden)

        # Input projection
        self.flatten = nn.Flatten(1)
        self.input_proj = nn.Linear(pool_dim, hidden)

        # Pre-norm residual blocks: h = h + Linear(LeakyReLU(LayerNorm(h)))
        self.norm1 = nn.LayerNorm(hidden)
        self.act1 = nn.LeakyReLU(0.2)
        self.res1 = nn.Linear(hidden, hidden)

        self.norm2 = nn.LayerNorm(hidden)
        self.act2 = nn.LeakyReLU(0.2)
        self.res2 = nn.Linear(hidden, hidden)

        # Output projection
        self.out_proj = nn.Linear(hidden, meas_flat_dim)

        # Kaiming init for hidden layers
        for m in [self.input_proj, self.res1, self.res2]:
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias)
        # Small init on output layer so decoder starts near zero
        nn.init.normal_(self.out_proj.weight, std=0.01)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h):
        h = self.flatten(self.pool(h))
        h = self.input_proj(h)
        # Pre-norm residual block 1
        h = h + self.res1(self.act1(self.norm1(h)))
        # Pre-norm residual block 2
        h = h + self.res2(self.act2(self.norm2(h)))
        return self.out_proj(h)


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


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention: Q from spatial features, K/V from context tokens.

    Lets the UNet attend to measurement-derived tokens at the bottleneck,
    rather than relying solely on input-level concatenation.
    Zero-initialized output projection so the block starts as identity.
    """

    def __init__(self, channels, context_dim, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.q_proj = nn.Conv1d(channels, channels, 1)
        self.kv_proj = nn.Linear(context_dim, channels * 2)
        self.out_proj = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x, context):
        """
        Args:
            x:       [B, C, H, W] spatial features
            context: [B, N_tokens, context_dim]
        """
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)                    # [B, C, HW]

        h = self.norm(x_flat)
        q = self.q_proj(h)                               # [B, C, HW]
        kv = self.kv_proj(context).transpose(1, 2)       # [B, 2C, N]
        k, v = kv.chunk(2, dim=1)

        q = q.reshape(b, self.num_heads, self.head_dim, -1)
        k = k.reshape(b, self.num_heads, self.head_dim, -1)
        v = v.reshape(b, self.num_heads, self.head_dim, -1)

        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q * scale, k)
        attn = attn.softmax(dim=-1)
        h = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        h = h.reshape(b, c, -1)

        h = self.out_proj(h)
        return (x_flat + h).reshape(b, c, *spatial)


class MeasurementTokenizer(nn.Module):
    """Encodes raw measurements into a sequence of tokens for cross-attention.

    Maps measurements of any shape (e.g. [B, 20, 360] complex) to
    [B, num_tokens, token_dim] via flatten -> Linear -> reshape.
    """

    def __init__(self, meas_flat_dim: int, token_dim: int, num_tokens: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        out_dim = token_dim * num_tokens
        hidden = min(meas_flat_dim, out_dim, 2048)
        self.proj = nn.Sequential(
            nn.Linear(meas_flat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, y):
        B = y.shape[0]
        if y.is_complex():
            y = torch.view_as_real(y).flatten(1).float()
        else:
            y = y.flatten(1).float()
        tokens = self.proj(y)
        return tokens.view(B, self.num_tokens, self.token_dim)


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
                 decoder_hidden: int = 2048,
                 num_res_blocks: int = 1,
                 num_tokens: int = 0,
                 enc_spatial_size: Optional[int] = None):
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
            num_res_blocks: Number of ResBlocks per encoder/decoder level (default 1).
            num_tokens: Number of cross-attention tokens for measurement conditioning.
                        0 disables cross-attention (default, backwards compatible).
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
        self.num_res_blocks = num_res_blocks
        self.num_tokens = num_tokens

        # Build measurement encoder for non-image observations
        if obs_shape is not None:
            if img_resolution is None:
                img_resolution = self.out_size[0]
            self.measurement_encoder = MeasurementEncoder(
                obs_shape, y_channels, img_resolution,
                enc_spatial_size=enc_spatial_size)
        else:
            self.measurement_encoder = None

        # Build measurement decoder to project spatial output to measurement space
        # Uses base_channels (not out_channels) because it's applied before out_conv
        if obs_shape is not None and meas_flat_dim is not None:
            self.measurement_decoder = MeasurementDecoder(
                base_channels, meas_flat_dim, hidden=decoder_hidden)
        else:
            self.measurement_decoder = None

        # Build measurement tokenizer + cross-attention for bottleneck
        if num_tokens > 0 and meas_flat_dim is not None:
            bot_ch_for_xattn = base_channels * channel_mult[-1]
            self.meas_tokenizer = MeasurementTokenizer(
                meas_flat_dim, token_dim=bot_ch_for_xattn,
                num_tokens=num_tokens)
            self.bot_cross_attn = CrossAttention(
                bot_ch_for_xattn, context_dim=bot_ch_for_xattn,
                num_heads=attn_heads)
        else:
            self.meas_tokenizer = None
            self.bot_cross_attn = None

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
            # Stack num_res_blocks ResBlocks per level
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                level_blocks.append(ResBlock(
                    prev_ch if j == 0 else cur_ch, cur_ch, emb_dim))
            self.enc_blocks.append(level_blocks)
            self.downsamples.append(nn.Conv2d(cur_ch, cur_ch, 3, stride=2, padding=1))
            prev_ch = cur_ch

        # -- bottleneck (ResBlock + SelfAttention [+ CrossAttention] + ResBlock)
        self.bot_res1 = ResBlock(bot_ch, bot_ch, emb_dim)
        self.bot_attn = SelfAttention(bot_ch, num_heads=attn_heads)
        self.bot_res2 = ResBlock(bot_ch, bot_ch, emb_dim)

        # -- decoder -----------------------------------------------------------
        self.upsample_layers = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        dec_prev_ch = bot_ch
        for i in reversed(range(num_levels)):
            skip_ch = ch_list[i]
            if i > 0:
                dec_out_ch = ch_list[i - 1]
            else:
                dec_out_ch = C
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            # Stack num_res_blocks ResBlocks per decoder level
            level_blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch = (dec_prev_ch + skip_ch) if j == 0 else dec_out_ch
                level_blocks.append(ResBlock(in_ch, dec_out_ch, emb_dim))
            self.dec_blocks.append(level_blocks)
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

        # --- tokenize measurements for cross-attention ---
        y_tokens = None
        if self.meas_tokenizer is not None:
            y_tokens = self.meas_tokenizer(y)             # [B, num_tokens, bot_ch]

        # --- input conv ---
        h = self.input_conv(torch.cat([x_t, y_in], dim=1))  # [B, C, 256, 256]

        # --- encoder ---
        skips = []
        for level_blocks, down in zip(self.enc_blocks, self.downsamples):
            for blk in level_blocks:
                h = blk(h, emb)
            skips.append(h)                               # save for decoder
            h = down(h)

        # --- bottleneck ---
        h = self.bot_res1(h, emb)
        h = self.bot_attn(h)
        if self.bot_cross_attn is not None:
            h = self.bot_cross_attn(h, y_tokens)
        h = self.bot_res2(h, emb)

        # --- decoder ---
        for up, level_blocks in zip(self.upsample_layers, self.dec_blocks):
            h = up(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            for blk in level_blocks:
                h = blk(h, emb)

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
    """Save classifier state_dict + architecture config + optional metadata.

    Dispatches on architecture type: 'unet' (MeasurementPredictor) or
    'transformer' (TransformerCBG).
    """
    if isinstance(classifier, TransformerCBG):
        config = {
            'arch': 'transformer',
            'img_resolution': classifier.img_resolution,
            'img_channels': classifier.img_channels,
            'obs_shape': list(classifier.obs_shape),
            'meas_flat_dim': classifier.meas_flat_dim,
            'embed_dim': classifier.embed_dim,
            'num_layers': classifier.num_layers,
            'num_heads': classifier.num_heads,
            'ffn_mult': classifier.ffn_mult,
            'patch_size': classifier.patch_size,
        }
        checkpoint = {
            'state_dict': classifier.state_dict(),
            'config': config,
        }
    else:
        config = {
            'arch': 'unet',
            'in_channels':   classifier.in_channels,
            'y_channels':    classifier.y_channels,
            'out_channels':  classifier.out_channels,
            'out_size':      list(classifier.out_size),
            'base_channels': classifier.base_channels,
            'emb_dim':       classifier.emb_dim,
            'channel_mult':  classifier.channel_mult,
            'num_res_blocks': classifier.num_res_blocks,
            'num_tokens':    classifier.num_tokens,
        }
        if classifier.obs_shape is not None:
            config['obs_shape'] = list(classifier.obs_shape)
            config['img_resolution'] = classifier.measurement_encoder.img_resolution
            config['enc_spatial_size'] = classifier.measurement_encoder.enc_spatial_size
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
    """Reconstruct classifier from a saved checkpoint.

    Dispatches on config['arch']: 'transformer' -> TransformerCBG,
    'unet' or missing -> MeasurementPredictor (backwards compatible).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint['config'])

    arch = config.pop('arch', 'unet')

    if arch == 'transformer':
        if 'obs_shape' in config:
            config['obs_shape'] = tuple(config['obs_shape'])
        classifier = TransformerCBG(**config).to(device)
        classifier.load_state_dict(checkpoint['state_dict'])
        classifier.eval()
        return classifier

    # UNet path (MeasurementPredictor) — original logic
    if 'obs_shape' in config:
        config['obs_shape'] = tuple(config['obs_shape'])
    classifier = MeasurementPredictor(**config).to(device)
    # Rebuild lazy measurement encoder before loading state_dict
    if 'meas_enc_in_dim' in checkpoint and classifier.measurement_encoder is not None:
        classifier.measurement_encoder._build(
            checkpoint['meas_enc_in_dim'], device)

    state_dict = checkpoint['state_dict']

    # Backwards compat: old checkpoints stored enc_blocks/dec_blocks as flat
    # ModuleList (enc_blocks.0.norm1), new code nests per-level ModuleLists
    # (enc_blocks.0.0.norm1). Remap if needed.
    if 'num_res_blocks' not in config:
        import re
        remapped = {}
        for k, v in state_dict.items():
            # enc_blocks.N.xxx -> enc_blocks.N.0.xxx
            m = re.match(r'^(enc_blocks|dec_blocks)\.(\d+)\.(.+)$', k)
            if m and not re.match(r'^\d+\.', m.group(3)):
                new_k = f'{m.group(1)}.{m.group(2)}.0.{m.group(3)}'
                remapped[new_k] = v
            else:
                remapped[k] = v
        state_dict = remapped

    classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier
