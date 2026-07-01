from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# TUH-19 channels (forced)
# =============================================================================

TUH_19_CHANNELS: List[str] = [
    "FP1", "FP2",
    "F3", "F4",
    "C3", "C4",
    "P3", "P4",
    "O1", "O2",
    "F7", "F8",
    "T3", "T4",
    "T5", "T6",
    "FZ", "CZ", "PZ",
]
CHANNEL_DICT: Dict[str, int] = {ch: i for i, ch in enumerate(TUH_19_CHANNELS)}

# =============================================================================
# Init helpers
# =============================================================================

@torch.no_grad()
def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """Truncated normal init (PyTorch-like)."""
    def norm_cdf(x: torch.Tensor | float) -> torch.Tensor:
        x = torch.as_tensor(x, device=tensor.device, dtype=tensor.dtype)
        return (1.0 + torch.erf(x / math.sqrt(2.0))) / 2.0

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.0)).add_(mean).clamp_(min=a, max=b)
    return tensor


def _init_transformer(m: nn.Module, std: float = 0.02) -> None:
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=std)
    elif isinstance(m, nn.Conv2d):
        trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _fix_residual_scaling(blocks: nn.ModuleList) -> None:
    """Rescale residual projection weights (common in ViT impls)."""
    @torch.no_grad()
    def rescale(p: torch.Tensor, layer_id: int) -> None:
        p.div_(math.sqrt(2.0 * layer_id))

    for lid, blk in enumerate(blocks):
        if hasattr(blk, "attn") and hasattr(blk.attn, "proj"):
            rescale(blk.attn.proj.weight.data, lid + 1)
        if hasattr(blk, "mlp") and hasattr(blk.mlp, "fc2"):
            rescale(blk.mlp.fc2.weight.data, lid + 1)

# =============================================================================
# Masking conventions (IMPORTANT)
# =============================================================================
# mask_x: LongTensor of FLAT INDICES of *VISIBLE/KEPT* (t,c) in [0, N*C).
#
# This file uses ONLY mask_x (uniform (t,c) masking).
# mask_t is not used by the encoder/predictor/reconstructor below.

def sample_mask_x_uniform(
    *,
    N: int,
    C: int,
    mask_ratio: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Uniform random masking over flattened (t,c) grid.
    Returns mask_x: [K] kept indices, K ~= (1-mask_ratio) * (N*C), clamped to [1, N*C].
    """
    mr = float(mask_ratio)
    if not (0.0 <= mr < 1.0):
        raise ValueError(f"mask_ratio must be in [0,1), got {mask_ratio}")

    T = int(N * C)
    K = int(round((1.0 - mr) * T))
    K = max(1, min(T, K))

    keep = torch.randperm(T, device=device, generator=generator)[:K].to(torch.long)
    keep, _ = torch.sort(keep)
    return keep


def _masked_bool_from_mask_x(
    mask_x: Optional[torch.Tensor],
    *,
    N: int,
    C: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns masked_bool [N,C] where True=masked, False=visible.
    Default: mask_x None/empty => all visible (all False).
    """
    masked = torch.zeros((N, C), dtype=torch.bool, device=device)
    if mask_x is None or mask_x.numel() == 0:
        return masked

    kept = mask_x.reshape(-1).to(device=device, dtype=torch.long)
    if (kept < 0).any() or (kept >= N * C).any():
        raise ValueError(f"mask_x indices out of range [0, {N*C})")

    t = (kept // C).long()
    c = (kept % C).long()

    masked[:] = True
    masked[t, c] = False
    return masked


# =========================== Mask helpers ============================ #
def apply_mask(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.long:
        raise TypeError("apply_mask: mask must be LongTensor of flat indices")

    B, N, C, D = x.shape
    x_flat = x.reshape(B, N * C, D).contiguous()

    if mask.ndim == 2:
        mN, mC = mask.shape
        idx = mask.reshape(1, mN * mC, 1).expand(B, mN * mC, D)
        out = torch.gather(x_flat, dim=1, index=idx).view(B, mN, mC, D)
    else:
        m = mask.numel()
        idx = mask.view(1, m, 1).expand(B, m, D)
        out = torch.gather(x_flat, dim=1, index=idx)

    return out.contiguous()


def apply_mask_t(mask_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if mask_t.dtype != torch.long:
        raise TypeError("apply_mask_t: mask_t must be LongTensor")
    B, N, D = x.shape
    if (mask_t < 0).any() or (mask_t >= N).any():
        raise ValueError("apply_mask_t: indices out of range")
    idx = mask_t.view(1, -1, 1).expand(B, mask_t.numel(), D)
    return torch.gather(x, dim=1, index=idx)


def apply_mask_with_token(mask: torch.Tensor, x: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.long:
        raise TypeError("apply_mask_with_token: mask must be LongTensor")

    B, N, C, D = x.shape
    x_flat = x.reshape(B, N * C, D).contiguous()

    if mask_token.ndim == 0:
        y_flat = mask_token.to(x.device, x.dtype).view(1, 1, 1).expand(B, N * C, D).clone()
    elif mask_token.ndim == 1 and mask_token.shape[0] == D:
        y_flat = mask_token.to(x.device, x.dtype).view(1, 1, D).expand(B, N * C, D).clone()
    else:
        y_flat = mask_token.to(x.device, x.dtype).expand(B, N, C, D).reshape(B, N * C, D).clone()

    K = mask.numel()
    idx = mask.view(1, K, 1).to(x.device).expand(B, K, D)
    kept = torch.gather(x_flat, dim=1, index=idx)
    y_flat.scatter_(dim=1, index=idx, src=kept)
    return y_flat.view(B, N, C, D).contiguous()


def apply_mask_t_with_token(mask_t: torch.Tensor, x: torch.Tensor, mask_token: torch.Tensor) -> torch.Tensor:
    B, N, D = x.shape
    if mask_token.ndim == 1 and mask_token.shape[0] == D:
        y = mask_token.to(x.device, x.dtype).view(1, 1, D).expand(B, N, D).clone()
    else:
        y = mask_token.to(x.device, x.dtype).expand(B, N, D).clone()

    idx = mask_t.view(1, -1, 1).to(x.device).expand(B, mask_t.numel(), D)
    kept = torch.gather(x, dim=1, index=idx)
    y.scatter_(dim=1, index=idx, src=kept)
    return y.contiguous()

# =============================================================================
# Summary-shape helpers (NEW)
# =============================================================================

def _as_bn_sd(x_sum: torch.Tensor) -> tuple[torch.Tensor, int, bool]:
    """
    Accepts:
      - [B,N,D]      => returns [B,N,1,D], S=1, squeeze_back=True
      - [B,N,S,D]    => returns as-is,      S, squeeze_back=False
    """
    if x_sum.ndim == 3:
        return x_sum.unsqueeze(2), 1, True
    if x_sum.ndim == 4:
        return x_sum, int(x_sum.shape[2]), False
    raise ValueError(f"x_sum must be [B,N,D] or [B,N,S,D], got {tuple(x_sum.shape)}")


def _flatten_time_summary_tokens(x4: torch.Tensor) -> torch.Tensor:
    """[B,N,S,D] -> [B,N*S,D] (t-major then s)."""
    B, N, S, D = x4.shape
    return x4.contiguous().view(B, N * S, D)

# =============================================================================
# Rotary embeddings (RoPE)
# =============================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).contiguous()
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int = 0,
    scale: float = 1.0,
) -> torch.Tensor:
    freqs = freqs.to(t.device, t.dtype)
    rot_dim = freqs.shape[-1]
    end = start_index + rot_dim
    if rot_dim > t.shape[-1]:
        raise ValueError(f"RoPE dim {rot_dim} > feature dim {t.shape[-1]}")

    left, mid, right = t[..., :start_index], t[..., start_index:end], t[..., end:]
    mid = (mid * freqs.cos() * scale) + (rotate_half(mid) * freqs.sin() * scale)
    return torch.cat((left, mid, right), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        learned_freq: bool = False,
        interpolate_factor: float = 1.0,
    ):
        super().__init__()
        if interpolate_factor < 1.0:
            raise ValueError("interpolate_factor must be >= 1.0")
        base = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.freqs = nn.Parameter(base, requires_grad=learned_freq)
        self.interpolate_factor = float(interpolate_factor)
        self._cache: Dict[Tuple[str, int], torch.Tensor] = {}

    def freqs_1d(
        self,
        L: int,
        device: torch.device,
        dtype: torch.dtype,
        offset: float = 0.0,
    ) -> torch.Tensor:
        key = ("freqs1d", int(L))
        if key in self._cache:
            return self._cache[key].to(device=device, dtype=dtype)
        pos = (torch.arange(L, device=device, dtype=dtype) + offset) / self.interpolate_factor
        freqs = torch.outer(pos.to(self.freqs.dtype), self.freqs).repeat_interleave(2, dim=-1)
        self._cache[key] = freqs
        return freqs.to(device=device, dtype=dtype)

# =============================================================================
# Transformer blocks
# =============================================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep + torch.rand(shape, device=x.device, dtype=x.dtype)
        rand.floor_()
        return x * rand / keep


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        is_causal: bool = False,
        use_rope: bool = False,
        return_attention: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.use_rope = bool(use_rope)
        self.is_causal = bool(is_causal)
        self.return_attention = bool(return_attention)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = float(attn_drop)

    def _q(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        kv = self.kv_proj(x).view(B, T, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        return kv[0], kv[1]

    @staticmethod
    def _prep_additive_mask(
        attn_mask: Optional[torch.Tensor],
        *,
        B: int,
        H: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Accepts:
          - bool mask: True means "block" => becomes -inf additive
          - float mask: additive already (0 or -inf typically)
        Shapes accepted:
          [Tq,Tk] or [B,Tq,Tk] or [B,H,Tq,Tk]
        """
        if attn_mask is None:
            return None

        m = attn_mask
        if m.dtype == torch.bool:
            m = torch.where(
                m,
                torch.tensor(float("-inf"), device=device, dtype=dtype),
                torch.tensor(0.0, device=device, dtype=dtype),
            )
        elif not torch.is_floating_point(m):
            m = m.to(device=device, dtype=dtype)

        if m.ndim == 2:
            m = m.unsqueeze(0).unsqueeze(0)  # [1,1,Tq,Tk]
        elif m.ndim == 3:
            m = m.unsqueeze(1)               # [B,1,Tq,Tk]
        elif m.ndim != 4:
            raise ValueError(f"attn_mask rank must be 2/3/4, got {m.shape}")

        if m.size(0) == 1 and B > 1:
            m = m.expand(B, -1, -1, -1)
        if m.size(1) == 1 and H > 1:
            m = m.expand(-1, H, -1, -1)
        return m.to(device=device, dtype=dtype)

    def forward(
        self,
        x_q: torch.Tensor,                           # [B,Tq,D]
        *,
        freqs: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,    # additive or bool
        x_kv: Optional[torch.Tensor] = None,         # [B,Tk,D]
    ) -> torch.Tensor:
        B, Tq, D = x_q.shape
        x_kv = x_q if x_kv is None else x_kv
        if x_kv.shape[0] != B or x_kv.shape[2] != D:
            raise ValueError(f"x_kv must be [B,Tk,D]={B, 'Tk', D}, got {tuple(x_kv.shape)}")

        q = self._q(x_q)         # [B,H,Tq,dh]
        k, v = self._kv(x_kv)    # [B,H,Tk,dh]

        if self.use_rope and freqs is not None:
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        add_mask = self._prep_additive_mask(
            attn_mask, B=B, H=self.num_heads, device=x_q.device, dtype=torch.float32
        )

        if self.return_attention:
            scale = 1.0 / math.sqrt(q.size(-1))
            logits = (q @ k.transpose(-2, -1)) * scale
            if add_mask is not None:
                logits = logits + add_mask
            return torch.softmax(logits, dim=-1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=add_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=self.is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, Tq, D)
        return self.proj_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        hid = int(round(dim * float(mlp_ratio)))
        self.fc1 = nn.Linear(dim, hid)
        self.act = act_layer()
        self.fc2 = nn.Linear(hid, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        is_causal: bool = False,
        use_rope: bool = False,
        return_attention: bool = False,
    ):
        super().__init__()
        self.return_attention = bool(return_attention)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            is_causal=is_causal,
            use_rope=use_rope,
            return_attention=return_attention,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(
        self,
        x: torch.Tensor,                             # [B,T,D]
        *,
        freqs: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        x_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm1(x)
        y = self.attn(q, freqs=freqs, attn_mask=attn_mask, x_kv=x_kv)
        if self.return_attention:
            return y
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# =============================================================================
# Patch embedding (pretokenized TUH-19 only)
# =============================================================================

class PatchEmbedPreTokenized(nn.Module):
    """
    Expects x: [B, 19, N, D_raw] (pretoken patches per (channel,time))
    Returns:   [B, N, 19, D_model]
    """
    def __init__(
        self,
        *,
        img_size: Tuple[int, int] = (19, 30),     # (C, N)
        patch_size: int = 200,                   # D_raw
        embed_dim: int = 768,
        temporal_channels: int = 25,
        k_temporal_first: int = 49,
        stride_temporal_first: int = 25,
        temporal_extra_convs: int = 2,
        pos_kernel_c: int = 19,
        pos_kernel_t: int = 7,
        dropout: float = 0.1,
        use_spectral: bool = True,
        use_dynamic_pe: bool = False,
        mask_token_trainable: bool = False,
    ):
        super().__init__()
        C, N = int(img_size[0]), int(img_size[1])
        if C != 19:
            raise ValueError(f"Forced TUH-19: img_size[0] must be 19, got {C}")
        D_raw = int(patch_size)

        self.num_patches = (C, N)
        self.embed_dim = int(embed_dim)
        self.patch_size = D_raw
        self.use_spectral = bool(use_spectral)
        self.use_dynamic_pe = bool(use_dynamic_pe)

        layers: List[nn.Module] = [
            nn.Conv2d(
                1, temporal_channels,
                kernel_size=(1, k_temporal_first),
                stride=(1, stride_temporal_first),
                padding=(0, k_temporal_first // 2),
            ),
            nn.GroupNorm(5, temporal_channels),
            nn.GELU(),
        ]
        for _ in range(int(temporal_extra_convs)):
            layers += [
                nn.Conv2d(temporal_channels, temporal_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
                nn.GroupNorm(5, temporal_channels),
                nn.GELU(),
            ]
        self.proj_in = nn.Sequential(*layers)

        def out_len(L: int, k: int, s: int, p: int, d: int = 1) -> int:
            return (L + 2 * p - d * (k - 1) - 1) // s + 1

        w = out_len(D_raw, k_temporal_first, stride_temporal_first, k_temporal_first // 2)
        for _ in range(int(temporal_extra_convs)):
            w = out_len(w, 3, 1, 1)
        self._flat_feat = temporal_channels * w

        self.to_dmodel = nn.Sequential(nn.Linear(self._flat_feat, embed_dim, bias=True), nn.Dropout(dropout))

        if self.use_spectral:
            n_freq = D_raw // 2 + 1
            self.spectral_proj = nn.Sequential(nn.Linear(n_freq, embed_dim, bias=True), nn.Dropout(dropout))

        if self.use_dynamic_pe:
            k_c = pos_kernel_c if (pos_kernel_c % 2 == 1) else (pos_kernel_c + 1)
            k_t = pos_kernel_t if (pos_kernel_t % 2 == 1) else (pos_kernel_t + 1)
            self.positional_encoding = nn.Conv2d(
                embed_dim, embed_dim,
                kernel_size=(k_c, k_t),
                padding=(k_c // 2, k_t // 2),
                groups=embed_dim,
                bias=True,
            )

        self.mask_encoding = nn.Parameter(torch.zeros(D_raw), requires_grad=bool(mask_token_trainable))

        # keep your original Kaiming init for these conv/linear blocks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x [B,19,N,D], got {tuple(x.shape)}")
        B, C, N, D = x.shape
        if C != 19:
            raise ValueError(f"Forced TUH-19: got C={C}")
        if D != self.patch_size:
            raise ValueError(f"Got D={D} but built with patch_size={self.patch_size}")

        # fold (C,N) into "height" so conv2d runs over time-axis only
        x2d = x.contiguous().view(B, 1, C * N, D)
        feat = self.proj_in(x2d)                           # [B, Tc, C*N, w]
        feat = feat.permute(0, 2, 1, 3).contiguous()       # [B, C*N, Tc, w]
        feat = feat.view(B, C, N, self._flat_feat)         # [B, C, N, flat]
        tok = self.to_dmodel(feat)                         # [B, C, N, Dm]

        if self.use_spectral:
            spec = torch.fft.rfft(x.view(B * C * N, D), dim=-1, norm="forward")
            spec = torch.abs(spec).view(B, C, N, D // 2 + 1)
            tok = tok + self.spectral_proj(spec)

        if self.use_dynamic_pe:
            pos = self.positional_encoding(tok.permute(0, 3, 1, 2).contiguous())
            tok = tok + pos.permute(0, 2, 3, 1).contiguous()

        return tok.permute(0, 2, 1, 3).contiguous()        # [B, N, C, Dm]

# =============================================================================
# Criss-cross masks
# =============================================================================

def _attn_mask_vis_only_token_axis(masked_bool_full: torch.Tensor, *, B: int) -> torch.Tensor:
    """
    masked_bool_full: [N, L], True=masked token, False=visible
    returns additive mask for token-axis attention: [B*N, L, L]
      - keys that are masked => -inf
      - if all keys masked for a timestep, unmask all keys to avoid NaNs
    """
    N, L = masked_bool_full.shape
    BN = B * N

    key_block = masked_bool_full.clone()
    all_masked_t = ~(~masked_bool_full).any(dim=1)  # [N]
    if all_masked_t.any():
        key_block = key_block.clone()
        key_block[all_masked_t] = False

    key_block = key_block.unsqueeze(0).expand(B, -1, -1).reshape(BN, L)  # [B*N, L]
    add = torch.zeros(BN, L, L, device=key_block.device, dtype=torch.float32)
    return add.masked_fill(key_block.unsqueeze(1), float("-inf")).contiguous()


def _attn_mask_vis_only_time_axis(masked_bool_full: torch.Tensor, *, B: int) -> torch.Tensor:
    """
    masked_bool_full: [N, L], True=masked token, False=visible
    returns additive mask for time-axis attention: [B*L, N, N]
      - for each token-axis item (L), block masked keys along time (N)
      - if an item is masked for all timesteps, unmask all to avoid NaNs
    """
    N, L = masked_bool_full.shape
    BL = B * L
    keep = ~masked_bool_full  # [N,L]

    all_masked_l = ~keep.any(dim=0)  # [L]
    if all_masked_l.any():
        keep = keep.clone()
        keep[:, all_masked_l] = True

    key_block = (~keep).transpose(0, 1).contiguous()            # [L,N] True=block
    key_block = key_block.unsqueeze(0).expand(B, -1, -1).reshape(BL, N)  # [B*L,N]

    add = torch.zeros(BL, N, N, device=key_block.device, dtype=torch.float32)
    return add.masked_fill(key_block.unsqueeze(1), float("-inf")).contiguous()

# =============================================================================
# Criss-cross stack
# =============================================================================

class CrissCrossBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        rope_time: bool = True,
        rope_chan: bool = True,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("CrissCross requires even model dim")

        self.D = int(dim)
        self.Dh = self.D // 2
        Htot = int(num_heads)

        # split heads between (token-axis) and (time-axis)
        Hc = max(1, Htot // 2)
        while Hc > 1 and (self.Dh % Hc) != 0:
            Hc -= 1
        Ht = max(1, Htot - Hc)
        while Ht > 1 and (self.Dh % Ht) != 0:
            Ht -= 1
        if (self.Dh % Hc) != 0 or (self.Dh % Ht) != 0:
            raise ValueError(f"Bad head split: Dh={self.Dh}, Hc={Hc}, Ht={Ht}")

        self.rope_time = bool(rope_time)
        self.rope_chan = bool(rope_chan)

        self.norm1 = norm_layer(self.D)
        self.attn_c = Attention(
            self.Dh, Hc,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_rope=self.rope_chan
        )
        self.attn_t = Attention(
            self.Dh, Ht,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_rope=self.rope_time
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(self.D)
        self.mlp = MLP(dim=self.D, mlp_ratio=mlp_ratio, drop=drop)

    def forward(
        self,
        x: torch.Tensor,                          # [B,N,L,D]
        *,
        freqs_c: Optional[torch.Tensor],
        freqs_t: Optional[torch.Tensor],
        add_mask_c: Optional[torch.Tensor],       # [B*N,L,L]
        add_mask_t: Optional[torch.Tensor],       # [B*L,N,N]
    ) -> torch.Tensor:
        B, N, L, _ = x.shape
        q = self.norm1(x)
        xs, xt = torch.split(q, [self.Dh, self.Dh], dim=-1)

        # token-axis attention per timestep: [B*N, L, Dh]
        xs_seq = xs.reshape(B * N, L, self.Dh).contiguous()
        ys = self.attn_c(xs_seq, freqs=freqs_c, attn_mask=add_mask_c).view(B, N, L, self.Dh)

        # time-axis attention per token-axis item: [B*L, N, Dh]
        xt_seq = xt.permute(0, 2, 1, 3).reshape(B * L, N, self.Dh).contiguous()
        yt = self.attn_t(xt_seq, freqs=freqs_t, attn_mask=add_mask_t).view(B, L, N, self.Dh)
        yt = yt.permute(0, 2, 1, 3).contiguous()

        y = torch.cat([ys, yt], dim=-1).contiguous()
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrissCrossStack(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        rope_time: bool = True,
        rope_chan: bool = True,
        rope_dim_frac: float = 1.0,
        rope_interp_time: float = 1.0,
        rope_interp_chan: float = 1.0,
    ):
        super().__init__()
        dprs = [x.item() for x in torch.linspace(0, drop_path, steps=int(depth))]
        self.blocks = nn.ModuleList(
            [
                CrissCrossBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dprs[i],
                    norm_layer=norm_layer,
                    rope_time=rope_time,
                    rope_chan=rope_chan,
                )
                for i in range(int(depth))
            ]
        )

        rope_dim_per_head = max(2, int((dim // num_heads) * float(rope_dim_frac)) // 2 * 2)
        self.rope_time = RotaryEmbedding(rope_dim_per_head, interpolate_factor=rope_interp_time)
        self.rope_chan = RotaryEmbedding(rope_dim_per_head, interpolate_factor=rope_interp_chan)
        self.use_rope_time = bool(rope_time)
        self.use_rope_chan = bool(rope_chan)

    def forward(
        self,
        x: torch.Tensor,
        *,
        add_mask_c: Optional[torch.Tensor],
        add_mask_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        _, N, L, _ = x.shape
        freqs_t = self.rope_time.freqs_1d(N, device=x.device, dtype=x.dtype) if self.use_rope_time else None
        freqs_c = self.rope_chan.freqs_1d(L, device=x.device, dtype=x.dtype) if self.use_rope_chan else None
        for blk in self.blocks:
            x = blk(x, freqs_c=freqs_c, freqs_t=freqs_t, add_mask_c=add_mask_c, add_mask_t=add_mask_t)
        return x

# =============================================================================
# Reconstructors
# =============================================================================

class LightReconstructor(nn.Module):
    """Simple per-token MLP head; can return full map or only masked positions."""
    def __init__(
        self,
        *,
        embed_dim: int,
        patch_size: int,
        hidden_ratio: float = 0.0,
        act: str = "gelu",
        recon_embed_dim: Optional[int] = None,
        use_internal_embed: bool = False,
    ):
        super().__init__()
        self.in_dim = int(embed_dim)
        self.dec_dim = int(recon_embed_dim) if (use_internal_embed and recon_embed_dim is not None) else self.in_dim
        self.patch_size = int(patch_size)

        self.norm = nn.LayerNorm(self.in_dim)
        self.proj_in = nn.Linear(self.in_dim, self.dec_dim) if self.dec_dim != self.in_dim else nn.Identity()

        if float(hidden_ratio) > 0.0:
            hid = int(round(self.dec_dim * float(hidden_ratio)))
            act_layer = nn.GELU() if act.lower() == "gelu" else nn.ReLU(inplace=True)
            self.head = nn.Sequential(nn.Linear(self.dec_dim, hid), act_layer, nn.Linear(hid, self.patch_size))
        else:
            self.head = nn.Linear(self.dec_dim, self.patch_size)

    def forward(self, x_tokens: torch.Tensor, mask_y_bool: torch.Tensor, *, return_full: bool = False):
        x = self.proj_in(self.norm(x_tokens))
        if return_full:
            return self.head(x)  # [B,N,C,P]

        B, N, C, D = x.shape
        idx_nc = torch.nonzero(mask_y_bool, as_tuple=False)
        if idx_nc.numel() == 0:
            empty = torch.empty(B, 0, self.patch_size, device=x.device, dtype=x.dtype)
            flat = torch.empty(0, dtype=torch.long, device=x.device)
            return empty, flat

        flat = idx_nc[:, 0] * C + idx_nc[:, 1]
        mask4 = mask_y_bool.unsqueeze(0).unsqueeze(-1).expand(B, N, C, D)
        toks = x[mask4].view(B, -1, D)
        return self.head(toks), flat


class ReconstructorTimeSummary(nn.Module):
    """
    Reconstruct masked (t,c) raw patches from encoder time-summary tokens.

    UPDATED: supports x_sum as [B,N,D] or [B,N,S,D] and *truly uses all S tokens*
    by building KV as [B, N*S, dec] and forming timestep queries from the S tokens.
    """
    def __init__(
        self,
        *,
        embed_dim: int,
        patch_size: int,
        num_channels: int = 19,
        recon_embed_dim: Optional[int] = None,
        use_internal_embed: bool = False,
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.in_dim = int(embed_dim)
        self.dec_dim = int(recon_embed_dim) if (use_internal_embed and recon_embed_dim is not None) else self.in_dim
        self.patch_size = int(patch_size)
        self.C = int(num_channels)

        self.norm_in = nn.LayerNorm(self.in_dim)
        self.proj_in = nn.Linear(self.in_dim, self.dec_dim) if self.dec_dim != self.in_dim else nn.Identity()

        # channel embedding used both for query conditioning and pooling across S
        self.chan_embed = nn.Embedding(self.C, self.dec_dim)
        self.query_bias = nn.Parameter(torch.zeros(1, 1, self.dec_dim))
        trunc_normal_(self.chan_embed.weight, std=float(init_std))

        dprs = [x.item() for x in torch.linspace(0, drop_path, steps=int(depth))]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.dec_dim,
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=bool(qkv_bias),
                    drop=float(drop),
                    attn_drop=float(attn_drop),
                    drop_path=dprs[i],
                    norm_layer=norm_layer,
                    is_causal=False,
                    use_rope=False,
                )
                for i in range(int(depth))
            ]
        )
        self.norm = norm_layer(self.dec_dim)
        self.out_proj = nn.Linear(self.dec_dim, self.patch_size)

        self.apply(lambda m: _init_transformer(m, std=0.02))
        _fix_residual_scaling(self.blocks)

    def forward(self, x_sum: torch.Tensor, mask_y_bool: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_sum:
          - [B,N,D] or [B,N,S,D]
        mask_y_bool:
          - [N,C] True=masked positions to reconstruct
        Returns:
          - pred: [B,M,P] where M=#masked positions
          - flat: [M] flat indices t*C + c (same order as pred's M)
        """
        x4, S, _squeeze_back = _as_bn_sd(x_sum)  # [B,N,S,Denc]
        if mask_y_bool.ndim != 2:
            raise ValueError(f"mask_y_bool must be [N,C], got {tuple(mask_y_bool.shape)}")

        B, N, S2, Denc = x4.shape
        if Denc != self.in_dim:
            raise ValueError(f"embed_dim mismatch: {Denc} != {self.in_dim}")
        if mask_y_bool.shape[0] != N or mask_y_bool.shape[1] != self.C:
            raise ValueError(f"mask_y_bool must be [N={N},C={self.C}], got {tuple(mask_y_bool.shape)}")

        # Project summaries to decoder dim
        kv_time = self.proj_in(self.norm_in(x4))           # [B,N,S,dec]
        kv_flat = _flatten_time_summary_tokens(kv_time)    # [B,N*S,dec]

        idx = torch.nonzero(mask_y_bool, as_tuple=False)   # [M,2] (t,c)
        if idx.numel() == 0:
            empty = torch.empty(B, 0, self.patch_size, device=x_sum.device, dtype=x_sum.dtype)
            flat = torch.empty(0, dtype=torch.long, device=x_sum.device)
            return empty, flat

        t_idx = idx[:, 0].long()
        c_idx = idx[:, 1].long()
        flat = t_idx * self.C + c_idx

        # --- Build queries from the S summary tokens at the masked timestep ---
        # kv_t: [B,M,S,dec]
        kv_t = kv_time[:, t_idx, :, :]

        # channel-conditioned pooling over S:
        # q_chan: [B,M,dec]
        q_chan = self.chan_embed(c_idx).to(device=kv_t.device, dtype=kv_t.dtype).unsqueeze(0)  # [1,M,dec]
        q_chan = q_chan.expand(B, -1, -1).contiguous()
        q_chan = q_chan + self.query_bias  # [B,M,dec]

        # scores: [B,M,S]
        scores = (kv_t * q_chan.unsqueeze(2)).sum(dim=-1) / math.sqrt(self.dec_dim)
        w = torch.softmax(scores, dim=2)  # [B,M,S]
        q = (w.unsqueeze(-1) * kv_t).sum(dim=2)  # [B,M,dec]
        q = q + q_chan  # keep channel conditioning

        # Cross-attend q into kv_flat (all summary tokens across all timesteps)
        for blk in self.blocks:
            q = blk(q, x_kv=kv_flat)

        q = self.norm(q)
        return self.out_proj(q), flat

# =============================================================================
# Predictor (time-summary) -- UPDATED for S>1
# =============================================================================

class TransformerPredictorTimeSummary(nn.Module):
    """
    Temporal refiner / predictor over time-summary tokens.

    Input :
      - x_sum [B,N,D] or [B,N,S,D]
    Output:
      - same shape as input (rank preserved)

    UPDATED: for S>1, runs self-attn over flattened [B, N*S, ·] so it truly uses all S tokens.
    """
    def __init__(
        self,
        *,
        num_timesteps: int,
        embed_dim: int = 768,
        use_inp_embed: bool = True,
        width_mult: float = 1.0,
        predictor_embed_dim: Optional[int] = None,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_std: float = 0.02,
        return_attention_layer: int = -1,
    ):
        super().__init__()
        self.N = int(num_timesteps)
        self.use_inp_embed = bool(use_inp_embed)

        hidden = int(predictor_embed_dim) if predictor_embed_dim is not None else max(
            32, int(round(embed_dim * float(width_mult)))
        )

        H = max(1, min(int(num_heads), hidden // 8))
        while H > 1 and (hidden % H) != 0:
            H -= 1

        self.inp = nn.Linear(embed_dim, hidden) if self.use_inp_embed else nn.Identity()

        dprs = [x.item() for x in torch.linspace(0, drop_path_rate, steps=int(depth))]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=hidden,
                    num_heads=H,
                    mlp_ratio=float(mlp_ratio),
                    qkv_bias=bool(qkv_bias),
                    drop=float(drop_rate),
                    attn_drop=float(attn_drop_rate),
                    drop_path=dprs[i],
                    norm_layer=norm_layer,
                    is_causal=False,
                    use_rope=False,
                    return_attention=((i + 1) == int(return_attention_layer)),
                )
                for i in range(int(depth))
            ]
        )
        self.norm = norm_layer(hidden)
        self.proj = nn.Linear(hidden, embed_dim)

        self.apply(lambda m: _init_transformer(m, std=float(init_std)))
        _fix_residual_scaling(self.blocks)

    def forward(self, x_sum: torch.Tensor):
        """
        Returns:
          - x_pred with same rank as x_sum: [B,N,D] or [B,N,S,D]
        If return_attention_layer is set, returns that attention tensor early (shape depends on Block).
        """
        x4, S, squeeze_back = _as_bn_sd(x_sum)  # [B,N,S,D]
        B, N, S2, D = x4.shape
        if N != self.N:
            raise ValueError(f"Expected N={self.N}, got {N}")

        x = self.inp(_flatten_time_summary_tokens(x4))  # [B,N*S,hidden]

        for blk in self.blocks:
            out = blk(x, x_kv=x)
            if blk.return_attention:
                return out
            x = out

        y = self.proj(self.norm(x))                   # [B,N*S,D_enc]
        y4 = y.view(B, N, S, D).contiguous()          # [B,N,S,D_enc]
        return y4[:, :, 0, :] if squeeze_back else y4

# =============================================================================
# Encoder (time-summary tokens, uniform (t,c) masking only)
# =============================================================================

class TransformerEncoderTimeSummary(nn.Module):
    """
    Encoder that outputs ONLY time-summary tokens (per timestep).

    Input:
      x: [B,19,N,D_raw]  (pretoken patches)
      mask_x: LongTensor kept flat indices in [0, N*19)

    Output:
      if summary_tokens==1: [B,N,D_model]
      else:                [B,N,S,D_model]
    """
    def __init__(
        self,
        *,
        img_size: Tuple[int, int] = (19, 30),
        patch_size: int = 200,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        patch_module: Optional[type[nn.Module]] = None,
        init_std: float = 0.02,
        return_attention_layer: int = -1,
        use_dynamic_pe: bool = False,
        use_criss_cross: bool = False,
        rope_time: bool = True,
        rope_chan: bool = True,
        rope_dim_frac: float = 1.0,
        rope_interp_time: float = 1.0,
        rope_interp_chan: float = 1.0,
        use_channel_embed: bool = True,
        summary_tokens: int = 1,
    ):
        super().__init__()
        patch_module = PatchEmbedPreTokenized if patch_module is None else patch_module

        C0, _ = int(img_size[0]), int(img_size[1])
        if C0 != 19:
            raise ValueError(f"Forced TUH-19: img_size[0] must be 19, got {C0}")

        self.embed_dim = int(embed_dim)
        self.summary_tokens = int(summary_tokens)
        if self.summary_tokens < 1:
            raise ValueError("summary_tokens must be >= 1")

        self.patch_embed = patch_module(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            use_dynamic_pe=use_dynamic_pe,
        )
        C, N = self.patch_embed.num_patches
        if C != 19:
            raise RuntimeError(f"patch_embed produced C={C}, expected 19")
        self.C = int(C)
        self.N = int(N)

        self.time_summary = nn.Parameter(torch.zeros(1, 1, self.summary_tokens, self.embed_dim))
        trunc_normal_(self.time_summary, std=float(init_std))

        self.use_channel_embed = bool(use_channel_embed)
        self.chan_embed = nn.Embedding(64, self.embed_dim) if self.use_channel_embed else None
        self.register_buffer("mask_tok_zero", torch.zeros(1, 1, 1, self.embed_dim), persistent=False)

        self.use_criss = bool(use_criss_cross)
        if self.use_criss:
            self.cc_stack = CrissCrossStack(
                depth=int(depth),
                dim=self.embed_dim,
                num_heads=int(num_heads),
                mlp_ratio=float(mlp_ratio),
                qkv_bias=bool(qkv_bias),
                drop=float(drop_rate),
                attn_drop=float(attn_drop_rate),
                drop_path=float(drop_path_rate),
                norm_layer=norm_layer,
                rope_time=bool(rope_time),
                rope_chan=bool(rope_chan),
                rope_dim_frac=float(rope_dim_frac),
                rope_interp_time=float(rope_interp_time),
                rope_interp_chan=float(rope_interp_chan),
            )
        else:
            dprs = [x.item() for x in torch.linspace(0, drop_path_rate, steps=int(depth))]
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=self.embed_dim,
                        num_heads=int(num_heads),
                        mlp_ratio=float(mlp_ratio),
                        qkv_bias=bool(qkv_bias),
                        drop=float(drop_rate),
                        attn_drop=float(attn_drop_rate),
                        drop_path=dprs[i],
                        norm_layer=norm_layer,
                        is_causal=False,
                        use_rope=False,
                        return_attention=((i + 1) == int(return_attention_layer)),
                    )
                    for i in range(int(depth))
                ]
            )

        self.norm = norm_layer(self.embed_dim)

        # --- init: DO NOT re-init patch_embed; init only transformer-y parts ---
        if self.chan_embed is not None:
            _init_transformer(self.chan_embed, std=0.02)

        if not self.use_criss:
            for blk in self.blocks:
                blk.apply(lambda m: _init_transformer(m, std=0.02))
            _init_transformer(self.norm, std=0.02)
            _fix_residual_scaling(self.blocks)

    def prepare_chan_ids(self, channels: Optional[List[str]]) -> torch.Tensor:
        device = next(self.parameters()).device
        if not channels:
            return torch.arange(self.C, device=device, dtype=torch.long)

        idxs: List[int] = []
        for ch in channels:
            key = str(ch).strip().upper().strip(".")
            if key not in CHANNEL_DICT:
                raise ValueError(f"Unknown channel '{ch}'. Allowed TUH-19: {TUH_19_CHANNELS}")
            idxs.append(CHANNEL_DICT[key])

        if len(idxs) != 19:
            raise ValueError(f"Expected 19 channel names; got {len(idxs)}. Allowed: {TUH_19_CHANNELS}")

        return torch.tensor(idxs, device=device, dtype=torch.long)

    def forward(
        self,
        x: torch.Tensor,                          # [B,19,N,D_raw]
        *,
        chan_ids: Optional[torch.Tensor] = None,
        mask_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_tok = self.patch_embed(x)               # [B,N,C,Dm]
        B, N, C, D = x_tok.shape
        if C != 19:
            raise RuntimeError(f"Forced TUH-19 at runtime: got C={C}")
        if N != self.N:
            self.N = int(N)

        if self.use_channel_embed and self.chan_embed is not None:
            if chan_ids is None:
                chan_ids = torch.arange(C, device=x_tok.device, dtype=torch.long)
            else:
                chan_ids = chan_ids.to(device=x_tok.device, dtype=torch.long).view(-1)
            if chan_ids.numel() != C:
                raise ValueError(f"chan_ids must have length C=19; got {chan_ids.numel()}")
            x_tok = x_tok + self.chan_embed(chan_ids).unsqueeze(0).unsqueeze(0)  # [B,N,C,D]

        masked_bool = _masked_bool_from_mask_x(mask_x, N=N, C=C, device=x_tok.device)  # [N,C]
        mask4 = masked_bool.unsqueeze(0).unsqueeze(-1).expand(B, N, C, D)
        mask_tok = self.mask_tok_zero.to(device=x_tok.device, dtype=x_tok.dtype).expand(B, N, C, D)

        x_q = torch.where(mask4, mask_tok, x_tok)
        x_kv = x_q

        summary = self.time_summary.to(device=x_tok.device, dtype=x_tok.dtype).expand(B, N, self.summary_tokens, D)
        x_q = torch.cat([summary, x_q], dim=2)    # [B,N,L,D], L=S+C
        x_kv = torch.cat([summary, x_kv], dim=2)
        L = x_q.shape[2]

        # summaries always visible
        prefix_vis = torch.zeros(N, self.summary_tokens, device=x_tok.device, dtype=torch.bool)  # False=visible
        masked_full = torch.cat([prefix_vis, masked_bool], dim=1)  # [N,L]

        add_mask_c = _attn_mask_vis_only_token_axis(masked_full, B=B)  # [B*N,L,L]
        add_mask_t = _attn_mask_vis_only_time_axis(masked_full, B=B)   # [B*L,N,N]

        if self.use_criss:
            y = self.cc_stack(x_q, add_mask_c=add_mask_c, add_mask_t=add_mask_t)  # [B,N,L,D]
        else:
            xq = x_q.view(B * N, L, D).contiguous()
            xkv = x_kv.view(B * N, L, D).contiguous()
            for blk in self.blocks:
                out = blk(xq, attn_mask=add_mask_c, x_kv=xkv)
                if blk.return_attention:
                    return out
                xq = out
            y = xq.view(B, N, L, D).contiguous()

        y = self.norm(y)
        if self.summary_tokens == 1:
            return y[:, :, 0, :]                  # [B,N,D]
        return y[:, :, : self.summary_tokens, :]  # [B,N,S,D]
