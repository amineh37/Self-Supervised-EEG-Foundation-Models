from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================== Channels ============================== #
# TUH 19-channel order matching preprocessing:
# FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ
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

# Strict mapping: only these 19 are valid.
CHANNEL_DICT: Dict[str, int] = {ch: i for i, ch in enumerate(TUH_19_CHANNELS)}


# ============================== Init ================================= #
@torch.no_grad()
def _no_grad_trunc_normal_(
    tensor: torch.Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
) -> torch.Tensor:
    def norm_cdf(x: torch.Tensor | float) -> torch.Tensor:
        x = torch.as_tensor(x, device=tensor.device, dtype=tensor.dtype)
        return (1.0 + torch.erf(x / math.sqrt(2.0))) / 2.0

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.0)).add_(mean).clamp_(min=a, max=b)
    return tensor


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# ============================== Utils ================================ #
def _masked_bool_from_masks(
    mask_x: Optional[torch.Tensor],
    C: int,
    N: int,
    device: torch.device,
) -> torch.Tensor:
    masked_bool = torch.ones((N, C), dtype=torch.bool, device=device)
    if mask_x is None or mask_x.numel() == 0:
        return masked_bool
    kept_flat = mask_x.reshape(-1).to(device=device, dtype=torch.long)
    kt = (kept_flat // C).long()
    kc = (kept_flat % C).long()
    masked_bool[kt, kc] = False
    return masked_bool


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


# =========================== Rotary Embeddings ======================== #
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
    end_index = start_index + rot_dim
    assert rot_dim <= t.shape[-1], f"feature dim {t.shape[-1]} < rope dim {rot_dim}"

    t_left, t_middle, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t_rot = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
    return torch.cat((t_left, t_rot, t_right), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: float = 10000,
        learned_freq: bool = False,
        interpolate_factor: float = 1.0,
    ):
        super().__init__()
        assert interpolate_factor >= 1.0, "interpolate_factor must be >= 1.0"
        self.freqs = nn.Parameter(
            1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
            requires_grad=learned_freq,
        )
        self.interpolate_factor = interpolate_factor
        self.cache: Dict[Tuple[str, int], torch.Tensor] = {}

    def prepare_freqs_1d(
        self,
        L: int,
        device: torch.device,
        dtype: torch.dtype,
        offset: float = 0.0,
    ) -> torch.Tensor:
        key = ("freqs1d", L)
        if key in self.cache:
            return self.cache[key].to(device=device, dtype=dtype)

        pos = (torch.arange(L, device=device, dtype=dtype) + offset) / self.interpolate_factor
        freqs_scaled = torch.outer(pos.type(self.freqs.dtype), self.freqs).repeat_interleave(2, dim=-1)
        self.cache[key] = freqs_scaled
        return freqs_scaled.to(device=device, dtype=dtype)


# ============================ Core blocks ============================= #
class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None):
        super().__init__()
        self.drop_prob = float(drop_prob or 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x * rand / keep


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        is_causal: bool = False,
        use_rope: bool = False,
        return_attention: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.use_rope = bool(use_rope)
        self.is_causal = bool(is_causal)
        self.return_attention = bool(return_attention)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = float(attn_drop)

    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        return self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _shape_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        kv = (
            self.kv_proj(x)
            .reshape(B, T, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        return kv[0], kv[1]

    @staticmethod
    def _prep_mask(
        mask: Optional[torch.Tensor],
        B: int,
        H: int,
        Tq: int,
        Tk: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None

        if mask.dtype == torch.bool:
            mask = torch.where(
                mask,
                torch.tensor(float("-inf"), device=device, dtype=dtype),
                torch.tensor(0.0, device=device, dtype=dtype),
            )
        elif not torch.is_floating_point(mask):
            mask = mask.to(device=device, dtype=dtype)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim != 4:
            raise ValueError(f"attn_mask rank must be 2, 3 or 4; got {mask.shape}")

        if mask.size(0) == 1 and B > 1:
            mask = mask.expand(B, -1, -1, -1)
        if mask.size(1) == 1 and H > 1:
            mask = mask.expand(-1, H, -1, -1)
        return mask.to(device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        x_for_kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self._shape_q(x)
        k, v = self._shape_kv(x_for_kv if x_for_kv is not None else x)

        if self.use_rope and freqs is not None:
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        mask_broadcast = self._prep_mask(attn_mask, B, self.num_heads, T, T, x.device, torch.float32)

        if self.return_attention:
            scale = 1.0 / math.sqrt(q.size(-1))
            attn_logits = torch.matmul(q, k.transpose(-2, -1)).contiguous() * scale
            if mask_broadcast is not None:
                attn_logits = attn_logits + mask_broadcast
            return torch.softmax(attn_logits, dim=-1)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask_broadcast,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=self.is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return self.proj_drop(y)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
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
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        x_for_kv: Optional[torch.Tensor] = None,
        return_q_in: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        q_in = self.norm1(x)
        y = self.attn(q_in, freqs=freqs, attn_mask=attn_mask, x_for_kv=x_for_kv)
        if self.return_attention:
            return y
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return (x, q_in) if return_q_in else x


# =========================== Patch encoders =========================== #
class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 1000),
        patch_size: int = 16,
        patch_stride: Optional[int] = None,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = int(patch_size)
        self.patch_stride = patch_stride

        if patch_stride is None:
            self.num_patches = (img_size[0], img_size[1] // patch_size)
            stride_t = patch_size
        else:
            self.num_patches = (img_size[0], (img_size[1] - patch_size) // patch_stride + 1)
            stride_t = patch_stride

        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=(1, patch_size),
            stride=(1, stride_t),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.proj(x).transpose(1, 3).contiguous()
        return x


class PatchEmbedPreTokenized(nn.Module):
    """
    Forced TUH-19: expects x [B, 19, N, D], returns [B, N, 19, embed_dim].
    """
    def __init__(
        self,
        img_size: Tuple[int, int] = (19, 30),  # (C, N)
        patch_size: int = 200,  # D
        patch_stride: Optional[int] = None,  # ignored
        embed_dim: int = 768,
        *,
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
            raise ValueError(f"[PatchEmbedPreTokenized] Forced TUH-19: img_size[0] must be 19, got {C}")
        D = int(patch_size)

        self.num_patches = (C, N)
        self.embed_dim = int(embed_dim)
        self.patch_size = D
        self.use_spectral = bool(use_spectral)
        self.use_dynamic_pe = bool(use_dynamic_pe)

        layers: List[nn.Module] = [
            nn.Conv2d(1, temporal_channels, kernel_size=(1, k_temporal_first), stride=(1, stride_temporal_first),
                      padding=(0, k_temporal_first // 2)),
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

        def _out_len(L: int, k: int, s: int, p: int, d: int = 1) -> int:
            return (L + 2 * p - d * (k - 1) - 1) // s + 1

        w = _out_len(D, k_temporal_first, stride_temporal_first, k_temporal_first // 2)
        for _ in range(int(temporal_extra_convs)):
            w = _out_len(w, 3, 1, 1)

        self._flat_feat = temporal_channels * w
        self.to_dmodel = nn.Sequential(nn.Linear(self._flat_feat, embed_dim, bias=True), nn.Dropout(dropout))

        if self.use_spectral:
            n_freq = D // 2 + 1
            self.spectral_proj = nn.Sequential(nn.Linear(n_freq, embed_dim, bias=True), nn.Dropout(dropout))

        if self.use_dynamic_pe:
            k_c = pos_kernel_c if (pos_kernel_c % 2 == 1) else (pos_kernel_c + 1)
            k_t = pos_kernel_t if (pos_kernel_t % 2 == 1) else (pos_kernel_t + 1)
            self.positional_encoding = nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=(k_c, k_t),
                stride=(1, 1),
                padding=(k_c // 2, k_t // 2),
                groups=embed_dim,
                bias=True,
            )

        self.mask_encoding = nn.Parameter(torch.zeros(D), requires_grad=bool(mask_token_trainable))

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
            raise ValueError(f"PatchEmbedPreTokenized expects [B,19,N,D], got {tuple(x.shape)}")
        B, C, N, D = x.shape
        if C != 19:
            raise ValueError(f"[PatchEmbedPreTokenized] Forced TUH-19: got C={C}")
        if D != self.patch_size:
            raise ValueError(f"Got D={D} but built with patch_size={self.patch_size}")

        x_2d = x.contiguous().view(B, 1, C * N, D)
        feat = self.proj_in(x_2d)
        feat = feat.permute(0, 2, 1, 3).contiguous()
        feat = feat.view(B, C, N, self._flat_feat)
        tok = self.to_dmodel(feat)

        if self.use_spectral:
            spec = torch.fft.rfft(x.view(B * C * N, D), dim=-1, norm="forward")
            spec = torch.abs(spec).view(B, C, N, D // 2 + 1)
            tok = tok + self.spectral_proj(spec)

        if self.use_dynamic_pe:
            pos = self.positional_encoding(tok.permute(0, 3, 1, 2).contiguous())
            tok = (tok + pos.permute(0, 2, 3, 1).contiguous()).contiguous()

        return tok.permute(0, 2, 1, 3).contiguous()  # [B, N, C, embed_dim]


# ========================= Time-axis masks ============================ #
def _build_time_vis_only_attn_mask(masked_bool: torch.Tensor, B: int) -> torch.Tensor:
    N, C = masked_bool.shape
    BC = B * C
    keep_bool = ~masked_bool

    all_masked_c = ~keep_bool.any(dim=0)
    if all_masked_c.any():
        keep_bool = keep_bool.clone()
        keep_bool[:, all_masked_c] = True

    key_block = (~keep_bool).transpose(0, 1).contiguous()
    key_block = key_block.unsqueeze(0).expand(B, -1, -1).reshape(BC, N)

    add = torch.zeros(BC, N, N, dtype=torch.float32, device=key_block.device)
    return add.masked_fill(key_block.unsqueeze(1), float("-inf")).contiguous()


# ============================ CrissCross ============================== #
class CrissCrossBlock(nn.Module):
    def __init__(
        self,
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
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, "CrissCross requires even model dim."
        self.D = int(dim)
        self.Dh = self.D // 2
        self.H_total = int(num_heads)

        self.Hc = max(1, self.H_total // 2)
        while self.Dh % self.Hc != 0 and self.Hc > 1:
            self.Hc -= 1
        self.Ht = max(1, self.H_total - self.Hc)
        while self.Dh % self.Ht != 0 and self.Ht > 1:
            self.Ht -= 1
        assert self.Dh % self.Hc == 0 and self.Dh % self.Ht == 0, f"Bad head split: Dh={self.Dh}, Hc={self.Hc}, Ht={self.Ht}"

        self.rope_time = bool(rope_time)
        self.rope_chan = bool(rope_chan)
        self.rope_dim_frac = float(rope_dim_frac)

        self.norm1 = norm_layer(dim)
        self.attn_c = Attention(self.Dh, self.Hc, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, is_causal=False, use_rope=self.rope_chan)
        self.attn_t = Attention(self.Dh, self.Ht, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, is_causal=False, use_rope=self.rope_time)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

    def forward(
        self,
        x: torch.Tensor,
        freqs_c: Optional[torch.Tensor],
        freqs_t: Optional[torch.Tensor],
        add_mask_c: Optional[torch.Tensor],
        add_mask_t: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, N, C, D = x.shape
        q_in = self.norm1(x)
        xs, xt = torch.split(q_in, [self.Dh, self.Dh], dim=-1)

        xs_seq = xs.contiguous().view(B * N, C, self.Dh)
        ys = self.attn_c(xs_seq, freqs=freqs_c, attn_mask=add_mask_c).contiguous().view(B, N, C, self.Dh)

        xt_seq = xt.permute(0, 2, 1, 3).contiguous().view(B * C, N, self.Dh)
        yt = self.attn_t(xt_seq, freqs=freqs_t, attn_mask=add_mask_t).contiguous().view(B, C, N, self.Dh).permute(0, 2, 1, 3).contiguous()

        y = torch.cat([ys, yt], dim=-1).contiguous()
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CrissCrossStack(nn.Module):
    def __init__(
        self,
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
    ) -> None:
        super().__init__()
        dprs = [x.item() for x in torch.linspace(0, drop_path, steps=depth)]
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
                    rope_dim_frac=rope_dim_frac,
                )
                for i in range(depth)
            ]
        )
        rope_dim_per_head = max(2, int((dim // num_heads) * rope_dim_frac) // 2 * 2)
        self.rope_time = RotaryEmbedding(dim=rope_dim_per_head, interpolate_factor=rope_interp_time)
        self.rope_chan = RotaryEmbedding(dim=rope_dim_per_head, interpolate_factor=rope_interp_chan)
        self.use_rope_time = bool(rope_time)
        self.use_rope_chan = bool(rope_chan)

    def forward(self, x: torch.Tensor, add_mask_c: Optional[torch.Tensor], add_mask_t: Optional[torch.Tensor]) -> torch.Tensor:
        B, N, C, D = x.shape
        freqs_t = self.rope_time.prepare_freqs_1d(N, device=x.device, dtype=x.dtype) if self.use_rope_time else None
        freqs_c = self.rope_chan.prepare_freqs_1d(C, device=x.device, dtype=x.dtype) if self.use_rope_chan else None
        for blk in self.blocks:
            x = blk(x, freqs_c=freqs_c, freqs_t=freqs_t, add_mask_c=add_mask_c, add_mask_t=add_mask_t)
        return x


# =========================== Reconstructor ============================ #
class LightReconstructor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        hidden_ratio: float = 0.0,
        act: str = "gelu",
        *,
        recon_embed_dim: Optional[int] = None,
        use_internal_embed: bool = False,
    ):
        super().__init__()
        self.in_embed_dim = int(embed_dim)
        self.use_internal_embed = bool(use_internal_embed)
        self.dec_embed_dim = int(recon_embed_dim) if (self.use_internal_embed and recon_embed_dim is not None) else self.in_embed_dim
        self.patch_size = int(patch_size)

        self.tok_norm = nn.LayerNorm(self.in_embed_dim)
        self.proj_in = nn.Linear(self.in_embed_dim, self.dec_embed_dim, bias=True) if self.dec_embed_dim != self.in_embed_dim else nn.Identity()

        if hidden_ratio and hidden_ratio > 0.0:
            hid = int(round(self.dec_embed_dim * hidden_ratio))
            self.head = nn.Sequential(
                nn.Linear(self.dec_embed_dim, hid, bias=True),
                nn.GELU() if act == "gelu" else nn.ReLU(inplace=True),
                nn.Linear(hid, patch_size, bias=True),
            )
        else:
            self.head = nn.Linear(self.dec_embed_dim, patch_size, bias=True)

    def forward(self, x_tokens: torch.Tensor, mask_y_bool: torch.Tensor, return_full: bool = False):
        x_tokens = self.tok_norm(x_tokens)
        B, N, C, D_in = x_tokens.shape
        assert D_in == self.in_embed_dim, f"embed_dim mismatch: {D_in} != {self.in_embed_dim}"

        x_dec = self.proj_in(x_tokens)
        D_rec = self.dec_embed_dim

        if return_full:
            return self.head(x_dec)

        idx_nc = torch.nonzero(mask_y_bool, as_tuple=False)
        flat = idx_nc[:, 0] * C + idx_nc[:, 1]

        mask4 = mask_y_bool.unsqueeze(0).unsqueeze(-1).expand(B, N, C, D_rec)
        toks_masked = x_dec[mask4].view(B, -1, D_rec)
        y_masked = self.head(toks_masked)
        return y_masked, flat

class TransformerReconstructor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        *,
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
        use_rope_chan: bool = True,
        rope_dim_frac: float = 1.0,
        rope_interp_chan: float = 1.0,
    ):
        super().__init__()
        self.in_embed_dim = int(embed_dim)
        self.use_internal_embed = bool(use_internal_embed)
        self.dec_embed_dim = (
            int(recon_embed_dim)
            if (self.use_internal_embed and recon_embed_dim is not None)
            else self.in_embed_dim
        )
        self.patch_size = int(patch_size)

        self.tok_norm_in = nn.LayerNorm(self.in_embed_dim)
        self.proj_in = (
            nn.Linear(self.in_embed_dim, self.dec_embed_dim, bias=True)
            if self.dec_embed_dim != self.in_embed_dim
            else nn.Identity()
        )

        self.use_rope_chan = bool(use_rope_chan)
        if self.use_rope_chan:
            rope_dim_per_head = int((self.dec_embed_dim // num_heads) * float(rope_dim_frac))
            rope_dim_per_head = max(2, (rope_dim_per_head // 2) * 2)
            self.rope_chan = RotaryEmbedding(dim=rope_dim_per_head, interpolate_factor=rope_interp_chan)
        else:
            self.rope_chan = None

        dpr = [x.item() for x in torch.linspace(0, drop_path, steps=depth)]
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=self.dec_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    is_causal=False,
                    use_rope=False,
                    return_attention=False,
                )
                for i in range(depth)
            ]
        )
        self.decoder_norm = norm_layer(self.dec_embed_dim)

        # output projection only
        self.out_proj = nn.Linear(self.dec_embed_dim, self.patch_size, bias=True)

    def _apply_channel_rope(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.use_rope_chan) or (self.rope_chan is None):
            return x
        B, N, C, D = x.shape
        x_seq = x.view(B * N, C, D)
        freqs_c = self.rope_chan.prepare_freqs_1d(L=C, device=x.device, dtype=x.dtype)
        x_seq = apply_rotary_emb(freqs_c, x_seq)
        return x_seq.view(B, N, C, D)

    def forward(
        self,
        x_combined: torch.Tensor,   # [B, N, C, D_in]
        mask_y_bool: torch.Tensor,  # [N, C] bool
    ):
        x_combined = self.tok_norm_in(x_combined)
        B, N, C, D_in = x_combined.shape
        assert D_in == self.in_embed_dim, f"embed_dim mismatch: {D_in} != {self.in_embed_dim}"

        x_dec = self.proj_in(x_combined)   # [B, N, C, D_dec]
        x_dec = self._apply_channel_rope(x_dec)

        idx_nc = torch.nonzero(mask_y_bool, as_tuple=False)  # [M, 2]
        if idx_nc.numel() == 0:
            empty = torch.empty(B, 0, self.patch_size, device=x_dec.device, dtype=x_dec.dtype)
            flat = torch.empty(0, dtype=torch.long, device=x_dec.device)
            return empty, flat

        flat_idx = idx_nc[:, 0] * C + idx_nc[:, 1]           # [M]
        x_flat = x_dec.view(B, N * C, self.dec_embed_dim)     # [B, NC, D]

        q = x_flat[:, flat_idx, :]                            # [B, M, D]
        vis_bool_flat = (~mask_y_bool).view(N * C)            # [NC]
        kv = x_flat[:, vis_bool_flat, :] if vis_bool_flat.any() else x_flat

        for blk in self.decoder_blocks:
            q = blk(q, freqs=None, attn_mask=None, x_for_kv=kv)

        q = self.decoder_norm(q)
        y_masked = self.out_proj(q)                           # [B, M, P]
        return y_masked, flat_idx



# --------------------------- Predictor patch ------------------------- #


class TransformerPredictor(nn.Module):
    """
    Extended with Criss-Cross dual-axis attention and RoPE on both axes.

    kv_mode options:
      - "global_vis_kv": queries (masked+visible) can attend only to visible keys.
      - "all":            NO additive mask — every query attends to every key.
    """

    def __init__(
        self,
        num_patches: Tuple[int, int],
        *,
        embed_dim: int = 768,
        use_inp_embed: bool = True,
        width_mult: float = 1.0,
        predictor_embed_dim: Optional[int] = None,
        num_heads_pred: Optional[int] = None,
        mlp_ratio_pred: Optional[float] = None,
        use_part_pred: bool = True,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_std: float = 0.02,
        interpolate_factor: float = 2.0,  # legacy RoPE factor
        return_attention_layer: int = -1,
        kv_mode: str = "global_vis_kv",
        use_criss_cross: bool = False,
        rope_time: bool = False,
        rope_chan: bool = False,
        rope_dim_frac: float = 1.0,
        rope_interp_time: float = 1.0,
        rope_interp_chan: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_part_pred = bool(use_part_pred)
        self.use_inp_embed = bool(use_inp_embed)
        self.num_patches = (int(num_patches[0]), int(num_patches[1]))  # (C, N)
        C, N = self.num_patches

        # width & heads
        hidden_dim = (
            int(predictor_embed_dim)
            if predictor_embed_dim is not None
            else max(32, int(round(embed_dim * float(width_mult))))
        )
        heads_req = num_heads if num_heads_pred is None else int(num_heads_pred)
        H = max(1, min(heads_req, hidden_dim // 8))
        while H > 1 and (hidden_dim % H) != 0:
            H -= 1
        self.num_heads = H
        self.head_dim = hidden_dim // H
        self.mlp_ratio = float(mlp_ratio if mlp_ratio_pred is None else mlp_ratio_pred)

        # input projection
        if self.use_inp_embed:
            self.predictor_embed = nn.Linear(embed_dim, hidden_dim, bias=True)
        internal_dim = hidden_dim

        # learned queries for masked positions
        self.mask_query_time = nn.Embedding(N, internal_dim)
        self.mask_query_chan = nn.Embedding(C, internal_dim)
        self.mask_query_bias = nn.Parameter(torch.zeros(1, 1, 1, internal_dim))

        nn.init.trunc_normal_(self.mask_query_time.weight, std=init_std)
        nn.init.trunc_normal_(self.mask_query_chan.weight, std=init_std)

        # ids buffers
        self.register_buffer("t_ids_buf", torch.arange(N, dtype=torch.long), persistent=False)
        self.register_buffer("c_ids_buf", torch.arange(C, dtype=torch.long), persistent=False)

        # RoPE for legacy path (channels)
        self.time_embed = RotaryEmbedding(
            dim=max(2, (internal_dim // self.num_heads) // 2 * 2),
            interpolate_factor=interpolate_factor,
        )

        # trunk(s)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, steps=depth)]
        self.use_criss = bool(use_criss_cross)

        if self.use_criss:
            self.cc_stack = CrissCrossStack(
                depth=depth,
                dim=internal_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                rope_time=rope_time,
                rope_chan=rope_chan,
                rope_dim_frac=rope_dim_frac,
                rope_interp_time=rope_interp_time,
                rope_interp_chan=rope_interp_chan,
            )
        else:
            self.predictor_blocks = nn.ModuleList(
                [
                    Block(
                        dim=internal_dim,
                        num_heads=self.num_heads,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        is_causal=False,
                        use_rope=True,
                        return_attention=(i + 1) == return_attention_layer,
                    )
                    for i in range(depth)
                ]
            )

        self.predictor_norm = norm_layer(internal_dim)
        self.predictor_proj = nn.Linear(internal_dim, embed_dim, bias=True)

        # init
        self.apply(self._init_weights)
        self._fix_init_weight()

        kv_mode = kv_mode.lower().strip()
        if kv_mode not in ("global_vis_kv", "all"):
            raise ValueError(f"kv_mode must be 'global_vis_kv' or 'all'; got {kv_mode!r}")
        self.kv_mode = kv_mode

    # ---- KV masks (channel) ----
    def _build_vis_only_attn_mask(self, masked_bool: torch.Tensor, B: int) -> torch.Tensor:
        N, C = masked_bool.shape
        BN = B * N
        key_block = masked_bool.clone()

        all_masked_t = ~(~masked_bool).any(dim=1)
        if all_masked_t.any():
            key_block = key_block.clone()
            key_block[all_masked_t] = False

        key_block = key_block.unsqueeze(0).expand(B, -1, -1).reshape(BN, C)
        add = torch.zeros(BN, C, C, dtype=torch.float32, device=key_block.device)
        return add.masked_fill(key_block.unsqueeze(1), float("-inf"))

    # ---- init helpers ----
    def _fix_init_weight(self) -> None:
        def rescale(p: torch.Tensor, layer_id: int) -> None:
            p.div_(math.sqrt(2.0 * layer_id))

        if hasattr(self, "predictor_blocks"):
            for lid, blk in enumerate(self.predictor_blocks):
                rescale(blk.attn.proj.weight.data, lid + 1)
                rescale(blk.mlp.fc2.weight.data, lid + 1)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ---- forward ----
    def forward(
        self,
        x: torch.Tensor,  # [B, N, C, D_enc] or after proj: [B, N, C, hidden]
        mask_x: Optional[torch.Tensor] = None,  # kept flat indices over [N*C]
        mask_t: Optional[torch.Tensor] = None,  # unused
    ) -> (
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        x_in = x
        if self.use_inp_embed:
            x = self.predictor_embed(x)  # -> [B, N, C, hidden_dim]

        B, N, C, Dp = x.shape
        masked_bool = _masked_bool_from_masks(mask_x, C=C, N=N, device=x.device)  # [N, C]
        mask4 = masked_bool.unsqueeze(0).unsqueeze(-1).expand(B, N, C, Dp)

        # per-(t,c) predictor queries
        q_t = self.mask_query_time(self.t_ids_buf.to(x.device)).view(1, N, 1, Dp)
        q_c = self.mask_query_chan(self.c_ids_buf.to(x.device)).view(1, 1, C, Dp)
        pred_q = (q_t + q_c + self.mask_query_bias).expand(B, -1, -1, -1)

        # use learned predictor queries at masked positions
        x_q = torch.where(mask4, pred_q, x)

        # ---- Channel/Time KV policy ----
        if self.kv_mode == "all":
            add_mask_c = None
            add_mask_t = None
        else:
            add_mask_c = self._build_vis_only_attn_mask(masked_bool, B=B)
            add_mask_t = _build_time_vis_only_attn_mask(masked_bool, B=B)

        if self.use_criss:
            y = self.cc_stack(x_q, add_mask_c=add_mask_c, add_mask_t=add_mask_t)  # [B, N, C, Dp]
        else:
            # legacy: attention over channels only (per time)
            freqs = self.time_embed.prepare_freqs_1d(C, device=x.device, dtype=x.dtype)
            x_q_seq = x_q.view(B * N, C, Dp).contiguous()
            x_kv_seq = x.view(B * N, C, Dp).contiguous()
            for blk in self.predictor_blocks:
                out = blk(x_q_seq, freqs=freqs, attn_mask=add_mask_c, x_for_kv=x_kv_seq)
                if isinstance(out, torch.Tensor):
                    x_q_seq = out
                else:
                    # attention return path (if enabled)
                    return out
            y = x_q_seq.view(B, N, C, Dp).contiguous()

        y = self.predictor_norm(y)
        x_pred = self.predictor_proj(y)  # [B, N, C, embed_dim]

        masked_times_full = torch.where(masked_bool.all(dim=1))[0]

        if self.use_part_pred and mask_x is not None:
            keep4 = (~masked_bool).unsqueeze(0).unsqueeze(-1).expand(B, N, C, x_pred.shape[-1])
            x_combined = torch.where(keep4, x_in, x_pred)
            return x_pred, x_combined, masked_bool, masked_times_full

        return x_pred, masked_bool, masked_times_full


# ---------------------------- Encoder patch -------------------------- #


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int] = (19, 30),
        patch_size: int = 200,
        patch_stride: Optional[int] = None,
        embed_dim: int = 768,
        embed_num: int = 1,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        patch_module: type[nn.Module] = PatchEmbedPreTokenized,
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
    ):
        super().__init__()
        C0, _ = int(img_size[0]), int(img_size[1])
        if C0 != 19:
            raise ValueError(f"[TransformerEncoder] Forced TUH-19: img_size[0] must be 19, got {C0}")

        self.init_std = float(init_std)
        self.num_features = self.embed_dim = int(embed_dim)
        self.embed_num = int(embed_num)
        self.num_heads = int(num_heads)
        self.use_channel_embed = bool(use_channel_embed)

        self.patch_embed = patch_module(
            img_size=img_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim,
            use_dynamic_pe=use_dynamic_pe,
        )
        self.num_patches = self.patch_embed.num_patches  # (C, N)
        C, N = self.num_patches
        if C != 19:
            raise RuntimeError(f"[TransformerEncoder] patch_embed produced C={C}, expected 19")

        D = self.embed_dim
        self.chan_embed = nn.Embedding(64, D) if self.use_channel_embed else None  # 64 is plenty for 19ch
        self.register_buffer("mask_tok_zero", torch.zeros(1, 1, 1, D), persistent=False)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, steps=depth)]
        self.use_criss = bool(use_criss_cross)

        if self.use_criss:
            self.cc_stack = CrissCrossStack(
                depth=depth,
                dim=D,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                rope_time=rope_time,
                rope_chan=rope_chan,
                rope_dim_frac=rope_dim_frac,
                rope_interp_time=rope_interp_time,
                rope_interp_chan=rope_interp_chan,
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=D,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        is_causal=False,
                        use_rope=False,
                        return_attention=(i + 1) == return_attention_layer,
                    )
                    for i in range(depth)
                ]
            )
        self.norm = norm_layer(D)
        self.apply(self._init_weights)
        self._fix_init_weight()

    def prepare_chan_ids(self, channels: Optional[List[str]]) -> torch.Tensor:
        device = next(self.parameters()).device
        C, _ = self.num_patches

        if not channels:
            # default: TUH-19 order
            return torch.arange(C, device=device, dtype=torch.long)

        idxs: List[int] = []
        for ch in channels:
            key = str(ch).strip().upper().strip(".")
            if key not in CHANNEL_DICT:
                raise ValueError(f"Unknown channel '{ch}'. Allowed TUH-19: {TUH_19_CHANNELS}")
            idxs.append(CHANNEL_DICT[key])

        if len(idxs) != 19:
            raise ValueError(f"Expected 19 channel names; got {len(idxs)}. Allowed: {TUH_19_CHANNELS}")

        return torch.tensor(idxs, dtype=torch.long, device=device)

    def _fix_init_weight(self) -> None:
        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))
        if hasattr(self, "blocks"):
            for lid, blk in enumerate(self.blocks):
                rescale(blk.attn.proj.weight.data, lid + 1)
                rescale(blk.mlp.fc2.weight.data, lid + 1)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _build_vis_only_attn_mask(self, masked_bool: torch.Tensor, B: int) -> torch.Tensor:
        N, C = masked_bool.shape
        BN = B * N
        key_block = masked_bool.clone()
        all_masked_t = ~(~masked_bool).any(dim=1)
        if all_masked_t.any():
            key_block = key_block.clone()
            key_block[all_masked_t] = False
        key_block = key_block.unsqueeze(0).expand(B, -1, -1).reshape(BN, C)
        add = torch.zeros(BN, C, C, dtype=torch.float32, device=key_block.device)
        return add.masked_fill(key_block.unsqueeze(1), float("-inf"))

    def forward(
        self,
        x: torch.Tensor,
        chan_ids: Optional[torch.Tensor] = None,
        mask_x: Optional[torch.Tensor] = None,
        mask_t: Optional[torch.Tensor] = None,
        *,
        return_final_for_alignment: bool = False,
    ):
        x = self.patch_embed(x)  # [B, N, C, D]
        B, N, C, D = x.shape
        if C != 19:
            raise RuntimeError(f"[TransformerEncoder] Forced TUH-19 at runtime: got C={C}")

        if self.use_channel_embed and self.chan_embed is not None:
            if chan_ids is None:
                chan_ids = torch.arange(0, C, device=x.device, dtype=torch.long)
            else:
                chan_ids = chan_ids.to(device=x.device, dtype=torch.long).view(-1)
            if chan_ids.numel() != C:
                raise ValueError(f"chan_ids must have length C=19; got {chan_ids.numel()}")
            chan_emb = self.chan_embed(chan_ids)  # [C, D]
            x = x + chan_emb.unsqueeze(0).unsqueeze(0)

        masked_bool = _masked_bool_from_masks(mask_x, C=C, N=N, device=x.device)
        mask4 = masked_bool.unsqueeze(0).unsqueeze(-1).expand(B, N, C, D)
        mask_tok_pos = self.mask_tok_zero.to(x.device, x.dtype).expand(B, N, C, D)
        x_q = torch.where(mask4, mask_tok_pos, x)
        x_kv = torch.where(mask4, mask_tok_pos, x)

        add_mask_c = self._build_vis_only_attn_mask(masked_bool, B=B)
        add_mask_t = _build_time_vis_only_attn_mask(masked_bool, B=B)

        if self.use_criss:
            y = self.cc_stack(x_q, add_mask_c=add_mask_c, add_mask_t=add_mask_t)
        else:
            x_q_seq = x_q.view(B * N, C, D).contiguous()
            x_kv_seq = x_kv.view(B * N, C, D).contiguous()
            for blk in self.blocks:
                x_q_seq = blk(x_q_seq, freqs=None, attn_mask=add_mask_c, x_for_kv=x_kv_seq)
            y = x_q_seq.view(B, N, C, D).contiguous()

        y = self.norm(y)

        if not return_final_for_alignment:
            return y

        flat_mask = torch.nonzero(masked_bool, as_tuple=False)
        if flat_mask.numel() > 0:
            t_idx = flat_mask[:, 0]
            c_idx = flat_mask[:, 1]
            align_feat = y[:, t_idx, c_idx, :]
            flat = (t_idx * C + c_idx).to(torch.long)
        else:
            align_feat = torch.empty(B, 0, D, device=y.device)
            flat = torch.empty(0, dtype=torch.long, device=y.device)

        return y, align_feat, flat