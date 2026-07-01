# modules/heads.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from .utils import LinearWithConstraint

# ---------- shape helper ----------
def _as_BNED(z: torch.Tensor) -> torch.Tensor:
    """
    Normalize inputs to [B, N, E, D]:
      - [B, D]       -> [B, 1, 1, D]
      - [B, N, D]    -> [B, N, 1, D]
      - [B, N, E, D] -> as-is
    """
    if z.dim() == 2:  # [B, D]
        return z[:, None, None, :]
    if z.dim() == 3:  # [B, N, D]
        return z[:, :, None, :]
    if z.dim() == 4:
        return z
    raise ValueError(f"Expected 2/3/4D tensor, got {tuple(z.shape)}")

# ---------- 1) Average pooling -> MLP ----------
class AvgPoolMLPHead(nn.Module):
    """ Global mean over (N,E), then MLP. """
    def __init__(self, *, embed_dim: int, num_classes: int,
                 hidden: int = 128, dropout: float = 0.5,
                 max_norm_fc1: float = 1.0, max_norm_fc2: float = 0.25):
        super().__init__()
        self.D = int(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)
        self.fc1 = LinearWithConstraint(self.D, hidden, max_norm=max_norm_fc1)
        self.fc2 = LinearWithConstraint(hidden, num_classes, max_norm=max_norm_fc2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = _as_BNED(z)                # [B,N,E,D]
        x = z.mean(dim=(1, 2))         # [B,D]
        h = self.act(self.fc1(self.drop(x)))
        return self.fc2(h)

# ---------- 2) Per-time attention pooling -> MLP ----------
class PerTimeAttnMLPHead(nn.Module):
    """
    For each time step: attend over tokens E using a shared query -> D,
    then pool over N (mean/max/meanmax) and classify with an MLP.
    """
    def __init__(self, *, embed_dim: int, num_classes: int,
                 hidden: int = 128, dropout: float = 0.5,
                 time_pool: str = "mean",            # 'mean' | 'max' | 'meanmax'
                 project_kv: bool = True,
                 temperature: Optional[float] = None,
                 max_norm_fc1: float = 1.0, max_norm_fc2: float = 0.25):
        super().__init__()
        self.D = int(embed_dim)
        self.time_pool = time_pool.lower()
        self.tau = float(temperature) if temperature is not None else (self.D ** 0.5)

        self.query = nn.Parameter(torch.randn(self.D) * 0.02)
        self.key = nn.Linear(self.D, self.D, bias=False) if project_kv else nn.Identity()
        self.val = nn.Linear(self.D, self.D, bias=False) if project_kv else nn.Identity()

        t_factor = 2 if self.time_pool == "meanmax" else 1
        in_dim = t_factor * self.D

        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)
        self.fc1 = LinearWithConstraint(in_dim, hidden, max_norm=max_norm_fc1)
        self.fc2 = LinearWithConstraint(hidden, num_classes, max_norm=max_norm_fc2)

    def _pool_time(self, z_bnd: torch.Tensor) -> torch.Tensor:
        if self.time_pool == "mean":
            return z_bnd.mean(dim=1)
        if self.time_pool == "max":
            return z_bnd.amax(dim=1)
        if self.time_pool == "meanmax":
            m = z_bnd.mean(dim=1)
            M = z_bnd.amax(dim=1)
            return torch.cat([m, M], dim=-1)
        raise ValueError(f"Unknown time_pool '{self.time_pool}'")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = _as_BNED(z)                             # [B,N,E,D]
        B, N, E, D = z.shape
        assert D == self.D

        x = z.reshape(B * N, E, D)                  # [B*N,E,D]
        q = self.query.view(1, 1, D).expand(B * N, 1, D)
        K = self.key(x)                             # [B*N,E,D]
        V = self.val(x)                             # [B*N,E,D]

        scores = torch.matmul(q, K.transpose(1, 2)) / self.tau  # [B*N,1,E]
        attn = torch.softmax(scores, dim=-1)                    # [B*N,1,E]
        pooled = torch.matmul(attn, V).squeeze(1)               # [B*N,D]

        per_time = pooled.view(B, N, D)            # [B,N,D]
        feat = self._pool_time(per_time)           # [B,D] or [B,2D]

        h = self.act(self.fc1(self.drop(feat)))
        return self.fc2(h)

# ---------- 3) Per-token MLP over all tokens, then mean ----------
class FlattenAllTokensMLPHead(nn.Module):
    """
    Reshape [B, N, E, D] -> [B, N*E, D], apply a token-wise MLP:
        D -> 4*D (ELU) -> D (ELU) -> num_classes
    By default, returns mean over tokens: [B, num_classes].
    Set return_token_logits=True to get [B, N*E, num_classes].
    """
    def __init__(self, *,
                 n_time: int, embed_slots: int, embed_dim: int,  # N, E, D
                 num_classes: int,
                 dropout: float = 0.5,
                 max_norm_fc1: float = 1.0,
                 max_norm_fc2: float = 0.25,
                 return_token_logits: bool = False):
        super().__init__()
        self.N = int(n_time)
        self.E = int(embed_slots)
        self.D = int(embed_dim)
        self.return_token_logits = return_token_logits

        h1 = 4 * self.D
        h2 = self.D

        self.mlp = nn.Sequential(
            LinearWithConstraint(self.D, h1, max_norm=max_norm_fc1),
            nn.ELU(),
            nn.Dropout(dropout),
            LinearWithConstraint(h1, h2, max_norm=max_norm_fc1),
            nn.ELU(),
            nn.Dropout(dropout),
            LinearWithConstraint(h2, num_classes, max_norm=max_norm_fc2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = _as_BNED(z)  # [B, N, E, D]
        B, N, E, D = z.shape
        if (N, E, D) != (self.N, self.E, self.D):
            raise ValueError(
                f"FlattenAllTokensMLPHead expects (B,N,E,D)=({B},{self.N},{self.E},{self.D}), "
                f"got ({B},{N},{E},{D})."
            )
        z = z.reshape(B, N * E, D)                # [B, N*E, D]
        z2 = z.reshape(B * (N * E), D)
        logits = self.mlp(z2).reshape(B, N * E, -1)
        if self.return_token_logits:
            return logits
        return logits.mean(dim=1)                 # [B, K]

# ---------- 4) Flatten-concat all patches -> MLP ----------
class AllPatchRepsMLPHead(nn.Module):
    """
    Match the 'all_patch_reps' head of CBraMod:

      [B, N, E, D] -> flatten to [B, N*E*D]
        -> Linear(N*E*D -> 4*D) -> ELU -> Dropout
        -> Linear(4*D -> D)     -> ELU -> Dropout
        -> Linear(D -> K)

    This generalizes your fixed (22 * 4 * 200) to (N * E * D) with hidden sizes tied to D.
    """
    def __init__(self, *,
                 n_time: int, embed_slots: int, embed_dim: int,
                 num_classes: int, dropout: float = 0.5,
                 max_norm_fc1: float = 1.0, max_norm_fc2: float = 0.25):
        super().__init__()
        self.N = int(n_time)
        self.E = int(embed_slots)
        self.D = int(embed_dim)

        in_dim = self.N * self.E * self.D
        h1 = 4 * self.D
        h2 = self.D

        self.net = nn.Sequential(
            LinearWithConstraint(in_dim, h1, max_norm=max_norm_fc1),
            nn.ELU(),
            nn.Dropout(dropout),
            LinearWithConstraint(h1, h2, max_norm=max_norm_fc1),
            nn.ELU(),
            nn.Dropout(dropout),
            LinearWithConstraint(h2, int(num_classes), max_norm=max_norm_fc2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = _as_BNED(z)                     # [B,N,E,D]
        B, N, E, D = z.shape
        if (N, E, D) != (self.N, self.E, self.D):
            raise ValueError(
                f"AllPatchRepsMLPHead expects (N,E,D)=({self.N},{self.E},{self.D}), "
                f"got ({N},{E},{D})."
            )
        x = z.reshape(B, N * E * D)         # [B, N*E*D]
        return self.net(x)                   # [B, K]

# ===========================================================
# ========== EEGPT-style probes (deterministic shapes) ======
# ===========================================================
class EEGPTProbeHead(nn.Module):
    """
    Match LitEEGPTCausal probe:
      z [B,N,E,D]
        -> flatten last two dims per time: [B,N,E*D]
        -> Dropout
        -> Linear(E*D -> P) per time (no nonlinearity)
        -> Flatten across time: [B, N*P]
        -> Linear(N*P -> num_classes)
    """
    def __init__(self, *,
                 n_time: int, embed_slots: int, embed_dim: int,
                 num_classes: int, proj_dim: int = 16,
                 dropout: float = 0.5,
                 max_norm_fc1: float = 1.0, max_norm_fc2: float = 0.25):
        super().__init__()
        self.N = int(n_time)
        self.E = int(embed_slots)
        self.D = int(embed_dim)
        self.P = int(proj_dim)

        self.drop = nn.Dropout(float(dropout))
        self.proj = LinearWithConstraint(self.E * self.D, self.P, max_norm=max_norm_fc1)
        self.cls  = LinearWithConstraint(self.N * self.P, num_classes, max_norm=max_norm_fc2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = _as_BNED(z)                 # [B,N,E,D]
        B, N, E, D = z.shape
        if (N, E, D) != (self.N, self.E, self.D):
            raise ValueError(
                f"EEGPTProbeHead expects (N,E,D)=({self.N},{self.E},{self.D}), got ({N},{E},{D})."
            )
        h = z.reshape(B, N, E * D)      # [B,N,E*D]
        h = self.proj(self.drop(h))     # [B,N,P]
        h = h.reshape(B, N * self.P)    # [B,N*P]
        return self.cls(h)              # [B,K]

# ---------- factory ----------
def make_head(
    head_type: str,
    *,
    num_classes: int,
    embed_dim: Optional[int] = None,   # required for 'avg' and 'per_time_attn'
    hidden: int = 128,
    dropout: float = 0.5,
    time_pool: str = "mean",
    project_kv: bool = True,
    temperature: Optional[float] = None,
    n_time: Optional[int] = None,      # required for 'flatten', 'all_patch_reps', 'eegpt_probe'
    embed_slots: Optional[int] = None, # required for 'flatten', 'all_patch_reps', 'eegpt_probe'
    proj_dim: int = 16,                # for eegpt_probe
) -> nn.Module:
    """
    Available head types (aliases):
      - 'avg' | 'average' | 'gap'                    -> AvgPoolMLPHead
      - 'per_time_attn' | 'pt_attn' | 'pertime_attention'
                                                    -> PerTimeAttnMLPHead
      - 'flatten' | 'flatten_mlp' | 'all_tokens'    -> FlattenAllTokensMLPHead
      - 'all_patch_reps' | 'concat_all'             -> AllPatchRepsMLPHead
      - 'eegpt_probe'                                -> EEGPTProbeHead
    """
    ht = head_type.lower()

    if ht in {"avg", "average", "gap"}:
        assert embed_dim is not None, "AvgPoolMLPHead requires embed_dim"
        return AvgPoolMLPHead(embed_dim=int(embed_dim),
                              num_classes=int(num_classes),
                              hidden=int(hidden),
                              dropout=float(dropout))

    if ht in {"per_time_attn", "pt_attn", "pertime_attention"}:
        assert embed_dim is not None, "PerTimeAttnMLPHead requires embed_dim"
        return PerTimeAttnMLPHead(embed_dim=int(embed_dim),
                                  num_classes=int(num_classes),
                                  hidden=int(hidden),
                                  dropout=float(dropout),
                                  time_pool=time_pool,
                                  project_kv=bool(project_kv),
                                  temperature=temperature)

    if ht in {"flatten", "flatten_mlp", "all_tokens"}:
        assert (n_time is not None) and (embed_slots is not None) and (embed_dim is not None), \
            "FlattenAllTokensMLPHead requires n_time, embed_slots, embed_dim"
        return FlattenAllTokensMLPHead(n_time=int(n_time),
                                       embed_slots=int(embed_slots),
                                       embed_dim=int(embed_dim),
                                       num_classes=int(num_classes),
                                       dropout=float(dropout))

    if ht in {"all_patch_reps", "concat_all"}:
        assert (n_time is not None) and (embed_slots is not None) and (embed_dim is not None), \
            "AllPatchRepsMLPHead requires n_time, embed_slots, embed_dim"
        return AllPatchRepsMLPHead(n_time=int(n_time),
                                   embed_slots=int(embed_slots),
                                   embed_dim=int(embed_dim),
                                   num_classes=int(num_classes),
                                   dropout=float(dropout))

    if ht == "eegpt_probe":
        assert (n_time is not None) and (embed_slots is not None) and (embed_dim is not None), \
            "EEGPTProbeHead requires n_time, embed_slots, embed_dim"
        return EEGPTProbeHead(n_time=int(n_time),
                              embed_slots=int(embed_slots),
                              embed_dim=int(embed_dim),
                              num_classes=int(num_classes),
                              proj_dim=int(proj_dim),
                              dropout=float(dropout))

    raise ValueError("Unknown head_type. Choose one of: "
                     "'avg', 'per_time_attn', 'flatten', 'all_patch_reps', 'eegpt_probe'.")
