# modules/utils.py
from __future__ import annotations
import torch
import torch.nn as nn
from torch.autograd import Function


# ----------------------- Positional embedding helper -----------------------
def create_1d_absolute_sin_cos_embedding(pos_len: int, dim: int) -> torch.Tensor:
    assert dim % 2 == 0, "Position dim must be even"
    position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
    i_matrix = torch.arange(dim // 2, dtype=torch.float)
    i_matrix /= (dim / 2)
    i_matrix = torch.pow(10000, i_matrix)
    i_matrix = 1 / i_matrix
    i_matrix = i_matrix.to(torch.long)
    pos_vec = torch.arange(pos_len).to(torch.long)
    out = pos_vec[:, None] @ i_matrix[None, :]
    emb_cos = torch.cos(out)
    emb_sin = torch.sin(out)
    position_emb[:, 0::2] = emb_sin
    position_emb[:, 1::2] = emb_cos
    return position_emb


# --------------------------- Gradient reversal -----------------------------
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


# ----------------------- Norm-constrained layers ---------------------------
class _BaseConstraint:
    doWeightNorm: bool
    max_norm: float

    def _maybe_renorm(self, weight: torch.Tensor) -> torch.Tensor:
        if self.doWeightNorm:
            with torch.no_grad():
                weight.data = torch.renorm(weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return weight


class Conv2dWithConstraint(nn.Conv2d, _BaseConstraint):
    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_renorm(self.weight)
        return super().forward(x)


class Conv1dWithConstraint(nn.Conv1d, _BaseConstraint):
    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_renorm(self.weight)
        return super().forward(x)


class LinearWithConstraint(nn.Linear, _BaseConstraint):
    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 1.0, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_renorm(self.weight)
        return super().forward(x)


# ------------------------------- SMMD Losses -------------------------------
def SMMDL_marginal(Cs: torch.Tensor, Ct: torch.Tensor) -> torch.Tensor:
    Cs = torch.mean(Cs, dim=0)
    Ct = torch.mean(Ct, dim=0)
    return torch.mean((Cs - Ct) ** 2)


def SMMDL_conditional(Cs: torch.Tensor, s_label: torch.Tensor,
                      Ct: torch.Tensor, t_label: torch.Tensor) -> torch.Tensor:
    s_label = s_label.reshape(-1)
    t_label = t_label.reshape(-1)
    classes = torch.unique(s_label)
    class_num = 0
    all_loss = torch.tensor(0.0, device=Cs.device, dtype=Cs.dtype)
    for c in classes:
        s_index = (s_label == c)
        t_index = (t_label == c)
        if torch.sum(t_index) == 0:
            continue
        class_num += 1
        m_Cs = torch.mean(Cs[s_index], dim=0)
        m_Ct = torch.mean(Ct[t_index], dim=0)
        all_loss = all_loss + torch.mean((m_Cs - m_Ct) ** 2)
    if class_num == 0:
        return all_loss
    return all_loss / class_num
