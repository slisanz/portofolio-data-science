"""Physics-Informed Neural Network for Q-Factor regression.

Soft physics constraint:
    For an idealized OOK/coherent link, Q_linear ~ sqrt(2 * OSNR_linear).
    In dB: Q[dB] ~ 10*log10(2 * OSNR_linear) / 2.
We map normalized OSNR in [0,1] to a synthetic OSNR_linear range and treat the
resulting Q as a soft anchor. The data loss handles the rest of the residual physics.
"""
from __future__ import annotations

import torch
from torch import nn

from .mlp import MLPRegressor


class PINNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 128, 64), dropout: float = 0.1):
        super().__init__()
        self.backbone = MLPRegressor(input_dim, hidden_dims, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def physics_anchor_q(osnr_norm: torch.Tensor, osnr_db_min: float = 5.0, osnr_db_max: float = 25.0) -> torch.Tensor:
    """Map normalized OSNR feature to an idealized Q-Factor anchor (dB)."""
    osnr_db = osnr_db_min + (osnr_db_max - osnr_db_min) * osnr_norm
    osnr_linear = 10.0 ** (osnr_db / 10.0)
    q_linear = torch.sqrt(2.0 * osnr_linear)
    return 20.0 * torch.log10(q_linear + 1e-9) / 4.0  # rescale to roughly match target range


def pinn_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    osnr_norm: torch.Tensor,
    lambda_phys: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data_loss = nn.functional.mse_loss(y_pred, y_true)
    anchor = physics_anchor_q(osnr_norm)
    phys_loss = nn.functional.mse_loss(y_pred, anchor)
    total = data_loss + lambda_phys * phys_loss
    return total, data_loss, phys_loss
