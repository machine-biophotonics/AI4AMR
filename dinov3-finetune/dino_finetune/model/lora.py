import torch
import torch.nn as nn
from typing import Optional


class LoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        alpha: float = 16.0,
        r: int = 8,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)
        self.scaling = alpha / r
        
        # Dropout layers (applied after linear_a)
        self.dropout_q = nn.Dropout(dropout)
        self.dropout_v = nn.Dropout(dropout)

        self.in_features = qkv.in_features
        self.out_features = qkv.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the original qkv
        qkv = self.qkv(x)  # Shape: (B, N, 3 * org_C)

        # Compute the new q and v components with dropout and scaling
        new_q = self.linear_b_q(self.dropout_q(self.linear_a_q(x))) * self.scaling
        new_v = self.linear_b_v(self.dropout_v(self.linear_a_v(x))) * self.scaling

        # Add new q and v components to the original qkv tensor
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v

        return qkv
