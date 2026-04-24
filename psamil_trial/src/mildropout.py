#!/usr/bin/env python3
"""
Mildropout: Random instance dropout for MIL (based on ASMIL-like approach)
Randomly drops top-k instances during training to prevent overfitting.

Reference: ASMIL - Attention-Stabilized Multiple Instance Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional


class Mildropout(nn.Module):
    """
    Mildropout: Randomly drops top-k instances during training.
    
    This helps prevent attention over-concentration on just a few instances.
    
    Args:
        topk: Number of top instances to consider for dropping (default: 3)
        kernel: Kernel size for attention smoothing (default: 7)
        dropout_rate: Probability of dropping an instance (default: 0.5)
    """
    
    def __init__(
        self,
        topk: int = 3,
        kernel: int = 7,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.topk = topk
        self.kernel = kernel
        self.dropout_rate = dropout_rate
        
        # Learnable attention for selecting which instances to drop
        self.dropout_attention = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x: torch.Tensor, return_mask: bool = False):
        """
        Args:
            x: Input tensor of shape (batch, num_instances, features)
            return_mask: Whether to return the dropout mask
            
        Returns:
            Output tensor with dropped instances, or (output, mask) if return_mask=True
        """
        batch, num_inst, feat_dim = x.shape
        
        # Compute attention scores
        attn = self.dropout_attention(x).squeeze(-1)  # (batch, num_inst)
        
        # Get top-k indices
        topk = min(self.topk, num_inst)
        _, top_indices = torch.topk(attn, topk, dim=1)  # (batch, topk)
        
        # Create mask
        mask = torch.ones_like(x)
        
        # Randomly drop some of the top-k instances during training
        if self.training:
            drop_mask = torch.rand(batch, topk, device=x.device) < self.dropout_rate
            # Apply to each batch
            for b in range(batch):
                for k in range(topk):
                    if drop_mask[b, k]:
                        inst_idx = top_indices[b, k]
                        mask[b, inst_idx] = 0.0
        
        # Apply mask
        x = x * mask
        
        if return_mask:
            return x, mask
        return x


class AdaptiveMildropout(nn.Module):
    """
    Adaptive Mildropout with learnable dropout intensity.
    """
    
    def __init__(
        self,
        num_instances: int = 25,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout_rate = nn.Parameter(torch.tensor(dropout_rate))
        self.num_instances = num_instances
    
    def forward(self, x: torch.Tensor):
        batch, num_inst, feat_dim = x.shape
        rate = torch.sigmoid(self.dropout_rate)
        
        # Random dropout mask
        mask = torch.rand(batch, num_inst, device=x.device) > rate
        mask = mask.float().unsqueeze(-1).expand_as(x)
        
        # Apply dropout during training only
        if self.training:
            x = x * mask
        
        return x


def create_mildropout(config: dict = None):
    """Factory function to create mildropout module."""
    if config is None:
        config = {}
    
    dropout_type = config.get('type', 'mildropout')
    
    if dropout_type == 'mildropout':
        return Mildropout(
            topk=config.get('topk', 3),
            kernel=config.get('kernel', 7),
            dropout_rate=config.get('dropout_rate', 0.5),
        )
    elif dropout_type == 'adaptive':
        return AdaptiveMildropout(
            dropout_rate=config.get('dropout_rate', 0.5),
        )
    else:
        raise ValueError(f"Unknown dropout type: {dropout_type}")


if __name__ == "__main__":
    # Test mildropout
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test Mildropout
    d = Mildropout(topk=3, kernel=7).to(device)
    d.train()
    x = torch.randn(4, 25, 1280).to(device)
    y = d(x)
    print(f"Mildropout train: {x.shape} -> {y.shape}")
    
    d.eval()
    y2 = d(x)
    print(f"Mildropout eval: {x.shape} -> {y2.shape}")
    
    # Test AdaptiveMildropout
    d2 = AdaptiveMildropout(dropout_rate=0.5).to(device)
    d2.train()
    y3 = d2(x)
    print(f"AdaptiveMildropout train: {x.shape} -> {y3.shape}")
    
    d2.eval()
    y4 = d2(x)
    print(f"AdaptiveMildropout eval: {x.shape} -> {y4.shape}")
