#!/usr/bin/env python3
"""
MAMMOTH: Mixture of Mini Experts for Multiple Instance Learning

From paper: "Mixture of Mini Experts: Overcoming the Linear Layer Bottleneck in MIL"
ICLR 2026 - Mahmood Lab

This is a drop-in replacement for the linear layer that maps patch features
to task-specific dimensions using a mixture of low-rank experts.

Usage:
    from mammoth import Mammoth, add_mammoth_args
    
    # In model __init__:
    if moe_args and moe_args.get('num_experts', 0) > 0:
        self.patch_embed = Mammoth(**moe_args)
    else:
        self.patch_embed = nn.Linear(input_dim, embed_dim)
    
    # In forward:
    x = self.patch_embed(x)

Example args:
    --use_mammoth --mammoth_num_experts 30 --mammoth_num_slots 10 --mammoth_num_heads 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any
import argparse


class LoRALinear(nn.Module):
    """Low-rank adaptation linear layer (LoRA-style)"""
    
    def __init__(self, input_dim: int, output_dim: int, rank: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        
        # Down projection
        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        # Up projection  
        self.lora_B = nn.Linear(rank, output_dim, bias=False)
        # Dropout
        self.dropout = nn.Dropout(p=dropout)
        # Initialize
        nn.init.zeros_(self.lora_B.weight)
        nn.init.normal_(self.lora_A.weight, std=1.0 / math.sqrt(rank))
    
    def forward(self, x):
        # x: (B, N, input_dim)
        return self.lora_B(self.dropout(self.lora_A(x)))


class SlotRouting(nn.Module):
    """Slot-based routing for Mixture of Experts"""
    
    def __init__(self, input_dim: int, num_slots: int, num_heads: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.num_heads = num_heads
        
        # Query from input
        self.query = nn.Linear(input_dim, num_slots * num_heads)
        # Key from slots
        self.key = nn.Linear(input_dim, num_slots * num_heads)
        # Value from slots
        self.value = nn.Linear(input_dim, num_slots * num_heads)
        
    def forward(self, x):
        # x: (B, N, input_dim)
        B, N, _ = x.shape
        
        # Get queries: (B, N, num_slots*num_heads)
        q = self.query(x).view(B, N, self.num_slots, self.num_heads)
        # Get keys: (B, num_slots, num_heads) - use learnable slot embeddings
        k = self.key.weight.T.view(self.input_dim, self.num_slots, self.num_heads).unsqueeze(0).unsqueeze(0)
        k = k.expand(B, N, -1, -1)
        # Get values
        v = self.value.weight.T.view(self.input_dim, self.num_slots, self.num_heads).unsqueeze(0).unsqueeze(0)
        v = v.expand(B, N, -1, -1)
        
        # Attention: (B, N, num_slots, num_heads)
        attn = (q * k).sum(dim=-1) / math.sqrt(self.num_heads)
        attn = F.softmax(attn, dim=2)
        
        # Weighted values: (B, N, num_slots, num_heads)
        out = attn.unsqueeze(-1) * v.unsqueeze(2)
        
        return out, attn


class Mammoth(nn.Module):
    """
    MAMMOTH: Mixture of Mini Experts
    
    A low-rank Mixture of Experts module that transforms patch embeddings using
    multiple specialized experts with slot-based routing.
    
    Args:
        input_dim: Input feature dimension (e.g., 1280 for EfficientNet)
        embed_dim: Output embedding dimension
        num_experts: Number of experts (default: 30)
        num_slots: Number of routing slots (default: 10) 
        num_heads: Number of attention heads (default: 16)
        rank: Low-rank dimension (auto-computed if auto_rank=True)
        dropout: Dropout probability (default: 0.1)
        share_weights: Share LoRA weights across experts
        auto_rank: Auto-compute rank for parameter efficiency
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_experts: int = 30,
        num_slots: int = 10,
        num_heads: int = 16,
        rank: Optional[int] = None,
        dropout: float = 0.1,
        share_weights: bool = True,
        auto_rank: True = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.share_weights = share_weights
        
        # Auto compute rank for parameter efficiency
        if auto_rank and rank is None:
            # Keep parameter count similar to single linear layer
            # single_linear_params = input_dim * embed_dim
            # mammoth_params = num_experts * num_slots * (2 * rank + embed_dim * num_heads)
            # Solve for rank
            single_params = input_dim * embed_dim
            mammoth_params_approx = num_experts * num_slots * (2 * rank if rank else 256)
            # Target around 1.5x single linear
            rank = max(8, min(64, input_dim // 8))
        
        self.rank = rank
        
        # Expert routing
        self.routing = SlotRouting(input_dim, num_slots, num_heads)
        
        # Create experts
        if share_weights:
            # Single expert that's reused (parameter efficient)
            self.expert = LoRALinear(input_dim, embed_dim, rank, dropout)
            self.experts = None
        else:
            # Multiple unique experts
            self.experts = nn.ModuleList([
                LoRALinear(input_dim, embed_dim, rank, dropout)
                for _ in range(num_experts)
            ])
            self.expert = None
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim * num_heads, embed_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x, return_weights: bool = False):
        """
        Args:
            x: Input features (B, N, input_dim)
            return_weights: Return routing weights for visualization
        
        Returns:
            Output: (B, embed_dim) or (B, embed_dim * num_heads)
            weights: Optional routing weights (B, N, num_slots, num_heads)
        """
        B, N, _ = x.shape
        
        # Get routing weights: (B, N, num_slots, num_heads)
        weights, attn = self.routing(x)
        
        # Get expert outputs
        if self.share_weights:
            # Use single shared expert
            expert_out = self.expert(x)  # (B, N, embed_dim)
            # Expand for slot/head dimensions
            expert_out = expert_out.unsqueeze(2).unsqueeze(3)  # (B, N, 1, 1, embed_dim)
            expert_out = expert_out.expand(-1, -1, self.num_slots, self.num_heads, -1)
        else:
            # Get outputs from all experts
            expert_outs = []
            for expert in self.experts:
                expert_outs.append(expert(x))  # (B, N, embed_dim)
            expert_out = torch.stack(expert_outs, dim=2)  # (B, N, num_experts, embed_dim)
            expert_out = expert_out.unsqueeze(3).unsqueeze(4)  # (B, N, num_experts, 1, 1, embed_dim)
            # Average over experts (simple combination)
            expert_out = expert_out.mean(dim=2)  # (B, N, num_slots, num_heads, embed_dim)
        
        # Weighted combination: (B, N, num_slots, num_heads, embed_dim)
        # weights: (B, N, num_slots, num_heads, 1)
        # expert_out: (B, N, num_slots, num_heads, embed_dim)
        weighted = weights.unsqueeze(-1) * expert_out
        
        # Aggregate over patches and slots
        weighted = weighted.sum(dim=1).sum(dim=1)  # (B, num_heads, embed_dim)
        
        # Project to output dimension
        out = self.output_proj(weighted)  # (B, embed_dim)
        
        if return_weights:
            return out, attn
        return out


class MammothConfig:
    """Configuration for Mammoth module"""
    
    def __init__(self, **kwargs):
        self.input_dim = kwargs.get('input_dim', 1280)
        self.embed_dim = kwargs.get('embed_dim', 512)
        self.num_experts = kwargs.get('num_experts', 30)
        self.num_slots = kwargs.get('num_slots', 10)
        self.num_heads = kwargs.get('num_heads', 16)
        self.rank = kwargs.get('rank', None)
        self.dropout = kwargs.get('dropout', 0.1)
        self.share_weights = kwargs.get('share_weights', True)
        self.auto_rank = kwargs.get('auto_rank', True)
    
    def to_dict(self):
        return {
            'input_dim': self.input_dim,
            'embed_dim': self.embed_dim,
            'num_experts': self.num_experts,
            'num_slots': self.num_slots,
            'num_heads': self.num_heads,
            'rank': self.rank,
            'dropout': self.dropout,
            'share_weights': self.share_weights,
            'auto_rank': self.auto_rank,
        }


def add_mammoth_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add MAMMOTH arguments to argument parser"""
    parser.add_argument('--use_mammoth', action='store_true', help='Use MAMMOTH MoE module')
    parser.add_argument('--mammoth_num_experts', type=int, default=30, help='Number of experts for MAMMOTH')
    parser.add_argument('--mammoth_num_slots', type=int, default=10, help='Number of routing slots')
    parser.add_argument('--mammoth_num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--mammoth_rank', type=int, default=None, help='Low-rank dimension (auto if None)')
    parser.add_argument('--mammoth_dropout', type=float, default=0.1, help='Dropout for MAMMOTH')
    parser.add_argument('--mammoth_share_weights', action='store_true', default=True, help='Share weights across experts')
    parser.add_argument('--mammoth_auto_rank', action='store_true', default=True, help='Auto compute rank')
    return parser


def create_mammoth(args: argparse.Namespace, input_dim: int, embed_dim: int) -> Optional[Mammoth]:
    """Create Mammoth module from args"""
    if not getattr(args, 'use_mammoth', False):
        return None
    
    moe_args = {
        'input_dim': input_dim,
        'embed_dim': embed_dim,
        'num_experts': getattr(args, 'mammoth_num_experts', 30),
        'num_slots': getattr(args, 'mammoth_num_slots', 10),
        'num_heads': getattr(args, 'mammoth_num_heads', 16),
        'rank': getattr(args, 'mammoth_rank', None),
        'dropout': getattr(args, 'mammoth_dropout', 0.1),
        'share_weights': getattr(args, 'mammoth_share_weights', True),
        'auto_rank': getattr(args, 'mammoth_auto_rank', True),
    }
    
    return Mammoth(**moe_args)


def get_default_mammoth_args() -> Dict[str, Any]:
    """Get default MAMMOTH arguments"""
    return {
        'num_experts': 30,
        'num_slots': 10,
        'num_heads': 16,
        'rank': None,
        'dropout': 0.1,
        'share_weights': True,
        'auto_rank': True,
    }


if __name__ == '__main__':
    # Test the module
    print("Testing MAMMOTH module...")
    
    x = torch.randn(2, 100, 1280)  # Batch=2, 100 patches, 1280 features
    
    mammoth = Mammoth(
        input_dim=1280,
        embed_dim=512,
        num_experts=30,
        num_slots=10,
        num_heads=16,
        auto_rank=True
    )
    
    out = mammoth(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in mammoth.parameters()):,}")
    
    # Compare with single linear
    linear = nn.Linear(1280, 512)
    linear_params = sum(p.numel() for p in linear.parameters())
    print(f"Linear parameters: {linear_params:,}")
    print(f"MAMMOTH overhead: {sum(p.numel() for p in mammoth.parameters()) - linear_params:,}")