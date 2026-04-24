#!/usr/bin/env python3
"""
MAMMOTH wrapper for mammoth-moe package (ICLR 2026).

Install: pip install mammoth-moe

This module provides create_mammoth() and add_mammoth_args() functions
for integrating MAMMOTH MoE into the MIL pipeline.

Usage:
    from mammoth_import import create_mammoth, add_mammoth_args
    parser = add_mammoth_args(parser)
    mammoth = create_mammoth(args, input_dim=1280, embed_dim=512)
    model = AttentionMILModel(num_classes=2, mammoth=mammoth)
"""

import argparse
from typing import Optional

import torch

try:
    from mammoth import Mammoth as _Mammoth
except ImportError:
    raise ImportError(
        "mammoth-moe package not found. Install with: pip install mammoth-moe"
    )


def add_mammoth_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add MAMMOTH arguments to parser."""
    parser.add_argument('--use_mammoth', action='store_true', default=False,
                    help='Use MAMMOTH mixture of experts')
    parser.add_argument('--mammoth_num_experts', type=int, default=30,
                    help='Number of experts')
    parser.add_argument('--mammoth_num_slots', type=int, default=10,
                    help='Number of slots')
    parser.add_argument('--mammoth_num_heads', type=int, default=16,
                    help='Number of heads')
    parser.add_argument('--mammoth_rank', type=int, default=None,
                    help='LoRA rank (auto if None)')
    parser.add_argument('--mammoth_dropout', type=float, default=0.1,
                    help='Dropout probability')
    parser.add_argument('--mammoth_share_weights', action='store_true', default=True,
                    help='Share LoRA weights across experts')
    parser.add_argument('--mammoth_auto_rank', action='store_true', default=True,
                    help='Auto compute rank')
    return parser


def create_mammoth(args: argparse.Namespace, input_dim: int, embed_dim: int) -> Optional[_Mammoth]:
    """Create MAMMOTH module from args."""
    if not getattr(args, 'use_mammoth', False):
        return None
    
    return _Mammoth(
        input_dim=input_dim,
        dim=embed_dim,
        num_experts=getattr(args, 'mammoth_num_experts', 30),
        num_slots=getattr(args, 'mammoth_num_slots', 10),
        num_heads=getattr(args, 'mammoth_num_heads', 16),
        lora_rank=getattr(args, 'mammoth_rank', None),
        dropout=getattr(args, 'mammoth_dropout', 0.1),
        share_lora_weights=getattr(args, 'mammoth_share_weights', True),
        auto_rank=getattr(args, 'mammoth_auto_rank', True),
        keep_slots=False,  # Output shape: (batch, seq, dim) instead of (batch, seq, num_experts * num_slots * head_dim)
    )