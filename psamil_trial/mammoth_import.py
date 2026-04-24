"""MAMMOTH wrapper - uses official mammoth-moe package."""
import argparse

try:
    from mammoth import Mammoth as _Mammoth
except ImportError:
    _Mammoth = None


def create_mammoth(
    input_dim: int,
    embed_dim: int,
    num_experts: int = 30,
    num_slots: int = 10,
    num_heads: int = 16,
    dropout: float = 0.1,
    slot_dropout: float = 0.0,
    slot_dim: int = 256,
    auto_rank: bool = True,
    keep_slots: bool = True,
    share_lora_weights: bool = True,
):
    if _Mammoth is None:
        raise ImportError("pip install mammoth-moe")
    return _Mammoth(
        input_dim=input_dim,
        dim=embed_dim,
        num_experts=num_experts,
        num_slots=num_slots,
        num_heads=num_heads,
        dropout=dropout,
        slot_dropout=slot_dropout,
        slot_dim=slot_dim,
        auto_rank=auto_rank,
        keep_slots=keep_slots,
        share_lora_weights=share_lora_weights,
    )


def add_mammoth_args(parser):
    parser.add_argument("--use_mammoth", action="store_true")
    parser.add_argument("--mammoth_num_experts", type=int, default=30)
    parser.add_argument("--mammoth_num_slots", type=int, default=10)
    parser.add_argument("--mammoth_num_heads", type=int, default=16)
    parser.add_argument("--mammoth_dropout", type=float, default=0.1)
    parser.add_argument("--mammoth_slot_dropout", type=float, default=0.0)
    parser.add_argument("--mammoth_auto_rank", action="store_true", default=True)
    parser.add_argument("--mammoth_keep_slots", action="store_true", default=True)
    return parser
