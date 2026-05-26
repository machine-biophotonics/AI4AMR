"""CellFlux UNetModel — exact reproduction of CellFlux (ICML 2025) architecture.

Modified from https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py
as used in https://github.com/yuhui-zh15/CellFlux

Key features:
    - OpenAI guided-diffusion UNet with FiLM (use_scale_shift_norm) conditioning
    - Timestep + perturbation condition injected via `mol_embed_transform`
    - `extra["concat_conditioning"]` carries the perturbation embedding (CellFlux API)
    - Flash attention for memory-efficient O(N) self-attention
    - Gradient checkpointing to reduce activation memory (~40% overhead for 2x memory savings)
"""
from abc import abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from nn import (
    avg_pool_nd, conv_nd, linear, normalization,
    timestep_embedding, zero_module,
)


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

    def forward(self, x, emb):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward, x, emb, use_reentrant=False)
        return self._forward(x, emb)


class AttentionBlock(nn.Module):
    """Multi-head attention with PyTorch flash attention (O(N) memory).

    Matches CellFlux's AttentionBlock structure (QKV projection, residual, norm).
    Replaces O(N^2) einsum with F.scaled_dot_product_attention for flash-attention speed.
    """
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.head_dim = channels // self.num_heads
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def _forward(self, x):
        b, c, *spatial = x.shape
        n = spatial[0] * spatial[1]
        x_flat = x.reshape(b, c, n)
        qkv = self.qkv(self.norm(x_flat))
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, self.head_dim, n).transpose(2, 3)
        k = k.reshape(b, self.num_heads, self.head_dim, n).transpose(2, 3)
        v = v.reshape(b, self.num_heads, self.head_dim, n).transpose(2, 3)
        # Cast to fp16 for flash attention; fallback fp32 math OOMs at high resolution
        q, k, v = q.half(), k.half(), v.half()
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        h = h.to(x.dtype)
        h = h.transpose(2, 3).reshape(b, c, n)
        h = self.proj_out(h)
        return (x + h.reshape(b, c, *spatial))

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class UNetModel(nn.Module):
    """CellFlux UNetModel — exact architecture from CellFlux (ICML 2025).

    Based on OpenAI guided-diffusion UNet.
    Conditioning via `extra["concat_conditioning"]` (CellFlux API).

    CellFlux BBBC021 config:
        in_channels=3, out_channels=3, model_channels=128,
        num_res_blocks=4, attention_resolutions=[2],
        channel_mult=[2,2,2], dropout=0.3, condition_dim=1024

    Our config (1-channel CRISPRi):
        in_channels=1, out_channels=1, condition_dim=512
    """

    def __init__(
        self,
        in_channels: int = 1,
        model_channels: int = 128,
        out_channels: int = 1,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int] = (2,),
        dropout: float = 0.3,
        channel_mult: Tuple[int] = (2, 2, 2),
        conv_resample: bool = False,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = True,
        num_heads: int = 1,
        num_head_channels: int = 64,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = False,
        condition_dim: int = 512,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        self.time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, self.time_embed_dim),
            nn.SiLU(),
            linear(self.time_embed_dim, self.time_embed_dim),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes + 1, self.time_embed_dim, padding_idx=num_classes)

        self.mol_embed_transform = nn.Linear(condition_dim, self.time_embed_dim)

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(ch, self.time_embed_dim, dropout,
                             out_channels=int(mult * model_channels),
                             dims=dims, use_scale_shift_norm=use_scale_shift_norm,
                             use_checkpoint=use_checkpoint)
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads,
                                       num_head_channels=num_head_channels,
                                       use_checkpoint=use_checkpoint)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch, self.time_embed_dim, dropout, out_channels=out_ch,
                                 dims=dims, use_scale_shift_norm=use_scale_shift_norm,
                                 use_checkpoint=use_checkpoint, down=True)
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, self.time_embed_dim, dropout, dims=dims,
                     use_scale_shift_norm=use_scale_shift_norm,
                     use_checkpoint=use_checkpoint),
            AttentionBlock(ch, num_heads=num_heads,
                           num_head_channels=num_head_channels,
                           use_checkpoint=use_checkpoint),
            ResBlock(ch, self.time_embed_dim, dropout, dims=dims,
                     use_scale_shift_norm=use_scale_shift_norm,
                     use_checkpoint=use_checkpoint),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, self.time_embed_dim, dropout,
                             out_channels=int(model_channels * mult),
                             dims=dims, use_scale_shift_norm=use_scale_shift_norm,
                             use_checkpoint=use_checkpoint)
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads,
                                       num_head_channels=num_head_channels,
                                       use_checkpoint=use_checkpoint)
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(ch, self.time_embed_dim, dropout, out_channels=out_ch,
                                 dims=dims, use_scale_shift_norm=use_scale_shift_norm,
                                 use_checkpoint=use_checkpoint, up=True)
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, int(channel_mult[0] * model_channels), out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, extra=None):
        """CellFlux forward pass.

        Args:
            x: (B, C, H, W) input at timestep t
            timesteps: (B,) timesteps in [0, 1]
            extra: dict with optional key "concat_conditioning": (B, D) embedding

        Returns:
            (B, C, H, W) predicted velocity
        """
        if extra is None:
            extra = {}

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(x))

        if self.num_classes is not None and "label" in extra:
            emb = emb + self.label_emb(extra["label"])

        if "concat_conditioning" in extra:
            mol_embedding = self.mol_embed_transform(extra["concat_conditioning"])
            emb = emb + mol_embedding

        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
