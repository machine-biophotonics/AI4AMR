"""DiT: Diffusion Transformer backbone for pixel-space flow matching.

Adapted from Peebles & Xie (2023) and Ma et al. (2024).
Designed for 1-channel 224×224 images (no VAE latent).

Model variants:
    DiT-S/16: hidden=384, depth=12, heads=6   (~33M params)
    DiT-B/16: hidden=768, depth=12, heads=12  (~131M params)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: scale-and-shift after LayerNorm."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbed(nn.Module):
    """Sinusoidal timestep embedding + MLP."""
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer('freqs', freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().unsqueeze(-1)  # (B, 1)
        freqs = self.freqs.to(t.device)  # (half,)
        emb = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)], dim=-1)  # (B, dim)
        return self.mlp(emb)


class PatchEmbed(nn.Module):
    """Image to patch tokens: (B, C, H, W) → (B, N, D)."""
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with fused QKV."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """Transformer block with AdaLN modulation for timestep+class conditioning."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=1)
        x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1))
        x = x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x


class FinalLayer(nn.Module):
    """Final AdaLN + linear projection to patch-wise output."""
    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    """Diffusion Transformer for pixel-space flow matching.

    Args:
        in_channels: input image channels (1 for grayscale)
        img_size: image size (224)
        patch_size: patch size (16 for 14×14 grid)
        hidden_size: transformer dimension
        depth: number of transformer blocks
        num_heads: number of attention heads
        num_classes: number of class embeddings (0 = unconditional)
        repa_return_layer: layer index from which to extract features for REPA
    """

    def __init__(
        self,
        in_channels: int = 1,
        img_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_classes: int = 0,
        repa_return_layer: int = -1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.num_classes = num_classes
        self.repa_return_layer = repa_return_layer

        self.x_embedder = PatchEmbed(in_channels, patch_size, hidden_size)
        self.t_embedder = TimestepEmbed(hidden_size)
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size) if num_classes > 0 else None

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.final = FinalLayer(hidden_size, patch_size, in_channels)

        # Resolve repa layer index once
        self._repa_idx = repa_return_layer if repa_return_layer >= 0 else depth + repa_return_layer

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_basic_init)

        # Zero-out AdaLN modulation and final layer for identity init
        for block in self.blocks:
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)
        nn.init.constant_(self.final.adaLN[-1].weight, 0)
        nn.init.constant_(self.final.adaLN[-1].bias, 0)
        nn.init.constant_(self.final.linear.weight, 0)
        nn.init.constant_(self.final.linear.bias, 0)

        # Class embedding
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.weight, std=0.02)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, P*P*C) → (B, C, H, W)"""
        B, N, _ = x.shape
        P = self.patch_size
        H = W = int(N ** 0.5)
        x = x.reshape(B, H, W, P, P, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, self.in_channels, H * P, W * P)
        return x

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        t_scaled = t * 1000.0
        t_emb = self.t_embedder(t_scaled)

        if self.y_embedder is not None:
            if class_labels is None:
                null_idx = torch.full((x_t.size(0),), self.num_classes,
                                      device=x_t.device, dtype=torch.long)
                y_emb = self.y_embedder(null_idx)
            else:
                safe_labels = class_labels.clamp(min=0, max=self.num_classes)
                y_emb = self.y_embedder(safe_labels)
            c = t_emb + y_emb
        else:
            c = t_emb

        x = self.x_embedder(x_t)  # (B, N, D)

        repa_feat = None
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if return_features and i == self._repa_idx:
                repa_feat = x.detach().clone()

        x = self.final(x, c)
        x = self._unpatchify(x)

        if return_features:
            if repa_feat is None:
                repa_feat = x
            return x, repa_feat

        return x

    def forward_with_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor | None,
        cfg_scale: float = 0.0,
        null_label: int = -1,
    ) -> torch.Tensor:
        if cfg_scale <= 0.0 or class_labels is None or self.y_embedder is None:
            return self.forward(x_t, t, class_labels=class_labels)

        v_cond = self.forward(x_t, t, class_labels=class_labels)
        v_uncond = self.forward(x_t, t, class_labels=None)  # no labels = null embedding

        return (1.0 + cfg_scale) * v_cond - cfg_scale * v_uncond


def build_dit(model_size: str = 'S', **kwargs) -> DiT:
    """Factory: build DiT by size name.

    Args:
        model_size: 'S' or 'B' (S=33M, B=131M params)
        **kwargs: overrides (e.g. num_classes=185, in_channels=1)
    """
    configs = {
        'S': dict(hidden_size=384, depth=12, num_heads=6),
        'B': dict(hidden_size=768, depth=12, num_heads=12),
    }
    cfg = configs.get(model_size.upper(), configs['S'])
    cfg.update(kwargs)
    return DiT(**cfg)
