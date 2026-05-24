import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """2D sin-cos positional embedding (from MAE paper)."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    pos = grid.reshape(2, -1).transpose(1, 0)
    pos_embed = np.zeros((grid_size * grid_size, embed_dim))

    i = 0
    for _ in range(2):
        for j in range(0, embed_dim // 4):
            theta = 10000.0 ** (-4.0 * j / embed_dim)
            pos_embed[:, i] = np.sin(pos[:, _] * theta)
            pos_embed[:, i + embed_dim // 4] = np.cos(pos[:, _] * theta)
            i += 1
    pos_embed = pos_embed[:, :embed_dim]

    if cls_token:
        pos_embed = np.concatenate([np.zeros((1, embed_dim)), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(nn.Module):
    """Image → patch embeddings."""
    def __init__(self, in_chans: int = 1, embed_dim: int = 384, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)  # (B, embed_dim, H/p, W/p)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    """ViT encoder — processes only VISIBLE patches."""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.initialize_weights(img_size, patch_size, num_patches)

    def initialize_weights(self, img_size, patch_size, num_patches):
        grid_size = img_size // patch_size
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.cls_token, std=0.02)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75, fg_binary: torch.Tensor = None,
                fg_mask_ratio: float = 0.95, bg_mask_ratio: float = 0.67):
        """Forward only visible patches.

        If fg_binary is provided, uses two-stage foreground-biased masking:
        - Masks fg_mask_ratio of foreground patches
        - Masks bg_mask_ratio of background patches
        - Overall masking rate ≈ fg_mask_ratio * fg_frac + bg_mask_ratio * (1 - fg_frac)

        Returns:
            x: encoded visible tokens w/ cls token (B, 1 + N_visible, embed_dim)
            mask: binary mask (B, N_patches) — 1 = masked, 0 = visible
            ids_restore: indices to restore original order
        """
        B = x.shape[0]
        patches = self.patch_embed(x)  # (B, embed_dim, 14, 14)
        patches = patches.flatten(2).transpose(1, 2)  # (B, 196, embed_dim)

        N = patches.shape[1]

        if fg_binary is not None:
            # Two-stage masking: different rates for FG vs BG patches
            fg_binary = fg_binary.bool()
            fg_noise = torch.rand(B, N, device=x.device)
            bg_noise = torch.rand(B, N, device=x.device)
            noise = torch.where(fg_binary,
                                fg_noise * (1 / fg_mask_ratio),  # FG: lower → visible
                                bg_noise * (1 / bg_mask_ratio))  # BG: lower → visible
            # Clamp to ensure valid range
            noise = noise.clamp(0, 1)
            n_keep = int(N * (1 - mask_ratio))
        else:
            # Standard random masking
            noise = torch.rand(B, N, device=x.device)
            n_keep = int(N * (1 - mask_ratio))

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :n_keep]

        # Keep only visible patches
        x = torch.gather(patches, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, patches.shape[-1]))

        # Add pos embed for visible positions
        pos_embed = self.pos_embed[:, 1:, :]  # (1, 196, embed_dim), shared across batch
        pos_keep = torch.gather(pos_embed.expand(B, -1, -1), dim=1,
                                index=ids_keep.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1]))
        x = x + pos_keep

        # Add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Build mask
        mask = torch.zeros(B, N, device=x.device)
        mask[:, n_keep:] = 1.0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x, mask, ids_restore


class MAEDecoder(nn.Module):
    """Lightweight decoder — reconstructs from encoded visible + mask tokens."""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 384,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        num_patches = (img_size // patch_size) ** 2

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        self.initialize_weights(img_size, patch_size, num_patches)

    def initialize_weights(self, img_size, patch_size, num_patches):
        grid_size = img_size // patch_size
        pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], grid_size, cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Decode to reconstruct full image.

        Args:
            x: encoded tokens from encoder (B, 1 + N_visible, embed_dim)
            ids_restore: indices to restore original patch order (B, N_patches)

        Returns:
            pred: predicted pixel values (B, N_patches, patch_size^2 * in_chans)
        """
        B = x.shape[0]
        N = ids_restore.shape[1]

        # Project to decoder dim
        x = self.decoder_embed(x)

        # Separate cls token and visible tokens
        cls_token = x[:, :1, :]
        x_visible = x[:, 1:, :]

        # Append mask tokens
        n_masked = N - x_visible.shape[1]
        mask_tokens = self.mask_token.repeat(B, n_masked, 1)
        x_full = torch.cat([x_visible, mask_tokens], dim=1)

        # Unshuffle to original order
        x_full = torch.gather(
            x_full, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[-1])
        )

        # Add cls token
        x = torch.cat([cls_token, x_full], dim=1)

        # Add positional embeddings
        x = x + self.decoder_pos_embed

        # Decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predict pixels (skip cls token)
        x = x[:, 1:, :]
        pred = self.decoder_pred(x)

        return pred


class MAE(nn.Module):
    """Masked Autoencoder: ViT encoder + lightweight decoder.

    Design follows He et al. 2022:
    - Asymmetric encoder-decoder (encoder only sees visible patches)
    - 75% masking ratio
    - norm_pix_loss (reconstruct per-patch normalized pixels)
    - Decoder is ~10% of encoder compute
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        encoder_embed_dim: int = 384,
        encoder_depth: int = 12,
        encoder_num_heads: int = 6,
        decoder_embed_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 3,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
        use_fg_loss: bool = False,
        use_fg_masking: bool = False,
        fg_temperature: float = 0.5,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.use_fg_loss = use_fg_loss
        self.use_fg_masking = use_fg_masking
        self.fg_temperature = fg_temperature
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
        )

        self.decoder = MAEDecoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
        )

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Split image into patches.

        imgs: (B, C, H, W)
        Returns: (B, N_patches, patch_size^2 * C)
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W and H % p == 0
        h = H // p
        x = imgs.reshape(B, C, h, p, h, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h * h, p * p * C)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from patches.

        x: (B, N_patches, patch_size^2 * C)
        Returns: (B, C, H, W)
        """
        p = self.patch_size
        B = x.shape[0]
        h = int(x.shape[1] ** 0.5)
        C = self.in_chans
        x = x.reshape(B, h, h, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, C, h * p, h * p)
        return x

    def forward(self, imgs: torch.Tensor) -> dict:
        B = imgs.shape[0]
        need_fg = self.use_fg_loss or self.use_fg_masking
        if need_fg:
            imgs_01 = imgs * 0.5 + 0.5
            p90 = torch.quantile(imgs_01.flatten(1), 0.90, dim=1).view(-1, 1, 1, 1)
            fg_pixels = (imgs_01 > p90).float()
            fg_patches = self.patchify(fg_pixels)
            fg_weight = fg_patches.mean(dim=-1)
            fg_binary = fg_weight > 0
        else:
            fg_weight = None
            fg_binary = None

        if self.use_fg_masking and fg_binary is not None:
            latent, mask, ids_restore = self.encoder(
                imgs, self.mask_ratio, fg_binary=fg_binary,
                fg_mask_ratio=0.95, bg_mask_ratio=0.67,
            )
        else:
            latent, mask, ids_restore = self.encoder(imgs, self.mask_ratio)

        pred = self.decoder(latent, ids_restore)
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True) + 1e-6
            target_norm = (target - mean) / torch.sqrt(var)
        else:
            target_norm = target

        loss = (pred - target_norm) ** 2
        loss = loss.mean(dim=-1)

        # AttG-style loss: exp(fg_weight / temperature) weighting
        # Background patches (fg_weight ≈ 0) → weight ≈ exp(0/τ) = 1
        # Foreground patches (fg_weight ≈ 1) → weight ≈ exp(1/τ), e.g. 7.4× for τ=0.5
        if self.use_fg_loss and fg_weight is not None:
            loss_weight = torch.exp(fg_weight / self.fg_temperature)
            loss = (loss * mask * loss_weight).sum() / (mask * loss_weight).sum().clamp(min=1)
        else:
            loss = (loss * mask).sum() / mask.sum()

        recon_norm = pred * mask.unsqueeze(-1) + target_norm * (1 - mask).unsqueeze(-1)

        if self.norm_pix_loss and mean is not None and var is not None:
            pred_pixel = pred * torch.sqrt(var) + mean
            target_pixel = target
            recon_pixel = pred_pixel * mask.unsqueeze(-1) + target_pixel * (1 - mask).unsqueeze(-1)
        else:
            recon_pixel = recon_norm

        return {
            'pred': pred,
            'mask': mask,
            'loss': loss,
            'recon': self.unpatchify(recon_norm),
            'recon_pixel': self.unpatchify(recon_pixel),
            'target_pixel': imgs,
            'fg_weight': fg_weight,
        }

    def encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode full image (no masking) — for downstream tasks.

        Returns: (B, N_patches + 1, embed_dim) — all tokens + cls
        """
        # Override: process all patches, no masking
        B = imgs.shape[0]
        patches = self.encoder.patch_embed(imgs)
        patches = patches.flatten(2).transpose(1, 2)

        N = patches.shape[1]
        pos_embed = self.encoder.pos_embed[:, 1:, :]
        x = patches + pos_embed

        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        return x

    def encode_pooled(self, imgs: torch.Tensor) -> torch.Tensor:
        """Pooled embedding (mean of patch tokens) for downstream tasks.

        Returns: (B, embed_dim)
        """
        x = self.encode(imgs)
        return x[:, 1:, :].mean(dim=1)  # exclude cls token, mean pool


def mae_vit_tiny(patch_size=16, in_chans=1, **kwargs):
    """ViT-Tiny MAE: 5M params, 12 blocks, 192-dim, 3 heads."""
    return MAE(
        patch_size=patch_size, in_chans=in_chans,
        encoder_embed_dim=192, encoder_depth=12, encoder_num_heads=3,
        decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
        **kwargs,
    )


def mae_vit_small(patch_size=16, in_chans=1, **kwargs):
    """ViT-Small MAE: 21M params, 12 blocks, 384-dim, 6 heads."""
    return MAE(
        patch_size=patch_size, in_chans=in_chans,
        encoder_embed_dim=384, encoder_depth=12, encoder_num_heads=6,
        decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=4,
        **kwargs,
    )


def mae_vit_base(patch_size=16, in_chans=1, **kwargs):
    """ViT-Base MAE: 86M params, 12 blocks, 768-dim, 12 heads."""
    return MAE(
        patch_size=patch_size, in_chans=in_chans,
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=384, decoder_depth=6, decoder_num_heads=6,
        **kwargs,
    )
