#!/usr/bin/env python3
"""Show how MAE masking looks on a single crop."""
import os, sys, warnings, glob
warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from mil_model import MAECropDataset
from mae_model import mae_vit_small

SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = SCRIPT_DIR

# Find one image
paths = glob.glob(os.path.join(PROJECT_ROOT, 'Drugs_Data', 'P1', '**', '*.tiff'), recursive=True)
if not paths:
    paths = glob.glob(os.path.join(PROJECT_ROOT, 'Mutants_Data', 'P1', '**', '*.tif'), recursive=True)
img_path = paths[0]
print(f"Image: {os.path.basename(img_path)}")

# Load crop using MAECropDataset (forces center crop via epoch=0, no augment)
ds = MAECropDataset([img_path], augment=False, seed=SEED)
ds.set_epoch(0)
crop = ds[0]  # (1, 224, 224) normalized to [-1, 1]
crop_np = crop.numpy().squeeze()  # (224, 224) in [-1, 1]
crop_display = crop_np * 0.5 + 0.5  # [0, 1]

# Create MAE model for masking
mae = mae_vit_small(in_chans=1, mask_ratio=0.75, norm_pix_loss=True)
mae.eval()

# Apply masking
with torch.no_grad():
    output = mae(crop.unsqueeze(0))

mask = output['mask'][0]  # (196,)
pred = output['pred'][0]  # (196, 256)
recon = output['recon'][0]  # (1, 224, 224)

# Build masked visualization
img_patches = mae.patchify(crop.unsqueeze(0))[0]  # (196, 256)
mask_3d = mask.unsqueeze(-1).expand(-1, img_patches.shape[-1])
# Gray out masked patches (value = 0 in [-1,1] → 0.5 in [0,1])
masked_patches = img_patches * (1 - mask_3d) + 0.0 * mask_3d  # 0.0 = mid-gray in [-1,1]
masked_img = mae.unpatchify(masked_patches.unsqueeze(0))
masked_display = (masked_img[0].numpy().squeeze() * 0.5 + 0.5).clip(0, 1)

# Reconstruction
recon_display = (recon.numpy().squeeze() * 0.5 + 0.5).clip(0, 1)

# Error map
error = np.abs(recon_display - crop_display)

# ---- Figure ----
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(crop_display, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('Original crop (224×224)', fontsize=10, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(masked_display, cmap='gray', vmin=0, vmax=1)
axes[0, 1].set_title(f'Masked input (75% patches hidden)\n{int(196*0.75)}/{196} masked', fontsize=10, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(recon_display, cmap='gray', vmin=0, vmax=1)
axes[0, 2].set_title('MAE reconstruction', fontsize=10, fontweight='bold')
axes[0, 2].axis('off')

# Patch grid overlay
ax = axes[1, 0]
ax.imshow(crop_display, cmap='gray', vmin=0, vmax=1)
for i in range(0, 225, 16):
    ax.axhline(i - 0.5, color='cyan', lw=0.3, alpha=0.3)
    ax.axvline(i - 0.5, color='cyan', lw=0.3, alpha=0.3)
ax.set_title('Patch grid (16×16 patches,\n14×14 = 196 total)', fontsize=10, fontweight='bold')
ax.axis('off')

# Show which patches are masked
ax = axes[1, 1]
mask_2d = mask.reshape(14, 14).numpy()
ax.imshow(mask_2d, cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
ax.set_title(f'Mask pattern\nblue=visible ({mask_2d.size - mask_2d.sum():.0f}) red=masked ({mask_2d.sum():.0f})',
             fontsize=10, fontweight='bold')
ax.axis('off')

ax = axes[1, 2]
im = ax.imshow(error, cmap='hot', vmin=0, vmax=0.3)
ax.set_title(f'Reconstruction error\nmean={error.mean():.4f} max={error.max():.4f}',
             fontsize=10, fontweight='bold')
ax.axis('off')
plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle('Masked Autoencoder: Input → Mask → Reconstruct', fontsize=13, y=0.98)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'mae_masking_demo.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()
print("Done")
