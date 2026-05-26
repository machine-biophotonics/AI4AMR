#!/usr/bin/env python3
"""Raw DIC images of drug control and WT NC_1 + feature-space traversal."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from vae_model import MILVAE

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/3 autoencoder approach/mil_vae_both/fold_P1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENTS_PATH = os.path.join(OUTPUT_DIR, 'test_latents_P1_20260523_222527.pt')
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint_mil_latest.pth')
N_STEPS = 11
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Raw image paths (from test latents records)
CTRL_RAW = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/Drugs_Data/P1/20260428_161449_035/WellB06_PointB06_0000_ChannelCam-DIA DIC Master Screening_Seq0378.tiff'
WTNC1_RAW = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/Mutants_Data/P1/TIFOCUS/WellE01_PointE01_0000_ChannelCam-DIA DIC Master Screening_Seq1008_tiff_sharpest_z001.tif'

print("=" * 60)
print("Control vs WT NC_1: Raw images + feature traversal")
print("=" * 60)

# 1. Load raw images
print("\n[1/4] Loading raw DIC images ...")
ctrl_raw = Image.open(CTRL_RAW)
wtnc1_raw = Image.open(WTNC1_RAW)
print(f"  Control: {ctrl_raw.size} {ctrl_raw.mode}")
print(f"  WT NC_1: {wtnc1_raw.size} {wtnc1_raw.mode}")

# Convert to numpy arrays
ctrl_arr = np.array(ctrl_raw).astype(np.float32)
wtnc1_arr = np.array(wtnc1_raw).astype(np.float32)

# Auto-contrast: clip at 2nd and 98th percentile
def auto_contrast(img, low=2, high=98):
    if img.ndim == 3: img = img.mean(axis=2)
    vmin, vmax = np.percentile(img[img > 0], [low, high]) if (img > 0).sum() > 100 else (img.min(), img.max())
    clipped = np.clip(img, vmin, vmax)
    return (clipped - vmin) / max(vmax - vmin, 1e-6)

ctrl_display = auto_contrast(ctrl_arr, 1, 99)
wtnc1_display = auto_contrast(wtnc1_arr, 1, 99)
print(f"  Contrast normalized: {ctrl_display.shape} -> [{ctrl_display.min():.3f}, {ctrl_display.max():.3f}]")

# 2. Load model and latents for traversal
print("\n[2/4] Loading model + latents ...")
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']
num_classes = sd['encoder.classifier.1.weight'].shape[0]
model = MILVAE(num_classes=num_classes, latent_dim=32, beta=0.1, num_heads=4,
               dropout=0.5, use_contrastive=True, num_channels=1,
               pretrained='imagenet', backbone='efficientnet_b0',
               pooling='attention', img_size=224,
               feature_decoder=True, pixel_decoder=True)
model.load_state_dict(sd)
model.to(DEVICE)
model.eval()

pt = torch.load(LATENTS_PATH, map_location='cpu', weights_only=False)
records = pt['records']
ctrl_mus = np.concatenate([r['mu'] for r in records if r['source'] == 'drug' and r['true_label'] == 'control'], axis=0).astype(np.float64)
wtnc1_mus = np.concatenate([r['mu'] for r in records if r['source'] == 'mutant' and r['true_label'] == 'WT NC_1'], axis=0).astype(np.float64)
c_ctrl = ctrl_mus.mean(axis=0)
c_wtnc1 = wtnc1_mus.mean(axis=0)
print(f"  Control latents: {ctrl_mus.shape[0]}")
print(f"  WT NC_1 latents: {wtnc1_mus.shape[0]}")

# 3. Compute traversal bag features
print("\n[3/4] Decoding traversal ...")
alphas = np.linspace(0, 1, N_STEPS)
decoded_bags = []
for alpha in alphas:
    z = (1 - alpha) * c_ctrl + alpha * c_wtnc1
    z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        bag = model.decode_bag(z_t)
    decoded_bags.append(bag.squeeze().cpu().numpy())
decoded_bags = np.array(decoded_bags)  # (11, 1280)

# Also get the real bag features for comparison
ctrl_real_bag = np.concatenate([r['bag'] for r in records if r['source'] == 'drug' and r['true_label'] == 'control'], axis=0).astype(np.float64)
wtnc1_real_bag = np.concatenate([r['bag'] for r in records if r['source'] == 'mutant' and r['true_label'] == 'WT NC_1'], axis=0).astype(np.float64)
c_ctrl_bag = ctrl_real_bag.mean(axis=0)
c_wtnc1_bag = wtnc1_real_bag.mean(axis=0)

# 4. Visualize
print("\n[4/4] Generating figures ...")

# --- FIGURE 1: Raw images side by side ---
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

for ax, img, title, well in [
    (axes[0], ctrl_display, 'Drug Control (water)', 'B06'),
    (axes[1], wtnc1_display, 'Mutant Control (WT NC_1)', 'E01'),
]:
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'{title}\nWell {well}', fontsize=13, fontweight='bold')
    ax.axis('off')

    # Overlay 3×3 crop grid (each crop ~224×224)
    # Raw image is ~1200×1200, crops span ~672px (3×224)
    h, w = img.shape
    grid_size = 224 * 3
    start_x = (w - grid_size) // 2
    start_y = (h - grid_size) // 2
    for row in range(3):
        for col in range(3):
            x0 = start_x + col * 224
            y0 = start_y + row * 224
            rect = Rectangle((x0, y0), 224, 224, linewidth=1.5,
                           edgecolor='cyan', facecolor='none', alpha=0.6)
            ax.add_patch(rect)
    ax.text(start_x + grid_size//2 - 60, start_y - 15,
            '3×3 crop neighborhood (9×224px)', color='cyan',
            fontsize=8, fontweight='bold')

plt.tight_layout()
raw_path = os.path.join(OUTPUT_DIR, 'raw_control_vs_wtnc1.png')
fig.savefig(raw_path, dpi=200, bbox_inches='tight')
print(f"  Raw images: {raw_path}")
plt.close()

# --- FIGURE 2: Bag feature profiles ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sort features by difference magnitude for cleaner display
diff = c_wtnc1_bag - c_ctrl_bag
order = np.argsort(-np.abs(diff))

for ax, feat, title, color in [
    (axes[0], c_ctrl_bag, 'Drug control (water)', '#E41A1C'),
    (axes[1], c_wtnc1_bag, 'WT NC_1', '#4DAF4A'),
]:
    ax.plot(feat[order], linewidth=0.3, color=color, alpha=0.7)
    ax.set_title(f'Mean bag feature: {title}', fontsize=12)
    ax.set_xlabel('Feature index (sorted by |Δ|)', fontsize=9)
    ax.set_ylabel('Value', fontsize=9)
    ax.set_ylim(-2, 2)

plt.tight_layout()
bag_path = os.path.join(OUTPUT_DIR, 'bag_features_control_vs_wtnc1.png')
fig.savefig(bag_path, dpi=200, bbox_inches='tight')
print(f"  Bag features: {bag_path}")
plt.close()

# --- FIGURE 3: Top differentiating features ---
fig, ax = plt.subplots(1, 1, figsize=(14, 4))
n_show = min(100, len(order))
top_idx = order[:n_show]
x = np.arange(n_show)
width = 0.4
ax.bar(x - width/2, c_ctrl_bag[top_idx], width, label='Drug control (water)', color='#E41A1C', alpha=0.8)
ax.bar(x + width/2, c_wtnc1_bag[top_idx], width, label='WT NC_1', color='#4DAF4A', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'f{i}' for i in top_idx], fontsize=5, rotation=90)
ax.set_ylabel('Value', fontsize=9)
ax.set_title(f'Top {n_show} most differentiating bag features (by |Δ|)', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
top_path = os.path.join(OUTPUT_DIR, 'top_diff_features_control_vs_wtnc1.png')
fig.savefig(top_path, dpi=200, bbox_inches='tight')
print(f"  Top features: {top_path}")
plt.close()

# --- FIGURE 4: Feature traversal as profiles ---
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
for step, alpha in enumerate(alphas):
    color = plt.cm.coolwarm(1 - alpha)  # blue→red
    label = f'α={alpha:.1f}' if step in [0, 2, 5, 8, 10] else None
    ax.plot(decoded_bags[step][order[:200]], linewidth=0.8,
            color=color, alpha=0.5 + 0.5 * alpha, label=label)
ax.set_title('Bag feature traversal: control (blue) → WT NC_1 (red)\n(top 200 features by |Δ|)',
             fontsize=12)
ax.set_xlabel('Feature index', fontsize=9)
ax.set_ylabel('Decoded value', fontsize=9)
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
traj_path = os.path.join(OUTPUT_DIR, 'bag_traversal_profiles.png')
fig.savefig(traj_path, dpi=200, bbox_inches='tight')
print(f"  Traversal profiles: {traj_path}")
plt.close()

# --- FIGURE 5: Combined summary with raw images + features ---
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.5, 1.5], height_ratios=[1, 1])

# Top row: raw images
for col, (img, title, well) in enumerate([
    (ctrl_display, 'Drug Control\n(water)', 'B06'),
    (wtnc1_display, 'WT NC_1', 'E01'),
]):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    h, w = img.shape
    grid_size = 224 * 3
    sx, sy = (w - grid_size) // 2, (h - grid_size) // 2
    for r in range(3):
        for c in range(3):
            rect = Rectangle((sx + c*224, sy + r*224), 224, 224,
                           linewidth=1, edgecolor='cyan', facecolor='none', alpha=0.5)
            ax.add_patch(rect)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.text(10, 20, f'Well {well}', color='white', fontsize=9,
            bbox=dict(boxstyle='round', fc='black', alpha=0.5))

# Top right: centroids bar
ax = fig.add_subplot(gs[0, 2:])
x = np.arange(32)
width = 0.35
ax.bar(x - width/2, c_ctrl, width, label='Drug control', color='#E41A1C', alpha=0.8)
ax.bar(x + width/2, c_wtnc1, width, label='WT NC_1', color='#4DAF4A', alpha=0.8)
ax.set_ylabel('μ', fontsize=10)
ax.set_title('32-dim latent centroids', fontsize=12)
ax.legend(fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f'z{i}' for i in range(32)], fontsize=5, rotation=45)
ax.axhline(0, color='gray', linewidth=0.5)

# Bottom: traversal
ax = fig.add_subplot(gs[1, :])
for step, alpha in enumerate(alphas):
    color = plt.cm.RdYlBu(1 - alpha)
    lbl = f'α={alpha:.1f}' if step in [0, 3, 5, 7, 10] else None
    ax.plot(decoded_bags[step][order[:100]], linewidth=0.7 + 0.5 * alpha,
            color=color, alpha=0.4 + 0.6 * alpha, label=lbl)
ax.set_title('Feature traversal (decoded 1280-dim bag, top 100 by |Δ|): blue=control → red=WT NC_1',
             fontsize=12)
ax.set_xlabel('Feature index', fontsize=9)
ax.set_ylabel('Decoded value', fontsize=9)
ax.legend(fontsize=8, ncol=3, loc='upper right')
ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
summary_path = os.path.join(OUTPUT_DIR, 'traversal_summary_with_raw.png')
fig.savefig(summary_path, dpi=200, bbox_inches='tight')
print(f"  Summary: {summary_path}")
plt.close()

print(f"\n{'=' * 60}")
print(f"All outputs in: {OUTPUT_DIR}")
print(f"{'=' * 60}")
