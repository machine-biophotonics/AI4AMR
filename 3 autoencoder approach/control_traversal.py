#!/usr/bin/env python3
"""Traverse from drug control (water) → mutant control (WT NC_1) in VAE latent space.

Decodes intermediate latents through pixel decoder to show morphological changes.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde

from vae_model import MILVAE

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/3 autoencoder approach/mil_vae_both/fold_P1'

os.makedirs(OUTPUT_DIR, exist_ok=True)

LATENTS_PATH = os.path.join(OUTPUT_DIR, 'test_latents_P1_20260523_222527.pt')
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint_mil_latest.pth')
PACMAP_PATH = os.path.join(OUTPUT_DIR, 'pacmap_embedding_all.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_STEPS = 11  # 0.0, 0.1, ..., 1.0

print("=" * 60)
print("Control → WT NC_1 latent traversal")
print("=" * 60)
print(f"Device: {DEVICE}")

# 1. Load model
print("\n[1/4] Loading model ...")
# Need num_classes to init; extract from checkpoint head
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']
# Infer num_classes from classifier weight shape
num_classes = sd['encoder.classifier.1.weight'].shape[0]
print(f"  num_classes = {num_classes}")

model = MILVAE(
    num_classes=num_classes,
    latent_dim=32,
    beta=0.1,
    num_heads=4,
    dropout=0.5,
    use_contrastive=True,
    num_channels=1,
    pretrained='imagenet',
    backbone='efficientnet_b0',
    pooling='attention',
    img_size=224,
    feature_decoder=True,
    pixel_decoder=True,
)
model.load_state_dict(sd)
model.to(DEVICE)
model.eval()
print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

# 2. Load latents
print("\n[2/4] Loading latents ...")
pt = torch.load(LATENTS_PATH, map_location='cpu', weights_only=False)
records = pt['records']

# Separate by class
drug_ctrl_mus = []
mut_ctrl_mus = []
drug_ctrl_bags = []
mut_ctrl_bags = []

for r in records:
    mu = r['mu']  # (100, 32) float16
    bag = r['bag']  # (100, 1280) float16
    lbl = r['true_label']
    src = r['source']
    if src == 'drug' and lbl == 'control':
        drug_ctrl_mus.append(mu)
        drug_ctrl_bags.append(bag)
    elif src == 'mutant' and lbl == 'WT NC_1':
        mut_ctrl_mus.append(mu)
        mut_ctrl_bags.append(bag)

drug_mu = np.concatenate(drug_ctrl_mus, axis=0).astype(np.float64)  # (N_d, 32)
mut_mu = np.concatenate(mut_ctrl_mus, axis=0).astype(np.float64)    # (N_m, 32)
drug_bag = np.concatenate(drug_ctrl_bags, axis=0).astype(np.float64)
mut_bag = np.concatenate(mut_ctrl_bags, axis=0).astype(np.float64)

print(f"  Drug control (water): {drug_mu.shape[0]} latents from {len(drug_ctrl_mus)} records")
print(f"  Mutant control (WT NC_1): {mut_mu.shape[0]} latents from {len(mut_ctrl_mus)} records")

# Centroids
c_drug = drug_mu.mean(axis=0)
c_mut = mut_mu.mean(axis=0)
print(f"  Drug centroid norm: {np.linalg.norm(c_drug):.4f}")
print(f"  Mutant centroid norm: {np.linalg.norm(c_mut):.4f}")
print(f"  Euclidean distance between centroids: {np.linalg.norm(c_drug - c_mut):.4f}")

# 3. Traversal
print("\n[3/4] Decoding traversal ...")
alphas = np.linspace(0, 1, N_STEPS)

decoded_images = []  # each (224, 224) in [0,1]
decoded_bags = []    # each (1280,)
traj_pts = []        # (N_steps, 2) PaCMAP-like projection

for step, alpha in enumerate(alphas):
    z = (1 - alpha) * c_drug + alpha * c_mut
    z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(DEVICE)  # (1, 32)
    with torch.no_grad():
        img = model.decode_img(z_t)  # (1, 1, 224, 224)
        bag = model.decode_bag(z_t)  # (1, 1280)
    img_np = img.squeeze().cpu().numpy()  # (224, 224)
    bag_np = bag.squeeze().cpu().numpy()  # (1280,)
    # tanh → [0, 1]
    img_np = np.clip(img_np * 0.5 + 0.5, 0, 1)
    decoded_images.append(img_np)
    decoded_bags.append(bag_np)
    traj_pts.append(z)
    if (step + 1) % 5 == 0:
        print(f"  Step {step+1}/{N_STEPS} (α={alpha:.1f})")

traj_pts = np.array(traj_pts)  # (11, 32)
decoded_bags = np.array(decoded_bags)  # (11, 1280)

# 4. PaCMAP trajectory (project traversal through the learned space)
# We need a 2D projection; we already have PaCMAP, let's use it if available
print("\n[4/4] Generating visualizations ...")

# --- Load PaCMAP ---
if os.path.exists(PACMAP_PATH):
    pac = torch.load(PACMAP_PATH, map_location='cpu', weights_only=False)
    pac_emb = pac['embedding'].astype(np.float64)
    pac_types = pac['src_types']
    pac_labels = pac['class_labels']
    print(f"  PaCMAP loaded: {pac_emb.shape[0]} points")

    # Find drug control and WT NC_1 indices in PaCMAP
    ctrl_idx = [i for i, (l, t) in enumerate(zip(pac_labels, pac_types))
                if t == 'drug' and l == 'control']
    wt_nc1_idx = [i for i, (l, t) in enumerate(zip(pac_labels, pac_types))
                  if t == 'mutant' and l == 'WT NC_1']
    print(f"  PaCMAP: {len(ctrl_idx)} control, {len(wt_nc1_idx)} WT NC_1")
    ctrl_pac = pac_emb[ctrl_idx]
    wt_nc1_pac = pac_emb[wt_nc1_idx]
    ctrl_pac_centroid = ctrl_pac.mean(axis=0)
    wt_nc1_pac_centroid = wt_nc1_pac.mean(axis=0)
else:
    pac_emb = None
    print("  No PaCMAP found, skipping trajectory overlay")

# --- FIGURE 1: PaCMAP overlay with trajectory ---
if pac_emb is not None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Background: subsample all drug and mutant
    rng = np.random.RandomState(SEED)
    all_idx = np.arange(len(pac_emb))
    n_plot = min(50000, len(all_idx))
    sidx = rng.choice(all_idx, n_plot, replace=False)
    colors = ['#E41A1C' if pac_types[i] == 'drug' else '#4DAF4A' for i in sidx]
    ax.scatter(pac_emb[sidx, 0], pac_emb[sidx, 1],
               c=colors, s=0.5, alpha=0.15, rasterized=True)

    # Highlight drug control (water) points
    ax.scatter(ctrl_pac[:, 0], ctrl_pac[:, 1],
               c='#FFD700', s=8, alpha=0.6, edgecolors='none',
               label=f'Drug control (water) n={len(ctrl_idx)}', zorder=4)

    # Highlight WT NC_1 points
    ax.scatter(wt_nc1_pac[:, 0], wt_nc1_pac[:, 1],
               c='#00FFFF', s=8, alpha=0.6, edgecolors='none',
               label=f'WT NC_1 n={len(wt_nc1_idx)}', zorder=4)

    # Project traversal 32-dim → PaCMAP via linear interpolation of centroids
    # The traversal in 32-dim space needs to be mapped to 2D PaCMAP.
    # We can't directly project it. But we can show the centroid trajectory.
    # Actually, we can just interpolate between centroids in PaCMAP space.

    # Interpolate between drug control centroid and WT NC_1 centroid in PaCMAP
    traj_pac = np.column_stack([
        np.linspace(ctrl_pac_centroid[0], wt_nc1_pac_centroid[0], N_STEPS),
        np.linspace(ctrl_pac_centroid[1], wt_nc1_pac_centroid[1], N_STEPS),
    ])

    # Draw trajectory
    ax.plot(traj_pac[:, 0], traj_pac[:, 1], 'w-', linewidth=2.5, zorder=5, alpha=0.8)
    ax.plot(traj_pac[:, 0], traj_pac[:, 1], 'k--', linewidth=1.5, zorder=5, alpha=0.4)

    # Mark steps with circles (color gradient from yellow → cyan)
    for step, (x, y) in enumerate(traj_pac):
        t = step / (N_STEPS - 1)
        color = (1 - t) * np.array([1, 0.84, 0]) + t * np.array([0, 1, 1])
        circle = plt.Circle((x, y), 0.15, color=color, ec='k', lw=0.8, zorder=6)
        ax.add_patch(circle)
        if step % 2 == 0:
            ax.annotate(f'α={alphas[step]:.1f}', (x, y),
                        xytext=(4, 8), textcoords='offset points',
                        fontsize=7, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))

    ax.set_title('Latent Traversal: Drug Control (water) → WT NC_1\n'
                 '(11-step linear interpolation in 32-dim VAE latent)',
                 fontsize=13)
    ax.set_xlabel('PaCMAP 1')
    ax.set_ylabel('PaCMAP 2')
    ax.legend(fontsize=9, markerscale=4, loc='upper right')
    ax.set_aspect('equal')
    plt.tight_layout()
    pacmap_path = os.path.join(OUTPUT_DIR, 'traversal_pacmap_overlay.png')
    fig.savefig(pacmap_path, dpi=200, bbox_inches='tight')
    print(f"  PaCMAP overlay: {pacmap_path}")
    plt.close(fig)

# --- FIGURE 2: Decoded images grid ---
fig, axes = plt.subplots(1, N_STEPS, figsize=(3 * N_STEPS, 3.2))
fig.suptitle('Decoded images along traversal: Drug control (water) → WT NC_1',
             fontsize=13, y=0.98)

for step, (alpha, img) in enumerate(zip(alphas, decoded_images)):
    ax = axes[step]
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'α={alpha:.1f}', fontsize=9, fontweight='bold')
    ax.axis('off')
    # Label end-points
    if step == 0:
        ax.set_xlabel('Drug control\n(water)', fontsize=8)
    elif step == N_STEPS - 1:
        ax.set_xlabel('WT NC_1\n(mutant control)', fontsize=8)

plt.tight_layout()
img_path = os.path.join(OUTPUT_DIR, 'traversal_decoded_images.png')
fig.savefig(img_path, dpi=200, bbox_inches='tight')
print(f"  Decoded images: {img_path}")
plt.close(fig)

# --- FIGURE 3: Decoded bag features heatmap ---
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
im = ax.imshow(decoded_bags, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax.set_yticks(range(N_STEPS))
ax.set_yticklabels([f'α={a:.1f}' for a in alphas], fontsize=8)
ax.set_xlabel('Feature dimension (1280-dim)', fontsize=10)
ax.set_title('Decoded 1280-dim bag features along traversal', fontsize=12)
plt.colorbar(im, ax=ax, label='Feature value', shrink=0.8)
plt.tight_layout()
bag_path = os.path.join(OUTPUT_DIR, 'traversal_decoded_bag_features.png')
fig.savefig(bag_path, dpi=200, bbox_inches='tight')
print(f"  Bag features: {bag_path}")
plt.close(fig)

# --- FIGURE 4: Latent space itself (32-dim) ---
fig, axes = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [1, 1.2]})

# Top: 32-dim latents of centroids
ax = axes[0]
x = np.arange(32)
width = 0.35
ax.bar(x - width/2, c_drug, width, label='Drug control (water)', color='#E41A1C', alpha=0.8)
ax.bar(x + width/2, c_mut, width, label='WT NC_1', color='#4DAF4A', alpha=0.8)
ax.set_ylabel('Mean μ', fontsize=10)
ax.set_title('VAE latent centroids (32-dim μ)', fontsize=12)
ax.legend(fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([f'z{i}' for i in range(32)], fontsize=6, rotation=45)

# Bottom: traversal in 32-dim as heatmap
ax = axes[1]
traj_mu = traj_pts  # (11, 32)
im = ax.imshow(traj_mu, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
ax.set_yticks(range(N_STEPS))
ax.set_yticklabels([f'α={a:.1f}' for a in alphas], fontsize=8)
ax.set_xlabel('Latent dimension', fontsize=10)
ax.set_xticks(range(32))
ax.set_xticklabels([f'z{i}' for i in range(32)], fontsize=6, rotation=45)
ax.set_title('Traversal in 32-dim latent space (color = μ value)', fontsize=12)
plt.colorbar(im, ax=ax, label='μ value', shrink=0.8)

plt.tight_layout()
latent_path = os.path.join(OUTPUT_DIR, 'traversal_latent_space.png')
fig.savefig(latent_path, dpi=200, bbox_inches='tight')
print(f"  Latent space: {latent_path}")
plt.close(fig)

# --- FIGURE 5: Combined summary ---
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1.5, 1])

# Top row: Latent centroids bar chart
ax1 = fig.add_subplot(gs[0, :])
x = np.arange(32)
width = 0.35
ax1.bar(x - width/2, c_drug, width, label='Drug control (water)', color='#E41A1C', alpha=0.8)
ax1.bar(x + width/2, c_mut, width, label='WT NC_1', color='#4DAF4A', alpha=0.8)
ax1.set_ylabel('μ', fontsize=10)
ax1.set_title('VAE Latent Centroids (32-dim μ)', fontsize=12)
ax1.legend(fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels([f'z{i}' for i in range(32)], fontsize=5, rotation=45)

# Middle row: Decoded images
for step in range(N_STEPS):
    ax = fig.add_subplot(gs[1, step // 4 if N_STEPS > 3 else 0])
    
# Actually let's just do a simpler layout
plt.close(fig)

# Simpler combined
fig, axes = plt.subplots(2, N_STEPS, figsize=(3 * N_STEPS, 6))
fig.suptitle('Latent Traversal: Drug control (water) → WT NC_1\n'
             '(top: decoded 224×224 image, bottom: 1280-dim bag feature)',
             fontsize=12, y=0.98)

vmin, vmax = decoded_bags.min(), decoded_bags.max()
for step in range(N_STEPS):
    # Top: decoded image
    ax = axes[0, step]
    ax.imshow(decoded_images[step], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'α={alphas[step]:.1f}', fontsize=9, fontweight='bold')
    ax.axis('off')
    if step == 0:
        ax.set_ylabel('Image\n224×224', fontsize=8)
    elif step == N_STEPS - 1:
        pass

    # Bottom: decoded bag feature
    ax = axes[1, step]
    ax.plot(decoded_bags[step], linewidth=0.3, color='#377EB8', alpha=0.7)
    ax.set_ylim(vmin - 0.5, vmax + 0.5)
    ax.set_xticks([])
    if step == 0:
        ax.set_ylabel('Bag\n1280-dim', fontsize=8)
    if step > 0:
        ax.set_yticks([])

plt.tight_layout()
combined_path = os.path.join(OUTPUT_DIR, 'traversal_combined.png')
fig.savefig(combined_path, dpi=200, bbox_inches='tight')
print(f"  Combined: {combined_path}")
plt.close(fig)

print(f"\n{'=' * 60}")
print(f"All outputs in: {OUTPUT_DIR}")
print(f"{'=' * 60}")
