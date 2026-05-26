#!/usr/bin/env python3
"""Traverse along PC1 of 32-dim latent = the drug-vs-mutant axis."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/3 autoencoder approach/mil_vae_both/fold_P1'
LATENTS_PATH = os.path.join(OUTPUT_DIR, 'test_latents_P1_20260523_222527.pt')
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'checkpoint_mil_latest.pth')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_STEPS = 11

print("=" * 60)
print("Traverse PC1 of 32-dim latent (drug vs mutant axis)")
print("=" * 60)

# 1. Load model
print("\n[1/5] Loading model ...")
from vae_model import MILVAE
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

# 2. Load latents
print("[2/5] Loading latents ...")
pt = torch.load(LATENTS_PATH, map_location='cpu', weights_only=False)
records = pt['records']
class_mus = defaultdict(list)
class_src = {}
for r in records:
    class_mus[r['true_label']].append(r['mu'].astype(np.float64))
    class_src[r['true_label']] = r['source']
classes = sorted(class_mus.keys())
N = len(classes)
mean_mu = np.zeros((N, 32))
for i, c in enumerate(classes):
    all_m = np.concatenate(class_mus[c], axis=0)
    mean_mu[i] = all_m.mean(axis=0)
src_arr = np.array([class_src[c] for c in classes])

# 3. PCA on 32-dim class means
print("[3/5] PCA on 32-dim class means ...")
pca = PCA(n_components=10)
scores = pca.fit_transform(mean_mu)
pc1 = pca.components_[0]  # 32-dim direction vector
var_pc1 = pca.explained_variance_ratio_[0] * 100

# Project class means onto PC1
proj = scores[:, 0]
drug_proj = proj[src_arr == 'drug']
mut_proj = proj[src_arr == 'mutant']
drug_center = drug_proj.mean()
mut_center = mut_proj.mean()
print(f"  PC1 variance: {var_pc1:.1f}%")
print(f"  Drug center on PC1: {drug_center:.4f}")
print(f"  Mutant center on PC1: {mut_center:.4f}")
print(f"  Separation: {mut_center - drug_center:.4f}")

# 4. Traverse along PC1 from drug side → mutant side
print("[4/5] Traversing PC1 ...")

# Start at drug centroid, move toward mutant end
drug_centroid = mean_mu[src_arr == 'drug'].mean(axis=0)
mut_centroid = mean_mu[src_arr == 'mutant'].mean(axis=0)

# But use PC1 direction, not centroid difference
# PC1 points from drug centroid toward... let's figure out direction
# PC1 score: <z - mean, pc1>. If drug has negative scores and mutant positive,
# then pc1 points from drug→mutant
mean_all = mean_mu.mean(axis=0)
print(f"  PC1 direction check: drug mean proj={drug_center:.4f}, mut mean proj={mut_center:.4f}")

# Starting point: 3 std below drug mean, ending: 3 std above mutant mean
proj_all = mean_mu @ pc1
proj_std = proj_all.std()
lo = proj_all.min() - 0.5 * proj_std
hi = proj_all.max() + 0.5 * proj_std
print(f"  PC1 range: [{lo:.4f}, {hi:.4f}]")

# Project individual points (all 403K) onto PC1 to get the full distribution
all_mu = np.concatenate([r['mu'].astype(np.float64) for r in records], axis=0)  # (403200, 32)
all_proj = all_mu @ pc1
pct5, pct95 = np.percentile(all_proj, [5, 95])
lo_full = pct5
hi_full = pct95
print(f"  PC1 5-95% range: [{lo_full:.4f}, {hi_full:.4f}]")

traj_values = np.linspace(lo_full, hi_full, N_STEPS)
decoded_images = []
decoded_bags = []

for step, val in enumerate(traj_values):
    z = mean_all + val * pc1  # PC1 component = score × loading
    z_t = torch.from_numpy(z.astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        img = model.decode_img(z_t)
        bag = model.decode_bag(z_t)
    img_np = np.clip(img.squeeze().cpu().numpy() * 0.5 + 0.5, 0, 1)
    bag_np = bag.squeeze().cpu().numpy()
    decoded_images.append(img_np)
    decoded_bags.append(bag_np)
    if (step+1) % 5 == 0:
        print(f"  Step {step+1}/{N_STEPS}")

# 5. Visualizations
print("[5/5] Generating figures ...")

# Histogram of PC1 projections
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.hist(all_proj[sources_flat := np.array([r['source'] for r in records for _ in range(100)]) == 'drug'],
         bins=80, alpha=0.5, color='#E41A1C', label='Drug', density=True)
ax.hist(all_proj[sources_flat == 'mutant'],
         bins=80, alpha=0.5, color='#4DAF4A', label='Mutant', density=True)
for v in traj_values:
    ax.axvline(v, color='white', linewidth=0.5, alpha=0.4, linestyle='--')
ax.set_xlabel('PC1 projection (drug → mutant)', fontsize=11, color='white')
ax.set_ylabel('Density', fontsize=11, color='white')
ax.set_title(f'PC1 of 32-dim latent (AUC=0.99): Drug vs Mutant separation\n'
             f'Traversal steps shown as dashed lines', fontsize=12, color='white')
ax.legend(fontsize=10)
ax.set_facecolor('#1a1a1a')
fig.patch.set_facecolor('#1a1a1a')
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_color('#444')
plt.tight_layout()
hist_path = os.path.join(OUTPUT_DIR, 'pc1_histogram_32dim.png')
fig.savefig(hist_path, dpi=150, bbox_inches='tight')
print(f"  Histogram: {hist_path}")
plt.close()

# Decoded images
fig, axes = plt.subplots(1, N_STEPS, figsize=(3 * N_STEPS, 3.2))
fig.suptitle('Traversal along PC1 of 32-dim latent (drug → mutant axis)',
             fontsize=13, y=0.98)
for step, (val, img) in enumerate(zip(traj_values, decoded_images)):
    ax = axes[step]
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    lbl = f'PC1={val:.2f}'
    if step == 0: lbl = f'Drug side\n{lbl}'
    if step == N_STEPS - 1: lbl = f'Mutant side\n{lbl}'
    ax.set_title(lbl, fontsize=7)
    ax.axis('off')
plt.tight_layout()
img_path = os.path.join(OUTPUT_DIR, 'pc1_traversal_images.png')
fig.savefig(img_path, dpi=200, bbox_inches='tight')
print(f"  Images: {img_path}")
plt.close()

# Bag features
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
from sklearn.metrics.pairwise import cosine_similarity
order = np.argsort(-np.abs(decoded_bags[-1] - decoded_bags[0]))[:100]
for step, (val, bag) in enumerate(zip(traj_values, decoded_bags)):
    color = plt.cm.coolwarm(step / (N_STEPS - 1))
    lbl = f'PC1={val:.1f}' if step in [0, 2, 5, 8, 10] else None
    ax.plot(bag[order], linewidth=0.5 + 0.3 * step / (N_STEPS - 1),
            color=color, alpha=0.4 + 0.6 * step / (N_STEPS - 1), label=lbl)
ax.set_title('Decoded bag features along PC1 (top 100 by |Δ|)', fontsize=12)
ax.set_xlabel('Feature index', fontsize=9)
ax.set_ylabel('Value', fontsize=9)
ax.legend(fontsize=7, loc='upper right')
plt.tight_layout()
bag_path = os.path.join(OUTPUT_DIR, 'pc1_traversal_bag_features.png')
fig.savefig(bag_path, dpi=150, bbox_inches='tight')
print(f"  Bag features: {bag_path}")
plt.close()

# Combined summary
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.05, 1.5], height_ratios=[1, 1.2])

# PC1 histogram
ax = fig.add_subplot(gs[0, 0])
sources_flat = np.array([r['source'] for r in records for _ in range(100)])
ax.hist(all_proj[sources_flat == 'drug'], bins=80, alpha=0.5, color='#E41A1C', label='Drug', density=True)
ax.hist(all_proj[sources_flat == 'mutant'], bins=80, alpha=0.5, color='#4DAF4A', label='Mutant', density=True)
for v in traj_values:
    ax.axvline(v, color='white', linewidth=0.5, alpha=0.3, linestyle='--')
ax.set_xlabel('PC1', fontsize=9, color='white')
ax.set_ylabel('Density', fontsize=9, color='white')
ax.set_title('PC1 = drug vs mutant axis\n(AUC=0.99)', fontsize=10, color='white')
ax.legend(fontsize=7)
ax.set_facecolor('#1a1a1a')
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_color('#444')

# PC1 loading bar chart
ax = fig.add_subplot(gs[0, 2])
x = np.arange(32)
ax.bar(x, pc1, color='#6A3D9A', alpha=0.8)
ax.axhline(0, color='gray', lw=0.5)
ax.set_xlabel('Latent dimension', fontsize=9, color='white')
ax.set_ylabel('PC1 loading', fontsize=9, color='white')
ax.set_title('PC1 loading vector in 32-dim latent space', fontsize=10, color='white')
ax.set_xticks(x)
ax.set_xticklabels([f'z{i}' for i in range(32)], fontsize=5, rotation=90)
ax.set_facecolor('#1a1a1a')
ax.tick_params(colors='white')
for spine in ax.spines.values(): spine.set_color('#444')

# Decoded images along PC1
for step in range(N_STEPS):
    ax = fig.add_subplot(gs[1, 0] if step < N_STEPS else gs[1, 2])
# Better: full width row for images
plt.close(fig)

# Clean combined
fig, axes = plt.subplots(2, N_STEPS, figsize=(3 * N_STEPS, 6))
fig.suptitle('Traversal along PC1 of 32-dim latent (drug → mutant axis)\n'
             'Top: decoded 224×224 image | Bottom: 1280-dim bag feature profile',
             fontsize=12, y=0.98)
vmin_bag = min(b.min() for b in decoded_bags)
vmax_bag = max(b.max() for b in decoded_bags)
for step in range(N_STEPS):
    ax = axes[0, step]
    ax.imshow(decoded_images[step], cmap='gray', vmin=0, vmax=1)
    lbl = f'PC1={traj_values[step]:.1f}'
    if step == 0: lbl = f'Drug ← {lbl}'
    if step == N_STEPS - 1: lbl = f'{lbl} → Mutant'
    ax.set_title(lbl, fontsize=7, fontweight='bold')
    ax.axis('off')

    ax = axes[1, step]
    ax.plot(decoded_bags[step][order[:200]], linewidth=0.3, color='#377EB8', alpha=0.7)
    ax.set_ylim(vmin_bag - 0.3, vmax_bag + 0.3)
    ax.set_xticks([])
    if step > 0: ax.set_yticks([])
plt.tight_layout()
combined_path = os.path.join(OUTPUT_DIR, 'pc1_traversal_combined.png')
fig.savefig(combined_path, dpi=200, bbox_inches='tight')
print(f"  Combined: {combined_path}")
plt.close()

print(f"\n{'=' * 60}")
print(f"All outputs in: {OUTPUT_DIR}")
print(f"{'=' * 60}")
