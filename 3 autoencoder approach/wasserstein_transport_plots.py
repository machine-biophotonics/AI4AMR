#!/usr/bin/env python3
"""
Wasserstein optimal transport visualizations
Inspired by: https://alexhwilliams.info/itsneuronalblog/2020/10/09/optimal-transport/

Generates:
  1. Transport cost matrix (drug vs mutant on 1D projection)
  2. Transport plan matrix + arrows in 2D t-SNE space
  3. Entropic regularization comparison
"""
import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import ot

SEED = 42
np.random.seed(SEED)

ANALYSIS_DIR = 'mil_vae_both/fold_Plate_1/analysis'
OUTPUT_DIR = os.path.join(ANALYSIS_DIR, 'wasserstein_transport')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
data = np.load(os.path.join(ANALYSIS_DIR, 'latent_codes.npz'))
z = data['z']
labels = data['labels']
class_names = data['class_names']

# Drug / Mutant classification based on class name pattern
# Drug names end with concentration suffix (e.g. _0.25x, _1x, _2x)
# 'control' is a no-treatment control well → drug
names = np.array([class_names[l] for l in labels])
drug_pattern = re.compile(r'_\d+(\.\d+)?x$')
is_drug = np.array([bool(drug_pattern.search(names[i])) or (names[i] == 'control') for i in range(len(labels))])
is_mutant = ~is_drug

print(f"Drug: {is_drug.sum()}, Mutant: {is_mutant.sum()}")

# Compute t-SNE on latent codes
z_scaled = StandardScaler().fit_transform(z)
tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
z_2d = tsne.fit_transform(z_scaled)

# =========================================================================
# 1. 1D PROJECTION + TRANSPORT COST/PLAN (like blog's 1D example)
# =========================================================================
print("\n--- 1D Transport: Drug vs Mutant ---")
# Project to the axis connecting the two centroids
drug_cent = z_2d[is_drug].mean(axis=0)
mut_cent = z_2d[is_mutant].mean(axis=0)
axis_vec = mut_cent - drug_cent
axis_vec = axis_vec / np.linalg.norm(axis_vec)
proj = z_2d @ axis_vec

n_bins = 50
x_min, x_max = proj.min(), proj.max()
bin_edges = np.linspace(x_min, x_max, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

p_drug, _ = np.histogram(proj[is_drug], bins=bin_edges, density=True)
p_mut, _ = np.histogram(proj[is_mutant], bins=bin_edges, density=True)
p_drug /= p_drug.sum()
p_mut /= p_mut.sum()

# Cost matrix (squared Euclidean distance between bin centers)
C_1d = (bin_centers[:, None] - bin_centers[None, :]) ** 2

# Normalize cost to [0, 1] for stable Sinkhorn
C_max = C_1d.max()
C_norm = C_1d / C_max

# Solve OT with Sinkhorn on normalized cost
reg = 0.02
T_sink = ot.sinkhorn(p_drug, p_mut, C_norm, reg, numItermax=5000)
wd_sink_scaled = np.sqrt(np.sum(T_sink * C_1d))

# Exact OT for reference
wd_exact = np.sqrt(ot.emd2(p_drug, p_mut, C_1d))

print(f"  Sinkhorn WD: {wd_sink_scaled:.4f}")
print(f"  Exact WD: {wd_exact:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Density functions
ax = axes[0]
ax.bar(bin_centers, p_drug, width=bin_centers[1]-bin_centers[0],
       alpha=0.6, color='tab:blue', label='Drug')
ax.bar(bin_centers, p_mut, width=bin_centers[1]-bin_centers[0],
       alpha=0.6, color='tab:orange', label='Mutant')
ax.set_xlabel('Projection onto drug→mutant axis')
ax.set_ylabel('Probability mass')
ax.set_title(f'1D Projection (WD={wd_exact:.3f})')
ax.legend()

# Panel 2: Cost matrix
ax = axes[1]
im = ax.imshow(C_1d, aspect='auto', cmap='viridis', origin='lower',
               extent=[x_min, x_max, x_min, x_max])
plt.colorbar(im, ax=ax, shrink=0.8, label='Cost (sq. distance)')
ax.set_xlabel('Mutant bins')
ax.set_ylabel('Drug bins')
ax.set_title('Transport Cost Matrix C')

# Panel 3: Transport plan
ax = axes[2]
eps = 1e-8
T_log = np.log(T_sink + eps)
vmin = np.percentile(T_log, 5)
vmax = np.percentile(T_log, 95)
im = ax.imshow(T_log, aspect='auto', cmap='hot', origin='lower',
               extent=[x_min, x_max, x_min, x_max],
               vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax, shrink=0.8, label='log(Transport mass)')
ax.set_xlabel('Mutant bins')
ax.set_ylabel('Drug bins')
ax.set_title(f'Transport Plan T* (Sinkhorn ε={reg})')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_1d_transport.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  1D transport saved")

# =========================================================================
# 2. 2D TRANSPORT ARROW DIAGRAM (like blog's 2D arrows)
# =========================================================================
print("\n--- 2D Arrow Diagram ---")
n_grid = 30
x2_min, x2_max = z_2d[:, 0].min(), z_2d[:, 0].max()
y2_min, y2_max = z_2d[:, 1].min(), z_2d[:, 1].max()
x_edges = np.linspace(x2_min, x2_max, n_grid + 1)
y_edges = np.linspace(y2_min, y2_max, n_grid + 1)
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
xv, yv = np.meshgrid(x_centers, y_centers)
grid_pts = np.column_stack([xv.ravel(), yv.ravel()])

# Bin drug and mutant into grid
h_drug, _, _ = np.histogram2d(z_2d[is_drug, 0], z_2d[is_drug, 1],
                               bins=[x_edges, y_edges], density=True)
h_mut, _, _ = np.histogram2d(z_2d[is_mutant, 0], z_2d[is_mutant, 1],
                              bins=[x_edges, y_edges], density=True)

p_drug_2d = h_drug.T.ravel()
p_mut_2d = h_mut.T.ravel()
p_drug_2d /= p_drug_2d.sum()
p_mut_2d /= p_mut_2d.sum()

# Cost matrix (between grid cells)
C_2d = ot.dist(grid_pts, grid_pts, metric='sqeuclidean')
C_2d_max = C_2d.max()
C_2d_norm = C_2d / C_2d_max

# Sinkhorn
T_2d = ot.sinkhorn(p_drug_2d, p_mut_2d, C_2d_norm, reg=0.02, numItermax=5000)
T_mat = T_2d.reshape(n_grid * n_grid, n_grid * n_grid)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: 2D densities
ax = axes[0]
ax.scatter(z_2d[is_drug, 0], z_2d[is_drug, 1], c='tab:blue', s=3, alpha=0.3, label='Drug')
ax.scatter(z_2d[is_mutant, 0], z_2d[is_mutant, 1], c='tab:orange', s=3, alpha=0.3, label='Mutant')
# Contours for drug
if is_drug.sum() > 10:
    kde_drug = gaussian_kde(z_2d[is_drug].T)
    zi_drug = kde_drug(np.vstack([xv.ravel(), yv.ravel()])).reshape(xv.shape)
    ax.contour(xv, yv, zi_drug, levels=5, colors='tab:blue', linewidths=1, alpha=0.6)
if is_mutant.sum() > 10:
    kde_mut = gaussian_kde(z_2d[is_mutant].T)
    zi_mut = kde_mut(np.vstack([xv.ravel(), yv.ravel()])).reshape(xv.shape)
    ax.contour(xv, yv, zi_mut, levels=5, colors='tab:orange', linewidths=1, alpha=0.6)
ax.legend(fontsize=12)
ax.set_title('Drug (blue) vs Mutant (orange) densities', fontsize=13)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

# Panel 2: Transport arrows (top N flows)
ax = axes[1]
ax.scatter(z_2d[is_drug, 0], z_2d[is_drug, 1], c='tab:blue', s=3, alpha=0.2)
ax.scatter(z_2d[is_mutant, 0], z_2d[is_mutant, 1], c='tab:orange', s=3, alpha=0.2)

# Find top transport flows
n_arrows = 60
flat_T = T_2d.ravel()
top_idx = np.argsort(flat_T)[-n_arrows:]
src_idx = top_idx // (n_grid * n_grid)
dst_idx = top_idx % (n_grid * n_grid)

for s, d in zip(src_idx, dst_idx):
    src_pt = grid_pts[s]
    dst_pt = grid_pts[d]
    mass = flat_T[s * (n_grid * n_grid) + d]
    ax.arrow(src_pt[0], src_pt[1],
             dst_pt[0] - src_pt[0], dst_pt[1] - src_pt[1],
             head_width=0.3, head_length=0.3, fc='red', ec='red',
             alpha=np.clip(mass * 10, 0.1, 0.8), length_includes_head=True)

ax.set_title(f'Top {n_arrows} OT flows: Drug → Mutant', fontsize=13)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_2d_transport_arrows.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  2D arrow diagram saved")

# =========================================================================
# 3. ENTROPIC REGULARIZATION COMPARISON (like blog's ε comparison)
# =========================================================================
print("\n--- Entropic Regularization Comparison ---")
regs = [0.5, 0.1, 0.02]
n_reg = len(regs)

fig, axes = plt.subplots(2, n_reg, figsize=(5 * n_reg, 8))

for idx, reg_val in enumerate(regs):
    T_r = ot.sinkhorn(p_drug, p_mut, C_norm, reg_val, numItermax=5000)
    wd_r = np.sqrt(np.sum(T_r * C_1d))
    # Top: transport plan
    ax = axes[0, idx]
    T_log = np.log(T_r + 1e-10)
    vmin = np.percentile(T_log, 5)
    vmax = np.percentile(T_log, 95)
    im = ax.imshow(T_log, aspect='auto', cmap='hot', origin='lower',
                   extent=[x_min, x_max, x_min, x_max],
                   vmin=vmin, vmax=vmax)
    ax.set_title(f'ε={reg_val}  WD={wd_r:.3f}', fontsize=11)
    ax.set_xlabel('Mutant')
    ax.set_ylabel('Drug')
    plt.colorbar(im, ax=ax, shrink=0.7)

    # Bottom: marginal densities + transport overlay
    ax = axes[1, idx]
    ax.plot(bin_centers, p_drug, 'b-', linewidth=2, label='Drug')
    ax.plot(bin_centers, p_mut, 'orange', linewidth=2, label='Mutant')
    # Shade transported mass
    for i in range(n_bins):
        for j in range(n_bins):
            if T_r[i, j] > 0.005:
                ax.plot([bin_centers[i], bin_centers[j]],
                        [p_drug[i] + 0.01, p_mut[j] + 0.01],
                        'r-', alpha=T_r[i, j] * 3, linewidth=1)
    ax.set_xlabel('1D projection')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

plt.suptitle('Effect of Entropic Regularization on Transport Plan', fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_entropic_regularization.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Entropic regularization comparison saved")

# =========================================================================
# 4. DISCRETE CLASS-TO-CLASS TRANSPORT MAP
# =========================================================================
print("\n--- Drug-to-Mutant Class Transport Map ---")
# Get per-class mean latent codes
unique_cls = sorted(np.unique(labels))
cls_z = {}
cls_name_map = {}
drug_pattern = re.compile(r'_\d+(\.\d+)?x$')
for cls in unique_cls:
    mask = labels == cls
    name = str(class_names[cls])
    cls_name_map[cls] = name
    cls_z[cls] = z[mask]

# Select top drug and mutant classes by sample count
drug_cls = [c for c in unique_cls if drug_pattern.search(cls_name_map[c]) or (cls_name_map[c] == 'control')]
mut_cls = [c for c in unique_cls if not (drug_pattern.search(cls_name_map[c]) or (cls_name_map[c] == 'control'))]
drug_cls = sorted(drug_cls, key=lambda c: len(cls_z[c]), reverse=True)[:10]
mut_cls = sorted(mut_cls, key=lambda c: len(cls_z[c]), reverse=True)[:10]

# Build cost matrix between class centroids
drug_centroids = np.array([cls_z[c].mean(axis=0) for c in drug_cls])
mut_centroids = np.array([cls_z[c].mean(axis=0) for c in mut_cls])

C_cls = ot.dist(drug_centroids, mut_centroids, metric='sqeuclidean')

# Uniform weights
p_cls = np.ones(len(drug_cls)) / len(drug_cls)
q_cls = np.ones(len(mut_cls)) / len(mut_cls)

# Transport plan
T_cls = ot.sinkhorn(p_cls, q_cls, C_cls, reg=0.1, numItermax=1000)
wd_cls = np.sqrt(np.sum(T_cls * C_cls))

drug_names_short = [str(class_names[c])[:18] for c in drug_cls]
mut_names_short = [str(class_names[c])[:18] for c in mut_cls]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Panel 1: Cost matrix
ax = axes[0]
im = ax.imshow(np.sqrt(C_cls), aspect='auto', cmap='viridis')
plt.colorbar(im, ax=ax, shrink=0.8, label='√Cost')
ax.set_xticks(range(len(mut_cls)))
ax.set_yticks(range(len(drug_cls)))
ax.set_xticklabels(mut_names_short, rotation=90, fontsize=7)
ax.set_yticklabels(drug_names_short, fontsize=7)
ax.set_xlabel('Mutant classes')
ax.set_ylabel('Drug classes')
ax.set_title('Pairwise distance (sqrt cost)', fontsize=13)

# Panel 2: Transport plan
ax = axes[1]
T_clipped = np.clip(T_cls, 1e-10, None)
im = ax.imshow(T_clipped, aspect='auto', cmap='hot')
plt.colorbar(im, ax=ax, shrink=0.8, label='Transport mass')
ax.set_xticks(range(len(mut_cls)))
ax.set_yticks(range(len(drug_cls)))
ax.set_xticklabels(mut_names_short, rotation=90, fontsize=7)
ax.set_yticklabels(drug_names_short, fontsize=7)
ax.set_xlabel('Mutant classes')
ax.set_ylabel('Drug classes')
ax.set_title(f'Transport plan (WD={wd_cls:.3f})', fontsize=13)

# Add arrows for top flows
for i in range(len(drug_cls)):
    for j in range(len(mut_cls)):
        if T_cls[i, j] > 0.15:
            ax.annotate('', xy=(j, i), xytext=(j, i),
                        arrowprops=dict(arrowstyle='->', color='cyan', lw=2))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_class_to_class_transport.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  Class-to-class transport map saved")

print(f"\nAll plots saved to: {OUTPUT_DIR}")
