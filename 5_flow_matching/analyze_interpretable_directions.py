#!/usr/bin/env python3
"""Discover interpretable directions + shared drug/mutant phenotypes from bottleneck features.

Usage:
    python3 analyze_interpretable_directions.py
    python3 analyze_interpretable_directions.py --features latent_analysis_pacmap --checkpoint auto

Steps:
    1. PCA on class embeddings (from model checkpoint) → major variation axes
    2. PCA on per-class bottleneck centroids (from extracted features)
    3. HDBSCAN clustering on centroids → discover shared drug+mutant phenotype clusters
    4. For each PC, list top/bottom classes
    5. Save all results for downstream validation
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
import umap
import argparse

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--features_dir', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'latent_analysis_pacmap'))
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to flow_best.pth (auto-detect)')
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--dims', type=int, default=256, help='Bottleneck feature dimension')
parser.add_argument('--n_pcs', type=int, default=10, help='Number of PCs to analyze')
parser.add_argument('--min_cluster_size', type=int, default=3, help='HDBSCAN min cluster size')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = args.output_dir or os.path.join(SCRIPT_DIR, 'interpretable_directions')
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("Interpretable Directions + Phenotype Discovery")
print(f"Output: {output_dir}")
print("=" * 60)

# ── 1. Load existing bottleneck features ────────────────────────────────
print("\n[1/6] Loading existing bottleneck features ...")
feats_path = os.path.join(args.features_dir, 'feats_t10_cond.npy')
labels_path = os.path.join(args.features_dir, 'labels.npy')
names_path = os.path.join(args.features_dir, 'class_names.npy')
types_path = os.path.join(args.features_dir, 'class_types.npy')

feats = np.load(feats_path).astype(np.float32)
labels = np.load(labels_path)
class_names = np.load(names_path, allow_pickle=True)
class_types = np.load(types_path)
print(f"  Features: {feats.shape}")
print(f"  Classes: {len(np.unique(labels))}")

# Map: class name → short name (drug: remove dose, mutant: remove guide)
def short_name(name, ctype):
    if ctype == 0:
        parts = name.rsplit('_', 1)
        return parts[0]
    elif ctype == 1:
        parts = name.rsplit('_', 1)
        return parts[0] if len(parts) > 1 else name
    return name

n_classes = len(class_names)
type_counts = np.bincount(class_types[labels])
type_names = ['Drug', 'Mutant', 'Control']
for t in range(3):
    print(f"  {type_names[t]}: {type_counts[t] if t < len(type_counts) else 0} images")

# ── 2. Compute per-class centroids ─────────────────────────────────────
print("\n[2/6] Computing per-class centroids ...")
centroids = np.zeros((n_classes, args.dims), dtype=np.float32)
class_type_ids = np.zeros(n_classes, dtype=np.int32)
for cid in range(n_classes):
    mask = labels == cid
    if mask.sum() > 0:
        centroids[cid] = feats[mask].mean(0)
        class_type_ids[cid] = class_types[mask][0]
    else:
        class_type_ids[cid] = 2
print(f"  Centroids: {centroids.shape}")
for t in range(3):
    n = (class_type_ids == t).sum()
    print(f"    {type_names[t]}: {n} classes")

# ── 3. Load checkpoint → extract class embeddings ──────────────────────
print("\n[3/6] Loading checkpoint for class embeddings ...")
if args.checkpoint is None:
    run_dirs = sorted([d for d in os.listdir(SCRIPT_DIR)
                       if d.startswith('flow_run_') and os.path.isdir(os.path.join(SCRIPT_DIR, d))])
    for rd in reversed(run_dirs):
        candidate = os.path.join(SCRIPT_DIR, rd, 'flow_best.pth')
        if os.path.exists(candidate):
            args.checkpoint = candidate
            break

if args.checkpoint is None:
    print("  WARNING: No checkpoint found. Skipping class embedding analysis.")
    class_emb = None
else:
    print(f"  Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']

    emb_key = [k for k in sd.keys() if 'class_embedding.weight' in k and 'main' in k]
    if emb_key:
        class_emb = sd[emb_key[0]].numpy()
        print(f"  Class embeddings: {class_emb.shape}")
    else:
        emb_key = [k for k in sd.keys() if 'class_embedding.weight' in k]
        if emb_key:
            class_emb = sd[emb_key[0]].numpy()
            print(f"  Class embeddings: {class_emb.shape} (from {emb_key[0]})")
        else:
            print("  WARNING: No class embeddings found.")
            class_emb = None

    n_params = sum(p.numel() for p in ckpt['model_state_dict'].values())
    print(f"  Model params: {n_params:,}")

# ── 4. PCA on class embeddings ─────────────────────────────────────────
if class_emb is not None:
    print("\n[4/6] PCA on class embeddings ...")
    emb_scaler = StandardScaler(with_mean=True, with_std=False)
    emb_scaled = emb_scaler.fit_transform(class_emb)
    emb_pca = PCA(n_components=min(args.n_pcs, class_emb.shape[1]))
    emb_pcs = emb_pca.fit_transform(emb_scaled)
    print(f"  Explained variance (top-5): {emb_pca.explained_variance_ratio_[:5]}")
    print(f"  Cumulative (top-5): {emb_pca.explained_variance_ratio_[:5].sum():.3f}")

    # Plot: PCA on class embeddings
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = {0: '#E74C3C', 1: '#2ECC71', 2: '#3498DB'}
    for t in range(3):
        mask = class_type_ids == t
        if mask.sum() > 0:
            axes[0].scatter(emb_pcs[mask, 0], emb_pcs[mask, 1],
                           c=colors[t], s=30, alpha=0.7, label=f'{type_names[t]} ({mask.sum()})')
    axes[0].set_xlabel(f'PC1 ({emb_pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({emb_pca.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('Class Embedding PCA (by type)')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # PCA 1 vs 3
    if emb_pcs.shape[1] >= 3:
        for t in range(3):
            mask = class_type_ids == t
            if mask.sum() > 0:
                axes[1].scatter(emb_pcs[mask, 0], emb_pcs[mask, 2],
                               c=colors[t], s=30, alpha=0.7, label=f'{type_names[t]} ({mask.sum()})')
        axes[1].set_xlabel(f'PC1 ({emb_pca.explained_variance_ratio_[0]:.1%})')
        axes[1].set_ylabel(f'PC3 ({emb_pca.explained_variance_ratio_[2]:.1%})')
    axes[1].set_title('Class Embedding PCA (PC1 vs PC3)')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'class_embedding_pca.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: class_embedding_pca.png")

    # Per-PC top/bottom classes
    with open(os.path.join(output_dir, 'embedding_pc_top_classes.txt'), 'w') as f:
        for pc_id in range(min(5, emb_pcs.shape[1])):
            loadings = emb_pca.components_[pc_id]
            scores = emb_pcs[:, pc_id]
            top_idx = np.argsort(scores)[-10:][::-1]
            bot_idx = np.argsort(scores)[:10]
            f.write(f"\n{'='*60}\n")
            f.write(f"PC{pc_id+1} (var={emb_pca.explained_variance_ratio_[pc_id]:.2%})\n")
            f.write(f"{'='*60}\n")
            f.write("  Top 10 (positive loading):\n")
            for i in top_idx:
                f.write(f"    {class_names[i]:40s} [{type_names[class_type_ids[i]]}]\n")
            f.write("  Bottom 10 (negative loading):\n")
            for i in bot_idx:
                f.write(f"    {class_names[i]:40s} [{type_names[class_type_ids[i]]}]\n")

        # Top/bottom for each PC: check for mixed drug+mutant
        f.write(f"\n{'='*60}\n")
        f.write("PCs with mixed Drug+Mutant in top/bottom 10\n")
        f.write(f"{'='*60}\n")
        for pc_id in range(emb_pcs.shape[1]):
            scores = emb_pcs[:, pc_id]
            top10 = np.argsort(scores)[-10:]
            bot10 = np.argsort(scores)[:10]
            top_types = set(class_type_ids[top10])
            bot_types = set(class_type_ids[bot10])
            has_mixed = (0 in top_types and 1 in top_types) or (0 in bot_types and 1 in bot_types)
            if has_mixed:
                f.write(f"\nPC{pc_id+1}: MIXED Drug+Mutant\n")
                f.write("  Top 10:\n")
                for i in top10:
                    f.write(f"    {class_names[i]:40s} [{type_names[class_type_ids[i]]}]\n")
                f.write("  Bottom 10:\n")
                for i in bot10:
                    f.write(f"    {class_names[i]:40s} [{type_names[class_type_ids[i]]}]\n")

    print(f"  Saved: embedding_pc_top_classes.txt")

# ── 5. PCA on bottleneck centroids ────────────────────────────────────
print("\n[5/6] PCA on bottleneck centroids ...")
cent_scaler = StandardScaler(with_mean=True, with_std=False)
cent_scaled = cent_scaler.fit_transform(centroids)
cent_pca = PCA(n_components=min(args.n_pcs, args.dims))
cent_pcs = cent_pca.fit_transform(cent_scaled)
print(f"  Explained variance (top-5): {cent_pca.explained_variance_ratio_[:5]}")
print(f"  Cumulative (top-5): {cent_pca.explained_variance_ratio_[:5].sum():.3f}")

# ── 6. HDBSCAN clustering on centroids ─────────────────────────────────
print("\n[6/6] HDBSCAN clustering on centroids ...")
clusterer = HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=1, metric='euclidean')
cluster_labels = clusterer.fit_predict(cent_scaled)
n_clusters = len(set(cluster_labels) - {-1})
n_noise = (cluster_labels == -1).sum()
print(f"  Clusters: {n_clusters} (+ {n_noise} noise points)")
print(f"  Cluster sizes: {np.bincount(cluster_labels[cluster_labels >= 0])}")

# Per-cluster composition
cluster_composition = {}
for cl in set(cluster_labels):
    mask = cluster_labels == cl
    ids_in_cluster = np.where(mask)[0]
    type_counts_cl = np.bincount(class_type_ids[mask], minlength=3)
    names_in_cluster = [class_names[i] for i in ids_in_cluster]
    cluster_composition[cl] = {
        'n': mask.sum(),
        'drug': int(type_counts_cl[0]),
        'mutant': int(type_counts_cl[1]),
        'control': int(type_counts_cl[2]),
        'names': names_in_cluster,
    }

# Identify mixed clusters (drug + mutant together)
mixed_clusters = {cl: comp for cl, comp in cluster_composition.items()
                  if cl >= 0 and comp['drug'] > 0 and comp['mutant'] > 0}
print(f"\n  Mixed Drug+Mutant clusters: {len(mixed_clusters)}")
for cl, comp in sorted(mixed_clusters.items(), key=lambda x: x[1]['n'], reverse=True):
    print(f"    Cluster {cl}: {comp['drug']}D + {comp['mutant']}M + {comp['control']}C = {comp['n']} total")

# ── Visualizations ─────────────────────────────────────────────────────
colors_map = {0: '#E74C3C', 1: '#2ECC71', 2: '#3498DB'}

# 5a. PCA on centroids (2D grid)
fig = plt.figure(figsize=(20, 18))
gs = GridSpec(2, 2, figure=fig, hspace=0.2, wspace=0.2)

for idx, (pc_x, pc_y) in enumerate([(0, 1), (0, 2), (1, 2), (1, 3)]):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    for t in range(3):
        mask = class_type_ids == t
        if mask.sum() > 0:
            ax.scatter(cent_pcs[mask, pc_x], cent_pcs[mask, pc_y],
                      c=colors_map[t], s=40, alpha=0.7,
                      label=f'{type_names[t]} ({mask.sum()})', edgecolors='white', linewidth=0.3)
    ax.set_xlabel(f'PC{pc_x+1} ({cent_pca.explained_variance_ratio_[pc_x]:.1%})')
    ax.set_ylabel(f'PC{pc_y+1} ({cent_pca.explained_variance_ratio_[pc_y]:.1%})')
    ax.set_title(f'Bottleneck Centroids: PC{pc_x+1} vs PC{pc_y+1}')
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.3)

plt.suptitle('PCA on 185 Class Bottleneck Centroids', fontsize=14, y=0.98)
fig.savefig(os.path.join(output_dir, 'pca_centroids_grid.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: pca_centroids_grid.png")

# 5b. UMAP on centroids
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
cent_umap = reducer.fit_transform(cent_scaled)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
# By type
for t in range(3):
    mask = class_type_ids == t
    if mask.sum() > 0:
        axes[0].scatter(cent_umap[mask, 0], cent_umap[mask, 1],
                       c=colors_map[t], s=50, alpha=0.7,
                       label=f'{type_names[t]} ({mask.sum()})', edgecolors='white', linewidth=0.3)
axes[0].set_title('UMAP: Bottleneck Centroids (by type)')
axes[0].legend(fontsize=10, markerscale=2)
axes[0].grid(alpha=0.3)

# By cluster (only non-noisy)
cluster_cmap = plt.cm.tab20
for cl in sorted(set(cluster_labels)):
    if cl == -1:
        continue
    mask = cluster_labels == cl
    c = cluster_cmap(cl % 20)
    axes[1].scatter(cent_umap[mask, 0], cent_umap[mask, 1],
                   c=[c], s=50, alpha=0.7, label=f'C{cl}',
                   edgecolors='white', linewidth=0.3)
# Noise
noise_mask = cluster_labels == -1
if noise_mask.sum() > 0:
    axes[1].scatter(cent_umap[noise_mask, 0], cent_umap[noise_mask, 1],
                   c='gray', s=20, alpha=0.3, label='noise')
axes[1].set_title(f'UMAP: HDBSCAN clusters ({n_clusters} clusters)')
axes[1].legend(fontsize=8, markerscale=2, ncol=2)
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'umap_centroids.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: umap_centroids.png")

# 5c. Explained variance + cluster compositions
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.bar(range(1, min(11, len(cent_pca.explained_variance_ratio_)+1)),
       cent_pca.explained_variance_ratio_[:10], alpha=0.7, color='steelblue')
ax.plot(range(1, min(11, len(cent_pca.explained_variance_ratio_)+1)),
        np.cumsum(cent_pca.explained_variance_ratio_[:10]), 'ro-', markersize=4)
ax.set_xlabel('PC')
ax.set_ylabel('Explained variance ratio')
ax.set_title('PCA explained variance (bottleneck centroids)')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'pca_variance.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: pca_variance.png")

# ── Per-PC top/bottom classes ─────────────────────────────────────────
with open(os.path.join(output_dir, 'bottleneck_pc_top_classes.txt'), 'w') as f:
    f.write(f"{'='*80}\n")
    f.write("BOTTLENECK CENTROID PCA: Top/Bottom Classes per PC\n")
    f.write(f"{'='*80}\n")
    for pc_id in range(min(10, cent_pcs.shape[1])):
        scores = cent_pcs[:, pc_id]
        top_idx = np.argsort(scores)[-10:][::-1]
        bot_idx = np.argsort(scores)[:10]
        f.write(f"\n{'─'*80}\n")
        f.write(f"PC{pc_id+1} (var={cent_pca.explained_variance_ratio_[pc_id]:.2%})\n")
        f.write(f"{'─'*80}\n")
        f.write(f"  Top 10 (+):\n")
        for i in top_idx:
            short = class_names[i][:45]
            f.write(f"    {short:45s} [{type_names[class_type_ids[i]]:8s}] score={scores[i]:+.3f}\n")
        f.write(f"  Bottom 10 (-):\n")
        for i in bot_idx:
            short = class_names[i][:45]
            f.write(f"    {short:45s} [{type_names[class_type_ids[i]]:8s}] score={scores[i]:+.3f}\n")

    # Detect mixed PCs
    f.write(f"\n{'='*80}\n")
    f.write("PCs WITH MIXED DRUG+MUTANT in top/bottom 10\n")
    f.write(f"{'='*80}\n")
    for pc_id in range(cent_pcs.shape[1]):
        scores = cent_pcs[:, pc_id]
        top10 = np.argsort(scores)[-10:]
        bot10 = np.argsort(scores)[:10]
        top_types = set(class_type_ids[top10])
        bot_types = set(class_type_ids[bot10])
        has_mixed = (0 in top_types and 1 in top_types) or (0 in bot_types and 1 in bot_types)
        if has_mixed:
            f.write(f"\n  PC{pc_id+1} — MIXED! Drug+Mutant on same axis\n")
            f.write(f"    Top 10 (+): ")
            f.write(', '.join([class_names[i][:20] for i in top10]))
            f.write('\n')
            f.write(f"    Bot 10 (-): ")
            f.write(', '.join([class_names[i][:20] for i in bot10]))
            f.write('\n')

print(f"  Saved: bottleneck_pc_top_classes.txt")

# ── Cluster composition report ─────────────────────────────────────────
with open(os.path.join(output_dir, 'phenotype_clusters.txt'), 'w') as f:
    f.write(f"{'='*80}\n")
    f.write("PHENOTYPE CLUSTERS (from HDBSCAN on bottleneck centroids)\n")
    f.write(f"{'='*80}\n")
    f.write(f"min_cluster_size={args.min_cluster_size}\n")
    f.write(f"{n_clusters} clusters + {n_noise} noise\n\n")

    # Mixed clusters first
    f.write(f"─── MIXED Drug+Mutant Phenotype Clusters ───\n\n")
    for cl, comp in sorted(mixed_clusters.items(), key=lambda x: x[1]['n'], reverse=True):
        f.write(f"Cluster {cl}: {comp['drug']} Drugs + {comp['mutant']} Mutants + {comp['control']} Controls = {comp['n']} total\n")
        f.write(f"  " + "-"*60 + "\n")
        for name in comp['names']:
            ctype = class_type_ids[int(np.where(class_names == name)[0][0])]
            f.write(f"  [{type_names[ctype]:8s}] {name}\n")
        f.write("\n")

    # All clusters
    f.write(f"\n─── ALL CLUSTERS ───\n\n")
    for cl in sorted(set(cluster_labels)):
        comp = cluster_composition[cl]
        label = f"Cluster {cl}" if cl >= 0 else "Noise"
        f.write(f"{label}: {comp['drug']}D + {comp['mutant']}M + {comp['control']}C = {comp['n']} total\n")
        for name in comp['names']:
            ctype = class_type_ids[int(np.where(class_names == name)[0][0])]
            f.write(f"  [{type_names[ctype]:8s}] {name}\n")
        f.write("\n")

print(f"  Saved: phenotype_clusters.txt")

# ── Contrastive directions for each mixed cluster ─────────────────────
if mixed_clusters:
    print(f"\n  Computing contrastive directions for {len(mixed_clusters)} mixed clusters ...")
    contrastive_dirs = []
    for cl in sorted(mixed_clusters.keys()):
        mask = cluster_labels == cl
        cluster_centroid = centroids[mask].mean(0)
        global_centroid = centroids.mean(0)
        direction = cluster_centroid - global_centroid

        # Project all centroids onto this direction
        projections = cent_scaler.transform(centroids) @ cent_scaler.transform(direction.reshape(1, -1).astype(np.float32)).T
        projections = projections.flatten()
        top_by_proj = np.argsort(projections)[-15:][::-1]
        bot_by_proj = np.argsort(projections)[:5]

        contrastive_dirs.append({
            'cluster': cl,
            'direction_vector': direction,
            'top_projection': [class_names[i] for i in top_by_proj],
            'bottom_projection': [class_names[i] for i in bot_by_proj],
        })

    with open(os.path.join(output_dir, 'contrastive_directions.txt'), 'w') as f:
        f.write(f"{'='*80}\n")
        f.write("CONTRASTIVE DIRECTIONS\n")
        f.write("For each mixed cluster, the direction from global centroid to cluster centroid.\n")
        f.write("Project all 185 classes onto this direction to discover new members.\n")
        f.write(f"{'='*80}\n")
        for cd in contrastive_dirs:
            f.write(f"\nCluster {cd['cluster']} direction:\n")
            f.write(f"  Top 15 (most aligned with this phenotype):\n")
            for i, name in enumerate(cd['top_projection']):
                cid = int(np.where(class_names == name)[0][0])
                ct = class_type_ids[cid]
                f.write(f"    {name:45s} [{type_names[ct]:8s}]\n")
            f.write(f"  Bottom 5 (most opposed):\n")
            for name in cd['bottom_projection']:
                cid = int(np.where(class_names == name)[0][0])
                ct = class_type_ids[cid]
                f.write(f"    {name:45s} [{type_names[ct]:8s}]\n")

    print(f"  Saved: contrastive_directions.txt")

# ── Save key outputs for downstream use ────────────────────────────────
np.savez(os.path.join(output_dir, 'analysis_results.npz'),
         centroids=centroids,
         centroids_pca=cent_pcs,
         centroids_umap=cent_umap,
         class_type_ids=class_type_ids,
         cluster_labels=cluster_labels,
         pca_components=cent_pca.components_,
         pca_explained_var=cent_pca.explained_variance_ratio_)

print(f"\n  Saved: analysis_results.npz")
print(f"\n{'='*60}")
print(f"Done. All outputs in: {output_dir}")
print(f"  Key files:")
print(f"    pca_centroids_grid.png    — PCA on 185 bottleneck centroids")
print(f"    umap_centroids.png         — UMAP colored by type + clusters")
print(f"    bottleneck_pc_top_classes.txt — Top/bottom classes per PC")
print(f"    phenotype_clusters.txt     — HDBSCAN cluster assignments")
print(f"    contrastive_directions.txt — Directions for each mixed cluster")
print(f"    analysis_results.npz       — All data for downstream use")
print(f"{'='*60}")
