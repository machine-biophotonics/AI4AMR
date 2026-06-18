#!/usr/bin/env python3
"""
Remove PC1 from per-image features and re-cluster to find
shared drug+mutant phenotype clusters.
Fixed: infer types from name patterns (class_types.npy is broken).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import hdbscan
from collections import Counter, defaultdict
import os, json, re

def infer_type(name):
    name = str(name)
    if name == 'control' or 'NC_' in name or 'WT NC' in name:
        return 'Control'
    if re.search(r'_\d+\.?\d*x$', name):
        return 'Drug'
    return 'Mutant'

OUT_DIR = "interpretable_directions/pc1_removed"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ──
feats = np.load("latent_analysis_pacmap/feats_t10_cond.npy").astype(np.float64)
labels = np.load("latent_analysis_pacmap/labels.npy")
class_names = np.load("latent_analysis_pacmap/class_names.npy", allow_pickle=True)

N, D = feats.shape
print(f"Features: {N} images x {D} dim")

# Build per-class type map from names
class_type_map = {i: infer_type(n) for i, n in enumerate(class_names)}
type_counts = Counter(class_type_map.values())
print(f"Class composition: {dict(type_counts)}")

# ── PCA on per-image features ──
feats_c = feats - feats.mean(axis=0, keepdims=True)
pca = PCA(n_components=20)
feats_pca = pca.fit_transform(feats_c)
var_ratio = pca.explained_variance_ratio_
print(f"PC1: {var_ratio[0]:.3%}, PC2: {var_ratio[1]:.3%}, PC3: {var_ratio[2]:.3%}")

# ── Remove PC1 ──
pc1_recon = feats_pca[:, 0:1] @ pca.components_[0:1]
feats_no_pc1 = feats_c - pc1_recon

# ── HDBSCAN on PC2-PC10 subspace ──
print("\n=== HDBSCAN on PC2-PC10 ===")
feats_pc2_10 = feats_pca[:, 1:11]

rng = np.random.RandomState(42)
idx = rng.choice(N, min(N, 15000), replace=False)
feats_sub = feats_pc2_10[idx]
labels_sub = labels[idx]

for mc in [10, 20, 30]:
    for ms in [3, 5, 10]:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mc, min_samples=ms,
            metric='euclidean', cluster_selection_epsilon=0.5
        )
        cl = clusterer.fit_predict(feats_sub)
        n_cl = len(set(cl) - {-1})
        n_noise = (cl == -1).sum()
        print(f"  mc={mc:2d} ms={ms:2d}: {n_cl} clusters, {n_noise} noise")

mc, ms = 20, 10
print(f"\nRunning full HDBSCAN with mc={mc}, ms={ms}...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=mc, min_samples=ms,
    metric='euclidean', cluster_selection_epsilon=0.5, prediction_data=True
)
cluster_labels = clusterer.fit_predict(feats_sub)

def analyze_clusters(cluster_labels, idx, labels, class_names, class_type_map):
    """Analyze clusters and return mixed cluster info."""
    cluster_info = defaultdict(lambda: defaultdict(lambda: Counter()))
    ci = defaultdict(int)
    for i, cl in enumerate(cluster_labels):
        if cl == -1: continue
        orig_idx = idx[i]
        cls_idx = int(labels[orig_idx])
        name = str(class_names[cls_idx])
        ctype = class_type_map[cls_idx]
        cluster_info[cl][ctype][name] += 1
        ci[cl] += 1

    n_clusters = len(cluster_info)
    mixed = []
    for cl in sorted(cluster_info.keys()):
        info = cluster_info[cl]
        nd = len(info.get('Drug', {}))
        nm = len(info.get('Mutant', {}))
        nc = len(info.get('Control', {}))
        types_present = []
        if nd > 0: types_present.append(f"{nd} drugs")
        if nm > 0: types_present.append(f"{nm} mutants")
        if nc > 0: types_present.append(f"{nc} controls")
        is_mixed = (nd > 0 and nm > 0)
        if is_mixed: mixed.append(cl)
        marker = " ★ MIXED" if is_mixed else ""
        print(f"\nCluster {cl} ({ci[cl]} imgs): {', '.join(types_present)}{marker}")
        if nd > 0:
            print(f"  Drugs: {', '.join(sorted(info['Drug'].keys()))}")
        if nm > 0:
            print(f"  Mutants: {', '.join(sorted(info['Mutant'].keys()))}")
        if nc > 0:
            print(f"  Controls: {', '.join(sorted(info['Control'].keys()))}")
    return mixed, cluster_info, ci

n_clusters = len(set(cluster_labels) - {-1})
n_noise = (cluster_labels == -1).sum()
print(f"\nHDBSCAN: {n_clusters} clusters + {n_noise} noise")
mixed_hdb, hdb_info, hdb_counts = analyze_clusters(cluster_labels, idx, labels, class_names, class_type_map)
print(f"\nHDBSCAN mixed clusters: {len(mixed_hdb)}/{n_clusters}")

# ── GMM on PC2-PC10 ──
print(f"\n\n{'='*60}")
print("=== GMM on PC2-PC10 ===")
print(f"{'='*60}")

n_clusters_range = list(range(5, 31, 5)) + [40, 50]
best_bic = np.inf
best_gmm = None
best_n = 0

for k in n_clusters_range:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
    gmm.fit(feats_pc2_10)
    bic = gmm.bic(feats_pc2_10)
    print(f"  GMM k={k:2d}: BIC={bic:.1f}")
    if bic < best_bic:
        best_bic = bic
        best_gmm = gmm
        best_n = k

print(f"\nBest GMM: k={best_n}")
gmm_labels = best_gmm.predict(feats_pc2_10)

# Use full dataset for GMM
full_idx = np.arange(N)
mixed_gmm, gmm_info, gmm_counts = analyze_clusters(gmm_labels, full_idx, labels, class_names, class_type_map)
print(f"\nGMM mixed clusters: {len(mixed_gmm)}/{best_n}")

# ── Detail on mixed clusters ──
print(f"\n{'='*60}")
print("DETAILED SUMMARY OF MIXED CLUSTERS")
print(f"{'='*60}")

for cl in mixed_gmm:
    info = gmm_info[cl]
    nd = len(info.get('Drug', {}))
    nm = len(info.get('Mutant', {}))
    nc = len(info.get('Control', {}))
    print(f"\nGMM Cluster {cl} ({gmm_counts[cl]} imgs): {nd} drugs + {nm} mutants + {nc} controls")
    # List drugs with counts
    print(f"  Top drugs: {', '.join(sorted(info['Drug'].keys())[:15])}")
    print(f"  Mutants: {', '.join(sorted(info['Mutant'].keys()))}")
    if nc > 0:
        print(f"  Controls: {', '.join(sorted(info['Control'].keys()))}")

# ── Visualize PC1-PC2 colored by inferred type ──
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
colors_map = {'Drug': 'red', 'Mutant': 'blue', 'Control': 'green'}
for ax_idx, (x_pc, y_pc) in enumerate([(0, 1), (1, 2), (2, 3)]):
    ax = axes[ax_idx]
    for ctype, color in colors_map.items():
        mask = np.array([class_type_map[int(l)] == ctype for l in labels])
        ax.scatter(feats_pca[mask, x_pc], feats_pca[mask, y_pc],
                  c=color, label=ctype, s=3, alpha=0.3)
    ax.set_xlabel(f"PC{x_pc+1} ({var_ratio[x_pc]:.1%})")
    ax.set_ylabel(f"PC{y_pc+1} ({var_ratio[y_pc]:.1%})")
    ax.legend(markerscale=5)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/pca_by_type.png", dpi=150)
plt.close()

# Save
np.save(f"{OUT_DIR}/feats_no_pc1.npy", feats_no_pc1)
np.save(f"{OUT_DIR}/cluster_labels_hdbscan.npy", cluster_labels)
np.save(f"{OUT_DIR}/cluster_labels_gmm.npy", gmm_labels)

print(f"\nSaved to {OUT_DIR}/")
