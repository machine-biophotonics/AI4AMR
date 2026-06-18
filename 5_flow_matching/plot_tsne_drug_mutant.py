#!/usr/bin/env python3
"""t-SNE of bottleneck features with PC1 removed, showing drug-mutant relationships."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict
import re, os

OUT_DIR = "interpretable_directions/pc1_removed"
os.makedirs(OUT_DIR, exist_ok=True)

def infer_type(name):
    name = str(name)
    if name == 'control' or 'NC_' in name or 'WT NC' in name:
        return 'Control'
    if re.search(r'_\d+\.?\d*x$', name):
        return 'Drug'
    return 'Mutant'

# ── Load ──
feats = np.load("latent_analysis_pacmap/feats_t10_cond.npy").astype(np.float64)
labels = np.load("latent_analysis_pacmap/labels.npy")
class_names = np.load("latent_analysis_pacmap/class_names.npy", allow_pickle=True)

N = feats.shape[0]
class_type_map = {i: infer_type(n) for i, n in enumerate(class_names)}
class_name_map = {i: str(n) for i, n in enumerate(class_names)}

# PCA + remove PC1
print("Computing PCA...")
feats_c = feats - feats.mean(axis=0, keepdims=True)
pca = PCA(n_components=20)
feats_pca = pca.fit_transform(feats_c)
pc1_recon = feats_pca[:, 0:1] @ pca.components_[0:1]
feats_no_pc1 = feats_c - pc1_recon

# Subsample for t-SNE (15k would be slow, use 10k)
rng = np.random.RandomState(42)
n_plot = min(N, 10000)
idx = rng.choice(N, n_plot, replace=False)
feats_plot = feats_no_pc1[idx]

print(f"Running t-SNE on {n_plot} points (PC1-removed features)...")
tsne = TSNE(n_components=2, perplexity=50, random_state=42, verbose=1)
coords = tsne.fit_transform(feats_plot)
print("t-SNE done.")

# Get types for plotted points
plot_types = [class_type_map[int(labels[i])] for i in idx]
plot_names = [class_name_map[int(labels[i])] for i in idx]

# Build per-class centroid in t-SNE space
class_centroids = defaultdict(list)
for i, (t, n) in enumerate(zip(plot_types, plot_names)):
    class_centroids[n].append(coords[i])

centroid_coords = {}
centroid_types = {}
for n, pts in class_centroids.items():
    centroid_coords[n] = np.mean(pts, axis=0)
    centroid_types[n] = infer_type(n)

# ── FIGURE 1: t-SNE colored by type ──
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

color_map = {'Drug': '#E74C3C', 'Mutant': '#3498DB', 'Control': '#2ECC71'}
for ctype in ['Drug', 'Mutant', 'Control']:
    mask = np.array([t == ctype for t in plot_types])
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=color_map[ctype], label=ctype, s=4, alpha=0.3, edgecolors='none')

# Plot centroids with larger markers
for n, pt in centroid_coords.items():
    ctype = centroid_types[n]
    if ctype == 'Drug':
        ax.scatter(pt[0], pt[1], c=color_map[ctype], s=40, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=5)
    elif ctype == 'Mutant':
        ax.scatter(pt[0], pt[1], c=color_map[ctype], s=40, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=5)
    elif ctype == 'Control':
        ax.scatter(pt[0], pt[1], c=color_map[ctype], s=40, alpha=0.8, edgecolors='black', linewidth=0.5, zorder=5)

# Connect drugs to mutants that share a cluster (from GMM)
gmm_labels = np.load(f"{OUT_DIR}/cluster_labels_gmm.npy")

cluster_composition = defaultdict(lambda: {"Drug": set(), "Mutant": set()})
for i in range(N):
    cls_idx = int(labels[i])
    cl = int(gmm_labels[i])
    ctype = class_type_map[cls_idx]
    name = class_name_map[cls_idx]
    if ctype == 'Drug':
        cluster_composition[cl]["Drug"].add(name)
    elif ctype == 'Mutant':
        cluster_composition[cl]["Mutant"].add(name)

# For each centroid pair draw a thin line if they share a specific cluster
# (not promiscuous clusters with >50% of all types)
total_drugs = len([i for i in range(185) if class_type_map[i] == 'Drug'])
total_mutants = len([i for i in range(185) if class_type_map[i] == 'Mutant'])

connections = set()
for cl, comp in cluster_composition.items():
    nd = len(comp["Drug"])
    nm = len(comp["Mutant"])
    if nd > 0 and nm > 0 and nd < total_drugs * 0.5 and nm < total_mutants * 0.5:
        for d in comp["Drug"]:
            for m in comp["Mutant"]:
                if d in centroid_coords and m in centroid_coords:
                    connections.add((d, m))

print(f"Drawing {len(connections)} drug-mutant connections from specific clusters...")
for d, m in list(connections)[:500]:  # limit to 500 for clarity
    if d in centroid_coords and m in centroid_coords:
        ax.plot([centroid_coords[d][0], centroid_coords[m][0]],
                [centroid_coords[d][1], centroid_coords[m][1]],
                c='gray', alpha=0.15, linewidth=0.3, zorder=1)

ax.legend(fontsize=12, markerscale=3)
ax.set_title("t-SNE of PC1-removed bottleneck features\n(Drug=red, Mutant=blue, Control=green, lines=shared GMM cluster)", fontsize=13)
ax.axis('off')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/tsne_drug_mutant.png", dpi=200)
plt.close()
print(f"Saved {OUT_DIR}/tsne_drug_mutant.png")

# ── FIGURE 2: t-SNE colored by GMM cluster (40 clusters) ──
fig, ax = plt.subplots(1, 1, figsize=(14, 10))

cluster_labels_sub = gmm_labels[idx]
cluster_colors = plt.cm.tab20(np.linspace(0, 1, 20))
# Extend with tab20b and tab20c
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
all_colors = list(plt.cm.tab20(np.linspace(0, 1, 20)))
all_colors += list(plt.cm.tab20b(np.linspace(0, 1, 20)))
unique_cls = sorted(set(cluster_labels_sub) - {-1})
cl_to_color = {}
for ci, cl in enumerate(unique_cls):
    cl_to_color[cl] = all_colors[ci % len(all_colors)]

# Noise points
noise_mask = cluster_labels_sub == -1
ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
           c='gray', s=4, alpha=0.2, edgecolors='none', label='noise')

# Non-noise points
for cl in unique_cls:
    mask = cluster_labels_sub == cl
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=[cl_to_color[cl]], s=6, alpha=0.4, edgecolors='none')

ax.legend(unique_cls, fontsize=6, ncol=4)
ax.set_title("t-SNE colored by GMM cluster (after PC1 removal)", fontsize=13)
ax.axis('off')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/tsne_gmm_clusters.png", dpi=200)
plt.close()
print(f"Saved {OUT_DIR}/tsne_gmm_clusters.png")

# ── FIGURE 3: label zoom on ftsZ and β-lactam drugs ──
fig, ax = plt.subplots(1, 1, figsize=(16, 12))

for ctype, color in [('Drug', '#E74C3C'), ('Mutant', '#3498DB'), ('Control', '#2ECC71')]:
    mask = np.array([t == ctype for t in plot_types])
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=color, s=2, alpha=0.15, edgecolors='none')

# Label specific classes
highlight_drugs = {'Aztreonam_2x', 'Ceftriaxone_2x', 'Cefepim_2x', 'Penicillin_2x', 
                   'Sulbactam_2x', 'Meropenem_2x', 'Ceftriaxone_1x', 'Aztreonam_1x'}
highlight_mutants = {'ftsZ_1', 'ftsZ_2', 'ftsZ_3', 'ftsI_1', 'ftsI_2', 'ftsI_3',
                     'murC_1', 'murC_2', 'murC_3', 'lpxC_1', 'lpxC_2', 'lpxC_3'}

for name, pt in centroid_coords.items():
    if name in highlight_drugs:
        ax.scatter(pt[0], pt[1], c=color_map['Drug'], s=80, alpha=0.9, 
                   edgecolors='black', linewidth=0.8, zorder=10)
        ax.annotate(name, pt, fontsize=6, alpha=0.8, ha='left', va='bottom')
    elif name in highlight_mutants:
        ax.scatter(pt[0], pt[1], c=color_map['Mutant'], s=80, alpha=0.9,
                   edgecolors='black', linewidth=0.8, zorder=10)
        ax.annotate(name, pt, fontsize=6, alpha=0.8, ha='left', va='bottom')

ax.set_title("t-SNE: Highlighting β-lactam drugs (red) + ftsZ/ftsI/murC mutants (blue)", fontsize=12)
ax.axis('off')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/tsne_labeled_highlight.png", dpi=200)
plt.close()
print(f"Saved {OUT_DIR}/tsne_labeled_highlight.png")

# ── FIGURE 4: what the user asked — drug class vs mutant class proximity ──
# For each drug centroid, find nearest mutant centroids
print("\nComputing drug→mutant nearest neighbors in t-SNE space...")

drug_centroids = {n: pt for n, pt in centroid_coords.items() if centroid_types[n] == 'Drug'}
mutant_centroids = {n: pt for n, pt in centroid_coords.items() if centroid_types[n] == 'Mutant'}

drug_mutant_nn = {}
for dname, dpt in drug_centroids.items():
    distances = [(mname, np.linalg.norm(dpt - mpt)) for mname, mpt in mutant_centroids.items()]
    distances.sort(key=lambda x: x[1])
    drug_mutant_nn[dname] = distances[:5]

# Write to file
with open(f"{OUT_DIR}/drug_nearest_mutants.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("DRUG → NEAREST MUTANTS IN t-SNE SPACE (PC1 removed)\n")
    f.write("="*80 + "\n\n")
    for dname in sorted(drug_mutant_nn.keys()):
        f.write(f"{dname}:\n")
        for mname, dist in drug_mutant_nn[dname]:
            f.write(f"  → {mname} (dist={dist:.3f})\n")
        f.write("\n")

    # Orphan drugs: those whose nearest mutant is far
    f.write("="*80 + "\n")
    f.write("ORPHAN DRUGS (furthest from any mutant in t-SNE)\n")
    f.write("="*80 + "\n\n")
    sorted_by_dist = sorted(drug_mutant_nn.items(), key=lambda x: -x[1][0][1])
    for dname, nns in sorted_by_dist[:20]:
        f.write(f"  {dname}: nearest mutant = {nns[0][0]} (dist={nns[0][1]:.3f})\n")

print(f"Saved {OUT_DIR}/drug_nearest_mutants.txt")
print("\nDone.")
