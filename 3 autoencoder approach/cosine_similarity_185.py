#!/usr/bin/env python3
"""185×185 cosine similarity of mean class embeddings (bag features)."""
import os, sys, warnings, re
warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/3 autoencoder approach/mil_vae_both/fold_P1'
LATENTS_PATH = os.path.join(OUTPUT_DIR, 'test_latents_P1_20260523_222527.pt')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MOA_GROUPS = {
    'Fluoroquinolones': {'drugs': ['ciprofloxacin', 'levofloxacin', 'norfloxacin'],
                         'targets': ['gyra', 'gyrb', 'parc', 'pare']},
    'Rifamycins': {'drugs': ['rifampicin'], 'targets': ['rpoa', 'rpob']},
    'Folate_inhibitors': {'drugs': ['trimethoprim'], 'targets': ['fola', 'folp']},
    'Ribosome_50S': {'drugs': ['chloramphenicol', 'clarithromycin'], 'targets': ['rpla', 'rplc']},
    'Ribosome_30S': {'drugs': ['doxicyclin', 'kanamycin'], 'targets': ['rpsa', 'rpsl']},
    'Penems': {'drugs': ['penicillin', 'mecillinam', 'meropenem'], 'targets': ['mrca', 'mrcb', 'mrda', 'ftsi']},
    'Cephalosporins': {'drugs': ['cefepim', 'cefsulodin', 'ceftriaxone'], 'targets': ['ftsi', 'mrca', 'mrcb', 'mrda']},
    'Polymyxins': {'drugs': ['polymyxin_b', 'colistin'], 'targets': ['lpxa', 'lpxc', 'lpta', 'lptc', 'msba']},
}

print("=" * 60)
print("185×185 Cosine Similarity of class embeddings")
print("=" * 60)

pt = torch.load(LATENTS_PATH, map_location='cpu', weights_only=False)
records = pt['records']

# Accumulate bag features per class
class_bags = defaultdict(list)
class_sources = {}
for r in records:
    lbl = r['true_label']
    bag = r['bag'].astype(np.float64)  # (100, 1280)
    class_bags[lbl].append(bag)
    class_sources[lbl] = r['source']

# Compute mean bag per class
classes = sorted(class_bags.keys())
N = len(classes)
print(f"\n{len(classes)} classes")
mean_bags = np.zeros((N, 1280), dtype=np.float64)
for i, c in enumerate(classes):
    all_bags = np.concatenate(class_bags[c], axis=0)  # (N_i * 100, 1280)
    mean_bags[i] = all_bags.mean(axis=0)

# Normalize for cosine similarity
norms = np.linalg.norm(mean_bags, axis=1, keepdims=True)
norms[norms == 0] = 1
mean_bags_norm = mean_bags / norms

sim = mean_bags_norm @ mean_bags_norm.T  # 185×185
print(f"Similarity matrix: {sim.shape}")
print(f"Range: [{sim.min():.4f}, {sim.max():.4f}]")

# Assign groups for coloring
def drug_base(name):
    m = re.match(r'^(.+)_(\d+(?:\.\d+)?x)$', name)
    return m.group(1).lower() if m else name.lower()

def mutant_gene(lbl):
    m = re.match(r'^([a-zA-Z]+)', lbl)
    return m.group(1).lower() if m else lbl

def get_group(c, src):
    if src == 'drug':
        base = drug_base(c)
        if base == 'control':
            return 'Drug: Control (water)'
        for g in MOA_GROUPS:
            if base in MOA_GROUPS[g]['drugs']:
                return f'Drug: {g}'
        return 'Drug: Other'
    else:
        gene = mutant_gene(c)
        if 'nc' in gene or 'wt' in gene:
            return 'Mutant: Control'
        for g in MOA_GROUPS:
            if gene in MOA_GROUPS[g]['targets']:
                return f'Mutant: {g}'
        return 'Mutant: Other'

groups = [get_group(c, class_sources[c]) for c in classes]
unique_groups = sorted(set(groups))

# Color palette
palette = sns.color_palette('tab20', len(unique_groups))
group_color = dict(zip(unique_groups, palette))

# Simplify labels for display
def short_label(lbl, src):
    if src == 'drug':
        if lbl == 'control': return 'control'
        base = drug_base(lbl).replace('_', ' ')
        m = re.search(r'_(\d+(?:\.\d+)?x)$', lbl)
        conc = m.group(1) if m else ''
        return f'{base} {conc}'
    else:
        return lbl.replace('_', ' ')

short_labels = [short_label(c, class_sources[c]) for c in classes]
src_types = [class_sources[c] for c in classes]

# Build row/col color bars
row_colors = np.array([group_color[groups[i]] for i in range(N)])
col_colors = row_colors.copy()

print("\nPlotting clustered heatmap ...")
g = sns.clustermap(
    sim, row_cluster=True, col_cluster=True,
    method='ward', metric='euclidean',
    xticklabels=short_labels, yticklabels=short_labels,
    figsize=(max(24, N * 0.2), max(20, N * 0.2)),
    cmap='RdBu_r', vmin=-0.5, vmax=1.0,
    row_colors=[row_colors], col_colors=[col_colors],
    linewidths=0, rasterized=True,
    dendrogram_ratio=0.08,
    cbar_pos=(0.02, 0.8, 0.03, 0.15),
)
g.ax_heatmap.set_xlabel('Class', fontsize=10)
g.ax_heatmap.set_ylabel('Class', fontsize=10)
g.ax_heatmap.tick_params(labelsize=4)
g.fig.suptitle('Cosine Similarity of Mean Bag Features (1280-dim) — 185 classes',
               fontsize=14, y=1.01)
# Legend
patches = [mpatches.Patch(color=group_color[g], label=g) for g in unique_groups]
g.ax_heatmap.legend(handles=patches, loc='upper left', fontsize=5,
                    framealpha=0.8, bbox_to_anchor=(1.02, 1.0))
heatmap_path = os.path.join(OUTPUT_DIR, 'cosine_similarity_185x185.png')
g.savefig(heatmap_path, dpi=200, bbox_inches='tight')
print(f"  Heatmap: {heatmap_path}")
plt.close(g.fig)

# Also do for 32-dim latents
print("\nAlso computing with 32-dim latents ...")
class_mus = defaultdict(list)
for r in records:
    lbl = r['true_label']
    mu = r['mu'].astype(np.float64)  # (100, 32)
    class_mus[lbl].append(mu)

mean_mus = np.zeros((N, 32), dtype=np.float64)
for i, c in enumerate(classes):
    all_mu = np.concatenate(class_mus[c], axis=0)
    mean_mus[i] = all_mu.mean(axis=0)

norms_mu = np.linalg.norm(mean_mus, axis=1, keepdims=True)
norms_mu[norms_mu == 0] = 1
sim_mu = (mean_mus / norms_mu) @ (mean_mus / norms_mu).T

g2 = sns.clustermap(
    sim_mu, row_cluster=True, col_cluster=True,
    method='ward', metric='euclidean',
    xticklabels=short_labels, yticklabels=short_labels,
    figsize=(max(24, N * 0.2), max(20, N * 0.2)),
    cmap='RdBu_r', vmin=-0.5, vmax=1.0,
    row_colors=[row_colors], col_colors=[col_colors],
    linewidths=0, rasterized=True,
    dendrogram_ratio=0.08,
    cbar_pos=(0.02, 0.8, 0.03, 0.15),
)
g2.ax_heatmap.set_xlabel('Class', fontsize=10)
g2.ax_heatmap.set_ylabel('Class', fontsize=10)
g2.ax_heatmap.tick_params(labelsize=4)
g2.fig.suptitle('Cosine Similarity of Mean VAE Latents (32-dim) — 185 classes',
                fontsize=14, y=1.01)
patches2 = [mpatches.Patch(color=group_color[g], label=g) for g in unique_groups]
g2.ax_heatmap.legend(handles=patches2, loc='upper left', fontsize=5,
                     framealpha=0.8, bbox_to_anchor=(1.02, 1.0))
latent_path = os.path.join(OUTPUT_DIR, 'cosine_similarity_185x185_latent32.png')
g2.savefig(latent_path, dpi=200, bbox_inches='tight')
print(f"  Heatmap (latent): {latent_path}")
plt.close(g2.fig)

print(f"\n{'=' * 60}")
print("Done")
print(f"{'=' * 60}")
