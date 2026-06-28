#!/usr/bin/env python3
"""
UMAP and t-SNE visualization of projected embeddings.
Loads proj.npy and labels, reduces to 2D, colors by domain and class.
"""

import os, sys, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLD_DIR = os.path.join(SCRIPT_DIR, 'both', 'fold_Plate_6')

PROJ_FILE    = os.path.join(FOLD_DIR, 'proj.npy')
LABELS_FILE  = os.path.join(FOLD_DIR, 'labels.npy')
DOMAINS_FILE = os.path.join(FOLD_DIR, 'domains.npy')

for f in [PROJ_FILE, LABELS_FILE, DOMAINS_FILE]:
    if not os.path.exists(f):
        print(f"ERROR: {f} not found"); sys.exit(1)

proj = np.load(PROJ_FILE)
labels = np.load(LABELS_FILE)
domains = np.load(DOMAINS_FILE)

# Subsample for speed (100k max)
N = len(proj)
if N > 100000:
    idx = np.random.RandomState(42).choice(N, 100000, replace=False)
    proj = proj[idx]; labels = labels[idx]; domains = domains[idx]
    print(f"Subsampled to {len(proj)} points")
else:
    print(f"Using all {len(proj)} points")

print("Running UMAP...")
try:
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42, verbose=True)
    emb_umap = reducer.fit_transform(proj)
except Exception as e:
    print(f"UMAP failed: {e}")
    emb_umap = None

print("Running t-SNE...")
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=50, random_state=42, verbose=1,
            learning_rate='auto', init='random')
emb_tsne = tsne.fit_transform(proj)

# ── Labels ─────────────────────────────────────────────────────────────────
with open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')) as f:
    IC50 = json.load(f)
with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')) as f:
    MUT = json.load(f)

drug_set = set()
for plate, wells in IC50.items():
    for well, info in wells.items():
        ab, ic = info.get('antibiotic',''), info.get('ic50_multiple','')
        if ab and ic:
            drug_set.add(f'{ab.replace(" ", "_")}_{ic}' if ic != 'control' else 'control')
drug_idx_to_name = {i: n for i, n in enumerate(sorted(drug_set))}

mutant_set = set()
for plate, rows in MUT.items():
    for row, cols in rows.items():
        for col, info in cols.items():
            if 'id' in info:
                mutant_set.add(info['id'])
mutant_idx_to_name = {i: n for i, n in enumerate(sorted(mutant_set))}

def gene_name(n):
    if n.startswith('WT NC'): return 'WT NC'
    if n.startswith('NC_'): return 'NC'
    m = re.match(r'^([a-zA-Z]+)_\d+$', n)
    return m.group(1) if m else n

# Assign colors by domain (drug=blue, mutant=red) and by class
drug_colors = plt.cm.tab20(np.linspace(0, 1, 20))
mutant_colors = plt.cm.tab20b(np.linspace(0, 1, 20))

domain_c = np.array(['#1f77b4' if d == 0 else '#d62728' for d in domains])

# Per-class coloring (within each domain)
drug_classes = sorted(drug_idx_to_name.values())
mutant_genes = sorted(set(gene_name(n) for n in mutant_idx_to_name.values())
                      - {'NC', 'WT NC'})

drug_class_to_color = {c: drug_colors[i % 20] for i, c in enumerate(drug_classes)}
mutant_gene_to_color = {g: mutant_colors[i % 20] for i, g in enumerate(mutant_genes)}

# ── Plot function ──────────────────────────────────────────────────────────
def plot_2d(emb, title, fname, color_by_domain=True):
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    # Plot 1: colored by domain
    ax = axes[0]
    ax.scatter(emb[:, 0], emb[:, 1], c=domain_c, s=1, alpha=0.3, rasterized=True)
    handles = [mpatches.Patch(color='#1f77b4', label='Drug'),
               mpatches.Patch(color='#d62728', label='Mutant')]
    ax.legend(handles=handles, loc='upper right', markerscale=5, fontsize=12)
    ax.set_title(f'{title} — colored by domain', fontsize=14)
    ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')

    # Plot 2: colored by class (drug: tab20, mutant: tab20b)
    ax = axes[1]
    for i in range(len(emb)):
        d = domains[i]
        lbl = int(labels[i])
        if d == 0:
            c = drug_class_to_color.get(drug_idx_to_name.get(lbl, ''), (0.5, 0.5, 0.5))
        else:
            g = gene_name(mutant_idx_to_name.get(lbl, ''))
            c = mutant_gene_to_color.get(g, (0.5, 0.5, 0.5))
        ax.scatter(emb[i, 0], emb[i, 1], c=[c], s=1, alpha=0.3, rasterized=True)
    ax.set_title(f'{title} — colored by class', fontsize=14)
    ax.set_xlabel('Component 1'); ax.set_ylabel('Component 2')

    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print(f"Saved: {fname}")
    plt.close(fig)

if emb_umap is not None:
    plot_2d(emb_umap, 'UMAP', os.path.join(FOLD_DIR, 'umap_embeddings.png'))

plot_2d(emb_tsne, 't-SNE', os.path.join(FOLD_DIR, 'tsne_embeddings.png'))

print("\nDone.")
