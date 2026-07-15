"""
t-SNE of all 36k embeddings.
1) tsne_all_sources.png — colored by source (control=green, mutant=blue, drug=red)
2) tsne_controls_highlighted.png — only control samples colored, everything else gray
"""
import numpy as np, csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(BASE_DIR, "features_all.npz")
CSV_PATH = os.path.join(BASE_DIR, "features_metadata.csv")
OUT_DIR = os.path.join(BASE_DIR, "analysis_figures")
os.makedirs(OUT_DIR, exist_ok=True)

SRC_COLORS = {'control': '#2ecc71', 'mutant': '#3498db', 'drug': '#e74c3c'}
SRC_NAMES = {'control': 'Controls', 'mutant': 'Mutants', 'drug': 'Drugs'}
CTRL_CLASSES = {
    'control': None,
    'mutant': {'NC_1','NC_2','NC_3','NC_4','NC_5','NC_6',
               'WT NC_1','WT NC_2','WT NC_3','WT NC_4','WT NC_5','WT NC_6'},
    'drug': {'drug_control'},
}

data = np.load(NPZ_PATH)
embeddings = data["embeddings"]
with open(CSV_PATH) as f:
    metadata = list(csv.DictReader(f))

sources = np.array([m["source"] for m in metadata])
labels = np.array([m["label"] for m in metadata])

n = len(embeddings)
pca = PCA(n_components=min(50, n - 1), random_state=42)
emb_pca = pca.fit_transform(embeddings)
print(f"PCA: {emb_pca.shape[1]} dims (var={pca.explained_variance_ratio_.sum():.3f})")

print(f"Running t-SNE on {n} points...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            method='barnes_hut', verbose=1)
emb_2d = tsne.fit_transform(emb_pca)

# --- Plot 1: by source ---
fig, ax = plt.subplots(figsize=(14, 12))
for src in ['control', 'mutant', 'drug']:
    sm = sources == src
    ax.scatter(emb_2d[sm, 0], emb_2d[sm, 1],
               c=SRC_COLORS[src], label=SRC_NAMES[src],
               s=4, alpha=0.4, linewidths=0)

ax.set_title('t-SNE of All Embeddings by Source (36k points)', fontsize=16, fontweight='bold')
ax.legend(fontsize=14, markerscale=5)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(os.path.join(OUT_DIR, 'tsne_all_sources.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: tsne_all_sources.png")

# --- Plot 2: controls highlighted, rest gray ---
is_ctrl = np.zeros(n, dtype=bool)
for src in ['control', 'mutant', 'drug']:
    ctrl_set = CTRL_CLASSES[src]
    if ctrl_set is None:
        is_ctrl[sources == src] = True
    else:
        is_ctrl |= (sources == src) & np.array([l in ctrl_set for l in labels])

ctrl_colors = {'control': '#2ecc71', 'mutant': '#3498db', 'drug': '#e74c3c'}

fig, ax = plt.subplots(figsize=(14, 12))

# gray background (non-control)
ax.scatter(emb_2d[~is_ctrl, 0], emb_2d[~is_ctrl, 1],
           c='#cccccc', s=4, alpha=0.3, linewidths=0, label='Other')

# colored controls by original source
for src in ['control', 'mutant', 'drug']:
    ctrl_set = CTRL_CLASSES[src]
    if ctrl_set is None:
        sm = (sources == src) & is_ctrl
    else:
        sm = (sources == src) & np.array([l in ctrl_set for l in labels])
    ax.scatter(emb_2d[sm, 0], emb_2d[sm, 1],
               c=ctrl_colors[src], label=f'{src.capitalize()} controls',
               s=12, alpha=0.7, linewidths=0.3, edgecolors='black')

ax.set_title('t-SNE — Controls Highlighted by Source (36k points)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, markerscale=3)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(os.path.join(OUT_DIR, 'tsne_controls_highlighted.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: tsne_controls_highlighted.png")
