"""
t-SNE of control embeddings — 3 figures:
1) 28 classes individually colored
2) 2 strains (ACE-1 red, MG1655 blue)
3) 2 ATC conditions (minusATC vs plusATC)
"""
import numpy as np, csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(BASE_DIR, "features_all.npz")
CSV_PATH = os.path.join(BASE_DIR, "features_metadata.csv")
OUT_DIR = os.path.join(BASE_DIR, "analysis_figures")
os.makedirs(OUT_DIR, exist_ok=True)

data = np.load(NPZ_PATH)
embeddings = data["embeddings"]
with open(CSV_PATH) as f:
    metadata = list(csv.DictReader(f))

labels_arr = np.array([m["label"] for m in metadata])
sources = np.array([m["source"] for m in metadata])

mask = sources == 'control'
emb_src = embeddings[mask]
labels_src = labels_arr[mask]

unique_classes = sorted(set(labels_src))
print(f"Control samples: {len(emb_src)} points across {len(unique_classes)} classes")

pca = PCA(n_components=min(50, len(emb_src) - 1), random_state=42)
emb_pca = pca.fit_transform(emb_src)
print(f"PCA: {emb_pca.shape[1]} dims (var={pca.explained_variance_ratio_.sum():.3f})")

tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            method='barnes_hut', verbose=1)
emb_2d = tsne.fit_transform(emb_pca)

# ---- Figure 1: 28 classes individually colored ----
cmap = plt.cm.tab20
colors = [cmap(i % 20) for i in range(len(unique_classes))]

fig, ax = plt.subplots(figsize=(16, 14))
for cls, color in zip(unique_classes, colors):
    cm = labels_src == cls
    ax.scatter(emb_2d[cm, 0], emb_2d[cm, 1],
               c=[color], label=cls, s=6, alpha=0.6, linewidths=0)
ax.set_title('Control Embeddings by Class (28 classes)', fontsize=16, fontweight='bold')
ax.legend(fontsize=8, markerscale=2, loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'tsne_control_by_class.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

# ---- Figure 2: 2 strains ----
def get_strain(label):
    return label.split('_')[0]

strains = np.array([get_strain(l) for l in labels_src])
strain_colors = {'ACE-1': '#e74c3c', 'MG1655': '#3498db'}

fig, ax = plt.subplots(figsize=(14, 12))
for strain, color in strain_colors.items():
    sm = strains == strain
    ax.scatter(emb_2d[sm, 0], emb_2d[sm, 1],
               c=color, label=strain, s=8, alpha=0.6, linewidths=0)
ax.set_title('Control Embeddings by Strain (ACE-1 vs MG1655)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, markerscale=3)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'tsne_control_by_strain.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

# ---- Figure 3: 2 ATC conditions ----
def get_atc(label):
    return 'plusATC' if 'plusATC' in label else 'minusATC'

atc = np.array([get_atc(l) for l in labels_src])
atc_colors = {'minusATC': '#8B4513', 'plusATC': '#4682B4'}

fig, ax = plt.subplots(figsize=(14, 12))
for cond, color in atc_colors.items():
    ac = atc == cond
    ax.scatter(emb_2d[ac, 0], emb_2d[ac, 1],
               c=color, label=cond, s=8, alpha=0.6, linewidths=0)
ax.set_title('Control Embeddings by ATC Condition', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, markerscale=3)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'tsne_control_by_atc.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
