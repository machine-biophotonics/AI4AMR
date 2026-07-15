"""
t-SNE grid: 3 figures (control/mutant/drug) x 6 subplots (P1-P6).
Each subplot shows all 6 plates; the focus plate is colored, others grayed out.
"""
import numpy as np, csv, os, json
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

SRC_COLORS = {'control': '#2ecc71', 'mutant': '#3498db', 'drug': '#e74c3c'}
SRC_NAMES = {'control': 'Controls', 'mutant': 'Mutants', 'drug': 'Drugs'}
PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

data = np.load(NPZ_PATH)
embeddings = data["embeddings"]
with open(CSV_PATH) as f:
    metadata = list(csv.DictReader(f))

labels_arr = np.array([m["label"] for m in metadata])
sources = np.array([m["source"] for m in metadata])
plates = np.array([m["plate"] for m in metadata])

for src in ['control', 'mutant', 'drug']:
    print(f"\n{'='*60}")
    print(f"Processing {src} ({SRC_NAMES[src]})...")

    mask = sources == src
    emb_src = embeddings[mask]
    plates_src = plates[mask]

    n = len(emb_src)
    pca = PCA(n_components=min(50, n - 1), random_state=42)
    emb_pca = pca.fit_transform(emb_src)
    print(f"  PCA: {emb_pca.shape[1]} dims (var={pca.explained_variance_ratio_.sum():.3f})")

    print(f"  Running t-SNE on {n} points...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                method='barnes_hut', verbose=1)
    emb_2d = tsne.fit_transform(emb_pca)

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()

    for i, plate in enumerate(PLATES):
        ax = axes[i]
        pm = plates_src == plate

        ax.scatter(emb_2d[~pm, 0], emb_2d[~pm, 1],
                   c='#cccccc', s=4, alpha=0.25, linewidths=0)
        ax.scatter(emb_2d[pm, 0], emb_2d[pm, 1],
                   c=SRC_COLORS[src], s=8, alpha=0.7, linewidths=0)

        ax.set_title(f'{SRC_NAMES[src]} — {plate} in focus', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'tsne_grid_{src}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

print(f"\nDone! All figures saved to {OUT_DIR}/")
