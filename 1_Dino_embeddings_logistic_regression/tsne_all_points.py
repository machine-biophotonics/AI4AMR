"""
t-SNE of all DINOv3 center crop CLS embeddings (36k points).
Colored by data source (control/drug/mutant) and plate.
Saves static PNG + interactive HTML.
"""
import numpy as np
import json, os, csv, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(BASE_DIR, "features_all.npz")
CSV_PATH = os.path.join(BASE_DIR, "features_metadata.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=None, help='Subset size (default: all 36k)')
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("Loading features...")
    data = np.load(NPZ_PATH)
    embeddings = data["embeddings"]
    with open(CSV_PATH) as f:
        metadata = list(csv.DictReader(f))

    print(f"Total: {len(embeddings)} embeddings, {len(metadata)} metadata rows")

    # Subset if requested
    if args.n_samples and args.n_samples < len(embeddings):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(embeddings), args.n_samples, replace=False)
        embeddings = embeddings[idx]
        metadata = [metadata[i] for i in idx]
        print(f"Subsampled to {args.n_samples}")

    # PCA 1024 -> 50 for speed
    print("PCA 1024->50...")
    pca = PCA(n_components=min(50, embeddings.shape[0]-1), random_state=args.seed)
    emb_pca = pca.fit_transform(embeddings)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # t-SNE
    print(f"t-SNE (perp={args.perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed,
                method='barnes_hut', verbose=1)
    emb_2d = tsne.fit_transform(emb_pca)
    print("t-SNE done!")

    # Extract metadata arrays
    sources = np.array([m["source"] for m in metadata])
    plates = np.array([m["plate"] for m in metadata])
    labels = np.array([m["label"] for m in metadata])

    # Plot by source
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    colors_src = {'control': '#2196F3', 'drug': '#FF5722', 'mutant': '#4CAF50'}
    for src in ['control', 'drug', 'mutant']:
        mask = sources == src
        axes[0].scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=colors_src[src],
                        s=2, alpha=0.5, label=src, rasterized=True)
    axes[0].set_title("Colored by Data Source", fontsize=14)
    axes[0].legend(markerscale=5)

    # Plot by plate
    cmap_plate = plt.cm.tab10
    for i, pl in enumerate(['P1','P2','P3','P4','P5','P6']):
        mask = plates == pl
        axes[1].scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                        c=[cmap_plate(i/6)], s=2, alpha=0.5, label=pl, rasterized=True)
    axes[1].set_title("Colored by Plate", fontsize=14)
    axes[1].legend(markerscale=5)

    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_facecolor('#f5f5f5')

    plt.tight_layout()
    out = os.path.join(BASE_DIR, "tsne_all_points.png")
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")

    # Interactive HTML (Plotly)
    try:
        import plotly.express as px
        import pandas as pd
        df = pd.DataFrame({
            'tSNE1': emb_2d[:, 0], 'tSNE2': emb_2d[:, 1],
            'source': sources, 'plate': plates, 'label': labels,
        })
        fig = px.scatter(df, x='tSNE1', y='tSNE2', color='source',
                         hover_data=['plate', 'label'],
                         title='DINOv3 CLS Embeddings (center crop, 500px)')
        html_out = os.path.join(BASE_DIR, "tsne_all_points.html")
        fig.write_html(html_out)
        print(f"Saved: {html_out}")
    except ImportError:
        print("plotly not available, skipping HTML")

if __name__ == '__main__':
    main()
