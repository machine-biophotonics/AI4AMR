"""
t-SNE before and after alignment: all drug vs all mutant embeddings.
Applies mean shift vector (drug_control → NC) to all drug samples.
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

SHIFT_PATH = os.path.join(OUT_DIR, 'alignment_mean_shift_vector.npy')

data = np.load(NPZ_PATH)
embeddings = data["embeddings"]
with open(CSV_PATH) as f:
    metadata = list(csv.DictReader(f))

sources = np.array([m["source"] for m in metadata])
labels_arr = np.array([m["label"] for m in metadata])

drug_mask = sources == 'drug'
mutant_mask = sources == 'mutant'

emb_drug = embeddings[drug_mask]
emb_mutant = embeddings[mutant_mask]
print(f"Drug: {len(emb_drug)} pts, Mutant: {len(emb_mutant)} pts")

# Load or compute shift vector
if os.path.exists(SHIFT_PATH):
    shift_vector = np.load(SHIFT_PATH)
    print(f"Loaded shift vector (norm={np.linalg.norm(shift_vector):.4f})")
else:
    # Compute from drug_control and NC
    drug_ctrl = embeddings[(sources == 'drug') & (labels_arr == 'drug_control')]
    nc_labels = {f'NC_{i}' for i in range(1, 7)}
    nc_ctrl = embeddings[(sources == 'mutant') & np.array([l in nc_labels for l in labels_arr])]
    shift_vector = nc_ctrl.mean(axis=0) - drug_ctrl.mean(axis=0)
    np.save(SHIFT_PATH, shift_vector)
    print(f"Computed shift vector (norm={np.linalg.norm(shift_vector):.4f})")

# Apply shift to all drug embeddings
emb_drug_aligned = emb_drug + shift_vector

# ============ t-SNE ============
# We need two separate t-SNEs: one for before, one for after
# Total 24k points each

for tag, drug_emb, title in [
    ('before', emb_drug, 'Before Alignment'),
    ('after', emb_drug_aligned, 'After Mean Shift Alignment'),
]:
    all_emb = np.vstack([drug_emb, emb_mutant])
    groups = np.array(['Drug'] * len(drug_emb) + ['Mutant'] * len(emb_mutant))

    pca = PCA(n_components=min(50, len(all_emb) - 1), random_state=42)
    emb_pca = pca.fit_transform(all_emb)
    print(f"{tag}: PCA {emb_pca.shape[1]} dims (var={pca.explained_variance_ratio_.sum():.3f})")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                method='barnes_hut', verbose=1)
    emb_2d = tsne.fit_transform(emb_pca)

    fig, ax = plt.subplots(figsize=(14, 12))
    colors = {'Drug': '#e74c3c', 'Mutant': '#3498db'}
    for key in ['Mutant', 'Drug']:
        m = groups == key
        ax.scatter(emb_2d[m, 0], emb_2d[m, 1],
                   c=colors[key], label=key,
                   s=4, alpha=0.35, linewidths=0)
    ax.set_title(f'Drug vs Mutant — {title}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=14, markerscale=5)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'tsne_drug_vs_mutant_{tag}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
