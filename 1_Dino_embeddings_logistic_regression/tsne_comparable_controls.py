"""
t-SNE: drug_control vs ACE-1_minusATC (no ATC, no NC, no plasmid induction).
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

sources = np.array([m["source"] for m in metadata])
labels_arr = np.array([m["label"] for m in metadata])

drug_mask = (sources == 'drug') & (labels_arr == 'drug_control')
ace_mask = (sources == 'control') & (labels_arr == 'ACE-1_minusATC')
keep = drug_mask | ace_mask

emb_all = embeddings[keep]
group = np.empty(len(emb_all), dtype=object)
group[drug_mask[keep]] = 'drug_control'
group[ace_mask[keep]] = 'ACE-1_minusATC'

n = len(emb_all)
print(f"Total: {n} — drug_control: {(group == 'drug_control').sum()}, ACE-1_minusATC: {(group == 'ACE-1_minusATC').sum()}")

pca = PCA(n_components=min(50, n - 1), random_state=42)
emb_pca = pca.fit_transform(emb_all)
print(f"PCA: {emb_pca.shape[1]} dims (var={pca.explained_variance_ratio_.sum():.3f})")

tsne = TSNE(n_components=2, perplexity=30, random_state=42, method='barnes_hut', verbose=1)
emb_2d = tsne.fit_transform(emb_pca)

colors = {'drug_control': '#e74c3c', 'ACE-1_minusATC': '#2ecc71'}
names = {'drug_control': 'Drug control', 'ACE-1_minusATC': 'ACE-1 minusATC'}

fig, ax = plt.subplots(figsize=(14, 12))
for key in ['ACE-1_minusATC', 'drug_control']:
    sm = group == key
    ax.scatter(emb_2d[sm, 0], emb_2d[sm, 1],
               c=colors[key], label=names[key],
               s=10, alpha=0.6, linewidths=0)

ax.set_title('t-SNE: Drug Control vs ACE-1 (minusATC)', fontsize=16, fontweight='bold')
ax.legend(fontsize=14, markerscale=4)
ax.set_xticks([])
ax.set_yticks([])

out_path = os.path.join(OUT_DIR, 'tsne_drug_control_vs_ace1.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
