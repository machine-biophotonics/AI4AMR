"""
t-SNE of controls from all 3 sources: control plates, mutant plates (NC/WT NC), drug plates (drug_control).
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

MUTANT_CONTROL_LABELS = {'NC_1','NC_2','NC_3','NC_4','NC_5','NC_6',
                         'WT NC_1','WT NC_2','WT NC_3','WT NC_4','WT NC_5','WT NC_6'}

data = np.load(NPZ_PATH)
embeddings = data["embeddings"]
with open(CSV_PATH) as f:
    metadata = list(csv.DictReader(f))

sources = np.array([m["source"] for m in metadata])
labels_arr = np.array([m["label"] for m in metadata])

ctrl_src_mask = sources == 'control'
mut_ctrl_mask = (sources == 'mutant') & np.array([m['label'] in MUTANT_CONTROL_LABELS for m in metadata])
drug_ctrl_mask = (sources == 'drug') & (labels_arr == 'drug_control')

keep = ctrl_src_mask | mut_ctrl_mask | drug_ctrl_mask
emb_all = embeddings[keep]
src_labels = np.empty(len(emb_all), dtype=object)
src_labels[ctrl_src_mask[keep]] = 'control_plates'
src_labels[mut_ctrl_mask[keep]] = 'mutant_controls'
src_labels[drug_ctrl_mask[keep]] = 'drug_controls'

n = len(emb_all)
print(f"Total control samples: {n}")
print(f"  Control plates: {(src_labels == 'control_plates').sum()}")
print(f"  Mutant controls: {(src_labels == 'mutant_controls').sum()}")
print(f"  Drug controls: {(src_labels == 'drug_controls').sum()}")

COLORS = {'control_plates': '#2ecc71', 'mutant_controls': '#3498db', 'drug_controls': '#e74c3c'}
NAMES = {'control_plates': 'Control plates (ACE-1, MG1655)', 'mutant_controls': 'Mutant internal controls (NC, WT NC)', 'drug_controls': 'Drug internal control (drug_control)'}

pca = PCA(n_components=min(50, n - 1), random_state=42)
emb_pca = pca.fit_transform(emb_all)
print(f"PCA: {emb_pca.shape[1]} dims (var={pca.explained_variance_ratio_.sum():.3f})")

print(f"Running t-SNE on {n} points...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            method='barnes_hut', verbose=1)
emb_2d = tsne.fit_transform(emb_pca)

fig, ax = plt.subplots(figsize=(14, 12))
for key in ['control_plates', 'mutant_controls', 'drug_controls']:
    sm = src_labels == key
    ax.scatter(emb_2d[sm, 0], emb_2d[sm, 1],
               c=COLORS[key], label=NAMES[key],
               s=6, alpha=0.5, linewidths=0)

ax.set_title('t-SNE of Controls from All Sources', fontsize=16, fontweight='bold')
ax.legend(fontsize=11, markerscale=4)
ax.set_xticks([])
ax.set_yticks([])

out_path = os.path.join(OUT_DIR, 'tsne_controls_all_sources.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
