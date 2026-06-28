import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

FOLD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'both', 'fold_Plate_6')

proj = np.load(os.path.join(FOLD_DIR, 'proj.npy'))
domains = np.load(os.path.join(FOLD_DIR, 'domains.npy'))  # 0=drug, 1=mutant

N = len(proj)
idx = np.random.RandomState(42).choice(N, 30000, replace=False)
proj = proj[idx]; domains = domains[idx]
print(f"Subsampled to {len(proj)} points")

print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, verbose=1,
            learning_rate='auto', init='random')
emb = tsne.fit_transform(proj)

colors = np.where(domains == 0, '#e74c3c', '#2980b9')  # drug=red, mutant=blue

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=1, alpha=0.3, rasterized=True)
from matplotlib.patches import Patch
handles = [Patch(color='#e74c3c', label='Drug'),
           Patch(color='#2980b9', label='Mutant')]
ax.legend(handles=handles, loc='upper right', markerscale=5, fontsize=14)
ax.set_title('t-SNE of projected embeddings', fontsize=16)
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

plt.tight_layout()
out = os.path.join(FOLD_DIR, 'tsne_domain.png')
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()
