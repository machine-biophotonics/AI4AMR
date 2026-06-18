#!/usr/bin/env python3
"""t-SNE grid for unsupervised latents: t=0.5 and t=1.0 side by side.

Drug=red, mutant=green, control=blue.

Usage:
    python3 plot_tsne_unsupervised.py
    python3 plot_tsne_unsupervised.py --input_dir unsupervised_latents
"""
import os, sys, argparse, warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE

SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default=None)
parser.add_argument('--perplexity', type=float, default=50)
parser.add_argument('--tsne_iter', type=int, default=5000)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
input_dir = args.input_dir or os.path.join(SCRIPT_DIR, 'unsupervised_latents')

def infer_type(name):
    name = str(name)
    if name == 'control' or 'NC_' in name or 'WT NC' in name:
        return 2  # control = blue
    if re.search(r'_\d+\.?\d*x$', name):
        return 0  # drug = red
    return 1  # mutant = green

print(f"Loading features from: {input_dir}")
labels = np.load(os.path.join(input_dir, 'labels.npy'))
class_names = np.load(os.path.join(input_dir, 'class_names.npy'), allow_pickle=True)

class_types = np.array([infer_type(class_names[l]) for l in labels])
n_drug = (class_types == 0).sum()
n_mutant = (class_types == 1).sum()
n_control = (class_types == 2).sum()
print(f"  Drug={n_drug}, Mutant={n_mutant}, Control={n_control}")

colors = {0: (1, 0, 0, 0.4), 1: (0, 0.8, 0, 0.4), 2: (0, 0, 1, 0.4)}
group_names = ('Drug', 'Mutant', 'Control')

settings = [
    ('t05', 't=0.5'),
    ('t10', 't=1.0'),
]

fig = plt.figure(figsize=(12, 5.5))
gs = GridSpec(1, 2, figure=fig, wspace=0.1)

for idx, (name, title) in enumerate(settings):
    feats = np.load(os.path.join(input_dir, f'feats_{name}.npy'))
    print(f"  t-SNE {name}: {feats.shape} ...", end=' ', flush=True)

    reducer = TSNE(n_components=2, perplexity=args.perplexity, max_iter=args.tsne_iter,
                   random_state=SEED, method='barnes_hut', verbose=0)
    embedding = reducer.fit_transform(feats)
    np.save(os.path.join(input_dir, f'tsne_{name}.npy'), embedding)
    print("done")

    ax = fig.add_subplot(gs[0, idx])
    for t in (0, 1, 2):
        mask = class_types == t
        if mask.sum() == 0:
            continue
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=colors[t], s=2, alpha=0.5, rasterized=True,
                   label=f'{group_names[t]} ({mask.sum()})')

    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, markerscale=8, loc='best')
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle(f't-SNE: Unsupervised Flow Model Bottleneck (perp={args.perplexity})',
             fontsize=12, y=0.98)
fig.savefig(os.path.join(input_dir, 'tsne_grid.png'),
            dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {os.path.join(input_dir, 'tsne_grid.png')}")
