#!/usr/bin/env python3
"""Permutation test: is the multivariate AUC significantly > 0.5?"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='output_all_plates/all_plates_features.csv')
parser.add_argument('--output', default='output_all_plates')
parser.add_argument('--region', default='center1128')
parser.add_argument('--n_perm', type=int, default=1000)
args = parser.parse_args()

OUT = args.output
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(args.input)

ALL_FEATS = ['mean', 'std', 'snr', 'entropy', 'p1', 'p99', 'median']
cols = [f'{args.region}_mp_{f}' for f in ALL_FEATS]
cols = [c for c in cols if c in df.columns]
if not cols:
    cols = [f'{args.region}_raw_{f}' for f in ALL_FEATS if f'{args.region}_raw_{f}' in df.columns]

X = df[cols].values.astype(np.float64)
y = (df['type'] == 'drug').astype(int).values

# Observed AUC
Xs = StandardScaler().fit_transform(X)
clf = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs').fit(Xs, y)
fpr, tpr, _ = roc_curve(y, clf.predict_proba(Xs)[:, 1])
obs_auc = auc(fpr, tpr)

# Permutation test
rng = np.random.RandomState(42)
null_aucs = np.zeros(args.n_perm)
for i in range(args.n_perm):
    y_shuff = y[rng.permutation(len(y))]
    Xs_s = StandardScaler().fit_transform(X)
    clf_p = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs').fit(Xs_s, y_shuff)
    fpr_p, tpr_p, _ = roc_curve(y_shuff, clf_p.predict_proba(Xs_s)[:, 1])
    null_aucs[i] = auc(fpr_p, tpr_p)

p_val = (np.sum(null_aucs >= obs_auc) + 1) / (args.n_perm + 1)
print(f'Observed AUC: {obs_auc:.4f}')
print(f'Permutation p-value: {p_val:.4f} ({args.n_perm} permutations)')
print(f'Null mean ± std: {null_aucs.mean():.4f} ± {null_aucs.std():.4f}')

fig, ax = plt.subplots(figsize=(9, 6))
ax.hist(null_aucs, bins=30, color='steelblue', edgecolor='white', alpha=0.8, density=True)
ax.axvline(obs_auc, color='red', ls='--', lw=2.5, label=f'Observed AUC = {obs_auc:.3f}')
ax.axvline(0.5, color='gray', ls=':', lw=1.5, alpha=0.7, label='Chance (AUC = 0.5)')
ax.set_xlabel('AUC under shuffled labels', fontsize=13)
ax.set_ylabel('Density', fontsize=13)
ax.set_title(f'Permutation Test — {args.region} (all 7 stats)\n'
             f'p = {p_val:.4f}  |  null μ±σ = {null_aucs.mean():.3f}±{null_aucs.std():.3f}', fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
outpath = os.path.join(OUT, 'roc_permutation.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {outpath}')
