#!/usr/bin/env python3
"""ROC curves for multivariate classifier (all 7 pixel stats).
   No overwrite — saves as roc_mv_full.png etc.
"""

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
args = parser.parse_args()

OUT = args.output
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(args.input)

ALL_FEATS = ['mean', 'std', 'snr', 'entropy', 'p1', 'p99', 'median']
PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
REGIONS = ['full', 'center1128', 'center224']
PLATE_COLORS = {'P1': '#e41a1c', 'P2': '#377eb8', 'P3': '#4daf4a', 'P4': '#984ea3', 'P5': '#ff7f00', 'P6': '#a65628',
                 'pooled': 'black'}

def lr_roc(df_subset, feat_cols):
    X = df_subset[feat_cols].values.astype(np.float64)
    y = (df_subset['type'] == 'drug').astype(int).values
    if len(np.unique(y)) < 2 or len(y) < 4:
        return None, None, 0.5
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs').fit(Xs, y)
    fpr, tpr, _ = roc_curve(y, clf.predict_proba(Xs)[:, 1])
    return fpr, tpr, auc(fpr, tpr)

for region in REGIONS:
    cols = [f'{region}_mp_{f}' for f in ALL_FEATS]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = [f'{region}_raw_{f}' for f in ALL_FEATS if f'{region}_raw_{f}' in df.columns]

    fig, ax = plt.subplots(figsize=(9, 8))

    # pooled first (thick black)
    fpr, tpr, auc_val = lr_roc(df, cols)
    if fpr is not None:
        ax.plot(fpr, tpr, color='black', lw=3, label=f'Pooled (AUC = {auc_val:.3f})')

    # per plate
    for plate in PLATES:
        fpr, tpr, auc_val = lr_roc(df[df['plate'] == plate], cols)
        if fpr is not None:
            ax.plot(fpr, tpr, color=PLATE_COLORS[plate], lw=1.5, label=f'{plate} (AUC = {auc_val:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(f'Drug vs Mutant — Logistic Regression (all 7 stats)\nRegion: {region}', fontsize=14)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_aspect('equal')
    plt.tight_layout()
    outpath = os.path.join(OUT, f'roc_mv_{region}.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outpath}')

# Comparison across regions
fig, ax = plt.subplots(figsize=(9, 8))
styles = {'full': ('-', 3), 'center1128': ('--', 2.5), 'center224': (':', 2)}
colors = {'full': '#d62728', 'center1128': '#2ca02c', 'center224': '#1f77b4'}
for region in REGIONS:
    cols = [f'{region}_mp_{f}' for f in ALL_FEATS]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = [f'{region}_raw_{f}' for f in ALL_FEATS if f'{region}_raw_{f}' in df.columns]
    fpr, tpr, auc_val = lr_roc(df, cols)
    if fpr is not None:
        ls, lw = styles[region]
        ax.plot(fpr, tpr, color=colors[region], ls=ls, lw=lw, label=f'{region} (AUC = {auc_val:.3f})')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('Drug vs Mutant — Region Comparison (all 7 stats, pooled)', fontsize=14)
ax.legend(fontsize=11, loc='lower right')
ax.set_aspect('equal')
plt.tight_layout()
outpath = os.path.join(OUT, 'roc_mv_region_comparison.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {outpath}')
