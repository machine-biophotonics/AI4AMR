#!/usr/bin/env python3
"""ROC curves per plate for each pixel statistic + combined (all 7 stats).
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

SINGLE_FEATS = ['mean', 'std', 'snr', 'entropy', 'p1', 'p99', 'median']
ALL_FEATS = ['mean', 'std', 'snr', 'entropy', 'p1', 'p99', 'median']
PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
PLATE_COLORS = {'P1': '#e41a1c', 'P2': '#377eb8', 'P3': '#4daf4a', 'P4': '#984ea3', 'P5': '#ff7f00', 'P6': '#a65628'}
REGIONS = ['full', 'center1128', 'center224']
METRIC_LABELS = {'mean': 'Mean', 'std': 'Std Dev', 'snr': 'SNR', 'entropy': 'Entropy',
                 'p1': 'P1', 'p99': 'P99', 'median': 'Median'}


def compute_multivariate_roc(df_subset, feat_cols):
    """Logistic regression ROC on all feat_cols, returns fpr, tpr, auc."""
    X = df_subset[feat_cols].values.astype(np.float64)
    y = (df_subset['type'] == 'drug').astype(int).values
    if len(np.unique(y)) < 2 or len(y) < 4:
        return np.array([0, 1]), np.array([0, 1]), 0.5
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs')
    clf.fit(Xs, y)
    fpr, tpr, _ = roc_curve(y, clf.predict_proba(Xs)[:, 1])
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


# =====================================================
# FIGURE 1: 5x7 grid per region (4 single + 1 combined)
# =====================================================
for region in REGIONS:
    prefix = region
    cols = [f'{prefix}_mp_{f}' for f in ALL_FEATS]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        # fallback to raw
        cols = [f'{prefix}_raw_{f}' for f in ALL_FEATS if f'{prefix}_raw_{f}' in df.columns]
    single_cols = [c for c in cols if c.split('_')[-1] in SINGLE_FEATS]

    nrows = len(SINGLE_FEATS) + 1  # 7 single + 1 combined
    fig, axes = plt.subplots(nrows, 7, figsize=(28, 24))
    fig.suptitle(f'Drug vs Mutant Pixel Statistics — Region: {region}', fontsize=18, y=0.97)

    # Rows 0-6: single features
    for fi, feat in enumerate(SINGLE_FEATS):
        col = f'{prefix}_mp_{feat}'
        if col not in df.columns:
            col = f'{prefix}_raw_{feat}'
        if col not in df.columns:
            continue

        y_true = (df['type'] == 'drug').astype(int).values
        pooled_fpr, pooled_tpr, _ = roc_curve(y_true, df[col].values)
        pooled_auc = auc(pooled_fpr, pooled_tpr)
        ax = axes[fi, 0]
        ax.plot(pooled_fpr, pooled_tpr, color='black', lw=2.5, label=f'AUC = {pooled_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        if fi == 0: ax.set_title('All plates (pooled)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_aspect('equal')

        for pi, plate in enumerate(PLATES):
            mask = df['plate'] == plate
            y_p = (df.loc[mask, 'type'] == 'drug').astype(int).values
            fpr_p, tpr_p, _ = roc_curve(y_p, df.loc[mask, col].values)
            auc_p = auc(fpr_p, tpr_p)
            ax = axes[fi, pi + 1]
            ax.plot(fpr_p, tpr_p, color=PLATE_COLORS[plate], lw=2.5, label=f'AUC = {auc_p:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            if fi == 0: ax.set_title(plate, fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, loc='lower right')
            ax.set_aspect('equal')

    # Last row: combined (all 7 stats, logistic regression)
    fi = nrows - 1
    pooled_fpr, pooled_tpr, pooled_auc = compute_multivariate_roc(df, cols)
    ax = axes[fi, 0]
    ax.plot(pooled_fpr, pooled_tpr, color='black', lw=2.5, label=f'AUC = {pooled_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title('All plates (pooled)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_aspect('equal')

    for pi, plate in enumerate(PLATES):
        mask = df['plate'] == plate
        fpr_p, tpr_p, auc_p = compute_multivariate_roc(df[mask], cols)
        ax = axes[fi, pi + 1]
        ax.plot(fpr_p, tpr_p, color=PLATE_COLORS[plate], lw=2.5, label=f'AUC = {auc_p:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_title(plate, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_aspect('equal')

    # Y-axis labels
    for fi, feat in enumerate(SINGLE_FEATS):
        axes[fi, 0].set_ylabel(METRIC_LABELS[feat], fontsize=12, fontweight='bold')
    axes[nrows - 1, 0].set_ylabel('ALL 7 stats\n(LR combined)', fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(OUT, f'roc_{region}.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outpath}')


# =====================================================
# FIGURE 2: Region comparison — combined AUC per plate
# =====================================================
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(PLATES))
w = 0.25
for ri, region in enumerate(REGIONS):
    prefix = region
    cols = [f'{prefix}_mp_{f}' for f in ALL_FEATS]
    cols = [c for c in cols if c in df.columns] or [f'{prefix}_raw_{f}' for f in ALL_FEATS if f'{prefix}_raw_{f}' in df.columns]
    aucs = []
    for plate in PLATES:
        _, _, roc_auc = compute_multivariate_roc(df[df['plate'] == plate], cols)
        aucs.append(roc_auc)
    bars = ax.bar(x + ri * w, aucs, w, label=region, alpha=0.85)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.2f}',
                ha='center', va='bottom', fontsize=8)

ax.set_xticks(x + w)
ax.set_xticklabels(PLATES)
ax.set_ylabel('Combined AUC (all 7 stats)')
ax.set_title('Region Comparison: Combined Pixel-Stat AUC per Plate')
ax.legend()
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
outpath = os.path.join(OUT, 'roc_region_comparison.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {outpath}')


# =====================================================
# Summary table
# =====================================================
print('\nMultivariate AUC (all 7 stats, logistic regression):')
for region in REGIONS:
    prefix = region
    cols = [f'{prefix}_mp_{f}' for f in ALL_FEATS]
    cols = [c for c in cols if c in df.columns] or [f'{prefix}_raw_{f}' for f in ALL_FEATS if f'{prefix}_raw_{f}' in df.columns]
    _, _, pooled_auc = compute_multivariate_roc(df, cols)
    plate_aucs = []
    for plate in PLATES:
        _, _, pa = compute_multivariate_roc(df[df['plate'] == plate], cols)
        plate_aucs.append(f'{pa:.3f}')
    print(f'  {region}: pooled={pooled_auc:.3f}  ' + '  '.join(f'{p}={a}' for p, a in zip(PLATES, plate_aucs)))
