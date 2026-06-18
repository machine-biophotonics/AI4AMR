#!/usr/bin/env python3
"""ROC curves per plate for each pixel statistic + combined (all 7 stats).
   FIXED: Uses 5-fold cross-validated AUC instead of training-set AUC.
"""

import argparse, os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
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

def cv_auc_single(df_subset, col):
    """5-fold cross-validated AUC for a single feature."""
    X = df_subset[col].values.astype(np.float64).reshape(-1, 1)
    y = (df_subset['type'] == 'drug').astype(int).values
    if len(np.unique(y)) < 2 or len(y) < 4:
        return 0.5
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    fpr_list, tpr_list = [], []
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        clf = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs')
        clf.fit(X_tr, y[train_idx])
        y_score = clf.predict_proba(X_va)[:, 1]
        fpr_v, tpr_v, _ = roc_curve(y[val_idx], y_score)
        aucs.append(auc(fpr_v, tpr_v))
        fpr_list.append(fpr_v)
        tpr_list.append(tpr_v)
    return np.mean(aucs), fpr_list, tpr_list

def cv_auc_multi(df_subset, cols):
    """5-fold cross-validated AUC for multiple features (logistic regression)."""
    X = df_subset[cols].values.astype(np.float64)
    y = (df_subset['type'] == 'drug').astype(int).values
    if len(np.unique(y)) < 2 or len(y) < 4:
        return 0.5, [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    fpr_list, tpr_list = [], []
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        clf = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs')
        clf.fit(X_tr, y[train_idx])
        y_score = clf.predict_proba(X_va)[:, 1]
        fpr_v, tpr_v, _ = roc_curve(y[val_idx], y_score)
        aucs.append(auc(fpr_v, tpr_v))
        fpr_list.append(fpr_v)
        tpr_list.append(tpr_v)
    return np.mean(aucs), fpr_list, tpr_list

# =====================================================
# FIGURE 1: 8x7 grid per region (7 single + 1 combined)
# =====================================================
for region in REGIONS:
    prefix = region
    cols = [f'{prefix}_mp_{f}' for f in ALL_FEATS]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = [f'{prefix}_raw_{f}' for f in ALL_FEATS if f'{prefix}_raw_{f}' in df.columns]
    single_cols = [c for c in cols if c.split('_')[-1] in SINGLE_FEATS]

    nrows = len(SINGLE_FEATS) + 1
    fig, axes = plt.subplots(nrows, 7, figsize=(28, 24))
    fig.suptitle(f'Drug vs Mutant Pixel Statistics (5-fold CV AUC) — Region: {region}', fontsize=18, y=0.97)

    # Rows 0-6: single features
    for fi, feat in enumerate(SINGLE_FEATS):
        col = f'{prefix}_mp_{feat}'
        if col not in df.columns:
            col = f'{prefix}_raw_{feat}'
        if col not in df.columns:
            continue

        # Pooled
        pooled_auc, _, _ = cv_auc_single(df, col)
        ax = axes[fi, 0]
        ax.text(0.5, 0.5, f'CV AUC\n={pooled_auc:.3f}', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if fi == 0: ax.set_title('All plates (pooled)', fontsize=10, fontweight='bold')
        ax.set_aspect('equal')

        for pi, plate in enumerate(PLATES):
            mask = df['plate'] == plate
            plate_auc, fpr_list, tpr_list = cv_auc_single(df[mask], col)
            ax = axes[fi, pi + 1]
            # Plot each fold's ROC, average color
            for fpr_v, tpr_v in zip(fpr_list, tpr_list):
                ax.plot(fpr_v, tpr_v, color=PLATE_COLORS[plate], lw=1, alpha=0.3)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            if fi == 0: ax.set_title(plate, fontsize=10, fontweight='bold')
            ax.text(0.6, 0.15, f'CV AUC\n={plate_auc:.3f}', ha='center', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7))
            ax.set_aspect('equal')

    # Last row: combined (all 7 stats, logistic regression CV)
    fi = nrows - 1
    pooled_auc, _, _ = cv_auc_multi(df, cols)
    ax = axes[fi, 0]
    ax.text(0.5, 0.5, f'CV AUC\n={pooled_auc:.3f}', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('All plates (pooled)', fontsize=10, fontweight='bold')
    ax.set_aspect('equal')

    for pi, plate in enumerate(PLATES):
        mask = df['plate'] == plate
        plate_auc, fpr_list, tpr_list = cv_auc_multi(df[mask], cols)
        ax = axes[fi, pi + 1]
        for fpr_v, tpr_v in zip(fpr_list, tpr_list):
            ax.plot(fpr_v, tpr_v, color=PLATE_COLORS[plate], lw=1, alpha=0.3)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_title(plate, fontsize=10, fontweight='bold')
        ax.text(0.6, 0.15, f'CV AUC\n={plate_auc:.3f}', ha='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7))
        ax.set_aspect('equal')

    # Y-axis labels
    for fi, feat in enumerate(SINGLE_FEATS):
        axes[fi, 0].set_ylabel(METRIC_LABELS[feat], fontsize=12, fontweight='bold')
    axes[nrows - 1, 0].set_ylabel('ALL 7 stats\n(LR CV)', fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(OUT, f'roc_{region}.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {outpath}')


# =====================================================
# FIGURE 2: Region comparison — CV AUC per plate
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
        plate_auc, _, _ = cv_auc_multi(df[df['plate'] == plate], cols)
        aucs.append(plate_auc)
    bars = ax.bar(x + ri * w, aucs, w, label=region, alpha=0.85)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.2f}',
                ha='center', va='bottom', fontsize=8)

ax.set_xticks(x + w)
ax.set_xticklabels(PLATES)
ax.set_ylabel('CV AUC (all 7 stats, 5-fold)')
ax.set_title('Region Comparison: Cross-validated Pixel-Stat AUC per Plate')
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
print('\nMultivariate CV AUC (all 7 stats, 5-fold logistic regression):')
for region in REGIONS:
    prefix = region
    cols = [f'{prefix}_mp_{f}' for f in ALL_FEATS]
    cols = [c for c in cols if c in df.columns] or [f'{prefix}_raw_{f}' for f in ALL_FEATS if f'{prefix}_raw_{f}' in df.columns]
    pooled_auc, _, _ = cv_auc_multi(df, cols)
    plate_aucs = []
    for plate in PLATES:
        pa, _, _ = cv_auc_multi(df[df['plate'] == plate], cols)
        plate_aucs.append(f'{pa:.3f}')
    print(f'  {region}: pooled={pooled_auc:.3f}  ' + '  '.join(f'{p}={a}' for p, a in zip(PLATES, plate_aucs)))
