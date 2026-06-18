#!/usr/bin/env python3
"""Leave-one-plate-out cross-validation: does the confound generalize?"""

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
PLATE_COLORS = {'P1': '#e41a1c', 'P2': '#377eb8', 'P3': '#4daf4a', 'P4': '#984ea3', 'P5': '#ff7f00', 'P6': '#a65628'}

results = []
for region in REGIONS:
    cols = [f'{region}_mp_{f}' for f in ALL_FEATS]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        cols = [f'{region}_raw_{f}' for f in ALL_FEATS if f'{region}_raw_{f}' in df.columns]

    for test_plate in PLATES:
        train_mask = df['plate'] != test_plate
        test_mask = df['plate'] == test_plate

        X_train = df.loc[train_mask, cols].values.astype(np.float64)
        y_train = (df.loc[train_mask, 'type'] == 'drug').astype(int).values
        X_test = df.loc[test_mask, cols].values.astype(np.float64)
        y_test = (df.loc[test_mask, 'type'] == 'drug').astype(int).values

        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs').fit(scaler.transform(X_train), y_train)
        fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(scaler.transform(X_test))[:, 1])
        held_out_auc = auc(fpr, tpr)

        # Within-plate (train+test on same plate)
        X_plate = df.loc[test_mask, cols].values.astype(np.float64)
        y_plate = (df.loc[test_mask, 'type'] == 'drug').astype(int).values
        if len(np.unique(y_plate)) >= 2 and len(y_plate) >= 4:
            Xs_p = StandardScaler().fit_transform(X_plate)
            clf_p = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs').fit(Xs_p, y_plate)
            fpr_p, tpr_p, _ = roc_curve(y_plate, clf_p.predict_proba(Xs_p)[:, 1])
            within_auc = auc(fpr_p, tpr_p)
        else:
            within_auc = np.nan

        results.append({'region': region, 'plate': test_plate, 'held_out': held_out_auc, 'within': within_auc})
        print(f'  {region} / {test_plate}: within={within_auc:.3f}  held-out={held_out_auc:.3f}')

df_res = pd.DataFrame(results)

# Plot: grouped bars per region
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
x = np.arange(len(PLATES))
w = 0.35

for ri, region in enumerate(REGIONS):
    ax = axes[ri]
    sub = df_res[df_res['region'] == region]
    within_vals = sub.set_index('plate').loc[PLATES, 'within'].values
    heldout_vals = sub.set_index('plate').loc[PLATES, 'held_out'].values

    bars1 = ax.bar(x - w/2, within_vals, w, label='Within-plate', color='#2ca02c', alpha=0.85)
    bars2 = ax.bar(x + w/2, heldout_vals, w, label='Held-out', color='#d62728', alpha=0.85)

    for bar, v in zip(bars1, within_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.2f}',
                    ha='center', va='bottom', fontsize=8)
    for bar, v in zip(bars2, heldout_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{v:.2f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(PLATES)
    ax.set_title(f'Region: {region}', fontsize=12)
    ax.set_ylabel('AUC')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
    if ri == 2:
        ax.legend(fontsize=9)

plt.suptitle('Leave-One-Plate-Out: Within-plate vs Held-out AUC (all 7 stats)', fontsize=14, y=1.02)
plt.tight_layout()
outpath = os.path.join(OUT, 'roc_crossfold.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved {outpath}')
