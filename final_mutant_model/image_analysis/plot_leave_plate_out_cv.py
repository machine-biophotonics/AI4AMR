#!/usr/bin/env python3
"""4/1/1 leave-one-plate-out: train on 4 plates, val on 1, test on 1.
All 7 pixel features → LogisticRegression. 6 ROC curves on one plot.
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
parser.add_argument('--region', default='center224', choices=['full', 'center1128', 'center224'])
parser.add_argument('--feat_prefix', default=None,
                    help='Manual feature prefix (e.g. center224_canny_raw_). Overrides auto-detect.')
args = parser.parse_args()

OUT = args.output
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(args.input)

ALL_FEATS = ['mean', 'std', 'snr', 'entropy', 'p1', 'p99', 'median']
PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
PLATE_COLORS = {'P1': '#e41a1c', 'P2': '#377eb8', 'P3': '#4daf4a',
                'P4': '#984ea3', 'P5': '#ff7f00', 'P6': '#a65628'}
PLATE_STYLES = {'P1': '-', 'P2': '--', 'P3': '-.', 'P4': ':', 'P5': '-', 'P6': '--'}

region = args.region
if args.feat_prefix:
    cols = [f'{args.feat_prefix}{f}' for f in ALL_FEATS]
    cols = [c for c in cols if c in df.columns]
else:
    prefix = region
    cols = [f'{prefix}_mp_{f}' for f in ALL_FEATS]
    cols = [c for c in cols if c in df.columns] or \
           [f'{prefix}_raw_{f}' for f in ALL_FEATS if f'{prefix}_raw_{f}' in df.columns]

if not cols:
    raise ValueError(f"No features found. Available columns with '{region}': "
                     f"{[c for c in df.columns if region in c]}")

fig, ax = plt.subplots(figsize=(10, 8))
all_aucs = []
val_aucs = []

for fi, test_plate in enumerate(PLATES):
    # Validation = next plate in cycle
    val_plate = PLATES[(fi + 1) % 6]
    train_plates = [p for p in PLATES if p not in (test_plate, val_plate)]

    train_mask = df['plate'].isin(train_plates)
    val_mask = df['plate'] == val_plate
    test_mask = df['plate'] == test_plate

    X_tr = df.loc[train_mask, cols].values.astype(np.float64)
    y_tr = (df.loc[train_mask, 'type'] == 'drug').astype(int).values
    X_va = df.loc[val_mask, cols].values.astype(np.float64)
    y_va = (df.loc[val_mask, 'type'] == 'drug').astype(int).values
    X_te = df.loc[test_mask, cols].values.astype(np.float64)
    y_te = (df.loc[test_mask, 'type'] == 'drug').astype(int).values

    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs')
    clf.fit(scaler.transform(X_tr), y_tr)

    # Validation AUC
    va_score = clf.predict_proba(scaler.transform(X_va))[:, 1]
    fpr_va, tpr_va, _ = roc_curve(y_va, va_score)
    val_auc_i = auc(fpr_va, tpr_va)
    val_aucs.append(val_auc_i)

    # Test AUC
    te_score = clf.predict_proba(scaler.transform(X_te))[:, 1]
    fpr_te, tpr_te, _ = roc_curve(y_te, te_score)
    test_auc_i = auc(fpr_te, tpr_te)
    all_aucs.append(test_auc_i)

    label = f'{test_plate} (test AUC={test_auc_i:.3f}, val={val_auc_i:.3f})'
    ax.plot(fpr_te, tpr_te, color=PLATE_COLORS[test_plate],
            ls=PLATE_STYLES[test_plate], lw=2, label=label)

mean_auc = np.mean(all_aucs)
std_auc = np.std(all_aucs)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
feat_label = args.feat_prefix.replace('_', ' ').strip() if args.feat_prefix else 'mp'
ax.set_title(f'4/1/1 Leave-Plate-Out CV — {region} ({feat_label})\n'
             f'Test AUC: {mean_auc:.3f} ± {std_auc:.3f} '
             f'(val: {np.mean(val_aucs):.3f} ± {np.std(val_aucs):.3f})',
             fontsize=14)
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_aspect('equal')
plt.tight_layout()

outpath = os.path.join(OUT, f'leave_plate_out_{region}.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {outpath}')
print(f'\nTest AUCs per fold:')
for p, a, v in zip(PLATES, all_aucs, val_aucs):
    print(f'  Train 4 → val={v:.3f}, test={p}: {a:.3f}')
print(f'\nMean test AUC: {mean_auc:.3f} ± {std_auc:.3f}')
