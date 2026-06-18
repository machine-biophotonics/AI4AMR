#!/usr/bin/env python3
"""Feature ablation with per-plate error bars.
   FIXED: Uses 5-fold cross-validated AUC per plate.
   For each k in 2..7: find best subset by mean plate CV AUC,
   show mean ± std across plates + individual plate values."""
import argparse, os, warnings, itertools
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
parser.add_argument('--region', default='center1128')
args = parser.parse_args()

OUT = args.output
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(args.input)

ALL_FEATS = ['mean', 'std', 'snr', 'entropy', 'p1', 'p99', 'median']
PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
PLATE_COLORS = {'P1': '#e41a1c', 'P2': '#377eb8', 'P3': '#4daf4a', 'P4': '#984ea3', 'P5': '#ff7f00', 'P6': '#a65628'}
prefix = args.region
cols_map = {f: f'{prefix}_mp_{f}' for f in ALL_FEATS}
cols_map = {k: v for k, v in cols_map.items() if v in df.columns}
if not cols_map:
    cols_map = {k: f'{prefix}_raw_{k}' for k in ALL_FEATS if f'{prefix}_raw_{k}' in df.columns}
feat_names = list(cols_map.keys())
feat_cols = list(cols_map.values())

def lr_cv_auc(df_subset, use_cols):
    """5-fold cross-validated AUC for logistic regression."""
    X = df_subset[use_cols].values.astype(np.float64)
    y = (df_subset['type'] == 'drug').astype(int).values
    if len(np.unique(y)) < 2 or len(y) < 4:
        return 0.5
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        try:
            clf = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs')
            clf.fit(X_tr, y[train_idx])
        except:
            clf = LogisticRegression(penalty='l2', C=1e6, max_iter=5000, solver='lbfgs')
            clf.fit(X_tr, y[train_idx])
        y_score = clf.predict_proba(X_va)[:, 1]
        fpr_v, tpr_v, _ = roc_curve(y[val_idx], y_score)
        aucs.append(auc(fpr_v, tpr_v))
    return np.mean(aucs)

# For each k, find best subset (by mean plate CV AUC) and record per-plate AUCs
results = {}
for k in range(2, 8):
    best_mean = -1
    best_subset = None
    best_plate_aucs = None
    for subset in itertools.combinations(feat_names, k):
        subcols = [cols_map[f] for f in subset]
        plate_aucs = []
        for plate in PLATES:
            mask = df['plate'] == plate
            a = lr_cv_auc(df[mask], subcols)
            plate_aucs.append(a)
        mean_p = np.mean(plate_aucs)
        if mean_p > best_mean:
            best_mean = mean_p
            best_subset = subset
            best_plate_aucs = plate_aucs
    results[k] = {
        'subset': best_subset,
        'plate_aucs': best_plate_aucs,
        'mean': best_mean,
        'std': np.std(best_plate_aucs, ddof=1) if len(best_plate_aucs) > 1 else 0,
    }
    print(f'k={k}: best={"+".join(sorted(best_subset)):30s}  mean={best_mean:.3f} ± {np.std(best_plate_aucs, ddof=1):.3f}')

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ks = list(range(2, 8))
means = [results[k]['mean'] for k in ks]
stds = [results[k]['std'] for k in ks]

bars = ax.bar(ks, means, yerr=stds, capsize=5, color='steelblue', alpha=0.7,
              edgecolor='navy', error_kw={'elinewidth': 1.5, 'capthick': 1.5})
ax.set_xticks(ks)
ax.set_xticklabels([f'k={k}\n({"+".join(sorted(results[k]["subset"]))})' for k in ks],
                   fontsize=8, rotation=45, ha='right')
ax.set_xlabel('Number of features', fontsize=12)
ax.set_ylabel('Mean CV AUC across 6 plates (LR, 5-fold per plate)', fontsize=12)
ax.set_title(f'Feature Ablation (5-fold CV) — {args.region}\nBest k-feature subset, LR, mean ± 1 std', fontsize=13)
ax.set_ylim(0.4, 1.05)
ax.axhline(0.5, color='gray', ls='--', alpha=0.5)

for ki, k in enumerate(ks):
    for pi, (plate, auc_val) in enumerate(zip(PLATES, results[k]['plate_aucs'])):
        jitter = np.random.uniform(-0.12, 0.12)
        ax.scatter(k + jitter, auc_val, color=PLATE_COLORS[plate], s=30, zorder=5,
                   edgecolors='white', linewidths=0.5, alpha=0.85)

from matplotlib.patches import Patch
legend_handles = [Patch(color=PLATE_COLORS[p], label=p) for p in PLATES]
ax.legend(handles=legend_handles, title='Plate', fontsize=8, title_fontsize=9, loc='lower right')

plt.tight_layout()
outpath = os.path.join(OUT, 'roc_feature_ablation_error.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved {outpath}')

print(f'\n{"k":>2}  {"subset":30s}  {"mean":>5}  {"std":>5}  {"all7":>6}')
print('-' * 55)
for k in ks:
    r = results[k]
    sub_str = '+'.join(sorted(r['subset']))
    print(f'{k:>2}  {sub_str:30s}  {r["mean"]:.3f}  {r["std"]:.3f}  ', end='')
    for a in r['plate_aucs']:
        print(f'{a:>6.3f}  ', end='')
    print()
