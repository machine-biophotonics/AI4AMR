#!/usr/bin/env python3
"""Generate per-sigma ROC figures + comparison bar chart from edge features CSV.

For each Canny sigma: edge_roc_canny_sigma{X}.png  (3 subplots)
Sobel (if present):  edge_roc_sobel.png
Comparison chart:     edge_auc_vs_sigma.png
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
parser.add_argument('--input', default='output_all_plates/all_plates_features_edge.csv')
parser.add_argument('--output', default='output_all_plates')
args = parser.parse_args()

OUT = args.output
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(args.input)

ALL_FEATS = ['mean', 'std', 'snr', 'entropy', 'p1', 'p99', 'median']
PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
REGIONS = ['full', 'center1128', 'center224']
REGION_LABELS = ['Full (2720×2720)', 'Center 1128×1128', 'Center 224×224']
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

# Detect available Canny sigmas and Sobel from CSV columns
import re
canny_sigmas = set()
has_sobel = False
for c in df.columns:
    m = re.search(r'canny_sigma([\d.]+)_mp_', c)
    if m:
        canny_sigmas.add(float(m.group(1)))
    if '_sobel_mp_' in c:
        has_sobel = True
canny_sigmas = sorted(canny_sigmas)
print(f"Found Canny sigmas: {canny_sigmas}")
print(f"Sobel: {'yes' if has_sobel else 'no'}")

def run_lpo(df, prefix, feats=None):
    """4/1/1 leave-plate-out. Returns list of (fpr, tpr, auc, plate)."""
    if feats is None:
        feats = ALL_FEATS
    cols = [f'{prefix}{f}' for f in feats]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return []

    results = []
    for fi, test_plate in enumerate(PLATES):
        val_plate = PLATES[(fi + 1) % 6]
        train_plates = [p for p in PLATES if p not in (test_plate, val_plate)]

        train_mask = df['plate'].isin(train_plates)
        test_mask = df['plate'] == test_plate

        X_tr = df.loc[train_mask, cols].values.astype(np.float64)
        y_tr = (df.loc[train_mask, 'type'] == 'drug').astype(int).values
        X_te = df.loc[test_mask, cols].values.astype(np.float64)
        y_te = (df.loc[test_mask, 'type'] == 'drug').astype(int).values

        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs')
        clf.fit(scaler.transform(X_tr), y_tr)

        score = clf.predict_proba(scaler.transform(X_te))[:, 1]
        fpr, tpr, _ = roc_curve(y_te, score)
        roc_auc = auc(fpr, tpr)
        results.append((fpr, tpr, roc_auc, test_plate))

    return results

def make_interp_roc(results):
    all_fpr = np.linspace(0, 1, 100)
    tprs = []
    for fpr, tpr, _, _ in results:
        tprs.append(np.interp(all_fpr, fpr, tpr))
    tprs = np.array(tprs)
    return all_fpr, tprs.mean(axis=0), tprs.std(axis=0)

def plot_roc(feat_type, feat_name, filename):
    """Generate a 3-subplot ROC figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ri, region in enumerate(REGIONS):
        ax = axes[ri]
        prefix = f'{region}_{feat_type}'

        results = run_lpo(df, prefix)
        if not results:
            ax.text(0.5, 0.5, f'No features found', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        fold_aucs = []
        for fpr, tpr, roc_auc, plate in results:
            ax.plot(fpr, tpr, color=COLORS[PLATES.index(plate)],
                    lw=0.8, alpha=0.5, label=f'{plate} (AUC={roc_auc:.3f})')
            fold_aucs.append(roc_auc)

        interp_fpr, mean_tpr, std_tpr = make_interp_roc(results)
        ax.plot(interp_fpr, mean_tpr, 'k-', lw=2.5,
                label=f'Mean AUC={np.mean(fold_aucs):.3f}±{np.std(fold_aucs):.3f}')
        ax.fill_between(interp_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color='gray', alpha=0.15)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'{REGION_LABELS[ri]}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')

    fig.suptitle(f'{feat_name} — 4/1/1 Leave-Plate-Out CV (Model-Preprocessed)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    outpath = os.path.join(OUT, filename)
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {outpath}")

    mean_aucs = []
    for r in REGIONS:
        res = run_lpo(df, f'{r}_{feat_type}')
        if res:
            aucs = [x[2] for x in res]
            mean_aucs.append(f'{r}: {np.mean(aucs):.3f}+-{np.std(aucs):.3f}')
    print(f"  Mean AUCs: {', '.join(mean_aucs)}")
    return {r: [x[2] for x in run_lpo(df, f'{r}_{feat_type}')] for r in REGIONS}

# === Per-sigma Canny plots ===
all_sigma_results = {}
for sigma in canny_sigmas:
    feat_type = f'canny_sigma{sigma}_mp_'
    feat_name = f'Canny σ={sigma} (edge density)'
    fname = f'edge_roc_canny_sigma{sigma}.png'
    results = plot_roc(feat_type, feat_name, fname)
    all_sigma_results[sigma] = results

# === Sobel plot (if available) ===
if has_sobel:
    plot_roc('sobel_mp_', 'Sobel (gradient magnitude)', 'edge_roc_sobel.png')

# === Comparison bar chart: AUC vs sigma for center224 ===
fig, ax = plt.subplots(figsize=(10, 6))
sigmas = list(all_sigma_results.keys())
mean_aucs = [np.mean(all_sigma_results[s].get('center224', [0.5])) for s in sigmas]
std_aucs = [np.std(all_sigma_results[s].get('center224', [0])) for s in sigmas]

bars = ax.bar([str(s) for s in sigmas], mean_aucs, yerr=std_aucs,
              capsize=5, color='steelblue', alpha=0.8)

ax.axhline(0.5, color='k', ls='--', alpha=0.4, label='Random (0.5)')
ax.set_ylabel('Mean Test AUC', fontsize=12)
ax.set_xlabel('Canny Sigma', fontsize=12)
ax.set_title('Edge Density Confound vs Canny Blur Level\n(center224, 4/1/1 LPO)',
             fontsize=14, fontweight='bold')
ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=10)

for bar, val in zip(bars, mean_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', fontsize=10)

outpath = os.path.join(OUT, 'edge_auc_vs_sigma.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved {outpath}")
