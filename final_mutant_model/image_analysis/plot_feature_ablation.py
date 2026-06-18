#!/usr/bin/env python3
"""Feature ablation: find minimal subsets of pixel stats that give strong confound AUC."""

import argparse, os, warnings, itertools
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
args = parser.parse_args()

OUT = args.output
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(args.input)

ALL_FEATS = ['mean', 'std', 'snr', 'entropy', 'p1', 'p99', 'median']
PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
prefix = args.region
cols = {f: f'{prefix}_mp_{f}' for f in ALL_FEATS}
cols = {k: v for k, v in cols.items() if v in df.columns}
if not cols:
    cols = {k: f'{prefix}_raw_{k}' for k in ALL_FEATS if f'{prefix}_raw_{k}' in df.columns}
feat_names = list(cols.keys())
feat_cols = list(cols.values())


def lr_auc(df_subset, use_cols):
    X = df_subset[use_cols].values.astype(np.float64)
    y = (df_subset['type'] == 'drug').astype(int).values
    if len(np.unique(y)) < 2 or len(y) < 4:
        return 0.5
    Xs = StandardScaler().fit_transform(X)
    try:
        clf = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs').fit(Xs, y)
    except:
        clf = LogisticRegression(penalty='l2', C=1e6, max_iter=5000, solver='lbfgs').fit(Xs, y)
    return auc(*roc_curve(y, clf.predict_proba(Xs)[:, 1])[:2])


def run_ablation(k):
    """Test all k-feature subsets, return sorted results."""
    results = []
    for subset in itertools.combinations(feat_names, k):
        subcols = [cols[f] for f in subset]
        pooled_auc = lr_auc(df, subcols)
        plate_aucs = [lr_auc(df[df['plate'] == p], subcols) for p in PLATES]
        mean_plate = np.mean(plate_aucs)
        results.append((set(subset), pooled_auc, mean_plate, plate_aucs))
    results.sort(key=lambda x: -x[1])
    return results


print(f"Feature ablation on {args.region} (all 7 stats, {len(ALL_FEATS)} total)")
print()

# All 7 as baseline
baseline_auc = lr_auc(df, feat_cols)
print(f"All 7 features: pooled AUC = {baseline_auc:.4f}\n")

# k=1
print("=== Best SINGLE features ===")
singles = run_ablation(1)
for subset, pool, mean_p, pl_aucs in singles[:3]:
    print(f"  {subset}: pooled={pool:.4f}, mean_plate={mean_p:.4f}")
print()

# k=2
print("=== Best PAIRS (which 2 are enough?) ===")
pairs = run_ablation(2)
for subset, pool, mean_p, pl_aucs in pairs[:5]:
    print(f"  {subset}: pooled={pool:.4f}, mean_plate={mean_p:.4f}")
top2 = pairs[0]
print()

# k=3
print("=== Best TRIPLETS (which 3 are enough?) ===")
triplets = run_ablation(3)
for subset, pool, mean_p, pl_aucs in triplets[:5]:
    print(f"  {subset}: pooled={pool:.4f}, mean_plate={mean_p:.4f}")
top3 = triplets[0]
print(f"  -- All 7: pooled={baseline_auc:.4f}")
print()

# =====================================================
# Plot: ALL subsets for k=2..6
# =====================================================
all_results = {}
for k in range(2, 7):
    all_results[k] = run_ablation(k)

fig, axes = plt.subplots(1, 5, figsize=(30, 10))
for ki, k in enumerate([2, 3, 4, 5, 6]):
    ax = axes[ki]
    results = all_results[k]
    labels = ['+'.join(sorted(s)) for s, _, _, _ in results]
    aucs = [r[1] for r in results]
    colors = plt.cm.RdYlGn(np.interp(aucs, [0.5, 1.0], [0, 1]))
    bars = ax.barh(range(len(labels)), aucs, color=colors, edgecolor='gray')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0.4, 1.02)
    ax.axvline(0.5, color='gray', ls='--', alpha=0.5)
    ax.axvline(baseline_auc, color='red', ls=':', lw=1.5, label=f'All 7 = {baseline_auc:.3f}')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title(f'All {k}-feature subsets\n({len(results)} total)', fontsize=11)
    ax.invert_yaxis()
    # Annotate only best/worst to avoid clutter
    if len(bars) > 0:
        ax.text(bars[0].get_width() + 0.005, bars[0].get_y() + bars[0].get_height()/2,
                f'{aucs[0]:.3f}', va='center', fontsize=7, fontweight='bold')
        ax.text(bars[-1].get_width() + 0.005, bars[-1].get_y() + bars[-1].get_height()/2,
                f'{aucs[-1]:.3f}', va='center', fontsize=7, fontweight='bold')

plt.suptitle(f'Feature Ablation — {args.region}', fontsize=14, y=1.02)
plt.tight_layout()
outpath = os.path.join(OUT, 'roc_feature_ablation.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {outpath}')

# Per-plate AUC table for best subsets of each size
print("Per-plate comparison (best subset of each size):")
header = f"{'plate':>6}"
for k in range(1, 8):
    header += f"  {'best'+str(k):>8}"
print(header)
best_by_k = {}
for k in range(1, 7):
    res = run_ablation(k)
    best_by_k[k] = ([cols[f] for f in res[0][0]], res[0][1])
best_by_k[7] = (feat_cols, baseline_auc)
for plate in PLATES:
    mask = df['plate'] == plate
    parts = [f"{plate:>6}"]
    for k in range(1, 8):
        a = lr_auc(df[mask], best_by_k[k][0])
        parts.append(f"{a:>8.3f}")
    print("  ".join(parts))
