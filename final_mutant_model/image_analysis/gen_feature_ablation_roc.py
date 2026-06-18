#!/usr/bin/env python3
"""Leave-one-plate-out ROC with every individual feature + every combination of 2..7 features.
Output: image_analysis/1_feature/ {roc_curves/, individual_bar.png, combination_bar.png, results.csv}
"""

import argparse, os, warnings, itertools, csv
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
parser.add_argument('--output', default='output_all_plates/1_feature')
parser.add_argument('--region', default='center224', choices=['full', 'center1128', 'center224'])
parser.add_argument('--cols', default='mp', choices=['raw', 'mp'],
                    help='Use raw or model-preprocessed columns')
args = parser.parse_args()

OUT = args.output
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, 'roc_curves'), exist_ok=True)

df = pd.read_csv(args.input)

ALL_FEATS = ['mean', 'std', 'snr', 'entropy', 'p1', 'p99', 'median']
FEAT_LABELS = {
    'mean': 'Mean', 'std': 'Std Dev', 'snr': 'SNR', 'entropy': 'Entropy',
    'p1': 'P1', 'p99': 'P99', 'median': 'Median'
}
FEAT_COLORS = dict(zip(ALL_FEATS, plt.cm.Set1(np.linspace(0, 1, 7))))

region = args.region
col_prefix = f'{region}_{args.cols}_'
feature_cols = [f'{col_prefix}{f}' for f in ALL_FEATS]
feature_cols = [c for c in feature_cols if c in df.columns]

if not feature_cols:
    # Fallback: try other prefix
    col_prefix2 = f'{region}_raw_'
    feature_cols = [f'{col_prefix2}{f}' for f in ALL_FEATS]
    feature_cols = [c for c in feature_cols if c in df.columns]
    col_prefix = col_prefix2

print(f"Region: {region}, Prefix: {col_prefix}")
print(f"Features found: {[c.split(col_prefix)[1] for c in feature_cols]}")

PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
PLATE_COLORS = {'P1': '#e41a1c', 'P2': '#377eb8', 'P3': '#4daf4a',
                'P4': '#984ea3', 'P5': '#ff7f00', 'P6': '#a65628'}
PLATE_STYLES = {'P1': '-', 'P2': '--', 'P3': '-.', 'P4': ':', 'P5': '-', 'P6': '--'}


def leave_one_plate_out_lr(cols, df, region_label=''):
    """4/1/1 leave-plate-out: train on 4, test on 1. Returns list of (test_plate, auc, fpr, tpr)."""
    results = []
    for fi, test_plate in enumerate(PLATES):
        val_plate = PLATES[(fi + 1) % 6]
        train_plates = [p for p in PLATES if p not in (test_plate, val_plate)]

        X_tr = df.loc[df['plate'].isin(train_plates), cols].values.astype(np.float64)
        y_tr = (df.loc[df['plate'].isin(train_plates), 'type'] == 'drug').astype(int).values
        X_te = df.loc[df['plate'] == test_plate, cols].values.astype(np.float64)
        y_te = (df.loc[df['plate'] == test_plate, 'type'] == 'drug').astype(int).values

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            results.append((test_plate, 0.5, np.array([0, 1]), np.array([0, 1])))
            continue

        scaler = StandardScaler().fit(X_tr)
        clf = LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs')
        clf.fit(scaler.transform(X_tr), y_tr)
        y_score = clf.predict_proba(scaler.transform(X_te))[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_score)
        fold_auc = auc(fpr, tpr)
        results.append((test_plate, fold_auc, fpr, tpr))
    return results


def plot_roc_curves(results, title, outpath, show_legend=True):
    """Plot ROC curves from leave-one-plate-out results."""
    fig, ax = plt.subplots(figsize=(10, 8))
    aucs = []
    for test_plate, fold_auc, fpr, tpr in results:
        aucs.append(fold_auc)
        ax.plot(fpr, tpr, color=PLATE_COLORS[test_plate],
                ls=PLATE_STYLES[test_plate], lw=2,
                label=f'Test={test_plate} (AUC={fold_auc:.3f})')
    mean_auc = np.mean(aucs)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.5)')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(f'{title}\nMean AUC = {mean_auc:.3f} ± {np.std(aucs):.3f}', fontsize=14)
    if show_legend:
        ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return mean_auc, np.std(aucs), aucs


# ========================================================================
# 1. INDIVIDUAL FEATURES — full ROC + bar plot
# ========================================================================
print("\n" + "="*60)
print("INDIVIDUAL FEATURES (k=1)")
print("="*60)

individual_results = {}  # feature_name -> (mean_auc, std_auc, [plate_aucs])

for feat_short, col in zip(ALL_FEATS, feature_cols):
    feat_name = FEAT_LABELS[feat_short]
    print(f"\n  {feat_name} ({col})...")
    results = leave_one_plate_out_lr([col], df, region)
    outpath = os.path.join(OUT, 'roc_curves', f'{feat_short}.png')
    title = f'Leave-Plate-Out — {feat_name}\n{region} ({args.cols} pixels)'
    mean_auc, std_auc, aucs = plot_roc_curves(results, title, outpath)
    individual_results[feat_short] = {
        'name': feat_name, 'mean': mean_auc, 'std': std_auc, 'aucs': aucs,
        'n_feats': 1, 'combination': [feat_short]
    }
    print(f"    Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}")

# Individual bar plot
fig, ax = plt.subplots(figsize=(12, 6))
feat_names = [individual_results[f]['name'] for f in ALL_FEATS]
means = [individual_results[f]['mean'] for f in ALL_FEATS]
stds = [individual_results[f]['std'] for f in ALL_FEATS]
colors = [FEAT_COLORS[f] for f in ALL_FEATS]
x_pos = np.arange(len(ALL_FEATS))

bars = ax.bar(x_pos, means, yerr=stds, capsize=6, color=colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(feat_names, fontsize=12)
ax.set_ylabel('Mean AUC (leave-one-plate-out)', fontsize=13)
ax.set_title(f'Individual Feature AUC — {region} ({args.cols} pixels)\n(6-fold leave-plate-out)', fontsize=14)
ax.set_ylim(0.4, 1.05)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
ax.legend(fontsize=10)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{m:.3f}\n±{s:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
outpath = os.path.join(OUT, 'individual_features_bar.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved {outpath}")


# ========================================================================
# 2. COMBINATIONS OF k=2..7 features
# ========================================================================
all_combination_results = {}  # n_feats -> [(mean_auc, std, combination, aucs)]

for k in range(2, 8):
    print(f"\n" + "="*60)
    print(f"FEATURE COMBINATIONS (k={k}) — C(7,{k}) = {len(list(itertools.combinations(ALL_FEATS, k)))}")
    print("="*60)

    best_mean = -1
    best_info = None
    combo_results = []

    for combo in itertools.combinations(ALL_FEATS, k):
        cols = [f'{col_prefix}{f}' for f in combo]
        cols = [c for c in cols if c in df.columns]
        if len(cols) != k:
            continue
        results = leave_one_plate_out_lr(cols, df, region)
        aucs = [r[1] for r in results]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        combo_results.append((mean_auc, std_auc, combo, aucs))

        if mean_auc > best_mean:
            best_mean = mean_auc
            best_info = (mean_auc, std_auc, combo, aucs, results)

    # Plot best combination of size k
    if best_info is not None:
        mean_auc, std_auc, combo, aucs, results = best_info
        combo_short = '+'.join(combo)
        combo_label = ' + '.join(FEAT_LABELS[f] for f in combo)
        outpath = os.path.join(OUT, 'roc_curves', f'best_{k}_feat.png')
        title = f'Best {k}-Feature Combination — {combo_short}\n{region} ({args.cols} pixels)'
        plot_roc_curves(results, title, outpath)
        print(f"\n  Best {k}-feature: {combo_short}")
        print(f"    Mean AUC = {mean_auc:.3f} ± {std_auc:.3f}")

    all_combination_results[k] = combo_results

# Also do all 7 features
print(f"\n" + "="*60)
print("ALL 7 FEATURES (k=7)")
print("="*60)
cols_all = [f'{col_prefix}{f}' for f in ALL_FEATS]
results_all = leave_one_plate_out_lr(cols_all, df, region)
aucs_all = [r[1] for r in results_all]
mean_all = np.mean(aucs_all)
std_all = np.std(aucs_all)
outpath = os.path.join(OUT, 'roc_curves', 'all_7_features.png')
title = f'All 7 Features — {region} ({args.cols} pixels)'
plot_roc_curves(results_all, title, outpath)
print(f"  Mean AUC = {mean_all:.3f} ± {std_all:.3f}")


# ========================================================================
# 3. SAVE FULL RESULTS CSV
# ========================================================================
print(f"\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

csv_path = os.path.join(OUT, 'combination_results.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['k', 'combination', 'feature_names', 'mean_auc', 'std_auc',
                     'p1_auc', 'p2_auc', 'p3_auc', 'p4_auc', 'p5_auc', 'p6_auc'])

    # Individual
    for f in ALL_FEATS:
        r = individual_results[f]
        writer.writerow([1, f, r['name'], f'{r["mean"]:.4f}', f'{r["std"]:.4f}'] +
                        [f'{a:.4f}' for a in r['aucs']])

    # Combinations 2-7
    for k in range(2, 8):
        for mean_auc, std_auc, combo, aucs in all_combination_results[k]:
            combo_name = '+'.join(combo)
            combo_labels = ' + '.join(FEAT_LABELS[f] for f in combo)
            writer.writerow([k, combo_name, combo_labels, f'{mean_auc:.4f}', f'{std_auc:.4f}'] +
                            [f'{a:.4f}' for a in aucs])

    # All 7
    writer.writerow([7, '+'.join(ALL_FEATS), 'All 7 Features', f'{mean_all:.4f}', f'{std_all:.4f}'] +
                    [f'{a:.4f}' for a in aucs_all])

print(f"Saved {csv_path}")


# ========================================================================
# 4. FINAL BAR PLOT: Best of each k (1..7), ascending
# ========================================================================
print(f"\n" + "="*60)
print("FINAL BEST-K BAR PLOT")
print("="*60)

best_per_k = {}

# k=1
best_1 = max(individual_results.values(), key=lambda x: x['mean'])
best_per_k[1] = {
    'name': best_1['name'],
    'short': best_1['combination'][0],
    'mean': best_1['mean'], 'std': best_1['std'],
    'aucs': best_1['aucs'], 'n_feats': 1
}
print(f"  k=1: {best_1['name']} — AUC = {best_1['mean']:.3f} ± {best_1['std']:.3f}")

# k=2..6
for k in range(2, 7):
    combo_results = all_combination_results[k]
    best = max(combo_results, key=lambda x: x[0])
    best_mean, best_std, best_combo, best_aucs = best
    best_per_k[k] = {
        'name': ' + '.join(FEAT_LABELS[f] for f in best_combo),
        'short': '+'.join(best_combo),
        'mean': best_mean, 'std': best_std, 'aucs': best_aucs, 'n_feats': k
    }
    print(f"  k={k}: {best_per_k[k]['short']} — AUC = {best_mean:.3f} ± {best_std:.3f}")

# k=7
best_per_k[7] = {
    'name': 'All 7 Features',
    'short': '+'.join(ALL_FEATS),
    'mean': mean_all, 'std': std_all, 'aucs': aucs_all, 'n_feats': 7
}
print(f"  k=7: All 7 — AUC = {mean_all:.3f} ± {std_all:.3f}")

# Final bar plot
fig, ax = plt.subplots(figsize=(12, 7))
ks = list(range(1, 8))
k_means = [best_per_k[k]['mean'] for k in ks]
k_stds = [best_per_k[k]['std'] for k in ks]
k_labels = [best_per_k[k]['short'] for k in ks]
k_names = [best_per_k[k]['name'] for k in ks]

colors_k = plt.cm.Blues(np.linspace(0.4, 0.9, 7))
bars = ax.bar(ks, k_means, yerr=k_stds, capsize=6, color=colors_k,
              edgecolor='black', linewidth=0.5)
ax.set_xticks(ks)
ax.set_xticklabels(k_labels, fontsize=11, rotation=45, ha='right')
ax.set_xlabel('Number of Features (k)', fontsize=13)
ax.set_ylabel('Mean AUC (leave-one-plate-out)', fontsize=13)
ax.set_title(f'Best AUC vs Number of Features — {region} ({args.cols} pixels)\n'
             f'{len(PLATES)}-fold leave-plate-out', fontsize=14)
ax.set_ylim(0.4, 1.05)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
ax.legend(fontsize=10)

# Annotate bars
for bar, m, s, name in zip(bars, k_means, k_stds, k_names):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{m:.3f}\n±{s:.3f}', ha='center', va='bottom', fontsize=9)

# Add a second row showing the actual features
for i, (bar, name) in enumerate(zip(bars, k_names)):
    ax.text(bar.get_x() + bar.get_width()/2, 0.42,
            name, ha='center', va='bottom', fontsize=7, rotation=45)

plt.tight_layout()
outpath = os.path.join(OUT, 'best_k_combination.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved {outpath}")


# ========================================================================
# 5. SUMMARY PRINT
# ========================================================================
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'k':<5} {'Best Combination':<50} {'AUC':<10} {'Std':<10}")
print("-"*75)
for k in ks:
    info = best_per_k[k]
    print(f"{k:<5} {info['short']:<50} {info['mean']:<10.3f} {info['std']:<10.3f}")
print(f"\nAll output saved to: {OUT}")
