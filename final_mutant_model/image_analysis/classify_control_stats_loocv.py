#!/usr/bin/env python3
"""Leave-one-plate-out classification on multi-region image statistics.
   Random Forest classifier. Outputs confusion matrix + feature importance."""

import numpy as np
import pandas as pd
import os, json, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='control_7stats.csv')
parser.add_argument('--output_dir', default='control_loocv_results')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
inpath = os.path.join(SCRIPT_DIR, args.input)
outdir = os.path.join(SCRIPT_DIR, args.output_dir)
os.makedirs(outdir, exist_ok=True)

print("=" * 60)
print("Leave-One-Plate-Out Classification (3 Regions × 7 Stats)")
print("=" * 60)

df = pd.read_csv(inpath)
print(f"\nLoaded {len(df)} samples from {inpath}")
print(f"  Plates: {sorted(df['plate'].unique())}")
print(f"  Classes: {df['label'].nunique()}")

# Dynamically detect feature columns: all numeric columns except metadata
meta_cols = {'plate', 'well', 'label', 'image', 'path'}
all_cols = set(df.columns)
FEATURES = sorted([c for c in all_cols - meta_cols if c not in meta_cols])
print(f"  Features ({len(FEATURES)}): {FEATURES[:5]}...{FEATURES[-3:]}")

plates = sorted(df['plate'].unique())
all_true = []
all_pred = []
fold_results = []

for held_out in plates:
    print(f"\n{'='*40}")
    print(f"Held-out: {held_out}")
    print(f"{'='*40}")

    train_df = df[df['plate'] != held_out]
    test_df = df[df['plate'] == held_out]

    X_train = train_df[FEATURES].values
    y_train = train_df['label'].values
    X_test = test_df[FEATURES].values
    y_test = test_df['label'].values

    train_classes = set(y_train)
    test_mask = np.array([lbl in train_classes for lbl in y_test])
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    print(f"  Train: {len(X_train)} samples, {len(train_classes)} classes")
    print(f"  Test:  {len(X_test)} samples, {len(set(y_test))} classes")

    clf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    bal = balanced_accuracy_score(y_test, y_pred) * 100
    print(f"  Acc: {acc:.2f}%  Bal: {bal:.2f}%")

    fold_results.append({'held_out': held_out, 'n_train': len(X_train), 'n_test': len(X_test),
                          'accuracy': acc, 'balanced_accuracy': bal})
    all_true.extend(y_test.tolist())
    all_pred.extend(y_pred.tolist())

overall_acc = accuracy_score(all_true, all_pred) * 100
overall_bal = balanced_accuracy_score(all_true, all_pred) * 100

print(f"\n{'='*60}")
print(f"Overall Acc: {overall_acc:.2f}%  Bal: {overall_bal:.2f}%")
print(f"{'='*60}")
print(f"{'Held-out':>10} {'Train':>8} {'Test':>8} {'Acc%':>8} {'Bal%':>8}")
print('-' * 42)
for r in fold_results:
    print(f"{r['held_out']:>10} {r['n_train']:>8} {r['n_test']:>8} {r['accuracy']:>7.2f} {r['balanced_accuracy']:>7.2f}")

# === Aggregated confusion matrix ===
classes_all = sorted(set(all_true + all_pred))
n_classes = len(classes_all)
cm = confusion_matrix(all_true, all_pred, labels=classes_all)

fig, ax = plt.subplots(figsize=(max(14, n_classes * 0.55), max(12, n_classes * 0.5)))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes_all, yticklabels=classes_all,
            cbar_kws={'label': 'Count'}, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'LOOCV Confusion Matrix (3 Regions × 7 Stats)\nAcc={overall_acc:.1f}%, Bal={overall_bal:.1f}%')
plt.xticks(rotation=90, fontsize=6)
plt.yticks(fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'confusion_matrix.png'), dpi=200, bbox_inches='tight')
plt.close()

# Normalized
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(figsize=(max(14, n_classes * 0.55), max(12, n_classes * 0.5)))
sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues', xticklabels=classes_all, yticklabels=classes_all,
            vmin=0, vmax=1, cbar_kws={'label': 'Fraction'}, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'LOOCV Normalized Confusion Matrix\nAcc={overall_acc:.1f}%, Bal={overall_bal:.1f}%')
plt.xticks(rotation=90, fontsize=6)
plt.yticks(fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'confusion_matrix_normalized.png'), dpi=200, bbox_inches='tight')
plt.close()

# === Feature importance (grouped by region) ===
clf_all = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
clf_all.fit(df[FEATURES].values, df['label'].values)
importances = clf_all.feature_importances_

# Color by region
region_colors = {'full': '#2196F3', 'center1128': '#FF5722', 'center224': '#4CAF50'}
colors = [region_colors.get(f.split('_')[0], '#999') for f in FEATURES]

idx_sort = np.argsort(importances)
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(len(FEATURES)), importances[idx_sort], color=[colors[i] for i in idx_sort])
ax.set_yticks(range(len(FEATURES)))
ax.set_yticklabels([FEATURES[i] for i in idx_sort], fontsize=8)
ax.set_xlabel('Importance')
ax.set_title('Feature Importance (colored by region)')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=r) for r, c in region_colors.items()]
ax.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(outdir, 'feature_importance.png'), dpi=200, bbox_inches='tight')
plt.close()

# === Per-class recall ===
report = classification_report(all_true, all_pred, labels=classes_all, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(outdir, 'classification_report.csv'))

# === Save results ===
results = {
    'overall_accuracy': overall_acc,
    'overall_balanced_accuracy': overall_bal,
    'n_total': len(all_true), 'n_classes': n_classes,
    'n_features': len(FEATURES), 'features': FEATURES,
    'feature_importances': {f: float(v) for f, v in zip(FEATURES, importances)},
    'per_fold': fold_results,
}
with open(os.path.join(outdir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nOutputs in {outdir}/:")
for fname in ['confusion_matrix.png', 'confusion_matrix_normalized.png', 'feature_importance.png',
              'classification_report.csv', 'results.json']:
    print(f"  {fname}")
print("=" * 60)
