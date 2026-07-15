"""
Random Forest: classify all 96 mutant classes (12 NC + 84 gene KO) using DINOv2 embeddings.
6-fold plate-based CV, confusion matrix annotated with counts,
row-normalized (each row sums to 100%), diagonal = per-class accuracy.
"""
import numpy as np, csv, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(BASE_DIR, "features_all.npz")
CSV_PATH = os.path.join(BASE_DIR, "features_metadata.csv")
OUT_DIR = os.path.join(BASE_DIR, "analysis_figures")
os.makedirs(OUT_DIR, exist_ok=True)

PLATES = ['P1','P2','P3','P4','P5','P6']

data = np.load(NPZ_PATH)
embeddings = data["embeddings"]
with open(CSV_PATH) as f:
    metadata = list(csv.DictReader(f))

sources = np.array([m["source"] for m in metadata])
plates = np.array([m["plate"] for m in metadata])
labels = np.array([m["label"] for m in metadata])

keep = sources == 'mutant'
emb = embeddings[keep].astype(np.float32)
plate = plates[keep]
label = labels[keep]

classes = sorted(set(label))
class_to_idx = {c: i for i, c in enumerate(classes)}
y = np.array([class_to_idx[l] for l in label], dtype=np.int32)

n_total = len(emb)
n_classes = len(classes)
per_class_total = n_total // n_classes
print(f"{n_total} samples, {n_classes} mutant classes ({per_class_total} per class)")

folds = []
for i, test_plate in enumerate(PLATES):
    val_plate = PLATES[(i + 4) % 6]
    train_plates = [p for p in PLATES if p not in (test_plate, val_plate)]
    folds.append((train_plates, val_plate, test_plate))

accs = []
cm_sum = np.zeros((n_classes, n_classes), dtype=np.float32)

pbar = tqdm(total=len(folds), desc="RF", ncols=80)
for train_plates, val_plate, test_plate in folds:
    train_mask = np.array([p in train_plates for p in plate])
    val_mask = plate == val_plate
    test_mask = plate == test_plate

    X_tr = emb[train_mask]; y_tr = y[train_mask]
    X_va = emb[val_mask]; y_va = y[val_mask]
    X_te = emb[test_mask]; y_te = y[test_mask]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    clf = RandomForestClassifier(
        n_estimators=500, max_depth=40, random_state=42,
        n_jobs=-1, class_weight='balanced'
    )
    clf.fit(X_tr_s, y_tr)

    X_te_s = scaler.transform(X_te)
    preds = clf.predict(X_te_s)
    acc = accuracy_score(y_te, preds)
    accs.append(acc)

    cm = confusion_matrix(y_te, preds, labels=range(n_classes))
    cm_sum += cm.astype(np.float32)
    pbar.update(1)
pbar.close()

cm_mean = cm_sum / len(folds)

print(f"\n{'='*60}")
print(f"Test accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Per-fold: {[f'{a:.3f}' for a in accs]}")

class_short = [c.replace('_', ' ') for c in classes]

# Count how many rows have majority on diagonal (diag > off-diag max)
diag = np.diag(cm_mean)
off_diag_max = np.array([np.max(np.delete(cm_mean[i], i)) for i in range(n_classes)])
rows_diag_wins = np.sum(diag > off_diag_max)
print(f"\nRows where diagonal > max off-diagonal: {rows_diag_wins}/{n_classes}")

# Annotation: show counts in each cell
# For readability, only annotate cells with >= 1 count (mean across folds)
# Row-normalized confusion matrix
cm_norm = cm_mean / (cm_mean.sum(axis=1, keepdims=True) + 1e-10)
per_class_acc = np.diag(cm_norm)

# Counts + row-normalized %, annotated only on diagonal
annot_diag = np.empty_like(cm_mean, dtype=object)
annot_diag[:, :] = ''
for i in range(n_classes):
    cnt = cm_mean[i, i]
    pct = cm_norm[i, i]
    annot_diag[i, i] = f'{cnt:.1f}\n({pct*100:.0f}%)'

fig, ax = plt.subplots(figsize=(32, 28))
sns.heatmap(cm_norm, cmap='Blues', vmin=0, vmax=0.5,
            xticklabels=class_short, yticklabels=class_short,
            ax=ax, annot=annot_diag, fmt='',
            cbar_kws={'label': 'Fraction of true class'},
            linewidths=0.3, linecolor='gray',
            square=True)
ax.set_xlabel('Predicted Class', fontsize=14)
ax.set_ylabel('True Class', fontsize=14)
ax.set_title(f'Mutant — Random Forest Confusion Matrix (6-fold CV, row-normalized)', fontsize=15, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=5)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'rf_mutant_confusion_norm.png'), dpi=200, bbox_inches='tight')
plt.close()

# Annotated counts version: diagonal = count + pct, off-diagonal only if >=3 (>2% of row)
annot_all = np.empty_like(cm_mean, dtype=object)
for i in range(n_classes):
    for j in range(n_classes):
        cnt = cm_mean[i, j]
        if i == j:
            annot_all[i, j] = f'{cnt:.1f}\n({cm_norm[i,i]*100:.0f}%)'
        elif cnt >= 3.0:
            annot_all[i, j] = f'{cnt:.0f}'
        else:
            annot_all[i, j] = ''

fig, ax = plt.subplots(figsize=(32, 28))
sns.heatmap(cm_mean, cmap='Blues', vmin=0, vmax=per_class_total,
            xticklabels=class_short, yticklabels=class_short,
            ax=ax, annot=annot_all, fmt='',
            cbar_kws={'label': 'Mean predicted count (6-fold)'},
            linewidths=0.3, linecolor='gray',
            square=True)
ax.set_xlabel('Predicted Class', fontsize=14)
ax.set_ylabel('True Class', fontsize=14)
ax.set_title('Mutant — Random Forest Confusion Matrix (6-fold CV, annotated counts)', fontsize=15, fontweight='bold')
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=5)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'rf_mutant_confusion_counts.png'), dpi=200, bbox_inches='tight')
plt.close()

# Per-class accuracy bar chart
fig, ax = plt.subplots(figsize=(18, 7))
colors = ['#e41a1c' if v < np.mean(per_class_acc) else '#377eb8' for v in per_class_acc]
bars = ax.bar(range(n_classes), per_class_acc, color=colors, edgecolor='white', linewidth=0.3)
ax.axhline(np.mean(per_class_acc), color='black', linestyle='--', linewidth=1,
           label=f'Mean={np.mean(per_class_acc):.3f}')
ax.set_xticks(range(n_classes))
ax.set_xticklabels(class_short, rotation=90, fontsize=5)
ax.set_ylabel('Per-Class Accuracy', fontsize=12)
ax.set_title(f'Mutant — Per-Class Accuracy (RF, mean={np.mean(per_class_acc):.3f})', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'rf_mutant_per_class_acc.png'), dpi=200, bbox_inches='tight')
plt.close()

# Print per-class accuracies
print(f"\nPer-class accuracy:")
top_idx = np.argsort(per_class_acc)[::-1]
for rank, i in enumerate(top_idx):
    marker = ' <<<' if per_class_acc[i] >= 0.8 else ''
    print(f"  {classes[i]:12s} {per_class_acc[i]:.3f}{marker}")
    if rank >= 95:
        break

summary = {
    'algorithm': 'RandomForest',
    'num_samples': int(n_total),
    'num_classes': n_classes,
    'n_estimators': 500,
    'max_depth': 40,
    'test_accuracy': {'mean': float(np.mean(accs)), 'std': float(np.std(accs)), 'per_fold': [float(v) for v in accs]},
    'rows_diag_majority': int(rows_diag_wins),
    'mean_per_class_acc': float(np.mean(per_class_acc)),
}
with open(os.path.join(OUT_DIR, 'rf_mutant_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)

np.savetxt(os.path.join(OUT_DIR, 'rf_mutant_confusion_norm.csv'), cm_norm, delimiter=',', fmt='%.6f',
           header=','.join(classes), comments='')

print(f"\n{'='*60}")
print("Outputs in analysis_figures/:")
print("  rf_mutant_confusion_counts.png  (annotated with counts)")
print("  rf_mutant_confusion_norm.png    (row-normalized, count+pct annotations)")
print("  rf_mutant_per_class_acc.png     (bar chart)")
print("  rf_mutant_results.json, rf_mutant_confusion_norm.csv")
