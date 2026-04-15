"""
Plot ROC and Precision-Recall curves for Logistic Regression model
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load labels mapping
with open(os.path.join(BASE_DIR, 'idx_to_label.json'), 'r') as f:
    idx_to_label = json.load(f)
idx_to_label = {int(k): v for k, v in idx_to_label.items()}
num_classes = len(idx_to_label)

# Load test predictions and labels
test_probs = np.load(os.path.join(BASE_DIR, 'test_probs.npy'))
test_labels = np.load(os.path.join(BASE_DIR, 'test_labels.npy'))

print(f"Test probabilities shape: {test_probs.shape}")
print(f"Test labels shape: {test_labels.shape}")
print(f"Number of classes: {num_classes}")

# Binarize labels for multi-class ROC/PR
test_labels_bin = label_binarize(test_labels, classes=range(num_classes))
if test_labels_bin.shape[1] == 1:
    # Binary case (should not happen)
    test_labels_bin = np.hstack([1 - test_labels_bin, test_labels_bin])

# Compute metrics for classes that have at least one positive sample
classes_with_samples = [i for i in range(test_labels_bin.shape[1]) if test_labels_bin[:, i].sum() > 0]
print(f"Classes with positive samples: {len(classes_with_samples)}")

# Compute ROC and PR curves
fpr = {}
tpr = {}
roc_auc = {}
precision = {}
recall = {}
ap = {}

for i in tqdm(classes_with_samples, desc="Computing metrics"):
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], test_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    precision[i], recall[i], _ = precision_recall_curve(test_labels_bin[:, i], test_probs[:, i])
    ap[i] = average_precision_score(test_labels_bin[:, i], test_probs[:, i])

# Get best 8 classes by AUC (ensure unique names)
sorted_by_auc = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)
best_classes = []
seen_names = set()
for i, auc_val in sorted_by_auc:
    name = idx_to_label[i]
    if name not in seen_names:
        best_classes.append(i)
        seen_names.add(name)
        if len(best_classes) >= 8:
            break

# Plot ROC curves
fig_roc, axes_roc = plt.subplots(2, 4, figsize=(16, 8))
axes_roc = axes_roc.flatten()

for idx, i in enumerate(best_classes):
    axes_roc[idx].plot(fpr[i], tpr[i], label=f'AUC = {roc_auc[i]:.2f}')
    axes_roc[idx].plot([0, 1], [0, 1], 'k--')
    axes_roc[idx].set_xlabel('False Positive Rate')
    axes_roc[idx].set_ylabel('True Positive Rate')
    axes_roc[idx].set_title(f'{idx_to_label[i]}')
    axes_roc[idx].legend(loc='lower right')

for j in range(len(best_classes), len(axes_roc)):
    axes_roc[j].axis('off')

plt.suptitle('ROC Curves (Best 8 Classes by AUC)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'roc_curves.png'), dpi=150)
plt.close()
print(f"Saved ROC curves to {os.path.join(BASE_DIR, 'roc_curves.png')}")

# Get best 8 classes by AP (ensure unique names)
sorted_by_ap = sorted(ap.items(), key=lambda x: x[1], reverse=True)
best_classes_ap = []
seen_names_ap = set()
for i, ap_val in sorted_by_ap:
    name = idx_to_label[i]
    if name not in seen_names_ap:
        best_classes_ap.append(i)
        seen_names_ap.add(name)
        if len(best_classes_ap) >= 8:
            break

# Precision-Recall Curve
fig_pr, axes_pr = plt.subplots(2, 4, figsize=(16, 8))
axes_pr = axes_pr.flatten()

for idx, i in enumerate(best_classes_ap):
    axes_pr[idx].plot(recall[i], precision[i], label=f'AP = {ap[i]:.2f}')
    axes_pr[idx].set_xlabel('Recall')
    axes_pr[idx].set_ylabel('Precision')
    axes_pr[idx].set_title(f'{idx_to_label[i]}')
    axes_pr[idx].legend(loc='lower left')

for j in range(len(best_classes_ap), len(axes_pr)):
    axes_pr[j].axis('off')

plt.suptitle('Precision-Recall Curves (Best 8 Classes by AP)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'precision_recall_curves.png'), dpi=150)
plt.close()
print(f"Saved Precision-Recall curves to {os.path.join(BASE_DIR, 'precision_recall_curves.png')}")

# Print average metrics
mean_roc_auc = np.mean([roc_auc[i] for i in classes_with_samples])
mean_ap = np.mean([ap[i] for i in classes_with_samples])

print(f"\n=== Performance Metrics (ALL {len(classes_with_samples)} classes) ===")
print(f"Average ROC AUC: {mean_roc_auc:.4f}")
print(f"Average Precision: {mean_ap:.4f}")

# Top and bottom performing classes by AUC
sorted_by_auc = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 5 Classes by AUC:")
for i, val in sorted_by_auc[:5]:
    print(f"  {idx_to_label[i]}: {val:.4f}")
print(f"Bottom 5 Classes by AUC:")
for i, val in sorted_by_auc[-5:]:
    print(f"  {idx_to_label[i]}: {val:.4f}")

# Top and bottom by AP
sorted_by_ap = sorted(ap.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 5 Classes by AP:")
for i, val in sorted_by_ap[:5]:
    print(f"  {idx_to_label[i]}: {val:.4f}")
print(f"Bottom 5 Classes by AP:")
for i, val in sorted_by_ap[-5:]:
    print(f"  {idx_to_label[i]}: {val:.4f}")

print("\nDone!")