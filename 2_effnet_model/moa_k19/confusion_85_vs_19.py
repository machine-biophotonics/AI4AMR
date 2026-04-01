#!/usr/bin/env python3
"""
Clean confusion matrix: 85 True Classes vs 19 MOA Clusters
No pathways - just raw class to cluster mapping
"""

import numpy as np
import json
import os
import re
import ast
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SEED = 42
np.random.seed(SEED)

BASE_DIR = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging'
EFFNET_DIR = os.path.join(BASE_DIR, 'effnet_model', 'eval_results')
OUTPUT_DIR = os.path.join(BASE_DIR, 'moa_k19')

# Load data
with open(os.path.join(EFFNET_DIR, 'idx_to_label.json'), 'r') as f:
    idx_to_label = {int(k): v for k, v in json.load(f).items()}

embeddings = np.load(os.path.join(EFFNET_DIR, 'test_embeddings.npy'))
labels = np.load(os.path.join(EFFNET_DIR, 'test_labels.npy'))

# Class centroids
class_labels = [idx_to_label.get(l, 'WT') for l in labels]
unique_classes = sorted(set(class_labels))

class_to_embeddings = defaultdict(list)
for emb, cls in zip(embeddings, class_labels):
    class_to_embeddings[cls].append(emb)

class_embeddings = {cls: np.mean(embs, axis=0) for cls, embs in class_to_embeddings.items()}
class_names = list(class_embeddings.keys())
X_centroids = np.array([class_embeddings[c] for c in class_names])

# Clustering k=19
BEST_K = 19
kmeans = KMeans(n_clusters=BEST_K, random_state=SEED, n_init=10)
class_cluster_labels = kmeans.fit_predict(X_centroids)
class_to_cluster = {class_names[i]: class_cluster_labels[i] for i in range(len(class_names))}

# Print cluster assignments
print("="*70)
print("85 CLASSES -> 19 MOA CLUSTERS")
print("="*70)

for cluster in range(BEST_K):
    classes = [class_names[i] for i in range(len(class_names)) if class_cluster_labels[i] == cluster]
    print(f"\nMOA Cluster {cluster} ({len(classes)} classes):")
    print(f"  {classes}")

# Map to crops
crop_moa = [class_to_cluster.get(class_labels[i], 0) for i in range(len(labels))]

# True labels and predictions
y_true = [idx_to_label.get(l, 'WT') for l in labels]
y_pred = [f'MOA-{c}' for c in crop_moa]

# Sort classes: WT first, then gene_alpha order
sorted_classes = sorted(class_names, key=lambda x: (x != 'WT', x))

# Confusion matrix - use crosstab to ensure all 19 columns
cluster_labels = [f'MOA-{i}' for i in range(BEST_K)]
cm = pd.crosstab(pd.Series(y_true, name='True'), pd.Series(y_pred, name='Pred'))
# Reindex to ensure all 85 rows and 19 columns
cm = cm.reindex(index=sorted_classes, columns=cluster_labels, fill_value=0)
cm = cm.values

# =============================================================================
# FIGURE 1: Raw counts heatmap
# =============================================================================
print("\nGenerating confusion matrix...")

fig, ax = plt.subplots(figsize=(24, 20))

sns.heatmap(cm, 
            cmap='Blues',
            xticklabels=[f'C{i}' for i in range(BEST_K)],
            yticklabels=sorted_classes,
            ax=ax,
            cbar_kws={'label': 'Count', 'shrink': 0.6},
            linewidths=0.1,
            linecolor='gray')

ax.set_title('Confusion Matrix: 85 True Classes vs 19 MOA Clusters\n(Row = True Class, Column = Predicted Cluster)', 
             fontsize=16, pad=20)
ax.set_xlabel('Predicted MOA Cluster (C0-C18)', fontsize=14)
ax.set_ylabel('True Class', fontsize=12)
ax.tick_params(axis='y', labelsize=7)

# Add cluster numbers at top
for i in range(BEST_K):
    ax.text(i + 0.5, -0.5, f'{i}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_85_vs_19_raw.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_85_vs_19_raw.png")

# =============================================================================
# FIGURE 2: Normalized (each row = 100%)
# =============================================================================
fig, ax = plt.subplots(figsize=(24, 20))

cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
cm_norm = np.nan_to_num(cm_norm)

sns.heatmap(cm_norm,
            cmap='YlOrRd',
            xticklabels=[f'C{i}' for i in range(BEST_K)],
            yticklabels=sorted_classes,
            ax=ax,
            annot=False,
            cbar_kws={'label': '% of True Class', 'shrink': 0.6},
            linewidths=0.1,
            linecolor='gray',
            vmin=0, vmax=100)

ax.set_title('Confusion Matrix (Normalized): 85 True Classes vs 19 MOA Clusters\n(Row = 100% of True Class samples)', 
             fontsize=16, pad=20)
ax.set_xlabel('Predicted MOA Cluster (C0-C18)', fontsize=14)
ax.set_ylabel('True Class', fontsize=12)
ax.tick_params(axis='y', labelsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_85_vs_19_normalized.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_85_vs_19_normalized.png")

# =============================================================================
# FIGURE 3: Compact version with annotation
# =============================================================================
fig, ax = plt.subplots(figsize=(26, 22))

# Create annotation array (only show non-zero values)
annot = np.where(cm > 0, cm.astype(str), '')

sns.heatmap(cm,
            cmap='Blues',
            xticklabels=[f'MOA-{i}' for i in range(BEST_K)],
            yticklabels=sorted_classes,
            ax=ax,
            annot=annot,
            fmt='s',
            annot_kws={'size': 6},
            cbar_kws={'label': 'Count', 'shrink': 0.6},
            linewidths=0.5,
            linecolor='lightgray')

ax.set_title('85 True Classes vs 19 MOA Clusters (k=19)\nCell values = number of crops assigned to cluster', 
             fontsize=16, pad=20)
ax.set_xlabel('Discovered MOA Cluster', fontsize=14)
ax.set_ylabel('True Class (gene_guide)', fontsize=14)
ax.tick_params(axis='y', labelsize=6)
ax.tick_params(axis='x', labelsize=9)

# Draw horizontal lines to separate genes
prev_gene = None
for i, cls in enumerate(sorted_classes):
    gene = cls.rsplit('_', 1)[0] if '_' in cls else cls
    if gene != prev_gene and prev_gene is not None:
        ax.axhline(y=i, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    prev_gene = gene

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_85_vs_19_annotated.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_85_vs_19_annotated.png")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*70)
print("CLASS -> CLUSTER MAPPING SUMMARY")
print("="*70)

summary = []
for cls in sorted_classes:
    cluster = class_to_cluster[cls]
    summary.append({'Class': cls, 'MOA_Cluster': cluster})

df_summary = pd.DataFrame(summary)
df_summary.to_csv(os.path.join(OUTPUT_DIR, 'class_to_cluster_85_19.csv'), index=False)
print("Saved: class_to_cluster_85_19.csv")

# Print cluster sizes
print("\nCluster sizes:")
for cluster in range(BEST_K):
    n = sum(1 for c in class_cluster_labels if c == cluster)
    classes = [class_names[i] for i in range(len(class_names)) if class_cluster_labels[i] == cluster]
    print(f"  MOA-{cluster:2d}: {n:2d} classes -> {classes}")

print("\n" + "="*70)
print("DONE - 3 confusion matrices generated")
print("="*70)
