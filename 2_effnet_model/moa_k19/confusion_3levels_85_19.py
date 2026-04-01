#!/usr/bin/env python3
"""
Confusion matrices at 3 levels: Crop, Image, Well
True Classes (85) vs MOA Clusters (19) with accuracy
"""

import numpy as np
import json
import os
import re
import ast
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
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

with open(os.path.join(EFFNET_DIR, 'crop_to_image_mapping.json'), 'r') as f:
    crop_mapping_raw = json.load(f)

embeddings = np.load(os.path.join(EFFNET_DIR, 'test_embeddings.npy'))
labels = np.load(os.path.join(EFFNET_DIR, 'test_labels.npy'))

# Parse crop mapping
crop_mapping = {}
for k, v in crop_mapping_raw.items():
    idx = int(k)
    filename = v.get('filename', '')
    if filename.startswith('['):
        try:
            parsed = ast.literal_eval(filename)
            if isinstance(parsed, list) and len(parsed) == 4:
                filename = parsed[3]
        except: pass
    match = re.search(r'Well(\w\d+)_', filename) if filename else None
    well = match.group(1) if match else ''
    crop_mapping[idx] = {'filename': filename, 'well': well}

def majority_vote(items):
    return Counter(items).most_common(1)[0][0]

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

# =============================================================================
# BUILD DATA AT 3 LEVELS
# =============================================================================
print("Building data at 3 levels...")

# CROP LEVEL
crop_y_true = [idx_to_label.get(l, 'WT') for l in labels]
crop_y_pred = [class_to_cluster.get(class_labels[i], 0) for i in range(len(labels))]

# IMAGE LEVEL
image_preds = defaultdict(list)
image_labels = defaultdict(list)
for crop_idx in range(len(labels)):
    meta = crop_mapping.get(crop_idx, {})
    filename = meta.get('filename', '')
    image_preds[filename].append(crop_y_pred[crop_idx])
    image_labels[filename].append(crop_y_true[crop_idx])

img_y_true = [image_labels[k][0] for k in sorted(image_labels.keys())]
img_y_pred = [majority_vote(image_preds[k]) for k in sorted(image_labels.keys())]

# WELL LEVEL
well_preds = defaultdict(list)
well_labels = {}
for filename in sorted(image_preds.keys()):
    match = re.search(r'Well(\w\d+)_', filename)
    well = match.group(1) if match else ''
    if well:
        well_preds[well].extend(image_preds[filename])
        if well not in well_labels:
            well_labels[well] = image_labels[filename][0]

well_y_true = [well_labels[w] for w in sorted(well_preds.keys())]
well_y_pred = [majority_vote(well_preds[w]) for w in sorted(well_preds.keys())]

print(f"Crops: {len(crop_y_true)}")
print(f"Images: {len(img_y_true)}")
print(f"Wells: {len(well_y_true)}")

# =============================================================================
# PLOT CONFUSION MATRICES
# =============================================================================
sorted_classes = sorted(class_names, key=lambda x: (x != 'WT', x))
cluster_labels = [f'C{i}' for i in range(BEST_K)]

def plot_confusion(y_true, y_pred, level, n_samples, filename):
    # Build confusion matrix manually
    cm = np.zeros((len(sorted_classes), BEST_K), dtype=int)
    true_to_idx = {c: i for i, c in enumerate(sorted_classes)}
    
    for t, p in zip(y_true, y_pred):
        if t in true_to_idx:
            cm[true_to_idx[t], p] += 1
    
    # Purity (what % of each class lands in its dominant cluster)
    class_accuracies = []
    for i, cls in enumerate(sorted_classes):
        row_sum = cm[i, :].sum()
        if row_sum > 0:
            dominant_cluster = np.argmax(cm[i, :])
            class_accuracies.append(cm[i, dominant_cluster] / row_sum)
        else:
            class_accuracies.append(0)
    avg_accuracy = np.mean(class_accuracies) * 100
    
    # Cluster purity (what % of cluster is one class)
    purity_scores = []
    for j in range(BEST_K):
        col_sum = cm[:, j].sum()
        if col_sum > 0:
            purity_scores.append(cm[:, j].max() / col_sum)
    avg_purity = np.mean(purity_scores) * 100
    
    print(f"\n{level}: {n_samples} samples, Avg Class Accuracy={avg_accuracy:.1f}%, Avg Purity={avg_purity:.1f}%")
    
    # Plot
    fig, ax = plt.subplots(figsize=(24, 22))
    
    # Annotate only non-zero values
    annot = np.where(cm > 0, cm.astype(str), '')
    
    sns.heatmap(cm,
                cmap='Blues',
                xticklabels=cluster_labels,
                yticklabels=sorted_classes,
                ax=ax,
                annot=annot,
                fmt='s',
                annot_kws={'size': 5},
                cbar_kws={'label': 'Count', 'shrink': 0.6},
                linewidths=0.3,
                linecolor='lightgray')
    
    ax.set_title(f'{level}: 85 True Classes vs 19 MOA Clusters\n(n={n_samples}, Accuracy={avg_accuracy:.1f}%, Purity={avg_purity:.1f}%)', 
                 fontsize=16, pad=20)
    ax.set_xlabel('Discovered MOA Cluster', fontsize=14)
    ax.set_ylabel('True Class', fontsize=12)
    ax.tick_params(axis='y', labelsize=5)
    ax.tick_params(axis='x', labelsize=9)
    
    # Draw lines between genes
    prev_gene = None
    for i, cls in enumerate(sorted_classes):
        gene = cls.rsplit('_', 1)[0] if '_' in cls else cls
        if gene != prev_gene and prev_gene is not None:
            ax.axhline(y=i, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        prev_gene = gene
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
    
    return avg_accuracy, avg_purity

# Plot all 3 levels
print("\n" + "="*70)
print("CONFUSION MATRICES: 85 CLASSES vs 19 MOA CLUSTERS")
print("="*70)

acc_crop, pur_crop = plot_confusion(crop_y_true, crop_y_pred, 'CROP LEVEL', len(crop_y_true), 'confusion_crop_85_19.png')
acc_img, pur_img = plot_confusion(img_y_true, img_y_pred, 'IMAGE LEVEL', len(img_y_true), 'confusion_image_85_19.png')
acc_well, pur_well = plot_confusion(well_y_true, well_y_pred, 'WELL LEVEL', len(well_y_true), 'confusion_well_85_19.png')

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*70)
print("ACCURACY SUMMARY")
print("="*70)

summary = pd.DataFrame({
    'Level': ['Crop', 'Image', 'Well'],
    'Samples': [len(crop_y_true), len(img_y_true), len(well_y_true)],
    'Avg_Class_Accuracy': [f'{acc_crop:.1f}%', f'{acc_img:.1f}%', f'{acc_well:.1f}%'],
    'Avg_Cluster_Purity': [f'{pur_crop:.1f}%', f'{pur_img:.1f}%', f'{pur_well:.1f}%']
})
print(summary.to_string(index=False))

# Save summary
summary.to_csv(os.path.join(OUTPUT_DIR, 'accuracy_summary_3levels.csv'), index=False)
print("\nSaved: accuracy_summary_3levels.csv")

# =============================================================================
# CLUSTER COMPOSITION TABLE
# =============================================================================
print("\n" + "="*70)
print("CLUSTER COMPOSITION (which classes map to which cluster)")
print("="*70)

for cluster in range(BEST_K):
    classes_in_cluster = [class_names[i] for i in range(len(class_names)) if class_cluster_labels[i] == cluster]
    genes = sorted(set([c.rsplit('_', 1)[0] if '_' in c else c for c in classes_in_cluster]))
    print(f"C{cluster:2d} ({len(classes_in_cluster):2d} classes): {genes}")

print("\n" + "="*70)
print("ALL FILES IN moa_k19/")
print("="*70)
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
    print(f"  {f:40s} {size/1024:.0f} KB")
