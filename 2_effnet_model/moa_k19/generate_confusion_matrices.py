#!/usr/bin/env python3
"""
Generate improved confusion matrices for MOA k=19 analysis
True Classes (85) vs Discovered MOA Clusters (19)
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

# Class centroids
class_labels = [idx_to_label.get(l, 'WT') for l in labels]
unique_classes = sorted(set(class_labels))

class_to_embeddings = defaultdict(list)
for emb, cls in zip(embeddings, class_labels):
    class_to_embeddings[cls].append(emb)

class_embeddings = {cls: np.mean(embs, axis=0) for cls, embs in class_to_embeddings.items()}
class_names = list(class_embeddings.keys())
X_centroids = np.array([class_embeddings[c] for c in class_names])

# Clustering
BEST_K = 19
kmeans = KMeans(n_clusters=BEST_K, random_state=SEED, n_init=10)
class_cluster_labels = kmeans.fit_predict(X_centroids)
class_to_cluster = {class_names[i]: class_cluster_labels[i] for i in range(len(class_names))}

# Map to all levels
def majority_vote(items):
    return Counter(items).most_common(1)[0][0]

crop_moa_clusters = [class_to_cluster.get(class_labels[i], 0) for i in range(len(labels))]

image_moa_clusters = defaultdict(list)
image_labels_agg = defaultdict(list)
for crop_idx in range(len(embeddings)):
    meta = crop_mapping.get(crop_idx, {})
    filename = meta.get('filename', '')
    image_moa_clusters[filename].append(crop_moa_clusters[crop_idx])
    image_labels_agg[filename].append(labels[crop_idx])

image_level_moa = {k: majority_vote(v) for k, v in image_moa_clusters.items()}
image_level_labels = {k: v[0] for k, v in image_labels_agg.items()}

well_moa_clusters = defaultdict(list)
well_labels_agg = {}
for filename, moas in image_moa_clusters.items():
    match = re.search(r'Well(\w\d+)_', filename)
    well = match.group(1) if match else ''
    if well:
        well_moa_clusters[well].extend(moas)
        if well not in well_labels_agg:
            well_labels_agg[well] = image_level_labels[filename]

well_level_moa = {w: majority_vote(v) for w, v in well_moa_clusters.items()}

print("="*70)
print("IMPROVED CONFUSION MATRICES - k=19")
print("="*70)

# =============================================================================
# CONFUSION MATRIX 1: By Pathway (cleaner, 6x19)
# =============================================================================
print("\n1. Pathway-level confusion matrix (6 pathways vs 19 clusters)...")

def get_pathway(label):
    PATHWAYS = {
        'Cell wall': ['mrcA', 'mrcB', 'mrdA', 'ftsI', 'murA', 'murC'],
        'LPS': ['lpxA', 'lpxC', 'lptA', 'lptC', 'msbA'],
        'DNA': ['gyrA', 'gyrB', 'parC', 'parE', 'dnaB', 'dnaE'],
        'Transcription': ['rpoA', 'rpoB', 'rplA', 'rplC', 'rpsA', 'rpsL'],
        'Metabolism': ['folA', 'folP', 'secA', 'secY'],
        'Cell division': ['ftsZ'],
    }
    gene_to_pathway = {}
    for pathway, genes in PATHWAYS.items():
        for gene in genes:
            gene_to_pathway[gene] = pathway
    
    if label == 'WT':
        return 'Control'
    gene = label.rsplit('_', 1)[0] if '_' in label else label
    return gene_to_pathway.get(gene, 'Unknown')

# Crop level - pathway vs MOA
crop_true_pathway = [get_pathway(idx_to_label.get(l, 'WT')) for l in labels]
crop_pred_moa = [f'MOA-{c}' for c in crop_moa_clusters]

pathway_order = ['Control', 'Cell wall', 'LPS', 'DNA', 'Transcription', 'Metabolism', 'Cell division', 'Unknown']
cluster_order = [f'MOA-{i}' for i in range(BEST_K)]

cm_pathway = confusion_matrix(crop_true_pathway, crop_pred_moa, labels=pathway_order)
cm_pathway_pct = np.divide(cm_pathway, cm_pathway.sum(axis=1, keepdims=True), 
                           where=cm_pathway.sum(axis=1, keepdims=True)>0) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

# Raw counts
sns.heatmap(cm_pathway, annot=True, fmt='d', cmap='Blues', 
            xticklabels=cluster_order, yticklabels=pathway_order, ax=ax1,
            cbar_kws={'label': 'Count'})
ax1.set_title('Crop Level: Raw Counts (Pathway vs MOA Cluster)', fontsize=14)
ax1.set_xlabel('Discovered MOA Cluster')
ax1.set_ylabel('True Pathway')
ax1.tick_params(axis='x', rotation=90)

# Percentage
sns.heatmap(cm_pathway_pct, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=cluster_order, yticklabels=pathway_order, ax=ax2,
            cbar_kws={'label': '% of True Pathway'})
ax2.set_title('Crop Level: Percentage (each row = 100%)', fontsize=14)
ax2.set_xlabel('Discovered MOA Cluster')
ax2.set_ylabel('True Pathway')
ax2.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_pathway_level.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: confusion_pathway_level.png")

# =============================================================================
# CONFUSION MATRIX 2: Image-level pathway
# =============================================================================
print("\n2. Image-level pathway confusion matrix...")

img_true_pathway = [get_pathway(idx_to_label.get(image_level_labels[k], 'WT')) for k in sorted(image_level_moa.keys())]
img_pred_moa = [f'MOA-{image_level_moa[k]}' for k in sorted(image_level_moa.keys())]

cm_img = confusion_matrix(img_true_pathway, img_pred_moa, labels=pathway_order)
cm_img_pct = np.divide(cm_img, cm_img.sum(axis=1, keepdims=True), 
                       where=cm_img.sum(axis=1, keepdims=True)>0) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

sns.heatmap(cm_img, annot=True, fmt='d', cmap='Blues',
            xticklabels=cluster_order, yticklabels=pathway_order, ax=ax1,
            cbar_kws={'label': 'Count'})
ax1.set_title(f'Image Level: Raw Counts (n={len(img_true_pathway)})', fontsize=14)
ax1.set_xlabel('Discovered MOA Cluster')
ax1.set_ylabel('True Pathway')
ax1.tick_params(axis='x', rotation=90)

sns.heatmap(cm_img_pct, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=cluster_order, yticklabels=pathway_order, ax=ax2,
            cbar_kws={'label': '% of True Pathway'})
ax2.set_title(f'Image Level: Percentage (n={len(img_true_pathway)})', fontsize=14)
ax2.set_xlabel('Discovered MOA Cluster')
ax2.set_ylabel('True Pathway')
ax2.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_image_pathway.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: confusion_image_pathway.png")

# =============================================================================
# CONFUSION MATRIX 3: Well-level pathway
# =============================================================================
print("\n3. Well-level pathway confusion matrix...")

well_true_pathway = [get_pathway(idx_to_label.get(well_labels_agg[w], 'WT')) for w in sorted(well_level_moa.keys())]
well_pred_moa = [f'MOA-{well_level_moa[w]}' for w in sorted(well_level_moa.keys())]

cm_well = confusion_matrix(well_true_pathway, well_pred_moa, labels=pathway_order)
cm_well_pct = np.divide(cm_well, cm_well.sum(axis=1, keepdims=True), 
                        where=cm_well.sum(axis=1, keepdims=True)>0) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

sns.heatmap(cm_well, annot=True, fmt='d', cmap='Blues',
            xticklabels=cluster_order, yticklabels=pathway_order, ax=ax1,
            cbar_kws={'label': 'Count'})
ax1.set_title(f'Well Level: Raw Counts (n={len(well_true_pathway)})', fontsize=14)
ax1.set_xlabel('Discovered MOA Cluster')
ax1.set_ylabel('True Pathway')
ax1.tick_params(axis='x', rotation=90)

sns.heatmap(cm_well_pct, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=cluster_order, yticklabels=pathway_order, ax=ax2,
            cbar_kws={'label': '% of True Pathway'})
ax2.set_title(f'Well Level: Percentage (n={len(well_true_pathway)})', fontsize=14)
ax2.set_xlabel('Discovered MOA Cluster')
ax2.set_ylabel('True Pathway')
ax2.tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_well_pathway.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: confusion_well_pathway.png")

# =============================================================================
# CONFUSION MATRIX 4: 85 Classes grouped by pathway
# =============================================================================
print("\n4. 85-class confusion matrix (grouped by pathway)...")

# Sort classes by pathway
PATHWAYS = {
    'Cell wall': ['mrcA', 'mrcB', 'mrdA', 'ftsI', 'murA', 'murC'],
    'LPS': ['lpxA', 'lpxC', 'lptA', 'lptC', 'msbA'],
    'DNA': ['gyrA', 'gyrB', 'parC', 'parE', 'dnaB', 'dnaE'],
    'Transcription': ['rpoA', 'rpoB', 'rplA', 'rplC', 'rpsA', 'rpsL'],
    'Metabolism': ['folA', 'folP', 'secA', 'secY'],
    'Cell division': ['ftsZ'],
}

# Create sorted class list
sorted_classes = ['WT']
for pathway in ['Cell wall', 'LPS', 'DNA', 'Transcription', 'Metabolism', 'Cell division']:
    for gene in PATHWAYS[pathway]:
        for guide in ['1', '2', '3']:
            class_name = f'{gene}_{guide}'
            if class_name in class_names:
                sorted_classes.append(class_name)

# Add any missing
for c in class_names:
    if c not in sorted_classes:
        sorted_classes.append(c)

# Crop level 85x19
crop_true = [idx_to_label.get(l, 'WT') for l in labels]
crop_pred = [f'MOA-{c}' for c in crop_moa_clusters]

cm_85 = confusion_matrix(crop_true, crop_pred, labels=sorted_classes)

# Create figure with pathway annotations
fig, ax = plt.subplots(figsize=(28, 22))

# Color by pathway for y-axis
PATHWAY_COLORS_MAP = {
    'Control': '#424242',
    'Cell wall': '#E57373',
    'LPS': '#4DB6AC',
    'DNA': '#5C6BC0',
    'Transcription': '#81C784',
    'Metabolism': '#AED581',
    'Cell division': '#F06292',
}

# Create custom colormap for heatmap
im = ax.imshow(cm_85, aspect='auto', cmap='Blues', interpolation='nearest')

# Set ticks
ax.set_xticks(np.arange(BEST_K))
ax.set_xticklabels([f'MOA-{i}' for i in range(BEST_K)], rotation=90, fontsize=8)
ax.set_yticks(np.arange(len(sorted_classes)))
ax.set_yticklabels(sorted_classes, fontsize=7)

# Color y-axis labels by pathway
for i, cls in enumerate(sorted_classes):
    if cls == 'WT':
        color = PATHWAY_COLORS_MAP['Control']
    else:
        gene = cls.rsplit('_', 1)[0]
        pathway = None
        for pw, genes in PATHWAYS.items():
            if gene in genes:
                pathway = pw
                break
        color = PATHWAY_COLORS_MAP.get(pathway, '#808080')
    ax.get_yticklabels()[i].set_color(color)

# Add pathway group labels on the right
pathway_ranges = {}
current_pathway = None
start_idx = 0
for i, cls in enumerate(sorted_classes):
    if cls == 'WT':
        pathway = 'Control'
    else:
        gene = cls.rsplit('_', 1)[0]
        pathway = None
        for pw, genes in PATHWAYS.items():
            if gene in genes:
                pathway = pw
                break
    
    if pathway != current_pathway:
        if current_pathway is not None:
            pathway_ranges[current_pathway] = (start_idx, i - 1)
        current_pathway = pathway
        start_idx = i
if current_pathway is not None:
    pathway_ranges[current_pathway] = (start_idx, len(sorted_classes) - 1)

# Draw pathway brackets
for pathway, (start, end) in pathway_ranges.items():
    mid = (start + end) / 2
    ax.text(BEST_K + 0.5, mid, pathway, fontsize=10, va='center',
            color=PATHWAY_COLORS_MAP.get(pathway, 'black'),
            fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.6)
cbar.set_label('Count', fontsize=12)

ax.set_title(f'Crop Level: 85 True Classes vs 19 MOA Clusters\n(Row color = Pathway)', fontsize=16)
ax.set_xlabel('Discovered MOA Cluster', fontsize=12)
ax.set_ylabel('True Class (sorted by pathway)', fontsize=12)

# Add gridlines for pathway groups
for pathway, (start, end) in pathway_ranges.items():
    ax.axhline(y=end + 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_85classes_grouped.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: confusion_85classes_grouped.png")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n5. Generating summary table...")

# Recreate cm_pathway with all clusters
cm_pathway_full = np.zeros((len(pathway_order), BEST_K), dtype=int)
for true, pred in zip(crop_true_pathway, crop_pred_moa):
    true_idx = pathway_order.index(true) if true in pathway_order else 0
    pred_cluster = int(pred.split('-')[1])
    cm_pathway_full[true_idx, pred_cluster] += 1

# Create summary table showing which pathway maps to which MOA clusters
summary_data = []
for pathway_idx, pathway in enumerate(pathway_order):
    for cluster in range(BEST_K):
        count = cm_pathway_full[pathway_idx, cluster]
        if count > 0:
            row_total = cm_pathway_full[pathway_idx, :].sum()
            pct = (count / row_total * 100) if row_total > 0 else 0
            summary_data.append({
                'Pathway': pathway,
                'MOA_Cluster': f'MOA-{cluster}',
                'Count': int(count),
                'Percentage': f'{pct:.1f}%'
            })

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(os.path.join(OUTPUT_DIR, 'pathway_cluster_summary.csv'), index=False)
print("  Saved: pathway_cluster_summary.csv")

print("\n" + "="*70)
print("CONFUSION MATRIX ANALYSIS COMPLETE!")
print("="*70)

print("\nKey Findings - Dominant MOA Cluster per Pathway:")
print("-" * 50)

# Find dominant MOA cluster for each pathway
for pathway_idx, pathway in enumerate(pathway_order):
    row_sum = cm_pathway_full[pathway_idx, :].sum()
    if row_sum > 0:
        dominant_cluster = np.argmax(cm_pathway_full[pathway_idx])
        pct = cm_pathway_full[pathway_idx, dominant_cluster] / row_sum * 100
        print(f"  {pathway:15s} -> MOA-{dominant_cluster:2d} ({pct:.1f}% of samples, n={row_sum})")
    else:
        print(f"  {pathway:15s} -> No samples")

print("\nGenerated files:")
print("  - confusion_pathway_level.png")
print("  - confusion_image_pathway.png")
print("  - confusion_well_pathway.png")
print("  - confusion_85classes_grouped.png")
print("  - pathway_cluster_summary.csv")
