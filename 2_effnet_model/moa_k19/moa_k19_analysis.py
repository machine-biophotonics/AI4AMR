#!/usr/bin/env python3
"""
MOA Discovery Analysis with k=19 (optimal from silhouette)
"""

import numpy as np
import json
import os
import re
import ast
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 8
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

SEED = 42
np.random.seed(SEED)

BASE_DIR = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging'
EFFNET_DIR = os.path.join(BASE_DIR, 'effnet_model', 'eval_results')
OUTPUT_DIR = os.path.join(BASE_DIR, 'moa_k19')
os.makedirs(OUTPUT_DIR, exist_ok=True)

GENE_COLORS = {
    'mrcA': '#E57373', 'mrcB': '#EF5350', 'mrdA': '#F06292', 'ftsI': '#EC407A',
    'murA': '#FFB74D', 'murC': '#FFA726', 'lpxA': '#4DB6AC', 'lpxC': '#26A69A',
    'lptA': '#4DD0E1', 'lptC': '#26C6DA', 'msbA': '#80DEEA', 'gyrA': '#5C6BC0',
    'gyrB': '#3F51B5', 'parC': '#7986CB', 'parE': '#9FA8DA', 'dnaE': '#9575CD',
    'dnaB': '#B39DDB', 'rpoA': '#81C784', 'rpoB': '#66BB6A', 'rpsA': '#FFF176',
    'rpsL': '#FFEE58', 'rplA': '#FFD54F', 'rplC': '#FFCA28', 'folA': '#AED581',
    'folP': '#9CCC65', 'secY': '#80CBC4', 'secA': '#4DB6AC', 'ftsZ': '#F06292',
    'WT': '#424242'
}

PATHWAY_COLORS = {
    'Cell wall': '#E57373', 'LPS': '#4DB6AC', 'DNA': '#5C6BC0',
    'Transcription': '#81C784', 'Metabolism': '#AED581', 'Cell division': '#F06292',
    'WT': '#424242'
}

PATHWAYS = {
    'Cell wall': ['mrcA', 'mrcB', 'mrdA', 'ftsI', 'murA', 'murC'],
    'LPS': ['lpxA', 'lpxC', 'lptA', 'lptC', 'msbA'],
    'DNA': ['gyrA', 'gyrB', 'parC', 'parE', 'dnaB', 'dnaE'],
    'Transcription': ['rpoA', 'rpoB', 'rplA', 'rplC', 'rpsA', 'rpsL'],
    'Metabolism': ['folA', 'folP', 'secA', 'secY'],
    'Cell division': ['ftsZ'],
}

GENE_TO_PATHWAY = {}
for pathway, genes in PATHWAYS.items():
    for gene in genes:
        GENE_TO_PATHWAY[gene] = pathway

GENE_FAMILIES = {
    'mrc': ['mrcA', 'mrcB'], 'mrd': ['mrdA'], 'mur': ['murA', 'murC'],
    'fts': ['ftsI', 'ftsZ'], 'lpx': ['lpxA', 'lpxC'], 'lpt': ['lptA', 'lptC'],
    'msbA': ['msbA'], 'gyr': ['gyrA', 'gyrB'], 'par': ['parC', 'parE'],
    'dna': ['dnaB', 'dnaE'], 'rpo': ['rpoA', 'rpoB'], 'rpl': ['rplA', 'rplC'],
    'rps': ['rpsA', 'rpsL'], 'fol': ['folA', 'folP'], 'sec': ['secA', 'secY'],
}

GENE_TO_FAMILY = {}
for family, genes in GENE_FAMILIES.items():
    for gene in genes:
        GENE_TO_FAMILY[gene] = family

def get_base_gene(label):
    if label == 'WT': return 'WT'
    if '_' in str(label): return str(label).rsplit('_', 1)[0]
    return str(label)

def get_guide(label):
    if label == 'WT': return 'WT'
    if '_' in str(label):
        parts = str(label).rsplit('_', 1)
        return parts[1] if len(parts) > 1 else 'WT'
    return 'WT'

def get_pathway(label):
    if label == 'WT': return 'WT'
    return GENE_TO_PATHWAY.get(get_base_gene(label), 'Unknown')

def get_family(label):
    if label == 'WT': return 'WT'
    return GENE_TO_FAMILY.get(get_base_gene(label), get_base_gene(label))

def majority_vote(items):
    return Counter(items).most_common(1)[0][0]

print("="*70)
print("MOA DISCOVERY ANALYSIS - k=19 (OPTIMAL)")
print("="*70)

# Load data
print("\nLoading data...")
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

class_labels = [idx_to_label.get(l, 'WT') for l in labels]
unique_classes = sorted(set(class_labels))

# Class centroids
class_to_embeddings = defaultdict(list)
for emb, cls in zip(embeddings, class_labels):
    class_to_embeddings[cls].append(emb)

class_embeddings = {cls: np.mean(embs, axis=0) for cls, embs in class_to_embeddings.items()}
class_names = list(class_embeddings.keys())
X_centroids = np.array([class_embeddings[c] for c in class_names])

print(f"Classes: {len(class_names)}")
print(f"Centroid matrix: {X_centroids.shape}")

# =============================================================================
# CLUSTERING WITH k=19
# =============================================================================
print("\n" + "="*70)
print(f"CLUSTERING WITH k=19")
print("="*70)

BEST_K = 19
kmeans = KMeans(n_clusters=BEST_K, random_state=SEED, n_init=10)
class_cluster_labels = kmeans.fit_predict(X_centroids)
class_to_cluster = {class_names[i]: class_cluster_labels[i] for i in range(len(class_names))}

# Cluster colors
CLUSTER_COLORS = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
cluster_color_map = {f'MOA-{i}': CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(BEST_K)}

print("\nCluster assignments:")
cluster_analysis = []
for cluster in range(BEST_K):
    classes_in_cluster = [class_names[i] for i in range(len(class_names)) if class_cluster_labels[i] == cluster]
    genes_in_cluster = sorted(set([get_base_gene(c) for c in classes_in_cluster]))
    pathways_in_cluster = [get_pathway(c) for c in classes_in_cluster]
    pathway_counts = Counter(pathways_in_cluster)
    dominant_pathway = pathway_counts.most_common(1)[0][0] if pathway_counts else 'Unknown'
    purity = pathway_counts[dominant_pathway] / len(pathways_in_cluster) if pathways_in_cluster else 0
    
    cluster_analysis.append({
        'Cluster': cluster,
        'Size': len(classes_in_cluster),
        'Classes': classes_in_cluster,
        'Genes': genes_in_cluster,
        'Dominant_Pathway': dominant_pathway,
        'Purity': purity,
        'Pathway_Distribution': dict(pathway_counts)
    })
    
    print(f"  MOA-{cluster} (n={len(classes_in_cluster)}, purity={purity:.2f}): {genes_in_cluster}")

df_clusters = pd.DataFrame(cluster_analysis)
df_clusters.to_csv(os.path.join(OUTPUT_DIR, 'moa_cluster_analysis.csv'), index=False)

# =============================================================================
# MAP CLUSTERS TO ALL LEVELS
# =============================================================================
print("\nMapping clusters to crop/image/well levels...")

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

print(f"Crops: {len(embeddings)}, Images: {len(image_level_moa)}, Wells: {len(well_level_moa)}")

# =============================================================================
# t-SNE VISUALIZATION
# =============================================================================
print("\n" + "="*70)
print("t-SNE VISUALIZATION")
print("="*70)

def create_tsne_plots(data, moa_clusters, true_labels, level_name, filename_prefix, sample_size=None, marker_size=8):
    if sample_size and len(data) > sample_size:
        idx = np.random.choice(len(data), sample_size, replace=False)
        data = data[idx]
        moa_clusters = [moa_clusters[i] for i in idx]
        true_labels = [true_labels[i] for i in idx]
    
    n = len(data)
    perplexity = min(50, max(5, int(np.sqrt(n))))
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=2000,
                learning_rate='auto', init='pca', random_state=SEED, method='barnes_hut')
    features_2d = tsne.fit_transform(data)
    
    moa_labels = [f'MOA-{c}' for c in moa_clusters]
    gene_labels = [get_base_gene(idx_to_label.get(l, 'WT')) for l in true_labels]
    guide_labels = [get_guide(idx_to_label.get(l, 'WT')) for l in true_labels]
    pathway_labels = [get_pathway(idx_to_label.get(l, 'WT')) for l in true_labels]
    
    df = pd.DataFrame({
        'tSNE_1': features_2d[:, 0],
        'tSNE_2': features_2d[:, 1],
        'MOA_Cluster': moa_labels,
        'Gene': gene_labels,
        'Guide': guide_labels,
        'Pathway': pathway_labels
    })
    
    # MOA cluster plot
    fig = px.scatter(df, x='tSNE_1', y='tSNE_2', color='MOA_Cluster',
                     title=f't-SNE: {level_name} - MOA Clusters (k={BEST_K}, n={n})',
                     color_discrete_map=cluster_color_map,
                     hover_data={'Gene': True, 'Guide': True, 'Pathway': True}, opacity=0.7)
    fig.update_traces(marker=dict(size=marker_size))
    fig.update_layout(width=1400, height=1000, legend_title_text='MOA Cluster')
    fig.write_html(os.path.join(OUTPUT_DIR, f'{filename_prefix}_moa.html'))
    print(f"  Saved: {filename_prefix}_moa.html")
    
    # Pathway plot
    fig2 = px.scatter(df, x='tSNE_1', y='tSNE_2', color='Pathway',
                      title=f't-SNE: {level_name} - Known Pathways (n={n})',
                      color_discrete_map=PATHWAY_COLORS,
                      hover_data={'Gene': True, 'Guide': True, 'MOA_Cluster': True}, opacity=0.7)
    fig2.update_traces(marker=dict(size=marker_size))
    fig2.update_layout(width=1400, height=1000)
    fig2.write_html(os.path.join(OUTPUT_DIR, f'{filename_prefix}_pathway.html'))
    print(f"  Saved: {filename_prefix}_pathway.html")
    
    # UMAP
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=2, metric='euclidean', random_state=SEED)
        features_umap = reducer.fit_transform(data)
        df_umap = pd.DataFrame({
            'UMAP_1': features_umap[:, 0], 'UMAP_2': features_umap[:, 1],
            'MOA_Cluster': moa_labels, 'Gene': gene_labels, 'Guide': guide_labels, 'Pathway': pathway_labels
        })
        fig3 = px.scatter(df_umap, x='UMAP_1', y='UMAP_2', color='MOA_Cluster',
                          title=f'UMAP: {level_name} - MOA Clusters (k={BEST_K}, n={n})',
                          color_discrete_map=cluster_color_map,
                          hover_data={'Gene': True, 'Guide': True}, opacity=0.7)
        fig3.update_traces(marker=dict(size=marker_size))
        fig3.update_layout(width=1400, height=1000)
        fig3.write_html(os.path.join(OUTPUT_DIR, f'{filename_prefix}_umap.html'))
        print(f"  Saved: {filename_prefix}_umap.html")

# Crop level
print("\nCrop level (10K sample)...")
create_tsne_plots(
    embeddings, crop_moa_clusters, labels,
    'Crop Level', 'tsne_01_crop', sample_size=10000, marker_size=4
)

# Image level 3D
print("\nImage level 3D...")
img_keys = sorted(image_level_moa.keys())
img_data = np.array([np.mean([embeddings[crop_idx] for crop_idx in range(len(embeddings)) 
                              if crop_mapping.get(crop_idx, {}).get('filename', '') == k], axis=0) 
                     for k in img_keys])

n = len(img_data)
perplexity = min(50, max(5, int(np.sqrt(n))))
tsne_3d = TSNE(n_components=3, perplexity=perplexity, max_iter=2000,
               learning_rate='auto', init='pca', random_state=SEED, method='barnes_hut')
features_3d = tsne_3d.fit_transform(img_data)

moa_labels_3d = [f'MOA-{image_level_moa[k]}' for k in img_keys]
gene_labels_3d = [get_base_gene(idx_to_label.get(image_level_labels[k], 'WT')) for k in img_keys]
guide_labels_3d = [get_guide(idx_to_label.get(image_level_labels[k], 'WT')) for k in img_keys]
pathway_labels_3d = [get_pathway(idx_to_label.get(image_level_labels[k], 'WT')) for k in img_keys]

df_3d = pd.DataFrame({
    'tSNE_1': features_3d[:, 0],
    'tSNE_2': features_3d[:, 1],
    'tSNE_3': features_3d[:, 2],
    'MOA_Cluster': moa_labels_3d,
    'Gene': gene_labels_3d,
    'Guide': guide_labels_3d,
    'Pathway': pathway_labels_3d
})

fig_3d = px.scatter_3d(df_3d, x='tSNE_1', y='tSNE_2', z='tSNE_3',
                       color='MOA_Cluster',
                       title=f'3D t-SNE: Image Level - MOA Clusters (k={BEST_K}, n={n})',
                       color_discrete_map=cluster_color_map,
                       hover_data={'Gene': True, 'Guide': True, 'Pathway': True},
                       opacity=0.7)
fig_3d.update_traces(marker=dict(size=5))
fig_3d.update_layout(width=1400, height=1000, legend_title_text='MOA Cluster')
fig_3d.write_html(os.path.join(OUTPUT_DIR, 'tsne_02_image_moa_3d.html'))
print("  Saved: tsne_02_image_moa_3d.html")

# Also 2D plots
create_tsne_plots(
    img_data,
    [image_level_moa[k] for k in img_keys],
    [image_level_labels[k] for k in img_keys],
    'Image Level', 'tsne_02_image', marker_size=10
)

# Well level
print("\nWell level...")
well_keys = sorted(well_level_moa.keys())
well_data = []
for w in well_keys:
    well_img_keys = [k for k in img_keys if re.search(r'Well(\w\d+)_', k) and re.search(r'Well(\w\d+)_', k).group(1) == w]
    if well_img_keys:
        well_embs = [np.mean([embeddings[crop_idx] for crop_idx in range(len(embeddings)) 
                              if crop_mapping.get(crop_idx, {}).get('filename', '') == k], axis=0) 
                     for k in well_img_keys]
        well_data.append(np.mean(well_embs, axis=0))
    else:
        well_data.append(np.zeros(1280))
well_data = np.array(well_data)

create_tsne_plots(
    well_data,
    [well_level_moa[w] for w in well_keys],
    [well_labels_agg[w] for w in well_keys],
    'Well Level', 'tsne_03_well', marker_size=15
)

# =============================================================================
# CONFUSION MATRICES (85 Classes vs MOA Clusters)
# =============================================================================
print("\n" + "="*70)
print("CONFUSION MATRICES (85 Classes vs MOA Clusters)")
print("="*70)

def plot_confusion_matrix(cm, xlabels, ylabels, title, filename, figsize=(22, 18)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, shrink=0.6)
    ax.set(xticks=np.arange(len(xlabels)), yticks=np.arange(len(ylabels)),
           xticklabels=xlabels, yticklabels=ylabels, title=title,
           ylabel='True Class (85)', xlabel='Discovered MOA Cluster (19)')
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black", fontsize=5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

# All 85 classes sorted
all_85_classes = sorted(class_names)
all_19_clusters = [f'MOA-{i}' for i in range(BEST_K)]

# Crop level confusion matrix
print("\nCrop level confusion matrix (85x19)...")
crop_true_class = [idx_to_label.get(l, 'WT') for l in labels]
crop_pred_moa = [f'MOA-{c}' for c in crop_moa_clusters]
cm_crop = confusion_matrix(crop_true_class, crop_pred_moa, labels=all_85_classes, normalize='true')
plot_confusion_matrix(cm_crop, all_19_clusters, all_85_classes,
                      f'Crop Level: True Classes (85) vs MOA Clusters (k={BEST_K})\nNormalized',
                      'confusion_crop_85classes.png', figsize=(24, 20))

# Image level confusion matrix
print("\nImage level confusion matrix (85x19)...")
img_true_class = [idx_to_label.get(image_level_labels[k], 'WT') for k in img_keys]
img_pred_moa = [f'MOA-{image_level_moa[k]}' for k in img_keys]
cm_img = confusion_matrix(img_true_class, img_pred_moa, labels=all_85_classes, normalize='true')
plot_confusion_matrix(cm_img, all_19_clusters, all_85_classes,
                      f'Image Level: True Classes (85) vs MOA Clusters (k={BEST_K})\nNormalized',
                      'confusion_image_85classes.png', figsize=(24, 20))

# Well level confusion matrix
print("\nWell level confusion matrix (85x19)...")
well_true_class = [idx_to_label.get(well_labels_agg[w], 'WT') for w in well_keys]
well_pred_moa = [f'MOA-{well_level_moa[w]}' for w in well_keys]
cm_well = confusion_matrix(well_true_class, well_pred_moa, labels=all_85_classes, normalize='true')
plot_confusion_matrix(cm_well, all_19_clusters, all_85_classes,
                      f'Well Level: True Classes (85) vs MOA Clusters (k={BEST_K})\nNormalized',
                      'confusion_well_85classes.png', figsize=(24, 20))

# =============================================================================
# DENDROGRAM
# =============================================================================
print("\n" + "="*70)
print("DENDROGRAM")
print("="*70)

Z = linkage(X_centroids, method='ward')
max_d = Z[-(BEST_K-1), 2]

fig, ax = plt.subplots(figsize=(24, 14))
dendrogram(Z, labels=class_names, leaf_rotation=90, leaf_font_size=7,
           color_threshold=max_d, above_threshold_color='gray', ax=ax)
ax.set_title(f'Hierarchical Clustering Dendrogram - MOA Discovery (k={BEST_K})', fontsize=14)
ax.set_xlabel('Class')
ax.set_ylabel('Distance')
ax.axhline(y=max_d, c='red', linestyle='--', label=f'Cut for k={BEST_K}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dendrogram.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: dendrogram.png")

# =============================================================================
# CLUSTER-PATHWAY HEATMAP
# =============================================================================
print("\n" + "="*70)
print("CLUSTER-PATHWAY HEATMAP")
print("="*70)

cross_data = []
for cluster in range(BEST_K):
    row = {'Cluster': f'MOA-{cluster}'}
    for pathway in PATHWAYS.keys():
        count = sum(1 for c in range(len(class_names)) if class_cluster_labels[c] == cluster 
                   and get_pathway(class_names[c]) == pathway)
        row[pathway] = count
    cross_data.append(row)

df_cross = pd.DataFrame(cross_data).set_index('Cluster')
df_cross.to_csv(os.path.join(OUTPUT_DIR, 'cluster_pathway_crosstab.csv'))

fig_heatmap = go.Figure(data=go.Heatmap(
    z=df_cross.values, x=df_cross.columns, y=df_cross.index,
    colorscale='Blues', text=df_cross.values, texttemplate='%{text}',
    textfont=dict(size=10)
))
fig_heatmap.update_layout(title=f'MOA Clusters (k={BEST_K}) vs Pathways',
                          xaxis_title='Pathway', yaxis_title='MOA Cluster',
                          width=1000, height=700)
fig_heatmap.write_html(os.path.join(OUTPUT_DIR, 'cluster_pathway_heatmap.html'))
print("  Saved: cluster_pathway_heatmap.html")

# =============================================================================
# GUIDE CONSISTENCY
# =============================================================================
print("\n" + "="*70)
print("GUIDE RNA CONSISTENCY")
print("="*70)

guide_data = []
for gene in sorted(GENE_TO_PATHWAY.keys()):
    gene_classes = [c for c in class_names if get_base_gene(c) == gene]
    if len(gene_classes) >= 2:
        clusters = [class_to_cluster[c] for c in gene_classes]
        guides = [c.split('_')[-1] if '_' in c else 'WT' for c in gene_classes]
        all_same = len(set(clusters)) == 1
        guide_data.append({'Gene': gene, 'Guides': guides, 'Clusters': clusters, 'Consistency': 'Same' if all_same else 'Different'})
        print(f"  {gene}: guides={guides} -> clusters={clusters} [{'SAME' if all_same else 'DIFFERENT'}]")

df_guide = pd.DataFrame(guide_data)
df_guide.to_csv(os.path.join(OUTPUT_DIR, 'guide_consistency.csv'), index=False)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY - k=19 ANALYSIS")
print("="*70)

# Count high purity clusters
high_purity = sum(1 for c in cluster_analysis if c['Purity'] >= 0.8)
medium_purity = sum(1 for c in cluster_analysis if 0.5 <= c['Purity'] < 0.8)
low_purity = sum(1 for c in cluster_analysis if c['Purity'] < 0.5)

print(f"\nCluster Purity Summary:")
print(f"  High purity (>=0.8): {high_purity} clusters")
print(f"  Medium purity (0.5-0.8): {medium_purity} clusters")
print(f"  Low purity (<0.5): {low_purity} clusters")

print(f"\nGenerated files in {OUTPUT_DIR}:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  - {f}")

print("\n" + "="*70)
print("DONE!")
print("="*70)
