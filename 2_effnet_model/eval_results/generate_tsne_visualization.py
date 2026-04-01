"""
t-SNE and UMAP Visualization for CRISPRi Screening Analysis
Creates interactive HTML plots at different aggregation levels.
"""

import numpy as np
import json
import os
import re
import ast
import random
from collections import Counter, defaultdict
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

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

FAMILY_COLORS = {
    'mrc': '#E57373', 'mrd': '#F06292', 'mur': '#FFA726', 'fts': '#EC407A',
    'lpx': '#4DB6AC', 'lpt': '#4DD0E1', 'msbA': '#80DEEA', 'gyr': '#5C6BC0',
    'par': '#7986CB', 'dna': '#9575CD', 'rpo': '#81C784', 'rpl': '#FFD54F',
    'rps': '#FFEE58', 'fol': '#AED581', 'sec': '#80CBC4', 'WT': '#424242'
}

PATHWAY_COLORS = {
    'Cell wall': '#E57373', 'LPS': '#4DB6AC', 'DNA': '#5C6BC0',
    'Transcription': '#81C784', 'Metabolism': '#AED581', 'Cell division': '#F06292',
    'WT': '#424242'
}

GENE_FAMILIES = {
    'mrc': ['mrcA', 'mrcB'],
    'mrd': ['mrdA'],
    'mur': ['murA', 'murC'],
    'fts': ['ftsI', 'ftsZ'],
    'lpx': ['lpxA', 'lpxC'],
    'lpt': ['lptA', 'lptC'],
    'msbA': ['msbA'],
    'gyr': ['gyrA', 'gyrB'],
    'par': ['parC', 'parE'],
    'dna': ['dnaB', 'dnaE'],
    'rpo': ['rpoA', 'rpoB'],
    'rpl': ['rplA', 'rplC'],
    'rps': ['rpsA', 'rpsL'],
    'fol': ['folA', 'folP'],
    'sec': ['secA', 'secY'],
}

GENE_TO_FAMILY = {}
for family, genes in GENE_FAMILIES.items():
    for gene in genes:
        GENE_TO_FAMILY[gene] = family

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

def get_base_gene(label):
    if label == 'WT':
        return 'WT'
    if '_' in label:
        return label.rsplit('_', 1)[0]
    return label

def get_family(label):
    if label == 'WT':
        return 'WT'
    base = get_base_gene(label)
    return GENE_TO_FAMILY.get(base, base)

def get_pathway(label):
    if label == 'WT':
        return 'WT'
    base = get_base_gene(label)
    return GENE_TO_PATHWAY.get(base, base)

print("Loading data...")
with open(os.path.join(BASE_DIR, 'idx_to_label.json'), 'r') as f:
    idx_to_label = json.load(f)
idx_to_label = {int(k): v for k, v in idx_to_label.items()}

with open(os.path.join(BASE_DIR, 'crop_to_image_mapping.json'), 'r') as f:
    crop_mapping_raw = json.load(f)

test_preds = np.load(os.path.join(BASE_DIR, 'test_preds.npy'))
test_labels = np.load(os.path.join(BASE_DIR, 'test_labels.npy'))
test_probs = np.load(os.path.join(BASE_DIR, 'test_probs.npy'))

crop_mapping = {}
for k, v in crop_mapping_raw.items():
    idx = int(k)
    filename = v.get('filename', '')
    if filename.startswith('['):
        try:
            parsed = ast.literal_eval(filename)
            if isinstance(parsed, list) and len(parsed) == 4:
                filename = parsed[3]
        except:
            pass
    match = re.search(r'Well(\w\d+)_', filename) if filename else None
    well = match.group(1) if match else ''
    crop_mapping[idx] = {'filename': filename, 'well': well}

print(f"Total crops: {len(test_preds)}")

image_preds = defaultdict(list)
image_labels = defaultdict(list)
image_probs = defaultdict(list)

for crop_idx in range(len(test_preds)):
    pred = test_preds[crop_idx]
    meta = crop_mapping.get(crop_idx, {})
    filename = meta.get('filename', '')
    image_preds[filename].append(pred)
    image_labels[filename].append(test_labels[crop_idx])
    image_probs[filename].append(test_probs[crop_idx])

def majority_vote(preds):
    return Counter(preds).most_common(1)[0][0]

image_level_predictions = {}
image_level_labels = {}
image_level_probs = {}

for key in image_preds:
    image_level_predictions[key] = majority_vote(image_preds[key])
    image_level_labels[key] = image_labels[key][0]
    image_level_probs[key] = np.mean(image_probs[key], axis=0)

well_preds = defaultdict(list)
well_labels = {}
well_probs = defaultdict(list)

for key in image_preds:
    filename = key
    match = re.search(r'Well(\w\d+)_', filename) if filename else None
    well = match.group(1) if match else ''
    if well:
        well_preds[well].append(image_level_predictions[key])
        if well not in well_labels:
            well_labels[well] = image_level_labels[key]
        well_probs[well].append(image_level_probs[key])

well_level_predictions = {}
well_level_labels = {}
well_level_probs = {}

for well in well_preds:
    well_level_predictions[well] = majority_vote(well_preds[well])
    well_level_labels[well] = well_labels[well]
    well_level_probs[well] = np.mean(well_probs[well], axis=0)

print(f"Unique images: {len(image_preds)}")
print(f"Unique wells: {len(well_preds)}")

def create_tsne_plot(data, labels, title, filename, color_map):
    print(f"  Creating {title}...")
    
    perplexity = min(50, max(5, int(np.sqrt(len(data)))))
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=2000,
        learning_rate='auto',
        init='pca',
        random_state=SEED,
        method='barnes_hut'
    )
    
    features_2d = tsne.fit_transform(data)
    
    df = pd.DataFrame({
        'tSNE_1': features_2d[:, 0],
        'tSNE_2': features_2d[:, 1],
        'Label': labels
    })
    
    fig = px.scatter(
        df,
        x='tSNE_1',
        y='tSNE_2',
        color='Label',
        title=title,
        color_discrete_map=color_map,
        opacity=0.7,
        hover_data={'Label': True}
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
    fig.update_layout(
        xaxis_title='t-SNE 1',
        yaxis_title='t-SNE 2',
        font=dict(size=12),
        width=1400,
        height=1000,
        legend_title_text='Label',
        legend=dict(itemsizing='constant', font=dict(size=10)),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    
    fig.write_html(os.path.join(BASE_DIR, filename))
    print(f"  Saved: {filename}")

def create_umap_plot(data, labels, title, filename, color_map):
    if not HAS_UMAP:
        print(f"  UMAP not available, skipping {title}")
        return
    
    print(f"  Creating {title}...")
    
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=SEED
    )
    
    features_2d = reducer.fit_transform(data)
    
    df = pd.DataFrame({
        'UMAP_1': features_2d[:, 0],
        'UMAP_2': features_2d[:, 1],
        'Label': labels
    })
    
    fig = px.scatter(
        df,
        x='UMAP_1',
        y='UMAP_2',
        color='Label',
        title=title,
        color_discrete_map=color_map,
        opacity=0.7,
        hover_data={'Label': True}
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
    fig.update_layout(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        font=dict(size=12),
        width=1400,
        height=1000,
        legend_title_text='Label',
        legend=dict(itemsizing='constant', font=dict(size=10)),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    
    fig.write_html(os.path.join(BASE_DIR, filename))
    print(f"  Saved: {filename}")

print("\n" + "="*70)
print("1. CROP LEVEL (no voting) - using prediction probabilities")
print("="*70)

crop_data = test_probs
crop_labels_gene = [idx_to_label.get(l, 'WT') for l in test_labels]
crop_labels_family = [get_family(l) for l in crop_labels_gene]
crop_labels_pathway = [get_pathway(l) for l in crop_labels_gene]

sample_size = min(10000, len(crop_data))
sample_idx = random.sample(range(len(crop_data)), sample_size)
crop_data_sample = crop_data[sample_idx]
crop_labels_gene_sample = [crop_labels_gene[i] for i in sample_idx]
crop_labels_family_sample = [crop_labels_family[i] for i in sample_idx]
crop_labels_pathway_sample = [crop_labels_pathway[i] for i in sample_idx]

create_tsne_plot(crop_data_sample, crop_labels_gene_sample, 
    f't-SNE: Crop Level - All 85 Classes (n={sample_size})', 
    'tsne_01_crop_level_all_85.html', GENE_COLORS)
create_tsne_plot(crop_data_sample, crop_labels_family_sample,
    f't-SNE: Crop Level - Family (n={sample_size})',
    'tsne_01_crop_level_family.html', FAMILY_COLORS)
create_tsne_plot(crop_data_sample, crop_labels_pathway_sample,
    f't-SNE: Crop Level - Pathway (n={sample_size})',
    'tsne_01_crop_level_pathway.html', PATHWAY_COLORS)

print("\n" + "="*70)
print("2. IMAGE LEVEL (majority vote of 144 crops)")
print("="*70)

img_data = np.array([image_level_probs[k] for k in sorted(image_level_probs.keys())])
img_labels_gene = [idx_to_label.get(image_level_labels[k], 'WT') for k in sorted(image_level_labels.keys())]
img_labels_family = [get_family(l) for l in img_labels_gene]
img_labels_pathway = [get_pathway(l) for l in img_labels_gene]

create_tsne_plot(img_data, img_labels_gene,
    f't-SNE: Image Level - All 85 Classes (n={len(img_data)})',
    'tsne_02_image_level_all_85.html', GENE_COLORS)
create_tsne_plot(img_data, img_labels_family,
    f't-SNE: Image Level - Family (n={len(img_data)})',
    'tsne_02_image_level_family.html', FAMILY_COLORS)
create_tsne_plot(img_data, img_labels_pathway,
    f't-SNE: Image Level - Pathway (n={len(img_data)})',
    'tsne_02_image_level_pathway.html', PATHWAY_COLORS)

print("\n" + "="*70)
print("3. WELL LEVEL (majority vote of all images)")
print("="*70)

well_data = np.array([well_level_probs[w] for w in sorted(well_level_probs.keys())])
well_labels_gene = [idx_to_label.get(well_level_labels[w], 'WT') for w in sorted(well_level_labels.keys())]
well_labels_family = [get_family(l) for l in well_labels_gene]
well_labels_pathway = [get_pathway(l) for l in well_labels_gene]

create_tsne_plot(well_data, well_labels_gene,
    f't-SNE: Well Level - All 85 Classes (n={len(well_data)})',
    'tsne_03_well_level_all_85.html', GENE_COLORS)
create_tsne_plot(well_data, well_labels_family,
    f't-SNE: Well Level - Family (n={len(well_data)})',
    'tsne_03_well_level_family.html', FAMILY_COLORS)
create_tsne_plot(well_data, well_labels_pathway,
    f't-SNE: Well Level - Pathway (n={len(well_data)})',
    'tsne_03_well_level_pathway.html', PATHWAY_COLORS)

print("\n" + "="*70)
print("4. UMAP VISUALIZATIONS")
print("="*70)

create_umap_plot(crop_data_sample, crop_labels_gene_sample,
    f'UMAP: Crop Level - All 85 Classes (n={sample_size})',
    'umap_01_crop_level_all_85.html', GENE_COLORS)
create_umap_plot(img_data, img_labels_gene,
    f'UMAP: Image Level - All 85 Classes (n={len(img_data)})',
    'umap_02_image_level_all_85.html', GENE_COLORS)
create_umap_plot(well_data, well_labels_gene,
    f'UMAP: Well Level - All 85 Classes (n={len(well_data)})',
    'umap_03_well_level_all_85.html', GENE_COLORS)

print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"\nAll HTML visualization files saved to: {BASE_DIR}")
print("\nGenerated files:")
print("  - tsne_01_crop_level_*.html (3 files)")
print("  - tsne_02_image_level_*.html (3 files)")
print("  - tsne_03_well_level_*.html (3 files)")
print("  - umap_01_crop_level_all_85.html")
print("  - umap_02_image_level_all_85.html")
print("  - umap_03_well_level_all_85.html")
