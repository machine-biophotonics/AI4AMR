import numpy as np
import json
import os
import re
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
import matplotlib
matplotlib.use('Agg')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Load plate maps for 85-class labels
with open(os.path.join(PROJECT_ROOT, 'plate maps', 'plate_well_id_path.json'), 'r') as f:
    plate_maps = json.load(f)

# Build filename -> plate mapping by scanning plate directories
# This is needed because same well has different labels in different plates
filename_to_plate: Dict[str, str] = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_dir = os.path.join(PROJECT_ROOT, plate)
    if os.path.exists(plate_dir):
        for fname in os.listdir(plate_dir):
            if fname.endswith(('.tif', '.tiff', '.png')):
                filename_to_plate[fname] = plate
print(f"Built filename->plate mapping: {len(filename_to_plate)} files across 6 plates")

# 29 gene classes
genes_29 = ['WT', 'dnaB', 'dnaE', 'folA', 'folP', 'ftsI', 'ftsZ', 'gyrA', 'gyrB', 
            'lptA', 'lptC', 'lpxA', 'lpxC', 'mrcA', 'mrcB', 'mrdA', 'msbA', 'murA', 
            'murC', 'parC', 'parE', 'rplA', 'rplC', 'rpoA', 'rpoB', 'rpsA', 'rpsL', 
            'secA', 'secY']
idx_to_gene = {i: g for i, g in enumerate(genes_29)}

# Colors for 85 classes (gene-based)
GENE_COLORS: Dict[str, str] = {
    'WT': '#424242',
    'dnaB': '#B39DDB', 'dnaE': '#9575CD',
    'folA': '#AED581', 'folP': '#9CCC65',
    'ftsI': '#EC407A', 'ftsZ': '#F06292',
    'gyrA': '#5C6BC0', 'gyrB': '#3F51B5',
    'lptA': '#4DD0E1', 'lptC': '#26C6DA',
    'lpxA': '#4DB6AC', 'lpxC': '#26A69A',
    'mrcA': '#E57373', 'mrcB': '#EF5350',
    'mrdA': '#F06292',
    'msbA': '#80DEEA',
    'murA': '#FFB74D', 'murC': '#FFA726',
    'parC': '#7986CB', 'parE': '#9FA8DA',
    'rplA': '#FFD54F', 'rplC': '#FFCA28',
    'rpoA': '#81C784', 'rpoB': '#66BB6A',
    'rpsA': '#FFF176', 'rpsL': '#FFEE58',
    'secA': '#4DB6AC', 'secY': '#80CBC4',
}

FAMILY_COLORS: Dict[str, str] = {
    'mrc': '#E57373', 'mrd': '#F06292', 'mur': '#FFA726', 'fts': '#EC407A',
    'lpx': '#4DB6AC', 'lpt': '#4DD0E1', 'msbA': '#80DEEA', 'gyr': '#5C6BC0',
    'par': '#7986CB', 'dna': '#9575CD', 'rpo': '#81C784', 'rpl': '#FFD54F',
    'rps': '#FFEE58', 'fol': '#AED581', 'sec': '#80CBC4', 'WT': '#424242'
}

PATHWAY_COLORS: Dict[str, str] = {
    'Cell wall': '#E57373', 'LPS': '#4DB6AC', 'DNA': '#5C6BC0',
    'Transcription': '#81C784', 'Metabolism': '#AED581', 'Cell division': '#F06292',
    'WT': '#424242'
}

# Gene families and pathways
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

# Helper functions
def extract_well_from_filename(filename: str) -> str | None:
    match = re.search(r'Well(\w\d+)', filename)
    return match.group(1) if match else None

def get_label_85_from_image(img_name: str) -> str:
    well = extract_well_from_filename(img_name)
    if not well:
        return 'WT'
    row = well[0]
    col = well[1:].lstrip('0') or '0'
    
    # Try to find plate from filename mapping
    plate = filename_to_plate.get(img_name)
    if plate and plate in plate_maps and row in plate_maps[plate]:
        for c in [well[1:], col]:
            if c in plate_maps[plate][row]:
                return plate_maps[plate][row][c]['id']
    
    # Fallback: iterate through plates
    for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        if plate in plate_maps and row in plate_maps[plate]:
            for c in [well[1:], col]:
                if c in plate_maps[plate][row]:
                    return plate_maps[plate][row][c]['id']
    return 'WT'

def extract_gene(label: str) -> str:
    if label == 'WT':
        return 'WT'
    return label.rsplit('_', 1)[0] if '_' in label else label

def get_family(gene: str) -> str:
    if gene == 'WT':
        return 'WT'
    return GENE_TO_FAMILY.get(gene, 'Unknown')

def get_pathway(gene: str) -> str:
    if gene == 'WT':
        return 'WT'
    return GENE_TO_PATHWAY.get(gene, 'Unknown')

def majority_vote(preds: List[int]) -> int:
    return Counter(preds).most_common(1)[0][0]

# Find most recent predictions file
preds_file = None
preds_mtime = 0
for f in os.listdir(SCRIPT_DIR):
    if f.startswith('test_predictions_') and f.endswith('.json'):
        fpath = os.path.join(SCRIPT_DIR, f)
        mtime = os.path.getmtime(fpath)
        if mtime > preds_mtime:
            preds_mtime = mtime
            preds_file = fpath

print(f"Loading predictions from: {preds_file}")
with open(preds_file, 'r') as f:
    results_data = json.load(f)
print(f"Loaded {len(results_data)} image predictions")

# Process data
crop_size = 224
grid_size = 12
stride = (2720 - crop_size) // (grid_size - 1)
positions = [(j * stride, i * stride) for i in range(grid_size) for j in range(grid_size)
             if j * stride + crop_size <= 2720 and i * stride + crop_size <= 2720]

print(f"Positions: {len(positions)}")

# Build data structures
crop_labels_85: List[str] = []
crop_labels_29: List[int] = []
crop_preds: List[int] = []
crop_probs: List[np.ndarray] = []
crop_image_names: List[str] = []

for result in results_data:
    img_name = result['image']
    # Use plate maps as single source of truth for all labels
    true_label_85 = get_label_85_from_image(img_name)
    true_gene = extract_gene(true_label_85)
    true_label_29 = genes_29.index(true_gene) if true_gene in genes_29 else 0
    per_crop_preds = result['per_crop_preds']
    avg_probs = np.array(result['avg_probs'])
    
    for crop_idx in range(len(positions)):
        crop_labels_85.append(true_label_85)
        crop_labels_29.append(true_label_29)
        crop_preds.append(per_crop_preds[crop_idx])
        crop_probs.append(avg_probs)
        crop_image_names.append(img_name)

crop_labels_85 = np.array(crop_labels_85)
crop_labels_29 = np.array(crop_labels_29)
crop_preds = np.array(crop_preds)
crop_probs = np.array(crop_probs)

print(f"Total crops: {len(crop_preds)}")

# Image-level aggregation
image_preds = defaultdict(list)
image_labels_85 = {}
image_labels_29 = {}
image_probs = defaultdict(list)

for i, img_name in enumerate(crop_image_names):
    image_preds[img_name].append(crop_preds[i])
    image_labels_85[img_name] = crop_labels_85[i]
    image_labels_29[img_name] = crop_labels_29[i]
    image_probs[img_name].append(crop_probs[i])

image_level_preds = {}
image_level_labels_85 = {}
image_level_labels_29 = {}
image_level_probs = {}

for img in image_preds:
    image_level_preds[img] = majority_vote(image_preds[img])
    image_level_labels_85[img] = image_labels_85[img]
    image_level_labels_29[img] = image_labels_29[img]
    image_level_probs[img] = np.mean(image_probs[img], axis=0)

# Well-level aggregation
well_preds_agg = defaultdict(list)
well_labels_85 = {}
well_labels_29 = {}
well_probs_agg = defaultdict(list)

for img in image_preds:
    well = extract_well_from_filename(img)
    if well:
        well_preds_agg[well].append(image_level_preds[img])
        well_labels_85[well] = image_level_labels_85[img]
        well_labels_29[well] = image_level_labels_29[img]
        well_probs_agg[well].append(image_level_probs[img])

well_level_preds = {}
well_level_labels_85 = {}
well_level_labels_29 = {}
well_level_probs = {}

for well in well_preds_agg:
    well_level_preds[well] = majority_vote(well_preds_agg[well])
    well_level_labels_85[well] = well_labels_85[well]
    well_level_labels_29[well] = well_labels_29[well]
    well_level_probs[well] = np.mean(well_probs_agg[well], axis=0)

# t-SNE plotting
def create_tsne_plot(data: np.ndarray, labels: List[str], title: str, 
                     filename: str, color_map: Dict[str, str]) -> None:
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
    
    fig.write_html(os.path.join(SCRIPT_DIR, filename))
    print(f"  Saved: {filename}")

# Generate color maps
# For 85-class labels, use gene-based colors (extract gene from label)
COLORS_85 = GENE_COLORS.copy()
for gene in genes_29[1:]:  # Add guide variants with same color
    for i in range(1, 4):
        COLORS_85[f'{gene}_{i}'] = GENE_COLORS[gene]

print("\n" + "="*70)
print("1. CROP LEVEL (no voting)")
print("="*70)

# Get crop-level labels
crop_true_85 = list(crop_labels_85)
crop_true_29 = [idx_to_gene[l] for l in crop_labels_29]
crop_pred_29 = [idx_to_gene[p] for p in crop_preds]
crop_true_family = [get_family(extract_gene(l)) for l in crop_labels_85]
crop_pred_family = [get_family(p) for p in crop_pred_29]
crop_true_pathway = [get_pathway(extract_gene(l)) for l in crop_labels_85]
crop_pred_pathway = [get_pathway(p) for p in crop_pred_29]

# Sample for 85-class (too many points)
sample_size = min(10000, len(crop_probs))
sample_idx = random.sample(range(len(crop_probs)), sample_size)

create_tsne_plot(crop_probs[sample_idx], [crop_true_85[i] for i in sample_idx],
    f't-SNE: Crop Level - 85 Classes (n={sample_size})',
    'tsne_01_crop_level_85.html', COLORS_85)

create_tsne_plot(crop_probs[sample_idx], [crop_true_29[i] for i in sample_idx],
    f't-SNE: Crop Level - 29 Classes (n={sample_size})',
    'tsne_01_crop_level_29.html', GENE_COLORS)

create_tsne_plot(crop_probs[sample_idx], [crop_true_family[i] for i in sample_idx],
    f't-SNE: Crop Level - 16 Family (n={sample_size})',
    'tsne_01_crop_level_family.html', FAMILY_COLORS)

create_tsne_plot(crop_probs[sample_idx], [crop_true_pathway[i] for i in sample_idx],
    f't-SNE: Crop Level - 7 Pathway (n={sample_size})',
    'tsne_01_crop_level_pathway.html', PATHWAY_COLORS)

print("\n" + "="*70)
print("2. IMAGE LEVEL (majority vote of 144 crops)")
print("="*70)

img_keys = sorted(image_level_preds.keys())
img_data = np.array([image_level_probs[k] for k in img_keys])
img_true_85 = [image_level_labels_85[k] for k in img_keys]
img_true_29 = [idx_to_gene[image_level_labels_29[k]] for k in img_keys]
img_true_family = [get_family(extract_gene(l)) for l in img_true_85]
img_true_pathway = [get_pathway(extract_gene(l)) for l in img_true_85]

create_tsne_plot(img_data, img_true_85,
    f't-SNE: Image Level - 85 Classes (n={len(img_data)})',
    'tsne_02_image_level_85.html', COLORS_85)

create_tsne_plot(img_data, img_true_29,
    f't-SNE: Image Level - 29 Classes (n={len(img_data)})',
    'tsne_02_image_level_29.html', GENE_COLORS)

create_tsne_plot(img_data, img_true_family,
    f't-SNE: Image Level - 16 Family (n={len(img_data)})',
    'tsne_02_image_level_family.html', FAMILY_COLORS)

create_tsne_plot(img_data, img_true_pathway,
    f't-SNE: Image Level - 7 Pathway (n={len(img_data)})',
    'tsne_02_image_level_pathway.html', PATHWAY_COLORS)

print("\n" + "="*70)
print("3. WELL LEVEL (majority vote of all images)")
print("="*70)

well_keys = sorted(well_level_preds.keys())
well_data = np.array([well_level_probs[w] for w in well_keys])
well_true_85_list = [well_level_labels_85[w] for w in well_keys]
well_true_29_list = [idx_to_gene[well_level_labels_29[w]] for w in well_keys]
well_true_family = [get_family(extract_gene(l)) for l in well_true_85_list]
well_true_pathway = [get_pathway(extract_gene(l)) for l in well_true_85_list]

create_tsne_plot(well_data, well_true_85_list,
    f't-SNE: Well Level - 85 Classes (n={len(well_data)})',
    'tsne_03_well_level_85.html', COLORS_85)

create_tsne_plot(well_data, well_true_29_list,
    f't-SNE: Well Level - 29 Classes (n={len(well_data)})',
    'tsne_03_well_level_29.html', GENE_COLORS)

create_tsne_plot(well_data, well_true_family,
    f't-SNE: Well Level - 16 Family (n={len(well_data)})',
    'tsne_03_well_level_family.html', FAMILY_COLORS)

create_tsne_plot(well_data, well_true_pathway,
    f't-SNE: Well Level - 7 Pathway (n={len(well_data)})',
    'tsne_03_well_level_pathway.html', PATHWAY_COLORS)

print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"\nAll HTML visualization files saved to: {SCRIPT_DIR}")
print("\nGenerated files:")
print("  - tsne_01_crop_level_*.html (4 files)")
print("  - tsne_02_image_level_*.html (4 files)")
print("  - tsne_03_well_level_*.html (4 files)")
