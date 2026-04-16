import numpy as np
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 8

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

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

# 29 gene classes (model output)
genes_29 = ['WT', 'dnaB', 'dnaE', 'folA', 'folP', 'ftsI', 'ftsZ', 'gyrA', 'gyrB', 
            'lptA', 'lptC', 'lpxA', 'lpxC', 'mrcA', 'mrcB', 'mrdA', 'msbA', 'murA', 
            'murC', 'parC', 'parE', 'rplA', 'rplC', 'rpoA', 'rpoB', 'rpsA', 'rpsL', 
            'secA', 'secY']
idx_to_gene = {i: g for i, g in enumerate(genes_29)}

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

def get_label_85_from_image(img_name: str, plate: str = None) -> str:
    well = extract_well_from_filename(img_name)
    if not well:
        return 'WT'
    row = well[0]
    col = well[1:].lstrip('0') or '0'
    
    # If plate is specified, use only that plate
    if plate:
        if plate in plate_maps and row in plate_maps[plate]:
            for c in [well[1:], col]:
                if c in plate_maps[plate][row]:
                    return plate_maps[plate][row][c]['id']
        return 'WT'
    
    # Otherwise, try to find plate from filename mapping, then fall back to iterating plates
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
    
    # For crop-level: we have 144 crops per image, all with same true labels
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

# Image-level aggregation (majority vote)
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
well_level_probs = {}

for well in well_preds_agg:
    well_level_preds[well] = majority_vote(well_preds_agg[well])
    well_level_probs[well] = np.mean(well_probs_agg[well], axis=0)

# Confusion matrix plotting
def plot_confusion_matrix(cm: np.ndarray, true_labels: List[str], pred_labels: List[str], 
                          title: str, filename: str, figsize: Tuple[int, int] = (14, 12)) -> None:
    if len(true_labels) <= 1 or len(pred_labels) <= 1:
        print(f"Skipped {filename}")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set(xticks=np.arange(len(pred_labels)), yticks=np.arange(len(true_labels)),
           xticklabels=pred_labels, yticklabels=true_labels, title=title,
           ylabel='True Label', xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black", fontsize=4)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def compute_confusion_matrix_85_to_29(true_labels_85: np.ndarray, pred_indices: np.ndarray,
                                       level: str) -> Tuple[np.ndarray, List[str], List[str], float]:
    """Compute confusion matrix and accuracy for 85->29 mapping.
    Rows: 85 guide-level labels (e.g., 'mrcA_1', 'mrcA_2', ..., 'WT')
    Columns: 29 gene-level labels (e.g., 'mrcA', 'mrcB', ..., 'WT')
    """
    pred_genes = [idx_to_gene[p] for p in pred_indices]
    
    # Get all unique 85-class labels from true labels
    all_85_labels = sorted(set(true_labels_85))
    all_29_labels = sorted(genes_29)  # Use full 29 gene list
    
    # Build 85 x 29 confusion matrix
    cm = np.zeros((len(all_85_labels), len(all_29_labels)), dtype=int)
    true_85_to_idx = {l: i for i, l in enumerate(all_85_labels)}
    pred_29_to_idx = {l: i for i, l in enumerate(all_29_labels)}
    
    for t, p in zip(true_labels_85, pred_genes):
        if t in true_85_to_idx and p in pred_29_to_idx:
            cm[true_85_to_idx[t], pred_29_to_idx[p]] += 1
    
    # Accuracy: true gene (extracted from 85-class) matches predicted gene
    true_genes = [extract_gene(l) for l in true_labels_85]
    accuracy = accuracy_score(true_genes, pred_genes) * 100
    
    return cm, list(all_85_labels), all_29_labels, accuracy

def compute_confusion_matrix_same(true_labels: np.ndarray, pred_labels: np.ndarray) -> Tuple[np.ndarray, List[str], List[str], float]:
    """Compute confusion matrix when true and pred have same label set."""
    labels = sorted(set(true_labels) | set(pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    accuracy = accuracy_score(true_labels, pred_labels) * 100
    return cm, labels, labels, accuracy

# Results storage
results = {}

print("\n" + "="*70)
print("1. CROP LEVEL")
print("="*70)

# 85->29
cm, true_l, pred_l, acc = compute_confusion_matrix_85_to_29(crop_labels_85, crop_preds, 'crop')
plot_confusion_matrix(cm, true_l, pred_l, f'Crop Level: 85->29\nAccuracy: {acc:.2f}%',
                     'confusion_matrix_01_crop_85_to_29.png', figsize=(28, 24))
crop_85_29 = acc

# 29->29
true_29_genes = [idx_to_gene[l] for l in crop_labels_29]
pred_29_genes = [idx_to_gene[p] for p in crop_preds]
cm, true_l, pred_l, acc = compute_confusion_matrix_same(true_29_genes, pred_29_genes)
plot_confusion_matrix(cm, true_l, pred_l, f'Crop Level: 29->29\nAccuracy: {acc:.2f}%',
                     'confusion_matrix_01_crop_29_to_29.png', figsize=(20, 18))
crop_29_29 = acc

# Family->Family
true_fam = [get_family(extract_gene(l)) for l in crop_labels_85]
pred_fam = [get_family(idx_to_gene[p]) for p in crop_preds]
cm, true_l, pred_l, acc = compute_confusion_matrix_same(true_fam, pred_fam)
plot_confusion_matrix(cm, true_l, pred_l, f'Crop Level: Family->Family\nAccuracy: {acc:.2f}%',
                     'confusion_matrix_01_crop_family.png', figsize=(12, 10))
crop_family = acc

# Pathway->Pathway
true_path = [get_pathway(extract_gene(l)) for l in crop_labels_85]
pred_path = [get_pathway(idx_to_gene[p]) for p in crop_preds]
cm, true_l, pred_l, acc = compute_confusion_matrix_same(true_path, pred_path)
plot_confusion_matrix(cm, true_l, pred_l, f'Crop Level: Pathway->Pathway\nAccuracy: {acc:.2f}%',
                     'confusion_matrix_01_crop_pathway.png', figsize=(10, 8))
crop_pathway = acc

results['crop'] = {'85_to_29': crop_85_29, '29_to_29': crop_29_29, 'family': crop_family, 'pathway': crop_pathway}

print("\n" + "="*70)
print("2. IMAGE LEVEL (majority vote of 144 crops)")
print("="*70)

img_keys = sorted(image_level_preds.keys())
img_true_85 = np.array([image_level_labels_85[k] for k in img_keys])
img_pred_indices = np.array([image_level_preds[k] for k in img_keys])
img_true_29 = np.array([image_level_labels_29[k] for k in img_keys])

# 85->29
cm, true_l, pred_l, acc = compute_confusion_matrix_85_to_29(img_true_85, img_pred_indices, 'image')
plot_confusion_matrix(cm, true_l, pred_l, f'Image Level: 85->29\nAccuracy: {acc:.2f}%',
                     'confusion_matrix_02_image_85_to_29.png', figsize=(28, 24))
img_85_29 = acc

# 29->29
true_29_genes = [idx_to_gene[l] for l in img_true_29]
pred_29_genes = [idx_to_gene[p] for p in img_pred_indices]
cm, true_l, pred_l, acc = compute_confusion_matrix_same(true_29_genes, pred_29_genes)
plot_confusion_matrix(cm, true_l, pred_l, f'Image Level: 29->29\nAccuracy: {acc:.2f}%',
                     'confusion_matrix_02_image_29_to_29.png', figsize=(20, 18))
img_29_29 = acc

# Family->Family
true_fam = [get_family(extract_gene(l)) for l in img_true_85]
pred_fam = [get_family(idx_to_gene[p]) for p in img_pred_indices]
cm, true_l, pred_l, acc = compute_confusion_matrix_same(true_fam, pred_fam)
plot_confusion_matrix(cm, true_l, pred_l, f'Image Level: Family->Family\nAccuracy: {acc:.2f}%',
                     'confusion_matrix_02_image_family.png', figsize=(12, 10))
img_family = acc

# Pathway->Pathway
true_path = [get_pathway(extract_gene(l)) for l in img_true_85]
pred_path = [get_pathway(idx_to_gene[p]) for p in img_pred_indices]
cm, true_l, pred_l, acc = compute_confusion_matrix_same(true_path, pred_path)
plot_confusion_matrix(cm, true_l, pred_l, f'Image Level: Pathway->Pathway\nAccuracy: {acc:.2f}%',
                     'confusion_matrix_02_image_pathway.png', figsize=(10, 8))
img_pathway = acc

results['image'] = {'85_to_29': img_85_29, '29_to_29': img_29_29, 'family': img_family, 'pathway': img_pathway}

print("\n" + "="*70)
print("3. WELL LEVEL (majority vote of all images in well)")
print("="*70)

well_keys = sorted(well_level_preds.keys())
well_true_85 = np.array([well_labels_85[w] for w in well_keys])
well_pred_indices = np.array([well_level_preds[w] for w in well_keys])
well_true_29 = np.array([well_labels_29[w] for w in well_keys])

# 85->29
cm, true_l, pred_l, acc = compute_confusion_matrix_85_to_29(well_true_85, well_pred_indices, 'well')
plot_confusion_matrix(cm, true_l, pred_l, f'Well Level: 85->29\nAccuracy: {acc:.2f}%',
                     'confusion_matrix_03_well_85_to_29.png', figsize=(20, 18))
well_85_29 = acc

# 29->29
true_29_genes = [idx_to_gene[l] for l in well_true_29]
pred_29_genes = [idx_to_gene[p] for p in well_pred_indices]
cm, true_l, pred_l, acc = compute_confusion_matrix_same(true_29_genes, pred_29_genes)
plot_confusion_matrix(cm, true_l, pred_l, f'Well Level: 29->29\nAccuracy: {acc:.2f}%',
                     'confusion_matrix_03_well_29_to_29.png', figsize=(20, 18))
well_29_29 = acc

# Family->Family
true_fam = [get_family(extract_gene(l)) for l in well_true_85]
pred_fam = [get_family(idx_to_gene[p]) for p in well_pred_indices]
cm, true_l, pred_l, acc = compute_confusion_matrix_same(true_fam, pred_fam)
plot_confusion_matrix(cm, true_l, pred_l, f'Well Level: Family->Family\nAccuracy: {acc:.2f}%',
                     'confusion_matrix_03_well_family.png', figsize=(12, 10))
well_family = acc

# Pathway->Pathway
true_path = [get_pathway(extract_gene(l)) for l in well_true_85]
pred_path = [get_pathway(idx_to_gene[p]) for p in well_pred_indices]
cm, true_l, pred_l, acc = compute_confusion_matrix_same(true_path, pred_path)
plot_confusion_matrix(cm, true_l, pred_l, f'Well Level: Pathway->Pathway\nAccuracy: {acc:.2f}%',
                     'confusion_matrix_03_well_pathway.png', figsize=(10, 8))
well_pathway = acc

results['well'] = {'85_to_29': well_85_29, '29_to_29': well_29_29, 'family': well_family, 'pathway': well_pathway}

# Summary
print("\n" + "="*70)
print("              FINAL ACCURACY SUMMARY")
print("="*70)
print("\n" + "-"*100)
print(f"{'Classification Level':<25} {'85->29':>12} {'29->29':>12} {'Family(16)':>12} {'Pathway(7)':>12}")
print("-"*100)
print(f"{'Crop (No Voting)':<25} {results['crop']['85_to_29']:>11.2f}% {results['crop']['29_to_29']:>11.2f}% {results['crop']['family']:>11.2f}% {results['crop']['pathway']:>11.2f}%")
print(f"{'Image (144 crop vote)':<25} {results['image']['85_to_29']:>11.2f}% {results['image']['29_to_29']:>11.2f}% {results['image']['family']:>11.2f}% {results['image']['pathway']:>11.2f}%")
print(f"{'Well (all image vote)':<25} {results['well']['85_to_29']:>11.2f}% {results['well']['29_to_29']:>11.2f}% {results['well']['family']:>11.2f}% {results['well']['pathway']:>11.2f}%")
print("-"*100)

print("\n" + "="*70)
print("All confusion matrix images saved to:", SCRIPT_DIR)
print("="*70)
