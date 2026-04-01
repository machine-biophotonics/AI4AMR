import numpy as np
import json
import os
import re
import ast
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = BASE_DIR

with open(os.path.join(output_dir, 'idx_to_label.json'), 'r') as f:
    idx_to_label = json.load(f)
idx_to_label = {int(k): v for k, v in idx_to_label.items()}

with open(os.path.join(output_dir, 'crop_to_image_mapping.json'), 'r') as f:
    crop_mapping_raw = json.load(f)

test_preds = np.load(os.path.join(output_dir, 'test_preds.npy'))
test_labels = np.load(os.path.join(output_dir, 'test_labels.npy'))
test_probs = np.load(os.path.join(output_dir, 'test_probs.npy'))

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
    
    crop_mapping[idx] = {
        'filename': filename,
        'well': well,
    }

unique_images = set()
unique_wells = set()
for idx, meta in crop_mapping.items():
    unique_images.add(meta['filename'])
    if meta['well']:
        unique_wells.add(meta['well'])

print(f"Total crops: {len(test_preds)}")
print(f"Unique images: {len(unique_images)}")
print(f"Unique wells: {len(unique_wells)}")

def get_base_gene(label_idx):
    label_str = idx_to_label.get(label_idx, 'WT')
    if label_str == 'WT':
        return 'WT'
    if '_' in label_str:
        return label_str.rsplit('_', 1)[0]
    return label_str

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

def get_family(label_idx):
    label_str = idx_to_label.get(label_idx, 'WT')
    if label_str == 'WT':
        return 'WT'
    base_gene = get_base_gene(label_idx)
    return GENE_TO_FAMILY.get(base_gene, base_gene)

def get_pathway(label_idx):
    label_str = idx_to_label.get(label_idx, 'WT')
    if label_str == 'WT':
        return 'WT'
    base_gene = get_base_gene(label_idx)
    return GENE_TO_PATHWAY.get(base_gene, base_gene)

image_preds = defaultdict(list)
image_labels = defaultdict(list)
image_probs = defaultdict(list)

for crop_idx in range(len(test_preds)):
    pred = test_preds[crop_idx]
    meta = crop_mapping.get(crop_idx, {})
    filename = meta.get('filename', '')
    key = filename
    image_preds[key].append(pred)
    image_labels[key].append(test_labels[crop_idx])
    image_probs[key].append(test_probs[crop_idx])

def majority_vote(preds):
    return Counter(preds).most_common(1)[0][0]

image_level_predictions = {}
image_level_labels = {}

for key in image_preds:
    image_level_predictions[key] = majority_vote(image_preds[key])
    image_level_labels[key] = image_labels[key][0]

well_preds = defaultdict(list)
well_labels = {}

for key in image_preds:
    filename = key
    match = re.search(r'Well(\w\d+)_', filename) if filename else None
    well = match.group(1) if match else ''
    
    if well:
        well_preds[well].append(image_level_predictions[key])
        if well not in well_labels:
            well_labels[well] = image_level_labels[key]

well_level_predictions = {}
well_level_labels = {}

for well in well_preds:
    well_level_predictions[well] = majority_vote(well_preds[well])

def plot_confusion_matrix(cm, labels, title, filename, figsize=(20, 18)):
    if len(labels) <= 1 or cm.shape[0] <= 1 or cm.shape[1] <= 1:
        print(f"Skipped {filename}")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

results = {}

print("\n" + "="*70)
print("1. CROP LEVEL (per-crop predictions, NO majority voting)")
print("="*70)

# Full 85 classes (WT + 28 genes × 3 guides)
crop_true_full = [idx_to_label.get(l, 'WT') for l in test_labels]
crop_pred_full = [idx_to_label.get(l, 'WT') for l in test_preds]
unique_full = sorted(set(crop_true_full + crop_pred_full))
cm = confusion_matrix(crop_true_full, crop_pred_full, labels=unique_full)
plot_confusion_matrix(cm, unique_full, 'Crop Level - All 85 Classes (No Voting)\nAccuracy: {:.2f}%'.format(
    accuracy_score(crop_true_full, crop_pred_full)*100), 
    'confusion_matrix_01_crop_level_all_85.png', figsize=(28, 24))

crop_true_guide = [get_base_gene(l) for l in test_labels]
crop_pred_guide = [get_base_gene(l) for l in test_preds]
unique_guide = sorted(set(crop_true_guide + crop_pred_guide))
cm = confusion_matrix(crop_true_guide, crop_pred_guide, labels=unique_guide)
plot_confusion_matrix(cm, unique_guide, 'Crop Level - Guide Level (No Voting)\nAccuracy: {:.2f}%'.format(
    accuracy_score(crop_true_guide, crop_pred_guide)*100), 
    'confusion_matrix_01_crop_level_guide.png', figsize=(24, 20))

crop_true_family = [get_family(l) for l in test_labels]
crop_pred_family = [get_family(l) for l in test_preds]
unique_family = sorted(set(crop_true_family + crop_pred_family))
cm = confusion_matrix(crop_true_family, crop_pred_family, labels=unique_family)
plot_confusion_matrix(cm, unique_family, 'Crop Level - Family Level (No Voting)\nAccuracy: {:.2f}%'.format(
    accuracy_score(crop_true_family, crop_pred_family)*100), 
    'confusion_matrix_01_crop_level_family.png', figsize=(12, 10))

crop_true_pathway = [get_pathway(l) for l in test_labels]
crop_pred_pathway = [get_pathway(l) for l in test_preds]
unique_pathway = sorted(set(crop_true_pathway + crop_pred_pathway))
cm = confusion_matrix(crop_true_pathway, crop_pred_pathway, labels=unique_pathway)
plot_confusion_matrix(cm, unique_pathway, 'Crop Level - Pathway Level (No Voting)\nAccuracy: {:.2f}%'.format(
    accuracy_score(crop_true_pathway, crop_pred_pathway)*100), 
    'confusion_matrix_01_crop_level_pathway.png', figsize=(10, 8))

results['crop'] = {
    'full_85': accuracy_score(crop_true_full, crop_pred_full)*100,
    'guide': accuracy_score(crop_true_guide, crop_pred_guide)*100,
    'family': accuracy_score(crop_true_family, crop_pred_family)*100,
    'pathway': accuracy_score(crop_true_pathway, crop_pred_pathway)*100
}

print("\n" + "="*70)
print("2. IMAGE LEVEL (majority vote of 144 crops per image)")
print("="*70)

# Full 85 classes (WT + 28 genes × 3 guides)
img_true_full = [idx_to_label.get(image_level_labels[k], 'WT') for k in sorted(image_level_predictions.keys())]
img_pred_full = [idx_to_label.get(image_level_predictions[k], 'WT') for k in sorted(image_level_predictions.keys())]
unique_full = sorted(set(img_true_full + img_pred_full))
cm = confusion_matrix(img_true_full, img_pred_full, labels=unique_full)
plot_confusion_matrix(cm, unique_full, 'Image Level - All 85 Classes (Majority Vote of 144 crops)\nAccuracy: {:.2f}%'.format(
    accuracy_score(img_true_full, img_pred_full)*100), 
    'confusion_matrix_02_image_level_all_85.png', figsize=(28, 24))

img_true_guide = [get_base_gene(image_level_labels[k]) for k in sorted(image_level_predictions.keys())]
img_pred_guide = [get_base_gene(image_level_predictions[k]) for k in sorted(image_level_predictions.keys())]
unique_guide = sorted(set(img_true_guide + img_pred_guide))
cm = confusion_matrix(img_true_guide, img_pred_guide, labels=unique_guide)
plot_confusion_matrix(cm, unique_guide, 'Image Level - Guide Level (Majority Vote of 144 crops)\nAccuracy: {:.2f}%'.format(
    accuracy_score(img_true_guide, img_pred_guide)*100), 
    'confusion_matrix_02_image_level_guide.png', figsize=(24, 20))

img_true_family = [get_family(image_labels[k][0]) for k in sorted(image_level_predictions.keys())]
img_pred_family = [get_family(image_level_predictions[k]) for k in sorted(image_level_predictions.keys())]
unique_family = sorted(set(img_true_family + img_pred_family))
cm = confusion_matrix(img_true_family, img_pred_family, labels=unique_family)
plot_confusion_matrix(cm, unique_family, 'Image Level - Family Level (Majority Vote of 144 crops)\nAccuracy: {:.2f}%'.format(
    accuracy_score(img_true_family, img_pred_family)*100), 
    'confusion_matrix_02_image_level_family.png', figsize=(12, 10))

img_true_pathway = [get_pathway(image_labels[k][0]) for k in sorted(image_level_predictions.keys())]
img_pred_pathway = [get_pathway(image_level_predictions[k]) for k in sorted(image_level_predictions.keys())]
unique_pathway = sorted(set(img_true_pathway + img_pred_pathway))
cm = confusion_matrix(img_true_pathway, img_pred_pathway, labels=unique_pathway)
plot_confusion_matrix(cm, unique_pathway, 'Image Level - Pathway Level (Majority Vote of 144 crops)\nAccuracy: {:.2f}%'.format(
    accuracy_score(img_true_pathway, img_pred_pathway)*100), 
    'confusion_matrix_02_image_level_pathway.png', figsize=(10, 8))

results['image'] = {
    'full_85': accuracy_score(img_true_full, img_pred_full)*100,
    'guide': accuracy_score(img_true_guide, img_pred_guide)*100,
    'family': accuracy_score(img_true_family, img_pred_family)*100,
    'pathway': accuracy_score(img_true_pathway, img_pred_pathway)*100
}

print("\n" + "="*70)
print("3. WELL LEVEL (majority vote of all images per well)")
print("="*70)

# Full 85 classes (WT + 28 genes × 3 guides)
well_true_full = [idx_to_label.get(well_labels[w], 'WT') for w in sorted(well_level_predictions.keys())]
well_pred_full = [idx_to_label.get(well_level_predictions[w], 'WT') for w in sorted(well_level_predictions.keys())]
unique_full = sorted(set(well_true_full + well_pred_full))
cm = confusion_matrix(well_true_full, well_pred_full, labels=unique_full)
plot_confusion_matrix(cm, unique_full, 'Well Level - All 85 Classes (Majority Vote of all images in well)\nAccuracy: {:.2f}%'.format(
    accuracy_score(well_true_full, well_pred_full)*100), 
    'confusion_matrix_03_well_level_all_85.png', figsize=(28, 24))

well_true_guide = [get_base_gene(well_labels[w]) for w in sorted(well_level_predictions.keys())]
well_pred_guide = [get_base_gene(well_level_predictions[w]) for w in sorted(well_level_predictions.keys())]
unique_guide = sorted(set(well_true_guide + well_pred_guide))
cm = confusion_matrix(well_true_guide, well_pred_guide, labels=unique_guide)
plot_confusion_matrix(cm, unique_guide, 'Well Level - Guide Level (Majority Vote of all images in well)\nAccuracy: {:.2f}%'.format(
    accuracy_score(well_true_guide, well_pred_guide)*100), 
    'confusion_matrix_03_well_level_guide.png', figsize=(24, 20))

well_true_family = [get_family(well_labels[w]) for w in sorted(well_level_predictions.keys())]
well_pred_family = [get_family(well_level_predictions[w]) for w in sorted(well_level_predictions.keys())]
unique_family = sorted(set(well_true_family + well_pred_family))
cm = confusion_matrix(well_true_family, well_pred_family, labels=unique_family)
plot_confusion_matrix(cm, unique_family, 'Well Level - Family Level (Majority Vote of all images in well)\nAccuracy: {:.2f}%'.format(
    accuracy_score(well_true_family, well_pred_family)*100), 
    'confusion_matrix_03_well_level_family.png', figsize=(12, 10))

well_true_pathway = [get_pathway(well_labels[w]) for w in sorted(well_level_predictions.keys())]
well_pred_pathway = [get_pathway(well_level_predictions[w]) for w in sorted(well_level_predictions.keys())]
unique_pathway = sorted(set(well_true_pathway + well_pred_pathway))
cm = confusion_matrix(well_true_pathway, well_pred_pathway, labels=unique_pathway)
plot_confusion_matrix(cm, unique_pathway, 'Well Level - Pathway Level (Majority Vote of all images in well)\nAccuracy: {:.2f}%'.format(
    accuracy_score(well_true_pathway, well_pred_pathway)*100), 
    'confusion_matrix_03_well_level_pathway.png', figsize=(10, 8))

results['well'] = {
    'full_85': accuracy_score(well_true_full, well_pred_full)*100,
    'guide': accuracy_score(well_true_guide, well_pred_guide)*100,
    'family': accuracy_score(well_true_family, well_pred_family)*100,
    'pathway': accuracy_score(well_true_pathway, well_pred_pathway)*100
}

print("\n" + "="*70)
print("="*70)
print("              FINAL ACCURACY SUMMARY")
print("="*70)

print("\n" + "-"*80)
print(f"{'Classification Level':<25} {'85 Classes':>12} {'Guide(29)':>12} {'Family(16)':>12} {'Pathway(7)':>12}")
print("-"*80)
print(f"{'Crop (No Voting)':<25} {results['crop']['full_85']:>11.2f}% {results['crop']['guide']:>11.2f}% {results['crop']['family']:>11.2f}% {results['crop']['pathway']:>11.2f}%")
print(f"{'Image (144 crop vote)':<25} {results['image']['full_85']:>11.2f}% {results['image']['guide']:>11.2f}% {results['image']['family']:>11.2f}% {results['image']['pathway']:>11.2f}%")
print(f"{'Well (all image vote)':<25} {results['well']['full_85']:>11.2f}% {results['well']['guide']:>11.2f}% {results['well']['family']:>11.2f}% {results['well']['pathway']:>11.2f}%")
print("-"*80)

print("\n" + "-"*80)
print("                    IMPROVEMENT WITH MAJORITY VOTING")
print("-"*80)
print(f"{'Level':<25} {'85 Classes':>12} {'Guide(29)':>12} {'Family(16)':>12} {'Pathway(7)':>12}")
print("-"*80)
print(f"{'Crop -> Image':<25} {results['image']['full_85']-results['crop']['full_85']:>+11.2f}% {results['image']['guide']-results['crop']['guide']:>+11.2f}% {results['image']['family']-results['crop']['family']:>+11.2f}% {results['image']['pathway']-results['crop']['pathway']:>+11.2f}%")
print(f"{'Crop -> Well':<25} {results['well']['full_85']-results['crop']['full_85']:>+11.2f}% {results['well']['guide']-results['crop']['guide']:>+11.2f}% {results['well']['family']-results['crop']['family']:>+11.2f}% {results['well']['pathway']-results['crop']['pathway']:>+11.2f}%")
print(f"{'Image -> Well':<25} {results['well']['full_85']-results['image']['full_85']:>+11.2f}% {results['well']['guide']-results['image']['guide']:>+11.2f}% {results['well']['family']-results['image']['family']:>+11.2f}% {results['well']['pathway']-results['image']['pathway']:>+11.2f}%")
print("-"*80)

print("\n" + "="*70)
print("All confusion matrix images saved to:", output_dir)
print("="*70)
