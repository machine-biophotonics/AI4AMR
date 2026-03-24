"""
Pathway-based Confusion Matrix
=============================
Groups genes by functional pathways:
- Cell wall synthesis (warm reds → oranges)
- LPS synthesis (teal → cyan → turquoise)
- DNA metabolism (indigo → violet → lavender)
- Transcription & translation (greens → golds)
- Metabolism & protein export (lime → mint)
- Cell division (magenta → rose)
- Control (WT)

Aggregation: 144 crops per image → average → confusion matrix
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import json
import re
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Gene colors from your original GENE_COLORS
GENE_COLORS = {
    'mrcA': '#E57373',
    'mrcB': '#EF5350',
    'mrdA': '#F06292',
    'ftsI': '#EC407A',
    'mreB': '#FF8A65',
    'murA': '#FFB74D',
    'murC': '#FFA726',
    'lpxA': '#4DB6AC',
    'lpxC': '#26A69A',
    'lptA': '#4DD0E1',
    'lptC': '#26C6DA',
    'msbA': '#80DEEA',
    'gyrA': '#5C6BC0',
    'gyrB': '#3F51B5',
    'parC': '#7986CB',
    'parE': '#9FA8DA',
    'dnaE': '#9575CD',
    'dnaB': '#B39DDB',
    'rpoA': '#81C784',
    'rpoB': '#66BB6A',
    'rpsA': '#FFF176',
    'rpsL': '#FFEE58',
    'rplA': '#FFD54F',
    'rplC': '#FFCA28',
    'folA': '#AED581',
    'folP': '#9CCC65',
    'secY': '#80CBC4',
    'secA': '#4DB6AC',
    'ftsZ': '#F06292',
    'minC': '#F48FB1',
    'WT': '#424242'
}

# Pathway groupings using original colors
PATHWAY_COLORS = {
    'Cell wall synthesis': {
        'genes': ['mrcA', 'mrcB', 'mrdA', 'ftsI', 'murA', 'murC'],
        'color': '#E53935',
        'hex_colors': ['#E57373', '#EF5350', '#F06292', '#EC407A', '#FFB74D', '#FFA726']
    },
    'LPS synthesis': {
        'genes': ['lpxA', 'lpxC', 'lptA', 'lptC', 'msbA'],
        'color': '#00897B',
        'hex_colors': ['#4DB6AC', '#26A69A', '#4DD0E1', '#26C6DA', '#80DEEA']
    },
    'DNA metabolism': {
        'genes': ['gyrA', 'gyrB', 'parC', 'parE', 'dnaE', 'dnaB'],
        'color': '#3949AB',
        'hex_colors': ['#5C6BC0', '#3F51B5', '#7986CB', '#9FA8DA', '#9575CD', '#B39DDB']
    },
    'Transcription & translation': {
        'genes': ['rpoA', 'rpoB', 'rpsA', 'rpsL', 'rplA', 'rplC'],
        'color': '#7CB342',
        'hex_colors': ['#81C784', '#66BB6A', '#FFF176', '#FFEE58', '#FFD54F', '#FFCA28']
    },
    'Metabolism & protein export': {
        'genes': ['folA', 'folP', 'secY', 'secA'],
        'color': '#43A047',
        'hex_colors': ['#AED581', '#9CCC65', '#80CBC4', '#4DB6AC']
    },
    'Cell division': {
        'genes': ['ftsZ', 'minC'],
        'color': '#D81B60',
        'hex_colors': ['#F06292', '#F48FB1']
    },
    'Control (WT)': {
        'genes': ['WT'],
        'color': '#424242',
        'hex_colors': ['#424242']
    }
}

# Build gene to pathway mapping
GENE_TO_PATHWAY = {}
for pathway, info in PATHWAY_COLORS.items():
    for gene in info['genes']:
        GENE_TO_PATHWAY[gene] = pathway

# All pathway names in order
PATHWAY_ORDER = [
    'Cell wall synthesis',
    'LPS synthesis', 
    'DNA metabolism',
    'Transcription & translation',
    'Metabolism & protein export',
    'Cell division',
    'Control (WT)'
]

def bin_pathway(label):
    """Bin label to pathway name"""
    if label is None:
        return None
    
    # Combine ALL NC_* and WT* variants into Control (WT)
    if label.startswith('NC_') or label.startswith('WT') or label.startswith('WT '):
        return 'Control (WT)'
    
    # Handle numbered genes (e.g., rpsA_1, rpsA_2, rpsA_3)
    base = label
    if base.endswith('_1') or base.endswith('_2') or base.endswith('_3'):
        base = base[:-2]
    
    # Map to pathway
    if base in GENE_TO_PATHWAY:
        return GENE_TO_PATHWAY[base]
    
    return base  # Return as-is if not in any pathway


def extract_well_from_filename(filename):
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        row = match.group(1)
        col = int(match.group(2))
        return f"{row}{col:02d}"
    return None


with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

# Get all unique labels from plate maps
all_labels = sorted(set(label for pm in plate_maps.values() for label in pm.values()))
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(all_labels)
print(f"Number of classes: {num_classes}")


def get_label_from_path(img_path):
    dirname = os.path.basename(os.path.dirname(img_path))
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if dirname in plate_maps and well in plate_maps[dirname]:
        return plate_maps[dirname][well]
    return None


class MultiPatchDataset(Dataset):
    """MultiPatch Dataset - 10 random crops from center region"""
    def __init__(self, image_paths, transform=None, n_patches=10):
        self.image_paths = image_paths
        self.transform = transform
        self.n_patches = n_patches
        self.patch_size = 224
        self.edge_margin = 200
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        patches = []
        center_w_start = self.edge_margin
        center_w_end = w - self.edge_margin
        center_h_start = self.edge_margin
        center_h_end = h - self.edge_margin
        
        center_w = center_w_end - center_w_start
        center_h = center_h_end - center_h_start
        
        for _ in range(self.n_patches):
            left = int(np.random.uniform(0, max(1, center_w - self.patch_size)))
            top = int(np.random.uniform(0, max(1, center_h - self.patch_size)))
            left = center_w_start + left
            top = center_h_start + top
            
            patch = image.crop((left, top, left + self.patch_size, top + self.patch_size))
            
            if self.transform:
                patch = self.transform(patch)
            patches.append(patch)
        
        return torch.stack(patches), img_path


class RandomCenterCrop:
    def __init__(self, size, edge_margin=200):
        self.size = size
        self.edge_margin = edge_margin
    
    def __call__(self, img):
        w, h = img.size
        center_w_start = self.edge_margin
        center_w_end = w - self.edge_margin
        center_h_start = self.edge_margin
        center_h_end = h - self.edge_margin
        
        max_top = center_h_end - self.size
        max_left = center_w_end - self.size
        
        if max_top <= 0 or max_left <= 0:
            left = (w - self.size) // 2
            top = (h - self.size) // 2
        else:
            top = random.randint(center_h_start, max_top)
            left = random.randint(center_w_start, max_left)
        
        return img.crop((left, top, left + self.size, top + self.size))


val_transform = Compose([
    RandomCenterCrop(224, edge_margin=200),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    print("Loading model...")
    checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model.pth'), map_location=device)
    
    # Use label mapping from checkpoint (97 classes)
    idx_to_label = checkpoint['idx_to_label']
    label_to_idx = checkpoint['label_to_idx']
    num_classes_checkpoint = checkpoint['num_classes']
    
    print(f"Model was trained with {num_classes_checkpoint} classes")
    
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = 1280
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes_checkpoint)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))
    print(f"Found {len(test_paths)} test images")
    
    test_dataset = MultiPatchDataset(test_paths, transform=val_transform, n_patches=10)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8,  # 8 images per batch = 80 crops
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )
    
    print("Getting predictions for each image (average 144 crops)...")
    image_probs = {}
    image_labels = {}
    
    with torch.no_grad():
        for patches, paths in tqdm(test_loader, desc="Inference"):
            batch_size, n_patches, C, H, W = patches.shape
            patches = patches.view(-1, C, H, W).to(device)
            
            outputs = model(patches)
            outputs = outputs.view(batch_size, n_patches, -1)
            probs = torch.softmax(outputs, dim=2)
            avg_probs = probs.mean(dim=1)
            
            for i in range(batch_size):
                img_path = paths[i]
                label = get_label_from_path(img_path)
                image_probs[img_path] = avg_probs[i].cpu().numpy()
                image_labels[img_path] = label
    
    # Convert to arrays for confusion matrix
    print("Computing pathway confusion matrix...")
    
    all_true_pathways = []
    all_pred_pathways = []
    
    for img_path, probs in image_probs.items():
        true_label = image_labels[img_path]
        true_pathway = bin_pathway(true_label)
        
        pred_idx = np.argmax(probs)
        pred_label = idx_to_label[pred_idx]
        pred_pathway = bin_pathway(pred_label)
        
        if true_pathway and pred_pathway:
            all_true_pathways.append(true_pathway)
            all_pred_pathways.append(pred_pathway)
    
    print(f"Total samples: {len(all_true_pathways)}")
    
    # Create confusion matrix at pathway level
    pathway_cm = confusion_matrix(all_true_pathways, all_pred_pathways, labels=PATHWAY_ORDER)
    
    # Calculate accuracy per pathway
    pathway_acc = {}
    for i, pathway in enumerate(PATHWAY_ORDER):
        if pathway_cm[i].sum() > 0:
            pathway_acc[pathway] = 100 * pathway_cm[i, i] / pathway_cm[i].sum()
        else:
            pathway_acc[pathway] = 0
    
    # Calculate overall accuracy
    total_correct = sum([pathway_cm[PATHWAY_ORDER.index(p), PATHWAY_ORDER.index(p)] for p in PATHWAY_ORDER])
    total_samples = sum([pathway_cm[PATHWAY_ORDER.index(p)].sum() for p in PATHWAY_ORDER])
    overall_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"Pathway Accuracy:")
    for pathway, acc in pathway_acc.items():
        print(f"  {pathway}: {acc:.1f}%")
    print(f"\n  OVERALL ACCURACY: {overall_acc:.1f}%")
    print(f"{'='*50}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Normalize confusion matrix
    pathway_cm_norm = pathway_cm.astype('float') / pathway_cm.sum(axis=1)[:, np.newaxis]
    pathway_cm_norm = np.nan_to_num(pathway_cm_norm) * 100
    
    # Create color map
    pathway_colors = [PATHWAY_COLORS[p]['color'] for p in PATHWAY_ORDER]
    
    # Plot with seaborn
    sns.heatmap(
        pathway_cm_norm,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=PATHWAY_ORDER,
        yticklabels=PATHWAY_ORDER,
        ax=ax,
        cbar_kws={'label': 'Accuracy (%)'},
        linewidths=0.5,
        linecolor='white',
        annot_kws={'size': 12, 'weight': 'bold'}
    )
    
    ax.set_xlabel('Predicted Pathway', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Pathway', fontsize=14, fontweight='bold')
    ax.set_title(f'Pathway-Level Confusion Matrix\nOverall Accuracy: {overall_acc:.1f}%', fontsize=16, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix_pathway.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved: confusion_matrix_pathway.png")
    
    # Create detailed version with gene counts
    fig2, ax2 = plt.subplots(figsize=(16, 14))
    
    # Add gene counts to labels
    pathway_labels = []
    for p in PATHWAY_ORDER:
        genes = PATHWAY_COLORS[p]['genes']
        pathway_labels.append(f"{p}\n({len(genes)} genes)")
    
    sns.heatmap(
        pathway_cm,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        xticklabels=pathway_labels,
        yticklabels=pathway_labels,
        ax=ax2,
        cbar_kws={'label': 'Count'},
        linewidths=0.5,
        linecolor='white',
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    
    ax2.set_xlabel('Predicted Pathway', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Pathway', fontsize=14, fontweight='bold')
    ax2.set_title(f'Pathway Confusion Matrix (Raw Counts)\nOverall Accuracy: {overall_acc:.1f}%', fontsize=16, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix_pathway_counts.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: confusion_matrix_pathway_counts.png")
    
    # Create per-pathway accuracy bar chart
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    pathways = list(pathway_acc.keys())
    accuracies = list(pathway_acc.values())
    colors = [PATHWAY_COLORS[p]['color'] for p in pathways]
    
    bars = ax3.barh(pathways, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1f}%', va='center', fontsize=12, fontweight='bold')
    
    ax3.set_xlabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, max(accuracies) * 1.15)
    ax3.set_title('Per-Pathway Classification Accuracy\n(Test Set: P6)', fontsize=16, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'pathway_accuracy.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: pathway_accuracy.png")
    
    # Save pathway metrics to CSV
    import csv
    with open(os.path.join(BASE_DIR, 'pathway_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Pathway', 'Accuracy (%)', 'Total Samples', 'Correct Predictions'])
        for pathway in PATHWAY_ORDER:
            total = pathway_cm[PATHWAY_ORDER.index(pathway)].sum()
            correct = pathway_cm[PATHWAY_ORDER.index(pathway), PATHWAY_ORDER.index(pathway)]
            writer.writerow([pathway, f'{pathway_acc[pathway]:.2f}', total, correct])
        writer.writerow([])
        writer.writerow(['OVERALL', f'{overall_acc:.2f}', total_samples, total_correct])
    
    print(f"Saved: pathway_metrics.csv")
    
    plt.close('all')
    print("\nDone!")


if __name__ == '__main__':
    main()
