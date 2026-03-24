"""
Comprehensive plotting script for train.py results
Generates: Training curves, t-SNE, Confusion matrix with PER-IMAGE AVERAGING
Usage: python generate_plots.py

Key change: Predictions are averaged across all crops per well (9 crops)
This gives better accuracy as it aggregates information from all parts of the well.
"""

import os
import sys

# Create results folder
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'baseline')
os.makedirs(RESULTS_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
from PIL import ImageOps
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
import json
import glob
import random
from tqdm import tqdm
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Gene colors
GENE_COLORS = {
    'mrcA': '#E57373', 'mrcB': '#EF5350', 'mrdA': '#F06292', 'ftsI': '#EC407A',
    'mreB': '#FF8A65', 'murA': '#FFB74D', 'murC': '#FFA726',
    'lpxA': '#4DB6AC', 'lpxC': '#26A69A', 'lptA': '#4DD0E1', 'lptC': '#26C6DA', 'msbA': '#80DEEA',
    'gyrA': '#5C6BC0', 'gyrB': '#3F51B5', 'parC': '#7986CB', 'parE': '#9FA8DA', 'dnaE': '#9575CD', 'dnaB': '#B39DDB',
    'rpoA': '#81C784', 'rpoB': '#66BB6A', 'rpsA': '#FFF176', 'rpsL': '#FFEE58', 'rplA': '#FFD54F', 'rplC': '#FFCA28',
    'folA': '#AED581', 'folP': '#9CCC65', 'secY': '#80CBC4', 'secA': '#4DB6AC',
    'ftsZ': '#F06292', 'minC': '#F48FB1',
    'WT': '#424242'
}

PATHWAY_COLORS = {
    'Control': '#424242',
    'Cell wall synthesis': '#E57373',
    'LPS synthesis': '#4DB6AC',
    'DNA metabolism': '#5C6BC0',
    'Transcription/translation': '#81C784',
    'Metabolism/export': '#AED581',
    'Cell division': '#F06292'
}

# Load pathway mapping
pathway_json_path = os.path.join(BASE_DIR, 'class_pathway_order.json')
pathway_mapping = {}
if os.path.exists(pathway_json_path):
    with open(pathway_json_path) as f:
        pathway_data = json.load(f)
    pathway_mapping = pathway_data.get('pathway_mapping', {})

def get_pathway_color(label):
    """Get color for a gene based on its pathway"""
    if label == 'WT':
        return PATHWAY_COLORS['Control']
    gene_base = label.rsplit('_', 1)[0] if '_' in label else label
    for pathway, info in pathway_mapping.items():
        if pathway == gene_base:
            return PATHWAY_COLORS.get(info.get('pathway', 'Control'), '#888888')
    return '#888888'


class WellCropDataset(Dataset):
    """Dataset that loads ALL crops for each well and returns averaged predictions"""
    def __init__(self, well_data, label, transform=None, n_crops=9):
        """
        well_data: list of (image_path, crop_indices) tuples
        label: the label for this well
        n_crops: number of crops to use per well
        """
        self.well_data = well_data  # List of (img_path, crop_row, crop_col)
        self.label = label
        self.transform = transform
        self.n_crops = n_crops
        
    def __len__(self):
        return len(self.well_data)
    
    def __getitem__(self, idx):
        img_path, crop_row, crop_col = self.well_data[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Calculate crop size and position
        img_w, img_h = img.size
        crop_w, crop_h = 224, 224
        
        # Calculate crop boundaries
        left = crop_col * crop_w
        top = crop_row * crop_h
        right = left + crop_w
        bottom = top + crop_h
        
        # Crop and transform
        crop = img.crop((left, top, right, bottom))
        
        if self.transform:
            crop = self.transform(crop)
        
        return crop, self.label


def load_data():
    """Load plate mapping and data paths"""
    with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
        plate_data = json.load(f)
    
    plate_maps = {}
    for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        plate_maps[plate] = {}
        for row, wells in plate_data[plate].items():
            for col, info in wells.items():
                well = f"{row}{int(col):02d}"
                plate_maps[plate][well] = info['id']
    
    all_labels = sorted(set(label for pm in plate_maps.values() for label in pm.values()))
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    num_classes = len(all_labels)
    
    return plate_maps, label_to_idx, idx_to_label, num_classes


def get_test_paths():
    """Get test image paths (per-crop, original approach)"""
    plate_maps, label_to_idx, idx_to_label, num_classes = load_data()
    
    test_paths = []
    test_labels = []
    
    for p in ['P6']:  # Test set
        for well, label_str in plate_maps[p].items():
            label = label_to_idx[label_str]
            
            img_files = glob.glob(os.path.join(BASE_DIR, p, f'*{well}*.tif'))
            
            for img_path in img_files:
                test_paths.append(img_path)
                test_labels.append(label)
    
    return test_paths, test_labels, idx_to_label


def get_well_data_for_testing(n_crops=9):
    """Get well data organized for per-image averaging"""
    plate_maps, label_to_idx, idx_to_label, num_classes = load_data()
    
    # Center 3x3 grid positions for 9 crops
    crop_positions = []
    for r in range(4, 7):  # rows 4, 5, 6
        for c in range(4, 7):  # cols 4, 5, 6
            crop_positions.append((r, c))
    
    well_data = []  # List of (well_label, list of crop data)
    
    for p in ['P6']:  # Test set
        for well, label_str in plate_maps[p].items():
            label = label_to_idx[label_str]
            
            img_files = glob.glob(os.path.join(BASE_DIR, p, f'*{well}*.tif'))
            if not img_files:
                continue
            
            # Use first image file for this well
            img_path = img_files[0]
            
            # Create list of all crops for this well
            crops = []
            for crop_row, crop_col in crop_positions:
                crops.append((img_path, crop_row, crop_col))
            
            well_data.append((label_str, label, crops))
    
    print(f"Test wells: {len(well_data)} wells with {n_crops} crops each")
    return well_data, idx_to_label, label_to_idx


def load_model():
    """Load trained model - handles both single-task and multi-task"""
    checkpoint_path = os.path.join(BASE_DIR, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: {checkpoint_path} not found!")
        print("Run training first to generate the model.")
        return None, None, None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load class mapping
    if 'label_to_idx' in checkpoint:
        label_to_idx = checkpoint['label_to_idx']
        idx_to_label = {v: k for k, v in label_to_idx.items()}
    else:
        _, label_to_idx, idx_to_label, _ = load_data()
    
    # Create model
    model = models.efficientnet_b0(weights=None)
    in_features = 1280
    
    # Check checkpoint architecture
    state_dict_keys = checkpoint['model_state_dict'].keys()
    classifier_keys = [k for k in state_dict_keys if 'classifier' in k]
    
    # Check if multi-task model
    has_multi_task = any('shared' in k or 'gene_head' in k for k in state_dict_keys)
    
    if has_multi_task:
        print("Loading multi-task model...")
        # Multi-task model class
        class MultiTaskEfficientNet(nn.Module):
            def __init__(self, num_classes, num_pathways=7, num_binary=2):
                super().__init__()
                self.backbone = models.efficientnet_b0(weights=None)
                self.backbone.classifier = nn.Identity()
                in_features = 1280
                self.shared = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(in_features, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.3)
                )
                self.gene_head = nn.Linear(256, num_classes)
                self.pathway_head = nn.Linear(256, num_pathways)
                self.binary_head = nn.Linear(256, num_binary)
            
            def forward(self, x):
                x = self.backbone(x)
                x = self.shared(x)
                gene_logits = self.gene_head(x)
                pathway_logits = self.pathway_head(x)
                binary_logits = self.binary_head(x)
                return gene_logits, pathway_logits, binary_logits
        
        model = MultiTaskEfficientNet(num_classes=len(idx_to_label))
    else:
        print("Loading single-task model...")
        # Single-task model
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, len(idx_to_label))
        )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    print(f"Best val acc: {checkpoint.get('best_val_acc', '?')}%")
    
    return model, idx_to_label, label_to_idx


def predict_with_averaging(well_data, model, transform, batch_size=16, n_crops=9):
    """
    Predict by averaging logits across all crops per well
    Returns: list of (true_label, predicted_label) for each well
    """
    all_well_preds = []
    all_well_labels = []
    all_well_features = []
    
    model.eval()
    with torch.no_grad():
        for well_name, true_label, crops in tqdm(well_data, desc="Predicting wells"):
            # Collect all crops for this well
            crop_tensors = []
            for img_path, crop_row, crop_col in crops:
                img = Image.open(img_path).convert('RGB')
                img_w, img_h = img.size
                crop_w, crop_h = 224, 224
                
                left = crop_col * crop_w
                top = crop_row * crop_h
                right = left + crop_w
                bottom = top + crop_h
                
                crop = img.crop((left, top, right, bottom))
                crop_tensor = transform(crop)
                crop_tensors.append(crop_tensor)
            
            # Stack crops
            batch = torch.stack(crop_tensors).to(device)
            
            # Forward pass
            outputs = model(batch)
            if isinstance(outputs, tuple):
                gene_logits = outputs[0]
            else:
                gene_logits = outputs
            
            # Average logits across crops
            avg_logits = gene_logits.mean(dim=0)
            
            # Get prediction
            pred = avg_logits.argmax().item()
            
            all_well_preds.append(pred)
            all_well_labels.append(true_label)
            all_well_features.append(avg_logits.cpu().numpy())
    
    return np.array(all_well_preds), np.array(all_well_labels), np.array(all_well_features)


def plot_training_curves():
    """Generate training curves from checkpoint"""
    checkpoint_path = os.path.join(BASE_DIR, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found, skipping training curves")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    train_losses = checkpoint.get('train_losses', [])
    train_accs = checkpoint.get('train_accs', [])
    val_losses = checkpoint.get('val_losses', [])
    val_accs = checkpoint.get('val_accs', [])
    
    if not train_losses:
        print("No training stats found in checkpoint")
        return
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gap
    gap = [train_accs[i] - val_accs[i] for i in range(len(train_accs))]
    axes[1, 0].plot(epochs, gap, 'g-', label='Train - Val Gap', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].fill_between(epochs, gap, 0, alpha=0.3, color=['green' if g > 0 else 'red' for g in gap])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gap (%)')
    axes[1, 0].set_title('Overfitting Gap (Train - Val)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary
    best_val_idx = val_accs.index(max(val_accs)) if val_accs else 0
    best_val_acc = val_accs[best_val_idx] if val_accs else 0
    
    axes[1, 1].text(0.5, 0.7, f"Training Summary", fontsize=16, fontweight='bold', 
                     ha='center', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.5, f"Total Epochs: {len(train_losses)}", fontsize=12, 
                     ha='center', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.4, f"Best Val Acc: {best_val_acc:.2f}%", fontsize=12, 
                     ha='center', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.3, f"Final Train Acc: {train_accs[-1]:.2f}%", fontsize=12, 
                     ha='center', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.2, f"Final Val Acc: {val_accs[-1]:.2f}%", fontsize=12, 
                     ha='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_tsne():
    """Generate t-SNE visualization (per-crop, original approach)"""
    print("\nGenerating t-SNE visualization (per-crop)...")
    
    model, idx_to_label, label_to_idx = load_model()
    if model is None:
        return
    
    # Use original approach: per-crop features
    test_paths, test_labels, _ = get_test_paths()
    
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Using {len(test_paths)} crop samples for t-SNE")
    
    # Load images and extract features
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for i, path in enumerate(tqdm(test_paths, desc="Extracting features")):
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)  # transform returns tensor
            img_tensor = img_tensor.unsqueeze(0).to(device)  # add batch dim and move to GPU
            
            outputs = model(img_tensor)
            if isinstance(outputs, tuple):
                gene_logits = outputs[0]
            else:
                gene_logits = outputs
            
            feat = gene_logits.squeeze().cpu().numpy()
            features.append(feat)
            labels.append(test_labels[i])
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Running t-SNE on {features.shape}...")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    # Create colors for each point
    colors = [get_pathway_color(idx_to_label[l]) for l in labels]
    
    # Interactive plot with plotly
    df = {
        'x': features_2d[:, 0],
        'y': features_2d[:, 1],
        'label': [idx_to_label[l] for l in labels],
        'color': colors
    }
    
    fig = px.scatter(
        df, x='x', y='y', color='label',
        title='t-SNE: Gene Classification (Per-Crop Features)',
        labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
        width=1200, height=800
    )
    
    output_path = os.path.join(RESULTS_DIR, 'tsne_interactive.html')
    fig.write_html(output_path)
    print(f"Saved: {output_path}")
    
    # Static plot
    plt.figure(figsize=(12, 10))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, s=5, alpha=0.6)
    
    # Add legend for pathways
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=8, label=pathway)
                      for pathway, color in PATHWAY_COLORS.items()]
    plt.legend(handles=legend_elements, loc='best', fontsize=8)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE: Gene Classification (Per-Crop Features)')
    
    output_path = os.path.join(RESULTS_DIR, 'tsne_static.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Static plot
    plt.figure(figsize=(12, 10))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, s=20, alpha=0.7)
    
    # Add legend for pathways
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=color, markersize=8, label=pathway)
                      for pathway, color in PATHWAY_COLORS.items()]
    plt.legend(handles=legend_elements, loc='best', fontsize=8)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f't-SNE: Gene Classification (Per-Well Averaged, Acc: {accuracy:.2f}%)')
    
    output_path = os.path.join(RESULTS_DIR, 'tsne_static.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_confusion_matrix():
    """Generate hierarchical confusion matrices with per-image averaging"""
    print("\nGenerating confusion matrices (per-image averaged)...")
    
    model, idx_to_label, label_to_idx = load_model()
    if model is None:
        return
    
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get well data
    well_data, _, _ = get_well_data_for_testing(n_crops=9)
    
    # Get predictions with averaging
    all_preds, all_labels, _ = predict_with_averaging(well_data, model, transform, n_crops=9)
    
    # Gene family mapping
    GENE_FAMILIES = {
        'mrcA': 'mr*', 'mrcB': 'mr*', 'mrdA': 'mr*', 'mreB': 'mr*',
        'murA': 'mur*', 'murC': 'mur*',
        'ftsI': 'fts*', 'ftsZ': 'fts*',
        'lpxA': 'lpx*', 'lpxC': 'lpx*',
        'lptA': 'lpt*', 'lptC': 'lpt*', 'msbA': 'lpt*',
        'gyrA': 'gyr*', 'gyrB': 'gyr*',
        'parC': 'par*', 'parE': 'par*',
        'rpoA': 'rpo*', 'rpoB': 'rpo*',
        'rpsA': 'rp*', 'rpsL': 'rp*', 'rplA': 'rp*', 'rplC': 'rp*',
        'folA': 'fol*', 'folP': 'fol*',
        'dnaE': 'dna*', 'dnaB': 'dna*',
        'secY': 'sec*', 'secA': 'sec*',
        'minC': 'min*',
        'WT': 'WT'
    }
    
    # Helper functions for hierarchy
    def get_gene_base(label):
        """Get gene name without number (e.g., mrcA_1 -> mrcA)"""
        idx_label = idx_to_label.get(label, str(label))
        if '_' in idx_label and idx_label != 'WT':
            return idx_label.rsplit('_', 1)[0]
        return idx_label
    
    def get_gene_family(label):
        """Get gene family (e.g., mrcA -> mr*)"""
        gene = get_gene_base(label)
        return GENE_FAMILIES.get(gene, gene)
    
    def get_pathway(label):
        """Get pathway for a label"""
        idx_label = idx_to_label.get(label, str(label))
        gene_base = get_gene_base(label)
        if gene_base in pathway_mapping:
            return pathway_mapping[gene_base].get('pathway', 'Unknown')
        return 'Control' if gene_base == 'WT' else 'Unknown'
    
    # Store accuracies for summary
    accuracies = {}
    
    # ============================================
    # LEVEL 1: All 85 individual classes
    # ============================================
    print("\n[1/4] Confusion matrix: All 85 classes...")
    
    acc_all = (all_preds == all_labels).mean() * 100
    accuracies['All 85 classes'] = acc_all
    
    unique_labels_list = sorted(set(all_labels) | set(all_preds))
    label_counts = Counter(all_labels)
    top_labels = [l for l, _ in label_counts.most_common(40)]
    
    cm = confusion_matrix(all_labels, all_preds, labels=top_labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm_normalized, cmap='Blues')
    
    ax.set_xticks(range(len(top_labels)))
    ax.set_yticks(range(len(top_labels)))
    ax.set_xticklabels([idx_to_label[l] for l in top_labels], rotation=90, fontsize=7)
    ax.set_yticklabels([idx_to_label[l] for l in top_labels], fontsize=7)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix: All Gene Classes (85)\nPer-Well Averaged Accuracy: {acc_all:.2f}%')
    
    plt.colorbar(im, ax=ax, label='Normalized Frequency')
    
    output_path = os.path.join(RESULTS_DIR, 'confusion_matrix_all_85.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path} (Acc: {acc_all:.2f}%)")
    
    # ============================================
    # LEVEL 2: By gene (ignoring replicate number)
    # ============================================
    print("\n[2/4] Confusion matrix: By gene (ignoring replicate)...")
    
    labels_by_gene = [get_gene_base(l) for l in all_labels]
    preds_by_gene = [get_gene_base(p) for p in all_preds]
    
    unique_genes = sorted(set(labels_by_gene) | set(preds_by_gene))
    
    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}
    labels_idx = [gene_to_idx[g] for g in labels_by_gene]
    preds_idx = [gene_to_idx[g] for g in preds_by_gene]
    
    cm = confusion_matrix(labels_idx, preds_idx, labels=range(len(unique_genes)))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    correct_gene = sum(1 for l, p in zip(labels_by_gene, preds_by_gene) if l == p)
    acc_gene = correct_gene / len(labels_by_gene) * 100
    accuracies['By gene (28)'] = acc_gene
    
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_normalized, cmap='Blues')
    
    ax.set_xticks(range(len(unique_genes)))
    ax.set_yticks(range(len(unique_genes)))
    ax.set_xticklabels(unique_genes, rotation=90, fontsize=9)
    ax.set_yticklabels(unique_genes, fontsize=9)
    
    ax.set_xlabel('Predicted Gene')
    ax.set_ylabel('True Gene')
    ax.set_title(f'Confusion Matrix: By Gene Name ({len(unique_genes)} genes)\nPer-Well Averaged Accuracy: {acc_gene:.2f}%')
    
    plt.colorbar(im, ax=ax)
    
    output_path = os.path.join(RESULTS_DIR, 'confusion_matrix_by_gene.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path} (Acc: {acc_gene:.2f}%)")
    
    # ============================================
    # LEVEL 3: By gene family
    # ============================================
    print("\n[3/4] Confusion matrix: By gene family...")
    
    labels_by_family = [get_gene_family(l) for l in all_labels]
    preds_by_family = [get_gene_family(p) for p in all_preds]
    
    unique_families = sorted(set(labels_by_family) | set(preds_by_family))
    
    family_to_idx = {f: i for i, f in enumerate(unique_families)}
    labels_idx = [family_to_idx[f] for f in labels_by_family]
    preds_idx = [family_to_idx[f] for f in preds_by_family]
    
    cm = confusion_matrix(labels_idx, preds_idx, labels=range(len(unique_families)))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    correct_family = sum(1 for l, p in zip(labels_by_family, preds_by_family) if l == p)
    acc_family = correct_family / len(labels_by_family) * 100
    accuracies['By family (14)'] = acc_family
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_normalized, cmap='Blues')
    
    ax.set_xticks(range(len(unique_families)))
    ax.set_yticks(range(len(unique_families)))
    ax.set_xticklabels(unique_families, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(unique_families, fontsize=10)
    
    for i in range(len(unique_families)):
        for j in range(len(unique_families)):
            val = cm_normalized[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color='white' if val > 0.5 else 'black', fontsize=10)
    
    ax.set_xlabel('Predicted Family')
    ax.set_ylabel('True Family')
    ax.set_title(f'Confusion Matrix: By Gene Family ({len(unique_families)} families)\nPer-Well Averaged Accuracy: {acc_family:.2f}%')
    
    plt.colorbar(im, ax=ax)
    
    output_path = os.path.join(RESULTS_DIR, 'confusion_matrix_by_family.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path} (Acc: {acc_family:.2f}%)")
    
    # ============================================
    # LEVEL 4: By pathway (7 classes)
    # ============================================
    print("\n[4/4] Confusion matrix: By pathway (7 categories)...")
    
    pathway_order = ['Control', 'Cell wall synthesis', 'LPS synthesis', 
                    'DNA metabolism', 'Transcription/translation', 
                    'Metabolism/export', 'Cell division']
    
    labels_by_pathway = [get_pathway(l) for l in all_labels]
    preds_by_pathway = [get_pathway(p) for p in all_preds]
    
    pathway_classes = pathway_order
    pathway_to_idx = {p: i for i, p in enumerate(pathway_classes)}
    
    labels_idx = [pathway_to_idx.get(g, 0) for g in labels_by_pathway]
    preds_idx = [pathway_to_idx.get(g, 0) for g in preds_by_pathway]
    
    cm = confusion_matrix(labels_idx, preds_idx, labels=range(len(pathway_classes)))
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    cm_normalized = np.nan_to_num(cm_normalized)
    
    correct_pathway = sum(1 for l, p in zip(labels_by_pathway, preds_by_pathway) if l == p)
    acc_pathway = correct_pathway / len(labels_by_pathway) * 100
    accuracies['By pathway (7)'] = acc_pathway
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, cmap='Blues')
    
    ax.set_xticks(range(len(pathway_classes)))
    ax.set_yticks(range(len(pathway_classes)))
    ax.set_xticklabels(pathway_classes, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(pathway_classes, fontsize=9)
    
    for i in range(len(pathway_classes)):
        for j in range(len(pathway_classes)):
            val = cm_normalized[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   color='white' if val > 0.5 else 'black', fontsize=11)
    
    ax.set_xlabel('Predicted Pathway')
    ax.set_ylabel('True Pathway')
    ax.set_title(f'Confusion Matrix: By Pathway ({len(pathway_classes)} categories)\nPer-Well Averaged Accuracy: {acc_pathway:.2f}%')
    
    plt.colorbar(im, ax=ax)
    
    output_path = os.path.join(RESULTS_DIR, 'confusion_matrix_by_pathway.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path} (Acc: {acc_pathway:.2f}%)")
    
    # ============================================
    # HIERARCHICAL ACCURACY SUMMARY
    # ============================================
    print("\n" + "="*60)
    print("HIERARCHICAL ACCURACY SUMMARY (Per-Well Averaged)")
    print("="*60)
    print(f"{'Level':<30} {'Classes':<10} {'Accuracy':<10}")
    print("-"*60)
    for level, acc in accuracies.items():
        classes = level.split('(')[1].split(')')[0] if '(' in level else '-'
        print(f"{level:<30} {classes:<10} {acc:.2f}%")
    print("-"*60)
    print(f"\nAccuracy Improvement:")
    print(f"  85 → 28 genes:      +{accuracies['By gene (28)'] - accuracies['All 85 classes']:.2f}%")
    print(f"  28 → 14 families:   +{accuracies['By family (14)'] - accuracies['By gene (28)']:.2f}%")
    print(f"  14 → 7 pathways:   +{accuracies['By pathway (7)'] - accuracies['By family (14)']:.2f}%")
    print(f"\nTotal improvement: {accuracies['By pathway (7)'] - accuracies['All 85 classes']:.2f}%")


def main():
    print("=" * 60)
    print("Generating Visualization Plots (Per-Image Averaging)")
    print("=" * 60)
    print("\nUsing 9 crops per well, averaged for predictions")
    
    # 1. Training curves
    print("\n[1/2] Generating training curves...")
    plot_training_curves()
    
    # 2. Confusion matrix
    print("\n[2/2] Generating confusion matrices...")
    generate_confusion_matrix()
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)
    print(f"\nOutput folder: {RESULTS_DIR}")
    print("\nOutput files:")
    print("  - training_curves.png")
    print("  - confusion_matrix_all_85.png")
    print("  - confusion_matrix_by_gene.png")
    print("  - confusion_matrix_by_family.png")
    print("  - confusion_matrix_by_pathway.png")
    print("\n(t-SNE: run 'python -c \"from generate_plots import generate_tsne; generate_tsne()\"' separately)")


if __name__ == '__main__':
    main()
