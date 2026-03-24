#!/usr/bin/env python3
"""
Hierarchical confusion matrices at different levels:
1. Gene variant level (dnaB_1, dnaB_2, dnaB_3)
2. Gene family level (dnaB, dnaE, etc.)
3. Pathway level (Cell wall, DNA metabolism, etc.)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMB_TRAIN_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_train_c512.npz")
EMB_VAL_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_val_c512.npz")
EMB_TEST_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_test_c512.npz")

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "dinov3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============== GENE HIERARCHY ==============
GENE_VARIANTS = {
    'dnaB': ['dnaB_1', 'dnaB_2', 'dnaB_3'],
    'dnaE': ['dnaE_1', 'dnaE_2', 'dnaE_3'],
    'folA': ['folA_1', 'folA_2', 'folA_3'],
    'folP': ['folP_1', 'folP_2', 'folP_3'],
    'ftsI': ['ftsI_1', 'ftsI_2', 'ftsI_3'],
    'ftsZ': ['ftsZ_1', 'ftsZ_2', 'ftsZ_3'],
    'gyrA': ['gyrA_1', 'gyrA_2', 'gyrA_3'],
    'gyrB': ['gyrB_1', 'gyrB_2', 'gyrB_3'],
    'lptA': ['lptA_1', 'lptA_2', 'lptA_3'],
    'lptC': ['lptC_1', 'lptC_2', 'lptC_3'],
    'lpxA': ['lpxA_1', 'lpxA_2', 'lpxA_3'],
    'lpxC': ['lpxC_1', 'lpxC_2', 'lpxC_3'],
    'mrcA': ['mrcA_1', 'mrcA_2', 'mrcA_3'],
    'mrcB': ['mrcB_1', 'mrcB_2', 'mrcB_3'],
    'mrdA': ['mrdA_1', 'mrdA_2', 'mrdA_3'],
    'msbA': ['msbA_1', 'msbA_2', 'msbA_3'],
    'murA': ['murA_1', 'murA_2', 'murA_3'],
    'murC': ['murC_1', 'murC_2', 'murC_3'],
    'parC': ['parC_1', 'parC_2', 'parC_3'],
    'parE': ['parE_1', 'parE_2', 'parE_3'],
    'rplA': ['rplA_1', 'rplA_2', 'rplA_3'],
    'rplC': ['rplC_1', 'rplC_2', 'rplC_3'],
    'rpoA': ['rpoA_1', 'rpoA_2', 'rpoA_3'],
    'rpoB': ['rpoB_1', 'rpoB_2', 'rpoB_3'],
    'rpsA': ['rpsA_1', 'rpsA_2', 'rpsA_3'],
    'rpsL': ['rpsL_1', 'rpsL_2', 'rpsL_3'],
    'secA': ['secA_1', 'secA_2', 'secA_3'],
    'secY': ['secY_1', 'secY_2', 'secY_3'],
}

PATHWAY_GROUPS = {
    'Cell wall synthesis': ['mrcA', 'mrcB', 'mrdA', 'ftsI', 'murA', 'murC'],
    'LPS synthesis': ['lpxA', 'lpxC', 'lptA', 'lptC', 'msbA'],
    'DNA metabolism': ['gyrA', 'gyrB', 'parC', 'parE', 'dnaE', 'dnaB'],
    'Transcription': ['rpoA', 'rpoB'],
    'Translation': ['rplA', 'rplC', 'rpsA', 'rpsL'],
    'Metabolism': ['folA', 'folP'],
    'Protein export': ['secA', 'secY'],
    'Cell division': ['ftsZ'],
    'Control': ['WT']
}

def get_gene_family(label):
    """Get gene family from label (e.g., 'dnaB_2' -> 'dnaB')"""
    if label == 'WT':
        return 'WT'
    if '_' in label:
        return label.rsplit('_', 1)[0]
    return label

def get_pathway(label):
    """Get pathway from gene family"""
    family = get_gene_family(label)
    for pathway, genes in PATHWAY_GROUPS.items():
        if family in genes:
            return pathway
    return 'Other'

# Load class mappings
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

print(f"Number of classes: {num_classes}")

# Load embeddings
print("Loading embeddings...")
train_data = np.load(EMB_TRAIN_PATH)
val_data = np.load(EMB_VAL_PATH)
test_data = np.load(EMB_TEST_PATH)

X_train = train_data['embeddings']
y_train = train_data['labels']
X_val = val_data['embeddings']
y_val = val_data['labels']
X_test = test_data['embeddings']
y_test = test_data['labels']

# Normalize
X_train_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
X_val_norm = X_val / (np.linalg.norm(X_val, axis=1, keepdims=True) + 1e-8)
X_test_norm = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train_norm).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val_norm).to(device)
y_val_t = torch.LongTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test_norm).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.5):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def train_model(hidden_dims, epochs=200, lr=0.001, dropout=0.5):
    model = MLPClassifier(1024, hidden_dims, num_classes, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    best_state = None
    patience = 25
    no_improve = 0
    
    for epoch in tqdm(range(epochs), desc=f"Training"):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_preds = val_outputs.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    model.load_state_dict(best_state)
    return model, best_val_acc


# Train with more regularization
print("\n" + "="*60)
print("Training MLP with better regularization...")
print("="*60)

best_model, best_val_acc = train_model([1024, 512, 256], epochs=200, lr=0.001, dropout=0.5)

# Get predictions
best_model.eval()
with torch.no_grad():
    train_preds = best_model(X_train_t).argmax(dim=1).cpu().numpy()
    test_preds = best_model(X_test_t).argmax(dim=1).cpu().numpy()

train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

print(f"\nTrain Accuracy: {train_acc*100:.2f}%")
print(f"Val Accuracy: {best_val_acc*100:.2f}%")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Convert predictions to labels
test_labels = [idx_to_label[idx] for idx in test_preds]
test_actual_labels = [idx_to_label[idx] for idx in y_test]

# ============== HIERARCHICAL CONFUSION MATRICES ==============

def create_grouped_labels(labels, groups):
    """Map labels to groups"""
    group_order = list(groups.keys())
    group_idx = {g: i for i, g in enumerate(group_order)}
    mapped = []
    for label in labels:
        found = False
        for group_name, gene_list in groups.items():
            if label in gene_list:
                mapped.append(group_name)
                found = True
                break
        if not found:
            mapped.append('Other')
    return mapped, group_order


def plot_confusion_matrix(y_true, y_pred, labels, title, filename, figsize=(12, 10)):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={'size': 8})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved {filename}")
    
    # Calculate accuracy
    cm_diag = np.diag(cm)
    cm_sum = cm.sum(axis=1)
    acc = cm_diag.sum() / cm_sum.sum() * 100
    return acc


# 1. GENE VARIANT LEVEL (dnaB_1, dnaB_2, dnaB_3)
print("\n" + "="*60)
print("1. GENE VARIANT LEVEL (85 classes)")
print("="*60)

gene_order = sorted(all_labels)
test_acc_var = plot_confusion_matrix(
    test_actual_labels, test_labels, gene_order,
    f'Gene Variant Level (85 classes)\nTest Accuracy: {test_acc*100:.2f}%',
    'confusion_variant.png', figsize=(24, 20)
)
print(f"Variant-level accuracy: {test_acc_var:.2f}%")


# 2. GENE FAMILY LEVEL (dnaB, dnaE, etc.)
print("\n" + "="*60)
print("2. GENE FAMILY LEVEL (~28 families)")
print("="*60)

# Create family mapping
family_mapping = {}
for family, variants in GENE_VARIANTS.items():
    for v in variants:
        family_mapping[v] = family
family_mapping['WT'] = 'WT'

# Get unique families
all_families = sorted(set(family_mapping.values()))
test_families = [family_mapping.get(l, 'Other') for l in test_labels]
test_actual_families = [family_mapping.get(l, 'Other') for l in test_actual_labels]

test_acc_family = plot_confusion_matrix(
    test_actual_families, test_families, all_families,
    f'Gene Family Level ({len(all_families)} families)\nTest Accuracy: {test_acc*100:.2f}% → Family: ???',
    'confusion_family.png', figsize=(16, 14)
)

# Calculate family accuracy manually
cm_family = confusion_matrix(test_actual_families, test_families, labels=all_families)
family_acc = np.diag(cm_family).sum() / cm_family.sum() * 100
print(f"Family-level accuracy: {family_acc:.2f}%")


# 3. PATHWAY LEVEL
print("\n" + "="*60)
print("3. PATHWAY LEVEL (9 pathways)")
print("="*60)

pathway_order = list(PATHWAY_GROUPS.keys())
test_pathways = [get_pathway(l) for l in test_labels]
test_actual_pathways = [get_pathway(l) for l in test_actual_labels]

test_acc_pathway = plot_confusion_matrix(
    test_actual_pathways, test_pathways, pathway_order,
    f'Pathway Level ({len(pathway_order)} pathways)\nTest Accuracy: {test_acc*100:.2f}%',
    'confusion_pathway.png', figsize=(12, 10)
)

# Calculate pathway accuracy manually
cm_pathway = confusion_matrix(test_actual_pathways, test_pathways, labels=pathway_order)
pathway_acc = np.diag(cm_pathway).sum() / cm_pathway.sum() * 100
print(f"Pathway-level accuracy: {pathway_acc:.2f}%")


# ============== SUMMARY ==============
print("\n" + "="*60)
print("HIERARCHICAL ACCURACY SUMMARY")
print("="*60)
print(f"Variant Level (85 classes): {test_acc*100:.2f}%")
print(f"Family Level (~28 families): {family_acc:.2f}%")
print(f"Pathway Level (9 pathways): {pathway_acc:.2f}%")

# Save results
results = {
    'variant_accuracy': float(test_acc),
    'family_accuracy': float(family_acc),
    'pathway_accuracy': float(pathway_acc)
}
with open(os.path.join(OUTPUT_DIR, 'hierarchical_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nAll files saved to {OUTPUT_DIR}")
