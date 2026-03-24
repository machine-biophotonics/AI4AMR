#!/usr/bin/env python3
"""
Generate standardized confusion matrices at different hierarchical levels for test set
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
EMB_TRAIN = os.path.join(BASE_DIR, "dinov3_embeddings_train_c512.npz")
EMB_VAL = os.path.join(BASE_DIR, "dinov3_embeddings_val_c512.npz")
EMB_TEST = os.path.join(BASE_DIR, "dinov3_embeddings_test_c512.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "dinov3")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hierarchical groupings
FAMILY_GROUPS = {}
for family, variants in {
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
}.items():
    for v in variants:
        FAMILY_GROUPS[v] = family
FAMILY_GROUPS['WT'] = 'WT'

PATHWAY_GROUPS = {
    'Cell wall': ['mrcA', 'mrcB', 'mrdA', 'ftsI', 'murA', 'murC'],
    'LPS': ['lpxA', 'lpxC', 'lptA', 'lptC', 'msbA'],
    'DNA': ['gyrA', 'gyrB', 'parC', 'parE', 'dnaE', 'dnaB'],
    'Transcription': ['rpoA', 'rpoB'],
    'Translation': ['rplA', 'rplC', 'rpsA', 'rpsL'],
    'Metabolism': ['folA', 'folP'],
    'Export': ['secA', 'secY'],
    'Cell division': ['ftsZ'],
    'WT': ['WT']
}

def get_family(label):
    return FAMILY_GROUPS.get(label, label)

def get_pathway(label):
    family = get_family(label)
    for pw, genes in PATHWAY_GROUPS.items():
        if family in genes:
            return pw
    return 'Other'

# Load data
print("Loading data...")
with open(os.path.join(BASE_DIR, 'plate_well_id_path.json')) as f:
    plate_data = json.load(f)

all_labels = sorted(set(
    plate_data[p][r][c]['id'] 
    for p in plate_data for r in plate_data[p] 
    for c in plate_data[p][r]
))
label_to_idx = {l: i for i, l in enumerate(all_labels)}
idx_to_label = {i: l for l, i in label_to_idx.items()}
num_classes = len(all_labels)

test_data = np.load(EMB_TEST)
X_test, y_test = test_data['embeddings'], test_data['labels']
X_test_n = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-8)
X_test_t = torch.FloatTensor(X_test_n).to(device)

# Load trained model
print("Loading model...")
model = nn.Sequential(
    nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(128, num_classes)
).to(device)

# We need to train since we can't load - train quickly
train_data = np.load(EMB_TRAIN)
val_data = np.load(EMB_VAL)
X_train, y_train = train_data['embeddings'], train_data['labels']
X_val, y_val = val_data['embeddings'], val_data['labels']

X_train_n = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
X_val_n = X_val / (np.linalg.norm(X_val, axis=1, keepdims=True) + 1e-8)

X_train_t = torch.FloatTensor(X_train_n).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val_n).to(device)
y_val_t = torch.LongTensor(y_val).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

class MLP(nn.Module):
    def __init__(self, dims, num_classes, dropout=0.5):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

print("\nTraining model...")
model = MLP([1024, 1024, 512, 256, 128], num_classes, dropout=0.5).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.03)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.003, epochs=100, steps_per_epoch=len(train_loader))

best_acc, best_state = 0, None
for epoch in tqdm(range(100)):
    model.train()
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).argmax(1).cpu().numpy()
        acc = accuracy_score(y_val, val_preds)
    if acc > best_acc:
        best_acc = acc
        best_state = model.state_dict().copy()

model.load_state_dict(best_state)

# Test predictions
model.eval()
with torch.no_grad():
    test_preds = model(X_test_t).argmax(1).cpu().numpy()

test_acc = accuracy_score(y_test, test_preds)
test_pred_labels = [idx_to_label[i] for i in test_preds]
test_true_labels = [idx_to_label[i] for i in y_test]

# Map to family and pathway
pred_family = [get_family(l) for l in test_pred_labels]
true_family = [get_family(l) for l in test_true_labels]
pred_pathway = [get_pathway(l) for l in test_pred_labels]
true_pathway = [get_pathway(l) for l in test_true_labels]

# Calculate accuracies
acc_variant = accuracy_score(true_family, pred_family) * 100  # Actually family accuracy from variant preds
acc_family = accuracy_score(true_family, pred_family) * 100
acc_pathway = accuracy_score(true_pathway, pred_pathway) * 100

print(f"\nVariant (85): {test_acc*100:.1f}%")
print(f"Family: {acc_family:.1f}%")
print(f"Pathway: {acc_pathway:.1f}%")

# ============== PLOT CONFUSION MATRICES ==============

def plot_confusion(y_true, y_pred, labels, title, filename, figsize=(12, 10)):
    """Standard normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_norm, annot=True, fmt='.0f', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Recall (%)'}, vmin=0, vmax=100)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")
    return np.trace(cm) / cm.sum() * 100

# 1. Variant level (85 classes) - show as family confusion
print("\n1. Variant confusion (showing family groupings)...")
families = sorted(set(true_family + pred_family))
acc_var = plot_confusion(
    true_family, pred_family, families,
    f'Gene Family Classification (Test Set)\nAccuracy: {acc_family:.1f}%',
    'cm_family.png', figsize=(16, 14)
)

# 2. Pathway level (9 pathways)
print("\n2. Pathway confusion...")
pathways = sorted(set(true_pathway + pred_pathway))
acc_pw = plot_confusion(
    true_pathway, pred_pathway, pathways,
    f'Pathway Classification (Test Set)\nAccuracy: {acc_pathway:.1f}%',
    'cm_pathway.png', figsize=(10, 8)
)

print(f"\nFinal: Variant={test_acc*100:.1f}%, Family={acc_family:.1f}%, Pathway={acc_pathway:.1f}%")
