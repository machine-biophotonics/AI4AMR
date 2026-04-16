#!/usr/bin/env python3
"""
Confusion Matrix: 85 classes → Mechanism of Action
Shows how well the model classifies mutants by their antibiotic-like MoA.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import re

BASE_DIR = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/logistic_regression_results'
OUTPUT_DIR = BASE_DIR

# CRISPRi mutants → Antibiotic-like Mechanism of Action mapping
MUTANT_TO_MOA = {
    'gyrA': 'DNA Gyrase (Fluoroquinolone-like)',
    'gyrB': 'DNA Gyrase (Fluoroquinolone-like)',
    'parC': 'DNA Topoisomerase IV (Fluoroquinolone-like)',
    'parE': 'DNA Topoisomerase IV (Fluoroquinolone-like)',
    'dnaB': 'DNA Replication (Quinolone-like)',
    'dnaE': 'DNA Replication (Quinolone-like)',
    
    'ftsI': 'Cell Wall - PBP3 (Cephalosporin-like)',
    'ftsZ': 'Cell Wall - FtsZ (Cell division)',
    'murA': 'Cell Wall - MurA (Fosfomycin-like)',
    'murC': 'Cell Wall - MurC (Bacitracin-like)',
    'mrcA': 'Cell Wall - PBP1a (Carbenicillin-like)',
    'mrcB': 'Cell Wall - PBP1b (Carbenicillin-like)',
    'mrdA': 'Cell Wall - PBP2 (Mecillinam-like)',
    
    'lpxA': 'LPS biosynthesis (Polymyxin-like)',
    'lpxC': 'LPS biosynthesis (Polymyxin-like)',
    'lptA': 'LPS transport (Polymyxin-like)',
    'lptC': 'LPS transport (Polymyxin-like)',
    'msbA': 'LPS transport (Polymyxin-like)',
    
    'rpoA': 'Transcription - RNAP (Rifampicin-like)',
    'rpoB': 'Transcription - RNAP (Rifampicin-like)',
    'rplA': 'Ribosome 50S (Macrolide-like)',
    'rplC': 'Ribosome 50S (Macrolide-like)',
    'rpsA': 'Ribosome 30S (Tetracycline-like)',
    'rpsL': 'Ribosome 30S (Streptomycin-like)',
    
    'folA': 'Folate pathway (Trimethoprim-like)',
    'folP': 'Folate pathway (Sulfonamide-like)',
    
    'secA': 'Secretion system',
    'secY': 'Secretion system',
    
    'WT': 'Wild-type Control',
}

def get_base_gene(label):
    """Extract base gene name from label like 'dnaB_1' -> 'dnaB'"""
    if label == 'WT':
        return 'WT'
    if '_' in label:
        return label.rsplit('_', 1)[0]
    return label

def get_moa(label):
    """Get antibiotic-like MoA for a label"""
    base = get_base_gene(label)
    return MUTANT_TO_MOA.get(base, f'Unknown ({base})')

print("Loading data...")

with open(os.path.join(BASE_DIR, 'idx_to_label.json'), 'r') as f:
    idx_to_label_raw = json.load(f)
    idx_to_label = {int(k): v for k, v in idx_to_label_raw.items()}

test_preds = np.load(os.path.join(BASE_DIR, 'test_preds.npy'))
test_labels = np.load(os.path.join(BASE_DIR, 'test_labels.npy'))

print(f"Total test samples: {len(test_preds)}")
print(f"Number of classes: {len(idx_to_label)}")

# Convert predictions and labels to MoA
true_moa = [get_moa(idx_to_label.get(l, 'WT')) for l in test_labels]
pred_moa = [get_moa(idx_to_label.get(p, 'WT')) for p in test_preds]

# Get unique MoA categories
unique_moa = sorted(set(true_moa + pred_moa))
print(f"\nUnique MoA categories: {len(unique_moa)}")
for moa in unique_moa:
    count = true_moa.count(moa)
    print(f"  {moa}: {count}")

# Create confusion matrix
cm = confusion_matrix(true_moa, pred_moa, labels=unique_moa)

# Plot
def plot_confusion_matrix(cm, labels, title, filename, figsize=(14, 12)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True Mechanism of Action',
           xlabel='Predicted Mechanism of Action')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

accuracy = accuracy_score(true_moa, pred_moa) * 100
print(f"\nMoA Classification Accuracy: {accuracy:.2f}%")

plot_confusion_matrix(
    cm, unique_moa, 
    f'85 Classes → Mechanism of Action\nAccuracy: {accuracy:.2f}%',
    'confusion_matrix_moa.png',
    figsize=(16, 14)
)

# Also create normalized version
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm)

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax, shrink=0.8)
ax.set(xticks=np.arange(cm_norm.shape[1]),
       yticks=np.arange(cm_norm.shape[0]),
       xticklabels=unique_moa, yticklabels=unique_moa,
       title=f'85 Classes → Mechanism of Action (Normalized)\nAccuracy: {accuracy:.2f}%',
       ylabel='True Mechanism of Action',
       xlabel='Predicted Mechanism of Action')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
thresh = cm_norm.max() / 2.
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        ax.text(j, i, f'{cm_norm[i, j]:.2f}',
               ha="center", va="center",
               color="white" if cm_norm[i, j] > thresh else "black",
               fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_moa_normalized.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix_moa_normalized.png")

# Save mapping table
mapping_table = {}
for moa in unique_moa:
    genes = [g for g, m in MUTANT_TO_MOA.items() if m == moa]
    mapping_table[moa] = genes

with open(os.path.join(OUTPUT_DIR, 'moa_mapping.json'), 'w') as f:
    json.dump(mapping_table, f, indent=2)
print("Saved: moa_mapping.json")

print("\nDone!")
