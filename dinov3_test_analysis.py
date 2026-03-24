#!/usr/bin/env python3
"""
Test set analysis: Confusion matrix, accuracy, t-SNE (test only)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMB_TRAIN_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_train_c512.npz")
EMB_VAL_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_val_c512.npz")
EMB_TEST_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_test_c512.npz")

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "dinov3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Pathway colors
PATHWAY_COLORS = {
    'mrcA': '#E57373', 'mrcB': '#EF5350', 'mrdA': '#F06292', 'ftsI': '#EC407A',
    'mreB': '#FF8A65', 'murA': '#FFB74D', 'murC': '#FFA726',
    'lpxA': '#4DB6AC', 'lpxC': '#26A69A', 'lptA': '#4DD0E1', 'lptC': '#26C6DA', 'msbA': '#80DEEA',
    'gyrA': '#5C6BC0', 'gyrB': '#3F51B5', 'parC': '#7986CB', 'parE': '#9FA8DA',
    'dnaE': '#9575CD', 'dnaB': '#B39DDB',
    'rpoA': '#81C784', 'rpoB': '#66BB6A', 'rpsA': '#FFF176', 'rpsL': '#FFEE58',
    'rplA': '#FFD54F', 'rplC': '#FFCA28',
    'folA': '#AED581', 'folP': '#9CCC65', 'secY': '#80CBC4', 'secA': '#4DB6AC',
    'ftsZ': '#F06292', 'minC': '#F48FB1',
    'WT': '#424242'
}

def get_gene_family(label):
    if label == 'WT':
        return 'WT'
    if '_' in label:
        return label.rsplit('_', 1)[0]
    return label

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

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

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
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
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


def train_model(hidden_dims, epochs=150, lr=0.001):
    model = MLPClassifier(1024, hidden_dims, num_classes, dropout=0.4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    best_state = None
    patience = 20
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


# Train model
print("\n" + "="*60)
print("Training MLP classifier...")
print("="*60)

best_model, best_val_acc = train_model([1024, 512, 256], epochs=150, lr=0.001)

# Get predictions
best_model.eval()
with torch.no_grad():
    train_preds = best_model(X_train_t).argmax(dim=1).cpu().numpy()
    val_preds = best_model(X_val_t).argmax(dim=1).cpu().numpy()
    test_preds = best_model(X_test_t).argmax(dim=1).cpu().numpy()

train_acc = accuracy_score(y_train, train_preds)
val_acc = accuracy_score(y_val, val_preds)
test_acc = accuracy_score(y_test, test_preds)

print(f"\n{'='*60}")
print("ACCURACY RESULTS")
print(f"{'='*60}")
print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Val Accuracy:   {val_acc*100:.2f}%")
print(f"Test Accuracy:  {test_acc*100:.2f}%")

# ============== 1. ACCURACY BAR CHART ==============
print("\nGenerating accuracy chart...")
fig, ax = plt.subplots(figsize=(10, 6))
accuracies = [train_acc*100, val_acc*100, test_acc*100]
colors = ['#4CAF50', '#2196F3', '#FF9800']
bars = ax.bar(['Train', 'Val', 'Test'], accuracies, color=colors)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title(f'DINOv3 MLP Classification Accuracy\nTest Accuracy: {test_acc*100:.2f}%', fontsize=14)
ax.set_ylim(0, 100)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{acc:.1f}%', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_chart.png'), dpi=150)
plt.close()
print("Saved accuracy_chart.png")

# ============== 2. CONFUSION MATRIX ==============
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, test_preds)

fig, ax = plt.subplots(figsize=(24, 20))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
            xticklabels=idx_to_label.values(), 
            yticklabels=idx_to_label.values(),
            ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix (Test Set)\nTest Accuracy: {test_acc*100:.2f}%', fontsize=14)
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_test.png'), dpi=150)
plt.close()
print("Saved confusion_matrix_test.png")

# ============== 3. t-SNE (TEST SET ONLY) ==============
print("\nGenerating t-SNE for test set only...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_test_norm)

fig, ax = plt.subplots(figsize=(14, 12))

# Color by gene family
for family in sorted(set(get_gene_family(idx_to_label[label]) for label in y_test)):
    color = PATHWAY_COLORS.get(family, '#888888')
    family_labels = [idx_to_label[l] for l in np.unique(y_test) if get_gene_family(idx_to_label[l]) == family]
    
    for label in family_labels:
        label_idx = label_to_idx[label]
        mask = y_test == label_idx
        if mask.sum() > 0:
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                      c=color, label=label, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

ax.set_xlabel('t-SNE 1', fontsize=12)
ax.set_ylabel('t-SNE 2', fontsize=12)
ax.set_title(f't-SNE Visualization (Test Set Only, n={len(y_test)})\nColored by Gene Family', fontsize=14)

# Legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5), 
          fontsize=6, ncol=1, framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_test_only.png'), dpi=150)
plt.close()
print("Saved tsne_test_only.png")

# ============== 4. PER-CLASS ACCURACY ==============
print("\nGenerating per-class accuracy...")
per_class_acc = {}
for label_idx in np.unique(y_test):
    label = idx_to_label[label_idx]
    mask = y_test == label_idx
    if mask.sum() > 0:
        per_class_acc[label] = (test_preds[mask] == y_test[mask]).mean() * 100

# Sort by accuracy
sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)

fig, ax = plt.subplots(figsize=(20, 10))
labels_list = [x[0] for x in sorted_classes]
accs_list = [x[1] for x in sorted_classes]
colors = ['#4CAF50' if a >= 50 else '#FF9800' if a >= 25 else '#F44336' for a in accs_list]

bars = ax.barh(range(len(labels_list)), accs_list, color=colors)
ax.set_yticks(range(len(labels_list)))
ax.set_yticklabels(labels_list, fontsize=6)
ax.set_xlabel('Accuracy (%)', fontsize=12)
ax.set_title('Per-Class Accuracy (Test Set)', fontsize=14)
ax.axvline(x=50, color='green', linestyle='--', alpha=0.5, label='50%')
ax.axvline(x=25, color='orange', linestyle='--', alpha=0.5, label='25%')
ax.legend()
ax.set_xlim(0, 110)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_accuracy.png'), dpi=150)
plt.close()
print("Saved per_class_accuracy.png")

# ============== SUMMARY ==============
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Train Accuracy: {train_acc*100:.2f}%")
print(f"Val Accuracy:   {val_acc*100:.2f}%")
print(f"Test Accuracy:  {test_acc*100:.2f}%")
print(f"\nFiles saved to {OUTPUT_DIR}:")
print("  - accuracy_chart.png")
print("  - confusion_matrix_test.png")
print("  - tsne_test_only.png")
print("  - per_class_accuracy.png")

# Save results
results = {
    'train_accuracy': float(train_acc),
    'val_accuracy': float(val_acc),
    'test_accuracy': float(test_acc),
    'per_class_accuracy': per_class_acc
}
with open(os.path.join(OUTPUT_DIR, 'test_analysis_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
