#!/usr/bin/env python3
"""
Train MLP classifier on DINOv3 embeddings using PyTorch with GPU
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMB_TRAIN_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_train_c512.npz")
EMB_VAL_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_val_c512.npz")
EMB_TEST_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_test_c512.npz")

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "dinov3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)


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


def train_model(hidden_dims, epochs=50, lr=0.001):
    model = MLPClassifier(1024, hidden_dims, num_classes, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    best_state = None
    
    for epoch in tqdm(range(epochs), desc=f"MLP {hidden_dims}"):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_preds = val_outputs.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_state)
    return model, best_val_acc


# Train different architectures
architectures = [
    [512, 256],
    [1024, 512],
]

best_model = None
best_val_acc = 0
best_arch = None

print("\n" + "="*60)
print("Training MLP classifiers (GPU accelerated)...")
print("="*60)

for arch in architectures:
    print(f"\n--- Architecture: {arch} ---")
    model, val_acc = train_model(arch, epochs=50, lr=0.001)
    
    # Get train accuracy
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_t)
        train_preds = train_outputs.argmax(dim=1).cpu().numpy()
        train_acc = accuracy_score(y_train, train_preds)
    
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Val Accuracy: {val_acc*100:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        best_arch = arch

print("\n" + "="*60)
print(f"Best architecture: {best_arch}")
print(f"Best val accuracy: {best_val_acc*100:.2f}%")
print("="*60)

# Evaluate on test set
print("\nEvaluating on test set...")
best_model.eval()
with torch.no_grad():
    test_outputs = best_model(X_test_t)
    test_preds = test_outputs.argmax(dim=1).cpu().numpy()

test_acc = accuracy_score(y_test, test_preds)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, test_preds, target_names=idx_to_label.values()))

# Save model
model_path = os.path.join(OUTPUT_DIR, 'dinov3_mlp_gpu.pth')
torch.save(best_model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")

# Save results
results = {
    'train_accuracy': float(accuracy_score(y_train, best_model(X_train_t).argmax(dim=1).cpu().numpy())),
    'val_accuracy': float(best_val_acc),
    'test_accuracy': float(test_acc),
    'best_architecture': best_arch,
}

results_path = os.path.join(OUTPUT_DIR, 'mlp_gpu_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_path}")
