#!/usr/bin/env python3
"""
Train MLP classifier on DINOv3 embeddings
"""

import os
import json
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMB_TRAIN_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_train_c512.npz")
EMB_VAL_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_val_c512.npz")
EMB_TEST_PATH = os.path.join(BASE_DIR, "dinov3_embeddings_test_c512.npz")

OUTPUT_DIR = os.path.join(BASE_DIR, "results", "dinov3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Normalize embeddings
print("Normalizing embeddings...")
X_train_norm = normalize(X_train)
X_val_norm = normalize(X_val)
X_test_norm = normalize(X_test)

# Try different MLP architectures (reduced for speed)
architectures = [
    (512, 256),      # Simple, fast
    (1024, 512),      # More capacity
]

best_val_acc = 0
best_clf = None
best_arch = None

print("\n" + "="*60)
print("Training MLP classifiers with different architectures...")
print("="*60)

from tqdm import tqdm

for arch in tqdm(architectures, desc="MLP Architectures"):
    print(f"\n--- Architecture: {arch} ---")
    print(f"Training MLP (max_iter=200, batch_size=512, lr=0.003)...")
    
    clf = MLPClassifier(
        hidden_layer_sizes=arch,
        max_iter=200,              # Reduced from 500
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,       # Reduced from 20
        random_state=42,
        learning_rate='adaptive',
        learning_rate_init=0.003,  # Increased for faster convergence
        batch_size=512,            # Increased from 256
        alpha=0.0001,
        verbose=False
    )
    
    clf.fit(X_train_norm, y_train)
    
    train_pred = clf.predict(X_train_norm)
    val_pred = clf.predict(X_val_norm)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Val Accuracy: {val_acc*100:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_clf = clf
        best_arch = arch

print("\n" + "="*60)
print(f"Best architecture: {best_arch}")
print(f"Best val accuracy: {best_val_acc*100:.2f}%")
print("="*60)

# Evaluate best model on test set
print("\nEvaluating on test set...")
y_pred = best_clf.predict(X_test_norm)
test_acc = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {test_acc*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=idx_to_label.values()))

# Save the model
model_path = os.path.join(OUTPUT_DIR, 'dinov3_mlp_classifier.joblib')
joblib.dump(best_clf, model_path)
print(f"\nModel saved to {model_path}")

# Save results
results = {
    'train_accuracy': float(accuracy_score(y_train, best_clf.predict(X_train_norm))),
    'val_accuracy': float(best_val_acc),
    'test_accuracy': float(test_acc),
    'best_architecture': best_arch,
}

import json as json_module
results_path = os.path.join(OUTPUT_DIR, 'mlp_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_path}")
