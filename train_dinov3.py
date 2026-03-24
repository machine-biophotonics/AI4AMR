#!/usr/bin/env python3
"""
DINOv3 Feature Extraction + Classification Training Script

This script:
1. Loads DINOv3 ViT-7B pretrained model (SAT-493M checkpoint)
2. Extracts embeddings from all microscopy images (train/val/test)
3. Trains a linear classifier (Logistic Regression) on embeddings
4. Evaluates and generates visualizations

Workflow based on: https://docs.voxel51.com/tutorials/dinov3.html#Classification-Tasks-with-DINOv3
"""

import argparse
import os
import sys
import json
import glob
import re
import random
from typing import Optional, List, Dict, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# Configuration
# =============================================================================
parser = argparse.ArgumentParser(description='DINOv3 Feature Extraction + Classification')
parser.add_argument('--n_crops', type=int, default=25, help='Number of crops per image (9 for quick, 25 for 5x5 grid)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for embedding extraction')
parser.add_argument('--classifier', type=str, default='logistic', choices=['logistic', 'mlp'], help='Classifier type')
parser.add_argument('--resume_embeddings', action='store_true', help='Resume from saved embeddings')
parser.add_argument('--model_size', type=str, default='vitl', choices=['vit7b', 'vitl', 'vitb', 'vits'], help='DINOv3 model size')
parser.add_argument('--crop_size', type=int, default=512, help='Crop size in pixels (default: 512)')
parser.add_argument('--crop_grid', type=int, default=5, help='Grid size for crop positions (e.g., 5x5)')
parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
args = parser.parse_args()

# DINOv3 checkpoint path (satellite pretrained)
# Using ViT-L (1.2GB, 300M params) - fits in 24GB GPU
# SAT-493M trained on 512x512 images resized to 256x256
DINOV3_CHECKPOINT = os.path.join(BASE_DIR, "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "dinov3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Embedding cache paths (with crop_size in name to avoid conflicts)
EMB_TRAIN_PATH = os.path.join(BASE_DIR, f"dinov3_embeddings_train_c{args.crop_size}.npz")
EMB_VAL_PATH = os.path.join(BASE_DIR, f"dinov3_embeddings_val_c{args.crop_size}.npz")
EMB_TEST_PATH = os.path.join(BASE_DIR, f"dinov3_embeddings_test_c{args.crop_size}.npz")

# =============================================================================
# Load plate data
# =============================================================================
with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps: dict[str, dict[str, str]] = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

def extract_well_from_filename(filename: str) -> Optional[str]:
    match = re.search(r'Well(\w\d+)_', filename)
    if match:
        return match.group(1)
    return None

all_labels = sorted(set(label for pm in plate_maps.values() for label in pm.values()))
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(all_labels)
print(f"Number of classes: {num_classes}")

# Save classes
with open(os.path.join(BASE_DIR, 'classes_dinov3.txt'), 'w') as f:
    for i, label in enumerate(all_labels):
        f.write(f"{i},{label}\n")

def get_label_from_path(img_path: str) -> Optional[str]:
    dirname = os.path.dirname(img_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        return plate_maps[plate][well]
    return None

# =============================================================================
# Image paths
# =============================================================================
train_paths = []
for p in ['P1', 'P2', 'P3', 'P4']:
    train_paths.extend(glob.glob(os.path.join(BASE_DIR, p, '*.tif')))

val_paths = glob.glob(os.path.join(BASE_DIR, 'P5', '*.tif'))
test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

# =============================================================================
# Dataset for embedding extraction - loads images in workers for parallel processing
# =============================================================================
class ImageCropDataset(Dataset):
    """Dataset that loads and extracts crops in parallel workers."""
    
    def __init__(self, image_paths: List[str], crop_size: int = 512, grid_size: int = 5, n_crops: int = 25):
        self.image_paths = image_paths
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.n_crops = n_crops
        
        # Calculate crop positions
        if n_crops == 9:
            center_start = grid_size // 2 - 1
            center_end = grid_size // 2 + 2
            self.crop_indices = []
            for i in range(center_start, center_end):
                for j in range(center_start, center_end):
                    self.crop_indices.append(i * grid_size + j)
        else:
            self.crop_indices = list(range(grid_size * grid_size))
        
        self.crops_per_image = len(self.crop_indices)
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        
        # Load and extract crops in worker
        img = Image.open(img_path).convert('RGB')
        crops = self._extract_crops(img)
        
        # Stack crops into tensor [n_crops, 3, 256, 256]
        crops_tensor = torch.stack(crops)
        
        label = get_label_from_path(img_path) or "Unknown"
        label_idx = label_to_idx.get(label, 0)
        
        return crops_tensor, label_idx
    
    def _extract_crops(self, img: Image.Image) -> List[torch.Tensor]:
        """Extract all crops from an image."""
        from torchvision.transforms import v2
        
        w, h = img.size
        step_w = (w - self.crop_size) / (self.grid_size - 1) if self.grid_size > 1 else 0
        step_h = (h - self.crop_size) / (self.grid_size - 1) if self.grid_size > 1 else 0
        
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((256, 256), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        crops = []
        for pos_idx in self.crop_indices:
            i = pos_idx // self.grid_size
            j = pos_idx % self.grid_size
            left = int(j * step_w)
            top = int(i * step_h)
            crop = img.crop((left, top, left + self.crop_size, top + self.crop_size))
            crop_tensor = transform(crop)
            crops.append(crop_tensor)
        
        return crops

# =============================================================================
# DINOv3 Model Loading
# =============================================================================
def load_dinov3_model(model_size: str = 'vitl'):
    """Load DINOv3 model using torch.hub with local checkpoint and cloned repo."""
    
    checkpoint_path = DINOV3_CHECKPOINT
    
    # Model configurations
    model_configs = {
        'vit7b': {
            'name': 'dinov3_vit7b16',
            'embed_dim': 1536,
        },
        'vitl': {
            'name': 'dinov3_vitl16', 
            'embed_dim': 1024,
        },
        'vitb': {
            'name': 'dinov3_vitb16',
            'embed_dim': 768,
        },
        'vits': {
            'name': 'dinov3_vits16',
            'embed_dim': 384,
        }
    }
    
    config = model_configs[model_size]
    model_name = config['name']
    embed_dim = config['embed_dim']
    
    # Path to cloned DINOv3 repo
    dinov3_repo_path = os.path.join(BASE_DIR, "dinov3")
    
    print(f"Loading DINOv3 {model_name}...")
    print(f"  Repo: {dinov3_repo_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    # Use torch.hub with local source - exactly as in documentation
    model = torch.hub.load(
        dinov3_repo_path,  # Local repo path
        model_name,         # Model name (e.g., 'dinov3_vitl16')
        source='local',     # Load from local repo
        weights=checkpoint_path,  # Local checkpoint path
    )
    
    print(f"Loaded {model_name} successfully!")
    return model, embed_dim

def get_transforms():
    """Get DINOv3 preprocessing transforms."""
    from torchvision.transforms import v2
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return transform

# =============================================================================
# Embedding Extraction
# =============================================================================
def extract_embeddings(model, dataset: Dataset, batch_size: int = 16, num_workers: int = 4) -> Tuple[np.ndarray, List[int]]:
    """Extract embeddings from all images using DINOv3 with parallel loading.
    
    Now with workers doing image loading + crop extraction + transform.
    """
    
    model.eval()
    
    image_embeddings = []
    image_labels = []
    
    # Create DataLoader - workers do all heavy lifting now
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    print(f"Extracting embeddings from {len(dataset)} images with {num_workers} workers (batch_size={batch_size})...")
    
    for batch_crops, batch_labels in tqdm(dataloader):
        # batch_crops: [batch_size, n_crops, 3, 256, 256]
        batch_size_actual = batch_crops.size(0)
        n_crops = batch_crops.size(1)
        
        # Reshape to process all crops at once: [batch * n_crops, 3, 256, 256]
        batch_crops = batch_crops.view(-1, 3, 256, 256).to(device)
        
        # Extract embeddings for all crops
        with torch.no_grad():
            outputs = model(batch_crops)
            
            # Get CLS token
            if hasattr(outputs, 'last_hidden_state'):
                cls_token = outputs.last_hidden_state[:, 0, :]  # [batch*n_crops, embed_dim]
            elif hasattr(outputs, 'pooler_output'):
                cls_token = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor):
                cls_token = outputs
            else:
                cls_token = outputs[0][:, 0, :] if hasattr(outputs[0], 'last_hidden_state') else outputs[0]
        
        # Reshape back and average per image: [batch_size, n_crops, embed_dim] -> [batch_size, embed_dim]
        cls_token = cls_token.cpu().numpy()
        cls_token = cls_token.reshape(batch_size_actual, n_crops, -1)
        avg_embeddings = cls_token.mean(axis=1)
        
        image_embeddings.append(avg_embeddings)
        image_labels.append(batch_labels.numpy())
    
    embeddings = np.vstack(image_embeddings)
    labels = np.concatenate(image_labels)
    
    return embeddings, labels

# =============================================================================
# Classifier Training
# =============================================================================
def train_classifier(X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray,
                     classifier_type: str = 'logistic'):
    """Train classifier on DINOv3 embeddings."""
    
    from sklearn.preprocessing import normalize, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    
    # Normalize embeddings
    X_train_norm = normalize(X_train)
    X_val_norm = normalize(X_val)
    
    print(f"\nTraining {classifier_type} classifier...")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    if classifier_type == 'logistic':
        clf = LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            n_jobs=-1,
            random_state=SEED,
            C=1.0,
            solver='lbfgs',
        )
        clf.fit(X_train_norm, y_train)
    else:
        # MLP classifier
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(
            hidden_layer_sizes=(512, 256),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
            learning_rate='adaptive',
        )
        clf.fit(X_train_norm, y_train)
    
    # Evaluate
    train_pred = clf.predict(X_train_norm)
    val_pred = clf.predict(X_val_norm)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"Train Accuracy: {train_acc*100:.2f}%")
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    
    return clf, train_acc, val_acc

# =============================================================================
# Main Execution
# =============================================================================
def main():
    print("\n" + "="*60)
    print("DINOv3 Feature Extraction + Classification")
    print("="*60)
    print(f"\nSettings:")
    print(f"  Model: DINOv3 ViT-L (SAT-493M)")
    print(f"  Crops: {args.n_crops} per image ({args.crop_grid}x{args.crop_grid} grid)")
    print(f"  Crop size: {args.crop_size}px -> resized to 256px for DINOv3")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers: {args.num_workers}")
    
    # Create datasets
    train_dataset = ImageCropDataset(train_paths, crop_size=args.crop_size, grid_size=args.crop_grid, n_crops=args.n_crops)
    val_dataset = ImageCropDataset(val_paths, crop_size=args.crop_size, grid_size=args.crop_grid, n_crops=args.n_crops)
    test_dataset = ImageCropDataset(test_paths, crop_size=args.crop_size, grid_size=args.crop_grid, n_crops=args.n_crops)
    
    print(f"\nDataset info:")
    print(f"  Train images: {len(train_paths)}, crops/image: {train_dataset.crops_per_image}")
    print(f"  Val images: {len(val_paths)}, crops/image: {val_dataset.crops_per_image}")
    print(f"  Test images: {len(test_paths)}, crops/image: {test_dataset.crops_per_image}")
    
    # Check for cached embeddings
    embeddings_available = all([
        os.path.exists(EMB_TRAIN_PATH),
        os.path.exists(EMB_VAL_PATH),
        os.path.exists(EMB_TEST_PATH),
    ])
    
    if args.resume_embeddings and embeddings_available:
        print("\nLoading cached embeddings...")
        train_data = np.load(EMB_TRAIN_PATH)
        val_data = np.load(EMB_VAL_PATH)
        test_data = np.load(EMB_TEST_PATH)
        
        X_train = train_data['embeddings']
        y_train = train_data['labels']
        X_val = val_data['embeddings']
        y_val = val_data['labels']
        X_test = test_data['embeddings']
        y_test = test_data['labels']
        
        print(f"Loaded embeddings: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    else:
        # Load DINOv3 model
        print(f"\nLoading DINOv3 model ({args.model_size})...")
        model, embed_dim = load_dinov3_model(args.model_size)
        model = model.to(device)
        model.eval()
        
        print(f"Embedding dimension: {embed_dim}")
        
        # Extract embeddings
        print("\n" + "="*40)
        print("Extracting embeddings...")
        print("="*40)
        
        X_train, y_train = extract_embeddings(model, train_dataset, args.batch_size, args.num_workers)
        X_val, y_val = extract_embeddings(model, val_dataset, args.batch_size, args.num_workers)
        X_test, y_test = extract_embeddings(model, test_dataset, args.batch_size, args.num_workers)
        
        # Save embeddings
        print("\nSaving embeddings to disk...")
        np.savez(EMB_TRAIN_PATH, embeddings=X_train, labels=y_train)
        np.savez(EMB_VAL_PATH, embeddings=X_val, labels=y_val)
        np.savez(EMB_TEST_PATH, embeddings=X_test, labels=y_test)
        
        print(f"Embeddings saved: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    # Train classifier
    print("\n" + "="*40)
    print("Training classifier...")
    print("="*40)
    
    clf, train_acc, val_acc = train_classifier(
        X_train, y_train, 
        X_val, y_val,
        args.classifier
    )
    
    # Evaluate on test set
    from sklearn.preprocessing import normalize
    from sklearn.metrics import accuracy_score, classification_report
    
    X_test_norm = normalize(X_test)
    y_pred = clf.predict(X_test_norm)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[idx_to_label[i] for i in range(num_classes)]))
    
    # Generate visualizations
    print("\n" + "="*40)
    print("Generating visualizations...")
    print("="*40)
    
    generate_visualizations(clf, X_train, y_train, X_val, y_val, X_test, y_test, y_pred)
    
    # Save results
    save_results(clf, train_acc, val_acc, test_acc)
    
    print("\n" + "="*60)
    print("DINOv3 Classification Complete!")
    print(f"Final Results: Train={train_acc*100:.2f}%, Val={val_acc*100:.2f}%, Test={test_acc*100:.2f}%")
    print("="*60)

def generate_visualizations(clf, X_train, y_train, X_val, y_val, X_test, y_test, y_pred):
    """Generate t-SNE, confusion matrix, etc."""
    
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    # t-SNE visualization
    print("Generating t-SNE plot...")
    
    # Combine all data
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    splits = ['Train']*len(X_train) + ['Val']*len(X_val) + ['Test']*len(X_test)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, len(X_all)-1))
    X_tsne = tsne.fit_transform(X_all)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # By split
    colors = {'Train': 'blue', 'Val': 'orange', 'Test': 'green'}
    for split in ['Train', 'Val', 'Test']:
        mask = [s == split for s in splits]
        axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors[split], label=split, alpha=0.5, s=10)
    axes[0].set_title('t-SNE by Split')
    axes[0].legend()
    
    # By class (top 10 only for clarity)
    unique_classes = np.unique(y_all)[:10]
    cmap = plt.cm.get_cmap('tab20', len(unique_classes))
    class_to_color = {c: cmap(i) for i, c in enumerate(unique_classes)}
    
    for c in unique_classes:
        mask = y_all == c
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[class_to_color[c]], 
                       label=idx_to_label[c][:15], alpha=0.5, s=10)
    axes[1].set_title('t-SNE by Class (top 10)')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_dinov3.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(20, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[idx_to_label[i][:10] for i in range(num_classes)])
    disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation=90)
    ax.set_title('DINOv3 Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_dinov3.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {OUTPUT_DIR}")

def save_results(clf, train_acc, val_acc, test_acc):
    """Save training results to JSON."""
    
    results = {
        "model": "DINOv3 ViT-7B (SAT-493M) + Logistic Regression",
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "num_classes": num_classes,
        "embedding_dim": 1536,
        "n_crops": args.n_crops,
    }
    
    results_path = os.path.join(OUTPUT_DIR, 'results_dinov3.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    main()