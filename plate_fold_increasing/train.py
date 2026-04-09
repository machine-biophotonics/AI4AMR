#!/usr/bin/env python3
"""
Plate Diversity Training Script

Tests how accuracy improves with plate diversity (NOT just more data):
- Fix total training crops = ~290K (same as 1 plate)
- Training plates increase while keeping crops equal
- Fixed validation: P5
- Fixed test: P6
- Same hyperparameters as plate_fold_no_aug
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import json
import re
import csv
import random
from datetime import datetime
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {device}")

# Default hyperparameters (same as plate_fold_no_aug)
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--warmup_epochs', type=int, default=6)
parser.add_argument('--patience', type=int, default=10)
args = parser.parse_args()

# Constants (same as plate_fold_no_aug)
CROP_SIZE = 224
GRID_SIZE = 12
NUM_TRAINING_PLATES = 4

# Load plate data
with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

all_genes = sorted(set(label for pm in plate_maps.values() for label in pm.values()))
gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
idx_to_gene = {idx: gene for gene, idx in gene_to_idx.items()}
num_classes = len(all_genes)
print(f"Number of classes: {num_classes}")


def get_image_paths_for_plate(plate):
    """Get all image paths for a plate."""
    plate_dir = os.path.join(BASE_DIR, plate)
    if not os.path.exists(plate_dir):
        return []
    
    paths = []
    for ext in ['*.tif', '*.tiff', '*.png']:
        paths.extend(glob.glob(os.path.join(plate_dir, '**', ext), recursive=True))
    
    valid_paths = []
    for path in paths:
        match = re.search(r'Well([A-H]\d{2})', os.path.basename(path))
        if match:
            well = match.group(1)
            if well in plate_maps.get(plate, {}):
                valid_paths.append(path)
    
    return valid_paths


def sample_n_images(paths, n):
    """Randomly sample n images from paths."""
    if len(paths) <= n:
        return paths
    indices = np.random.choice(len(paths), n, replace=False)
    return [paths[i] for i in indices]


class GrayscaleMixedCropDataset(Dataset):
    """Same dataset class as plate_fold_no_aug."""
    
    def __init__(self, image_paths, labels, crop_size=224, grid_size=12, augment=False, seed=42, epoch=0):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        # 144 positions
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if left + crop_size <= w and top + crop_size <= h:
                    positions.append((left, top))
        self.positions = positions
        
        # Augmentation (same as plate_fold_no_aug - minimal)
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                         scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, rotate=(-15, 15), p=0.5),
                # Geometric transforms simulating colony deformation and camera angles
                A.SomeOf([
                    A.ElasticTransform(alpha=50, sigma=5, p=1.0),
                    A.Perspective(scale=(0.02, 0.05), p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                    A.OpticalDistortion(distort_limit=0.05, p=1.0),
                ], n=1, replace=False, p=0.5),
                # Noise and blur (tighter ranges for microscopy)
                A.SomeOf([
                    A.GaussNoise(std_range=(0.05, 0.15), per_channel=False, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], n=1, replace=False, p=0.5),
                # Image quality artifacts
                A.ImageCompression(quality_range=(85, 100), p=0.3),
                A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.4),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        print(f"Gene Dataset: {len(self.positions)} positions, augment={augment}")
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        # Create per-image position hashmap (dict)
        # Each image gets unique position for first 144, then cycles with different order
        rng = random.Random(self.seed + epoch)
        shuffled = rng.sample(self.positions, len(self.positions))
        
        self.epoch_positions = {}
        num_images = len(self.image_paths)
        total_positions = len(self.positions)
        
        # Images 0-143: unique positions
        for idx in range(min(144, num_images)):
            self.epoch_positions[idx] = shuffled[idx]
        
        # Images 144+: cycle with different offset each cycle
        # Cycle k uses positions from offset k in shuffled
        for idx in range(144, num_images):
            cycle = idx // total_positions  # which cycle (0, 1, 2, ...)
            pos_in_cycle = idx % total_positions
            offset_pos = (pos_in_cycle + cycle) % total_positions
            self.epoch_positions[idx] = shuffled[offset_pos]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Use hashmap for per-image position in this epoch
        # First 144 images get unique positions, rest cycle with different shuffled order
        left, top = self.epoch_positions[idx]
        crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        crop = np.array(crop)
        crop = self.transform(image=crop)['image']
        
        return crop, self.labels[idx]


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal Loss (same as plate_fold_no_aug)."""
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()


def weighted_focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0):
    """Weighted Focal Loss."""
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    weighted = focal * weights
    return weighted.mean()


def train_and_evaluate(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, output_dir):
    """Train with fixed hyperparameters and return test accuracy."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create datasets (same as plate_fold_no_aug)
    train_dataset = GrayscaleMixedCropDataset(train_paths, train_labels, augment=True, seed=SEED)
    val_dataset = GrayscaleMixedCropDataset(val_paths, val_labels, augment=False, seed=SEED)
    test_dataset = GrayscaleMixedCropDataset(test_paths, test_labels, augment=False, seed=SEED)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model (same as plate_fold_no_aug)
    model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = True
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    model = model.to(device)
    
    # Optimizer (same as plate_fold_no_aug)
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': classifier_params, 'lr': args.lr}
    ], weight_decay=0.01)
    
    # LR scheduler (same as plate_fold_no_aug)
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / num_warmup_steps
        progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # CSV logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'training_metrics_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = focal_loss(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, LR={current_lr:.2e}")
        
        # Save to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, current_lr])
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    
    # Load best model and test
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    model.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    test_acc = 100. * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return test_acc


def main():
    all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    
    # Fixed validation and test plates
    VAL_PLATE = 'P5'
    TEST_PLATE = 'P6'
    
    # Calculate images needed per plate to equalize total crops
    IMAGES_PER_PLATE = 2016  # Each plate has 2016 images (full)
    CROPS_PER_IMAGE = 144     # 12x12 grid
    TOTAL_CROPS = 2016 * 144  # 290,304 total (fixed)
    
    results = {}
    
    # For each number of training plates (1, 2, 3, 4)
    for n_train in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Training with {n_train} plate(s)")
        print(f"{'='*60}")
        
        # Get training plates (exclude val and test)
        train_plates = [p for p in all_plates if p not in [VAL_PLATE, TEST_PLATE]][:n_train]
        
        # Calculate images per training plate to equalize total crops
        # TOTAL_CROPS is fixed at 290,304 (same as 1 full plate)
        # Formula: images_per_plate = TOTAL_CROPS / (n_plates * CROPS_PER_IMAGE)
        images_per_plate = TOTAL_CROPS // (n_train * CROPS_PER_IMAGE)
        
        print(f"Using {images_per_plate} images per plate ({images_per_plate * CROPS_PER_IMAGE} crops)")
        
        # Load and sample training data
        train_paths, train_labels = [], []
        for plate in train_plates:
            all_paths = get_image_paths_for_plate(plate)
            sampled_paths = sample_n_images(all_paths, images_per_plate)
            for path in sampled_paths:
                well = re.search(r'Well([A-H]\d{2})', os.path.basename(path)).group(1)
                gene = plate_maps[plate][well]
                train_paths.append(path)
                train_labels.append(gene_to_idx[gene])
        
        print(f"Training: {len(train_paths)} images")
        
        # Load validation and test data (full)
        val_paths = get_image_paths_for_plate(VAL_PLATE)
        val_labels = [gene_to_idx[plate_maps[VAL_PLATE][re.search(r'Well([A-H]\d{2})', os.path.basename(p)).group(1)]] for p in val_paths]
        
        test_paths = get_image_paths_for_plate(TEST_PLATE)
        test_labels = [gene_to_idx[plate_maps[TEST_PLATE][re.search(r'Well([A-H]\d{2})', os.path.basename(p)).group(1)]] for p in test_paths]
        
        print(f"Validation: {len(val_paths)} images")
        print(f"Test: {len(test_paths)} images")
        
        # Run training
        output_dir = os.path.join(SCRIPT_DIR, f'increase_{n_train}_plates')
        test_acc = train_and_evaluate(
            train_paths, train_labels,
            val_paths, val_labels,
            test_paths, test_labels,
            output_dir
        )
        
        results[n_train] = test_acc
        print(f"\nResult: {test_acc:.2f}%")
    
    # Save results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    df = pd.DataFrame({
        'n_training_plates': list(results.keys()),
        'test_accuracy': list(results.values())
    })
    df.to_csv(os.path.join(SCRIPT_DIR, 'diversity_results.csv'), index=False)
    print(df)
    
    # Plot bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_plates = list(results.keys())
    accs = list(results.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax.bar(range(len(n_plates)), accs, color=colors[:len(n_plates)], alpha=0.8)
    
    ax.set_xlabel('Number of Training Plates', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy vs Plate Diversity\n(Fixed Training Crops = 290K)', fontsize=14)
    ax.set_xticks(range(len(n_plates)))
    ax.set_xticklabels([f'{n}' for n in n_plates])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., acc + 0.5, 
               f'{acc:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'diversity_plot.png'), dpi=150)
    print(f"\nSaved: {os.path.join(SCRIPT_DIR, 'diversity_plot.png')}")


if __name__ == '__main__':
    main()