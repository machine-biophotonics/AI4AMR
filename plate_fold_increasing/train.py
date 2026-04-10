#!/usr/bin/env python3
"""
Plate Diversity Training Script

Tests how accuracy improves with plate diversity (NOT just more data):
- Fix total training crops = 290K (same as 1 plate)
- Training plates increase while keeping crops equal
- Fixed validation: P5
- Fixed test: P6
- Same as plate_fold with complex augmentations
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
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--warmup_epochs', type=int, default=6)
parser.add_argument('--patience', type=int, default=10)
args = parser.parse_args()

# Constants
CROP_SIZE = 224
GRID_SIZE = 12
NUM_TRAINING_PLATES = 4

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


def extract_well_from_filename(filename):
    match = re.search(r'Well([A-H]\d{2})', filename)
    return match.group(1) if match else None


def get_image_paths_for_plate(plate):
    plate_dir = os.path.join(BASE_DIR, plate)
    if not os.path.exists(plate_dir):
        return []
    
    paths = []
    for ext in ['*.tif', '*.tiff', '*.png']:
        paths.extend(glob.glob(os.path.join(plate_dir, '**', ext), recursive=True))
    
    valid_paths = []
    for path in paths:
        well = extract_well_from_filename(os.path.basename(path))
        if well and well in plate_maps.get(plate, {}):
            valid_paths.append(path)
    
    return valid_paths


def get_gene_from_path(img_path):
    dirname = os.path.dirname(img_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        return plate_maps[plate][well]
    return 'WT'


class GrayscaleMixedCropDataset(Dataset):
    def __init__(self, image_paths, labels, crop_size=224, grid_size=12, augment=True, seed=42, epoch=0):
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
        
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                         scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)}, rotate=(-15, 15), p=0.5),
                A.SomeOf([
                    A.ElasticTransform(alpha=50, sigma=5, p=1.0),
                    A.Perspective(scale=(0.02, 0.05), p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                    A.OpticalDistortion(distort_limit=0.05, p=1.0),
                ], n=1, replace=False, p=0.5),
                A.SomeOf([
                    A.GaussNoise(std_range=(0.05, 0.15), per_channel=False, p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=3, p=1.0),
                ], n=1, replace=False, p=0.5),
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
        
        print(f"Gene Dataset: {len(self.positions)} positions, {len(self.image_paths)} images")
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        # Create per-image position hashmap - unique for first 144 images, then cycle with offset
        rng = random.Random(self.seed + epoch)
        shuffled = rng.sample(self.positions, len(self.positions))
        
        self.epoch_positions = {}
        num_images = len(self.image_paths)
        total_positions = len(self.positions)
        
        # First 144 images: unique positions (no repeat)
        for idx in range(min(144, num_images)):
            self.epoch_positions[idx] = shuffled[idx]
        
        # Images 144+: cycle with different offset each cycle
        for idx in range(144, num_images):
            cycle = idx // total_positions
            pos_in_cycle = idx % total_positions
            offset_pos = (pos_in_cycle + cycle) % total_positions
            self.epoch_positions[idx] = shuffled[offset_pos]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        left, top = self.epoch_positions[idx]
        crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        crop = np.array(crop)
        crop = self.transform(image=crop)['image']
        
        return crop, self.labels[idx]


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()


def weighted_focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0):
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    weighted = focal * weights
    return weighted.mean()


def train_and_evaluate(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)
    
    train_plates = []
    for path in train_paths:
        plate = os.path.basename(os.path.dirname(path))
        train_plates.append(plate)
    train_plates = np.array(train_plates)
    
    class_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor([total / (num_classes * class_counts[i]) for i in range(num_classes)], device=device)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    plate_counts = Counter(train_plates)
    n_plates = len(plate_counts)
    domain_weights = {plate: 1.0 / np.sqrt(count) for plate, count in plate_counts.items()}
    dom_sum = sum(domain_weights.values())
    domain_weights = {k: v / dom_sum * n_plates for k, v in domain_weights.items()}
    print(f"Domain weights: {domain_weights}")
    
    def get_combined_weights(plates, labels):
        weights = []
        for plate, label in zip(plates, labels):
            class_w = class_weights[label].item()
            domain_w = domain_weights.get(plate, 1.0)
            weights.append(class_w * domain_w)
        weights = np.array(weights)
        weights = weights / weights.mean()
        return torch.tensor(weights, device=device)
    
    train_dataset = GrayscaleMixedCropDataset(train_paths, train_labels, augment=True, seed=SEED)
    val_dataset = GrayscaleMixedCropDataset(val_paths, val_labels, augment=False, seed=SEED)
    test_dataset = GrayscaleMixedCropDataset(test_paths, test_labels, augment=False, seed=SEED)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = True
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )
    model = model.to(device)
    
    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
    classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': classifier_params, 'lr': args.lr}
    ], weight_decay=0.01)
    
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / num_warmup_steps
        progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'training_metrics_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_balanced_acc', 'lr'])
    
    best_val_acc = 0.0
    best_val_balanced_acc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)
        test_dataset.set_epoch(epoch)
        
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)):
            images, labels = images.to(device), labels.to(device)
            
            batch_start = batch_idx * train_loader.batch_size
            batch_end = min(batch_start + labels.size(0), len(train_plates))
            batch_plates = train_plates[batch_start:batch_end]
            
            weights = get_combined_weights(batch_plates, labels.cpu().tolist())
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = weighted_focal_loss(outputs, labels, weights)
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
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        per_class_correct = [np.sum((all_preds == i) & (all_labels == i)) for i in range(num_classes)]
        per_class_total = [np.sum(all_labels == i) for i in range(num_classes)]
        balanced_acc = np.mean([per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0 for i in range(num_classes)])
        
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%, Balanced Acc={balanced_acc:.4f}, LR={current_lr:.2e}")
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, balanced_acc, current_lr])
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
        }, os.path.join(output_dir, 'last_model.pth'))
        
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(output_dir, f'checkpoint_e{epoch}.pth'))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs,
                'best_val_acc': best_val_acc,
            }, os.path.join(output_dir, 'best_model.pth'))
        
        if balanced_acc > best_val_balanced_acc + 0.001:
            best_val_balanced_acc = balanced_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_balanced_acc': best_val_balanced_acc,
            }, os.path.join(output_dir, 'best_model_balanced.pth'))
    
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    test_acc = 100. * np.mean(all_preds == all_labels)
    test_balanced_acc = 100. * np.mean([np.mean(all_preds[all_labels == i] == i) for i in range(num_classes) if np.sum(all_labels == i) > 0])
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Balanced Accuracy: {test_balanced_acc:.2f}%")
    
    return test_acc


def main():
    all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    
    VAL_PLATE = 'P5'
    TEST_PLATE = 'P6'
    
    # Load validation and test data
    val_paths = get_image_paths_for_plate(VAL_PLATE)
    val_labels = [gene_to_idx[get_gene_from_path(p)] for p in val_paths]
    
    test_paths = get_image_paths_for_plate(TEST_PLATE)
    test_labels = [gene_to_idx[get_gene_from_path(p)] for p in test_paths]
    
    print(f"Validation: {len(val_paths)}, Test: {len(test_paths)}")
    
    results = {}
    
    for n_train in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"Training with {n_train} plate(s)")
        print(f"{'='*60}")
        
        output_dir = os.path.join(SCRIPT_DIR, f'increase_{n_train}_plates')
        
        # Skip if already completed
        if os.path.exists(os.path.join(output_dir, 'best_model.pth')):
            print(f"Skipping {n_train} plates - already completed")
            results[n_train] = 0
            continue
        
        train_plates = [p for p in all_plates if p not in [VAL_PLATE, TEST_PLATE]][:n_train]
        
        # Use 21 images per class = 2016 total images = exact whole number
        # 21 images/class * 96 classes = 2016 images total
        # With 1 plate: all 2016 images, 2 plates: 1008 each, 3 plates: 672 each, 4 plates: 504 each
        # Note: Some classes will have slightly different count (21 vs 20/21/22) due to integer rounding
        images_per_plate = 2016 // n_train
        
        train_paths = []
        for plate in train_plates:
            paths = get_image_paths_for_plate(plate)
            # Shuffle and take exact number for reproducibility
            rng = random.Random(SEED + n_train)
            rng.shuffle(paths)
            train_paths.extend(paths[:images_per_plate])
        
        # Shuffle final train_paths for good mixing
        random.shuffle(train_paths)
        
        train_labels = [gene_to_idx[get_gene_from_path(p)] for p in train_paths]
        
        print(f"Training: {len(train_paths)} images")
        test_acc = train_and_evaluate(
            train_paths, train_labels,
            val_paths, val_labels,
            test_paths, test_labels,
            output_dir
        )
        
        results[n_train] = test_acc
    
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
    ax.set_title('Test Accuracy vs Plate Diversity', fontsize=14)
    ax.set_xticks(range(len(n_plates)))
    ax.set_xticklabels([f'{n}' for n in n_plates])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2., acc + 0.5, 
               f'{acc:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'diversity_plot.png'), dpi=150)


if __name__ == '__main__':
    main()