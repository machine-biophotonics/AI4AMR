#!/usr/bin/env python3
"""
EfficientNet-B0 Training for Gene-Level Classification
Trains on 96 guide-level classes (including WT and NC controls)
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
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, Normalize, Compose, RandomCrop, Lambda, ColorJitter, RandomRotation, RandomAffine
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import json
import re
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from typing import Optional, List, Dict
import random
from tqdm import tqdm
import csv
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from collections import Counter
import io

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
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.set_num_threads(16)

print(f"PyTorch version: {torch.__version__}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--warmup_epochs', type=int, default=6, help='Warmup epochs')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
parser.add_argument('--min_delta', type=float, default=0.001, help='Min delta')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
parser.add_argument('--test_only', action='store_true', help='Only run test evaluation (use with --resume)')
parser.add_argument('--test_plate', type=str, default='P6', help='Plate to use for test (other plates as train+val)')
parser.add_argument('--run_all_folds', action='store_true', help='Run all 6 folds (each plate as test once)')
parser.add_argument('--plot_fold_comparison', action='store_true', help='Generate combined plot of all folds')
args = parser.parse_args()


def plot_all_folds_comparison(folds_list, output_dir):
    """Generate combined plot with train/val accuracy across all folds with std deviation."""
    import seaborn as sns
    
    all_data = []
    
    for test_plate in folds_list:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
        csv_files = glob.glob(os.path.join(fold_dir, 'training_metrics_*.csv'))
        
        if not csv_files:
            print(f"Warning: No training_metrics CSV found for fold {test_plate}")
            continue
        
        csv_path = csv_files[0]
        df = pd.read_csv(csv_path)
        
        for epoch in range(len(df)):
            all_data.append({
                'fold': test_plate,
                'epoch': epoch,
                'train_acc': df.iloc[epoch]['train_acc'],
                'val_acc': df.iloc[epoch]['val_acc'],
            })
    
    if not all_data:
        print("No data found for plotting")
        return
    
    plot_df = pd.DataFrame(all_data)
    
    epochs = sorted(plot_df['epoch'].unique())
    
    train_means = []
    train_stds = []
    val_means = []
    val_stds = []
    
    for epoch in epochs:
        epoch_data = plot_df[plot_df['epoch'] == epoch]
        train_means.append(epoch_data['train_acc'].mean())
        train_stds.append(epoch_data['train_acc'].std())
        val_means.append(epoch_data['val_acc'].mean())
        val_stds.append(epoch_data['val_acc'].std())
    
    train_means = np.array(train_means)
    train_stds = np.array(train_stds)
    val_means = np.array(val_means)
    val_stds = np.array(val_stds)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs_arr = np.array(epochs)
    ax.fill_between(epochs_arr, train_means - train_stds, train_means + train_stds, 
                    alpha=0.2, color='blue')
    ax.plot(epochs_arr, train_means, 'b-', linewidth=2, label='Train Accuracy')
    
    ax.fill_between(epochs_arr, val_means - val_stds, val_means + val_stds, 
                    alpha=0.2, color='orange')
    ax.plot(epochs_arr, val_means, 'orange', linewidth=2, label='Validation Accuracy')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Train vs Validation Accuracy (n={len(folds_list)} folds)\nMean ± Std Dev', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(epochs))
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'all_folds_train_val_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    summary_df = pd.DataFrame({
        'epoch': epochs,
        'train_acc_mean': train_means,
        'train_acc_std': train_stds,
        'val_acc_mean': val_means,
        'val_acc_std': val_stds,
    })
    summary_path = os.path.join(output_dir, 'all_folds_metrics_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")


if args.plot_fold_comparison:
    all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    print("Generating combined fold comparison plot...")
    plot_all_folds_comparison(all_plates, SCRIPT_DIR)
    print("Done!")
    exit(0)

# Create output subfolder for this fold (skip if run_all_folds - will be set in loop)
if not args.run_all_folds:
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, f'fold_{args.test_plate}')
else:
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, f'fold_{args.test_plate}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# Determine train/val/test plates based on test_plate
all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
train_val_plates = [p for p in all_plates if p != args.test_plate]
# Randomize to avoid order bias
rng = random.Random(SEED + hash(args.test_plate) % 10000)
rng.shuffle(train_val_plates)
# Split train_val: first 4 plates for train, last 1 for val
train_plates = train_val_plates[:4]
val_plates = train_val_plates[4:]
test_plate = args.test_plate

# If run_all_folds, loop through all plates
if args.run_all_folds:
    all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    fold_results = []
    
    for test_plate in all_plates:
        print(f"\n{'='*60}")
        print(f"Running fold: test on {test_plate}")
        print(f"{'='*60}")
        
        # Determine train/val plates (exclude test plate)
        train_val = [p for p in all_plates if p != test_plate]
        # Shuffle to avoid order bias
        rng = random.Random(SEED + hash(test_plate) % 10000)
        rng.shuffle(train_val)
        fold_train = train_val[:4]
        fold_val = train_val[4:]
        
        # Update args for this fold
        args.test_plate = test_plate
        args.resume = None
        
        # Run training for this fold (skip if already done)
        OUTPUT_DIR = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
        best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
        result_file = os.path.join(OUTPUT_DIR, 'training_results.json')
        
        print(f"[Fold {test_plate}] Checking: {best_model_path}")
        
        if os.path.exists(best_model_path) and os.path.exists(result_file):
            print(f"Fold {test_plate} already complete (found best_model.pth and training_results.json), skipping...")
            # Still load results for summary
            try:
                with open(result_file, 'r') as f:
                    fold_results.append(json.load(f))
            except:
                pass
            continue
        
        # Create fresh script to run this fold
        import subprocess
        import sys
        cmd = [
            sys.executable, __file__,
            '--test_plate', test_plate,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--warmup_epochs', str(args.warmup_epochs),
            '--patience', str(args.patience),
        ]
        subprocess.run(cmd)
        
        # Load results for this fold
        result_file = os.path.join(OUTPUT_DIR, f'training_results.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                fold_results.append(json.load(f))
    
    # Print summary
    print(f"\n{'='*60}")
    print("FOLD RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"Test plate {r.get('config', {}).get('test_plate', '?')}: "
              f"Val Acc: {r.get('results', {}).get('best_val_acc', 0)/100:.2%}, "
              f"Test Acc: {r.get('results', {}).get('test_acc', 0)/100:.2%}")
    
else:
    SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

def extract_gene(label):
    """Extract gene name from label - for 96 classes, use full label including guide number"""
    return label

def extract_well_from_filename(filename):
    match = re.search(r'Well(\w\d+)_', filename)
    return match.group(1) if match else None

all_genes = sorted(set(extract_gene(label) for pm in plate_maps.values() for label in pm.values()))
gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
idx_to_gene = {idx: gene for gene, idx in gene_to_idx.items()}
num_classes = len(all_genes)
print(f"Number of gene classes: {num_classes}")
print(f"Genes: {all_genes}")

with open(os.path.join(SCRIPT_DIR, 'classes.txt'), 'w') as f:
    for i, gene in enumerate(all_genes):
        f.write(f"{i},{gene}\n")

def get_gene_from_path(img_path):
    dirname = os.path.dirname(img_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        label = plate_maps[plate][well]
        return extract_gene(label)
    return 'WT'

class GrayscaleMixedCropDataset(Dataset):
    """Dataset with random cropping (matching DINOv3 pipeline)"""
    
    def __init__(self, image_paths, labels, crop_size=224, grid_size=12, augment=True, seed=42, epoch=0, use_center_crop=False):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        self.use_center_crop = use_center_crop
        
        # Extract plate info from paths
        self.plates = []
        for path in image_paths:
            plate = os.path.basename(os.path.dirname(path))
            self.plates.append(plate)
        self.plates = np.array(self.plates)
        
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
        
        
        from albumentations.pytorch import ToTensorV2
        
        if augment:
            # Exact augmentations from Farrar et al. 2025 paper / KapanidisLab repo
            # NOTE: Shear and Blur REMOVED per paper findings - "shearing and blurring could cause distortions of the ribosome phenotype and hinder learning"
            
            # Geometric transforms (applied always)
            geometric_transform = A.Compose([
                A.RandomRotate90(p=1.0),
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Affine(scale=(0.6, 1.4), rotate=(-360, 360), translate_px=(-20, 20), p=1.0),
            ])
            
            # Pixel transforms (applied with probability)
            pixel_transform = A.Compose([
                A.GaussNoise(std_range=(0.01, 0.02), per_channel=True, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.5, p=0.5),
                A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(16, 64), hole_width_range=(16, 64), p=0.05),
            ])
            
            # Combined augmentations
            self.transform = A.Compose([
                geometric_transform,
                pixel_transform,
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
        
        # Use fixed center crop for val/test (deterministic)
        if self.use_center_crop:
            center_idx = len(self.positions) // 2
            center_pos = self.positions[center_idx]
            self.epoch_positions = {i: center_pos for i in range(len(self.image_paths))}
            return
        
        num_pos = len(self.positions)
        num_images = len(self.image_paths)
        
        # Cycle-based permutation: each epoch shifts the starting position
        cycle = epoch // num_pos
        pos_in_cycle = epoch % num_pos
        
        # Deterministic shuffle per cycle (seed ensures reproducibility)
        rng = random.Random(self.seed + cycle)
        shuffled = self.positions.copy()
        rng.shuffle(shuffled)
        
        # Each image gets a position based on (index + epoch) mod num_positions
        self.epoch_positions = {}
        for idx in range(num_images):
            assigned_idx = (idx + pos_in_cycle) % num_pos
            self.epoch_positions[idx] = shuffled[assigned_idx]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        left, top = self.epoch_positions[idx]
        crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        crop = np.array(crop)
        crop = self.transform(image=crop)['image']
        
        return crop, self.labels[idx], self.plates[idx]

def get_image_paths_for_plate(plate, base_dir):
    plate_dir = os.path.join(base_dir, plate)
    if not os.path.exists(plate_dir):
        return []
    
    patterns = ['*.tif', '*.tiff', '*.png']
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(plate_dir, '**', pattern), recursive=True))
    
    valid_paths = []
    for path in paths:
        well = extract_well_from_filename(os.path.basename(path))
        if well and well in plate_maps.get(plate, {}):
            valid_paths.append(path)
    
    return valid_paths

train_paths, train_labels = [], []
val_paths, val_labels = [], []
test_paths, test_labels = [], []

for plate in train_plates:
    paths = get_image_paths_for_plate(plate, BASE_DIR)
    for path in paths:
        gene = get_gene_from_path(path)
        train_paths.append(path)
        train_labels.append(gene_to_idx[gene])

for plate in val_plates:
    paths = get_image_paths_for_plate(plate, BASE_DIR)
    for path in paths:
        gene = get_gene_from_path(path)
        val_paths.append(path)
        val_labels.append(gene_to_idx[gene])

for plate in [test_plate]:
    paths = get_image_paths_for_plate(plate, BASE_DIR)
    for path in paths:
        gene = get_gene_from_path(path)
        test_paths.append(path)
        test_labels.append(gene_to_idx[gene])

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
print(f"Class distribution: {Counter(train_labels)}")

# Class weights only (inverse frequency, normalized, clipped)
class_counts = Counter(train_labels)
total = len(train_labels)
class_weights = torch.tensor([total / (num_classes * class_counts[i]) for i in range(num_classes)], device=device)
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = torch.clamp(class_weights, 0.5, 5.0)
print(f"Class weights range: {class_weights.min():.4f} - {class_weights.max():.4f}")

def weighted_ce_loss(logits, targets, weights, label_smoothing=0.1):
    """CrossEntropyLoss with label smoothing and class weights."""
    return nn.functional.cross_entropy(logits, targets, weight=weights, label_smoothing=label_smoothing)

train_dataset = GrayscaleMixedCropDataset(train_paths, train_labels, augment=True, seed=SEED)
val_dataset = GrayscaleMixedCropDataset(val_paths, val_labels, augment=False, seed=SEED, use_center_crop=True)
test_dataset = GrayscaleMixedCropDataset(test_paths, test_labels, augment=False, seed=SEED, use_center_crop=True)

# Initialize epoch once (deterministic for val/test)
val_dataset.set_epoch(0)
test_dataset.set_epoch(0)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')

# FULL FINETUNING - unfreeze entire model
for param in model.parameters():
    param.requires_grad = True

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(model.classifier[1].in_features, num_classes)
)
model = model.to(device)

# Different learning rates for backbone vs classifier
backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for backbone
    {'params': classifier_params, 'lr': args.lr}       # Higher LR for classifier
], weight_decay=0.01)
num_training_steps = len(train_loader) * args.epochs
num_warmup_steps = len(train_loader) * args.warmup_epochs

def lr_lambda(step):
    if step < num_warmup_steps:
        return step / num_warmup_steps
    progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = torch.amp.GradScaler()

# CSV logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(OUTPUT_DIR, f'training_metrics_{timestamp}.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_balanced_acc', 'val_auc', 'lr'])

best_val_acc = 0.0
best_val_balanced_acc = 0.0
best_val_auc = 0.0
best_val_loss = float('inf')
start_epoch = 0
train_losses, train_accs, val_losses, val_accs = [], [], [], []

if args.resume:
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'train_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        train_accs = checkpoint['train_accs']
        val_losses = checkpoint['val_losses']
        val_accs = checkpoint['val_accs']
        best_val_acc = checkpoint['best_val_acc']
        start_epoch = len(train_losses)
        print(f"Resuming from epoch {start_epoch}")

if args.test_only:
    print("Test-only mode: loading best model for evaluation...")
    
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Note: albumentations already imported at top
    
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    crop_size = 224
    grid_size = 12
    stride = (2720 - crop_size) // (grid_size - 1)
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            left = j * stride
            top = i * stride
            if left + crop_size <= 2720 and top + crop_size <= 2720:
                positions.append((left, top))
    
    results_data = []
    with torch.no_grad():
        for img_path, true_label in tqdm(zip(test_paths, test_labels), total=len(test_paths), desc="Processing"):
            image = Image.open(img_path).convert('RGB')
            img_name = os.path.basename(img_path)
            
            all_crops_preds = []
            for left, top in positions:
                crop = image.crop((left, top, left + crop_size, top + crop_size))
                crop = np.array(crop)
                crop = transform(image=crop)['image'].unsqueeze(0).to(device)
                
                output = model(crop)
                probs = torch.softmax(output, dim=1)
                all_crops_preds.append(probs.cpu().numpy()[0])
            
            all_crops_preds = np.array(all_crops_preds)
            avg_probs = all_crops_preds.mean(axis=0)
            pred = np.argmax(avg_probs)
            
            results_data.append({
                'image': img_name,
                'true_label': int(int(true_label)),
                'pred_label': int(int(pred)),
                'avg_probs': [float(x) for x in avg_probs.tolist()],
                'per_crop_preds': [int(x) for x in [np.argmax(p) for p in all_crops_preds]]
            })
    
    with open(os.path.join(SCRIPT_DIR, f'test_predictions_{timestamp}.json'), 'w') as f:
        json.dump(results_data, f, indent=2)
    
    correct = sum(1 for r in results_data if r['true_label'] == r['pred_label'])
    test_acc = 100. * correct / len(results_data)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Saved predictions to test_predictions_{timestamp}.json")
    print("Done!")
    exit(0)
else:
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type):
                outputs = model(images)
                loss = weighted_ce_loss(outputs, labels, class_weights, label_smoothing=0.1)
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
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
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.inference_mode():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = weighted_ce_loss(outputs, labels, class_weights, label_smoothing=0.1)
                probs = torch.softmax(outputs, dim=1)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        all_probs = np.vstack(all_probs)
        avg_val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        per_class_correct = [np.sum((all_preds == i) & (all_labels == i)) for i in range(num_classes)]
        per_class_total = [np.sum(all_labels == i) for i in range(num_classes)]
        per_class_acc = [per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else np.nan for i in range(num_classes)]
        balanced_acc = np.nanmean(per_class_acc)
        
        # Compute ROC AUC
        valid_classes = [i for i in range(num_classes) if per_class_total[i] > 0]
        if len(valid_classes) > 1:
            try:
                y_true_bin = label_binarize(all_labels, classes=np.arange(num_classes))
                val_auc = roc_auc_score(y_true_bin, all_probs, average='macro', multi_class='ovr')
            except ValueError:
                val_auc = 0.0
        else:
            val_auc = 0.0
        
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%, Balanced Acc={balanced_acc:.4f}, Val AUC={val_auc:.4f}, LR={current_lr:.2e}")
        
        # Write to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, balanced_acc, val_auc, current_lr])
        
        # Save LAST model (overwrite each epoch) - always save immediately
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'best_val_balanced_acc': best_val_balanced_acc,
            'best_val_auc': best_val_auc,
            'best_val_loss': best_val_loss,
        }, os.path.join(OUTPUT_DIR, 'last_model.pth'))
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(OUTPUT_DIR, f'checkpoint_e{epoch}.pth'))
        
        # Save based on val_acc (secondary metric)
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
                'best_val_balanced_acc': best_val_balanced_acc,
                'best_val_auc': best_val_auc,
                'best_val_loss': best_val_loss,
            }, os.path.join(OUTPUT_DIR, 'best_model_acc.pth'))
        
        # Save based on balanced accuracy
        if balanced_acc > best_val_balanced_acc + args.min_delta:
            best_val_balanced_acc = balanced_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_balanced_acc': best_val_balanced_acc,
            }, os.path.join(OUTPUT_DIR, 'best_model_balanced.pth'))
        
        # Save based on ROC AUC - PRIMARY best_model.pth
        if val_auc > best_val_auc + 0.001:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_auc': best_val_auc,
                'best_val_balanced_acc': best_val_balanced_acc,
                'best_val_loss': best_val_loss,
            }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_auc': best_val_auc,
            }, os.path.join(OUTPUT_DIR, 'best_model_auc.pth'))
        
        if avg_val_loss < best_val_loss - 0.001:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(OUTPUT_DIR, 'best_model_loss.pth'))

    print("Training complete. Generating test results...")

checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

test_acc = 100. * np.mean(all_preds == all_labels)
print(f"Test Accuracy: {test_acc:.2f}%")

test_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
all_probs = np.array(all_probs)

roc_auc = {}
ap = {}
for i in range(num_classes):
    if test_labels_bin[:, i].sum() > 0:
        roc_auc[i] = roc_auc_score(test_labels_bin[:, i], all_probs[:, i])
        ap[i] = average_precision_score(test_labels_bin[:, i], all_probs[:, i])

mean_roc_auc = np.mean(list(roc_auc.values()))
mean_ap = np.mean(list(ap.values()))
print(f"Mean ROC AUC: {mean_roc_auc:.4f}")
print(f"Mean AP: {mean_ap:.4f}")

results = {
    'timestamp': timestamp,
    'config': {
        'num_classes': num_classes,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'test_plate': args.test_plate,
    },
    'results': {
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'mean_roc_auc': float(mean_roc_auc),
        'mean_ap': float(mean_ap),
    }
}

with open(os.path.join(OUTPUT_DIR, 'training_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("Done!")
