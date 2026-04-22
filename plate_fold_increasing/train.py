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
import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import csv
import random
from datetime import datetime
from collections import Counter
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from PIL import Image
import glob
import hashlib

def stable_hash(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 10000

def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

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
parser.add_argument('--case', type=int, default=None, help='Case number: 1, 2, 3, or 4')
parser.add_argument('--run_subset', type=int, default=0, help='Subset run: 0, 1, 2, or 3')
parser.add_argument('--run_all', action='store_true', help='Run all 4 runs for a case')
parser.add_argument('--val_plate', type=str, default='P5', help='Validation plate (P1-P6)')
parser.add_argument('--test_plate', type=str, default='P6', help='Test plate (P1-P6)')
parser.add_argument('--all_experiments', action='store_true', help='Run all 6 val/test experiments')
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
        
        # Use fixed center crop for val/test (deterministic)
        if self.use_center_crop:
            center_idx = len(self.positions) // 2
            center_pos = self.positions[center_idx]
            self.epoch_positions = {i: center_pos for i in range(len(self.image_paths))}
            return
        
        num_pos = len(self.positions)
        num_images = len(self.image_paths)
        
        # Cycle-based permutation: each epoch shifts the starting position
        # Cycle 0: uses shuffled[0], shuffled[1], ... shuffled[143]
        # Cycle 1: uses shuffled[143], shuffled[0], ... (rotated by 1)
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
    
    train_dataset = GrayscaleMixedCropDataset(train_paths, train_labels, augment=True, seed=SEED)
    train_plates = train_dataset.plates
    
    class_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor([total / (num_classes * class_counts[i]) for i in range(num_classes)], device=device)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    def get_weights(labels):
        weights = [class_weights[label].item() for label in labels]
        return torch.tensor(weights, device=device)
    
    val_dataset = GrayscaleMixedCropDataset(val_paths, val_labels, augment=False, seed=SEED, use_center_crop=True)
    test_dataset = GrayscaleMixedCropDataset(test_paths, test_labels, augment=False, seed=SEED, use_center_crop=True)
    
    # Initialize epoch once (deterministic for val/test)
    val_dataset.set_epoch(0)
    test_dataset.set_epoch(0)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True)
    
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
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_balanced_acc', 'val_auc', 'lr'])
    
    best_val_acc = 0.0
    best_val_balanced_acc = 0.0
    best_val_auc = 0.0
    best_val_loss = float('inf')
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels, _ in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            weights = get_weights(labels.cpu().tolist())
            
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
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.inference_mode():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
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
        balanced_acc = np.mean([per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else np.nan for i in range(num_classes)])
        balanced_acc = np.nanmean(balanced_acc)
        
        # Compute ROC AUC (one-vs-rest) - binarize labels for multiclass
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
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, balanced_acc, val_auc, current_lr])
        
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
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
            }, os.path.join(output_dir, 'best_model.pth'))
    
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
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


def run_experiment(case_num, run_idx, val_plate, test_plate, all_train_plates):
    """Run experiment with case_num and run_idx"""
    exp_name = f"val{val_plate}test{test_plate}"
    
    if case_num == 1:
        combo_name = all_train_plates[run_idx]
    elif case_num == 2:
        idx = run_idx
        combo_name = f"{all_train_plates[idx]}{all_train_plates[(idx+1)%4]}"
    elif case_num == 3:
        idx = run_idx
        combo_name = f"{all_train_plates[idx]}{all_train_plates[(idx+1)%4]}{all_train_plates[(idx+2)%4]}"
    else:
        combo_name = ''.join(all_train_plates)
    
    output_dir = os.path.join(SCRIPT_DIR, exp_name, f'case_{case_num}_{combo_name}', f'run_{run_idx}')
    
    if os.path.exists(os.path.join(output_dir, 'best_model.pth')):
        print(f"Skipping {combo_name} {exp_name} run_{run_idx} - already completed")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    val_paths = get_image_paths_for_plate(val_plate)
    val_labels = [gene_to_idx[get_gene_from_path(p)] for p in val_paths]
    
    test_paths = get_image_paths_for_plate(test_plate)
    test_labels = [gene_to_idx[get_gene_from_path(p)] for p in test_paths]
    
    if case_num == 1:
        target_total = 2016
        train_plates_list = [all_train_plates[run_idx]]
    elif case_num == 2:
        target_total = 1920
        idx = run_idx
        train_plates_list = [all_train_plates[idx], all_train_plates[(idx+1)%4]]
    elif case_num == 3:
        target_total = 2016
        idx = run_idx
        train_plates_list = [all_train_plates[idx], all_train_plates[(idx+1)%4], all_train_plates[(idx+2)%4]]
    else:
        target_total = 1920
        train_plates_list = all_train_plates
    
    n_plates = len(train_plates_list)
    images_per_plate = target_total // n_plates
    
    train_paths = []
    for plate in train_plates_list:
        paths = get_image_paths_for_plate(plate)
        
        gene_to_paths = {}
        for p in paths:
            gene = get_gene_from_path(p)
            gene_to_paths.setdefault(gene, []).append(p)
        
        per_class = images_per_plate // num_classes
        
        for gene in sorted(gene_to_paths.keys()):
            gene_paths = gene_to_paths[gene]
            gene_paths.sort()
            start_idx = run_idx if case_num == 4 else 0
            for i in range(per_class):
                img_idx = (start_idx + i) % len(gene_paths)
                train_paths.append(gene_paths[img_idx])
    
    train_labels = [gene_to_idx[get_gene_from_path(p)] for p in train_paths]
    
    class_dist = Counter(train_labels)
    print(f"Case {case_num} {combo_name} {exp_name} run_{run_idx}: {len(train_paths)} images, min={min(class_dist.values())}, max={max(class_dist.values())}")
    
    test_acc = train_and_evaluate(
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        output_dir
    )
    
    return test_acc


def run_all_val_test_experiments():
    """Run all 6 val/test experiment combinations"""
    val_test_combos = [
        ('P1', 'P2'),
        ('P2', 'P3'),
        ('P3', 'P4'),
        ('P4', 'P5'),
        ('P5', 'P6'),
        ('P6', 'P1'),
    ]
    
    for val_plate, test_plate in val_test_combos:
        print(f"\n{'='*60}")
        print(f"Running experiment: val{val_plate} -> test{test_plate}")
        print(f"{'='*60}")
        
        all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
        train_plates = [p for p in all_plates if p != val_plate and p != test_plate]
        print(f"Training plates: {''.join(train_plates)}")
        
        for case_num in [1, 2, 3, 4]:
            if case_num == 4:
                num_runs = 4
            else:
                num_runs = 4
            
            for run_idx in range(num_runs):
                exp_name = f"val{val_plate}test{test_plate}"
                
                if case_num == 1:
                    combo_name = train_plates[run_idx]
                elif case_num == 2:
                    combo_name = f"{train_plates[run_idx]}{train_plates[(run_idx+1)%4]}"
                elif case_num == 3:
                    combo_name = f"{train_plates[run_idx]}{train_plates[(run_idx+1)%4]}{train_plates[(run_idx+2)%4]}"
                else:
                    combo_name = ''.join(train_plates)
                
                check_dir = os.path.join(SCRIPT_DIR, exp_name, f'case_{case_num}_{combo_name}', f'run_{run_idx}')
                model_file = os.path.join(check_dir, 'best_model.pth')
                
                if os.path.exists(model_file):
                    print(f"SKIP: case_{case_num}_{combo_name} run_{run_idx} - already completed")
                    continue
                
                print(f"Running case_{case_num}_{combo_name} run_{run_idx}...")
                try:
                    run_experiment(case_num, run_idx, val_plate, test_plate, train_plates)
                except Exception as e:
                    print(f"ERROR: case_{case_num}_{combo_name} run_{run_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


def main():
    if args.all_experiments:
        run_all_val_test_experiments()
    else:
        print("Please use --all_experiments to run all 6 val/test experiments")


if __name__ == '__main__':
    main()