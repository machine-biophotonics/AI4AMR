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
parser.add_argument('--run_all', action='store_true', help='Run all 16 experiments')
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
        
        if val_auc > best_val_auc + 0.001:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_balanced_acc': best_val_balanced_acc,
                'best_val_auc': best_val_auc,
            }, os.path.join(output_dir, 'best_model_auc.pth'))
        
        if avg_val_loss < best_val_loss - 0.001:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_balanced_acc': best_val_balanced_acc,
                'best_val_auc': best_val_auc,
                'best_val_loss': best_val_loss,
            }, os.path.join(output_dir, 'best_model_loss.pth'))
    
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


def main():
    all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
    
    VAL_PLATE = 'P5'
    TEST_PLATE = 'P6'
    
    val_paths = get_image_paths_for_plate(VAL_PLATE)
    val_labels = [gene_to_idx[get_gene_from_path(p)] for p in val_paths]
    
    test_paths = get_image_paths_for_plate(TEST_PLATE)
    test_labels = [gene_to_idx[get_gene_from_path(p)] for p in test_paths]
    
    print(f"Validation: {len(val_paths)}, Test: {len(test_paths)}")
    
    case_configs = {
        1: [['P1'], ['P2'], ['P3'], ['P4']],
        2: [['P1', 'P2'], ['P2', 'P3'], ['P3', 'P4'], ['P4', 'P1']],
        3: [['P1', 'P2', 'P3'], ['P2', 'P3', 'P4'], ['P3', 'P4', 'P1'], ['P4', 'P1', 'P2']],
        4: [['P1', 'P2', 'P3', 'P4'], ['P1', 'P2', 'P3', 'P4'], ['P1', 'P2', 'P3', 'P4'], ['P1', 'P2', 'P3', 'P4']],
    }
    
    plate_combo_names = {
        1: ['P1', 'P2', 'P3', 'P4'],
        2: ['P1P2', 'P2P3', 'P3P4', 'P4P1'],
        3: ['P1P2P3', 'P2P3P4', 'P3P4P1', 'P4P1P2'],
        4: ['P1P2P3P4', 'P1P2P3P4', 'P1P2P3P4', 'P1P2P3P4'],
    }
    
    def run_experiment(case_num, subset_idx):
        train_plates = case_configs[case_num][subset_idx]
        combo_name = plate_combo_names[case_num][subset_idx]
        
        output_dir = os.path.join(SCRIPT_DIR, f'case_{case_num}_{combo_name}', f'run_{subset_idx}')
        
        if os.path.exists(os.path.join(output_dir, 'best_model.pth')):
            print(f"Skipping case {case_num} {combo_name} run_{subset_idx} - already completed")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        if case_num == 1:
            target_total = 2016
        elif case_num == 2:
            target_total = 1920
        elif case_num == 3:
            target_total = 2016
        else:
            target_total = 1920
        
        n_plates = len(train_plates)
        images_per_plate = target_total // n_plates
        
        train_paths = []
        for plate in train_plates:
            paths = get_image_paths_for_plate(plate)
            
            gene_to_paths = {}
            for p in paths:
                gene = get_gene_from_path(p)
                gene_to_paths.setdefault(gene, []).append(p)
            
            per_class = images_per_plate // num_classes
            
            for gene in sorted(gene_to_paths.keys()):
                gene_paths = gene_to_paths[gene]
                gene_paths.sort()
                
                if case_num == 4:
                    start_idx = subset_idx
                    for i in range(per_class):
                        img_idx = (start_idx + i) % len(gene_paths)
                        train_paths.append(gene_paths[img_idx])
                else:
                    train_paths.extend(gene_paths[:per_class])
        
        train_labels = [gene_to_idx[get_gene_from_path(p)] for p in train_paths]
        
        class_dist = Counter(train_labels)
        print(f"Case {case_num} {combo_name} run_{subset_idx}: {len(train_paths)} images, min={min(class_dist.values())}, max={max(class_dist.values())}")
        
        test_acc = train_and_evaluate(
            train_paths, train_labels,
            val_paths, val_labels,
            test_paths, test_labels,
            output_dir
        )
        
        return test_acc
    
    def run_all_experiments():
        all_results = {}
        
        for case_num in [1, 2, 3, 4]:
            print(f"\n{'='*60}")
            print(f"CASE {case_num}: {len(case_configs[case_num])} runs")
            print(f"{'='*60}")
            
            case_results = []
            for subset_idx in range(4):
                print(f"\n--- Running subset {subset_idx} ---")
                test_acc = run_experiment(case_num, subset_idx)
                case_results.append(test_acc)
            
            case_results = [r for r in case_results if r is not None]
            if case_results:
                mean_acc = np.mean(case_results)
                std_acc = np.std(case_results)
                all_results[case_num] = {'mean': mean_acc, 'std': std_acc, 'runs': case_results}
                print(f"\nCase {case_num} Results: {mean_acc:.2f}% ± {std_acc:.2f}%")
        
        print("\n" + "="*60)
        print("FINAL RESULTS WITH STD")
        print("="*60)
        
        rows = []
        for case_num, data in all_results.items():
            rows.append({
                'case': case_num,
                'n_plates': case_num,
                'mean_accuracy': data['mean'],
                'std_accuracy': data['std'],
                'run_0': data['runs'][0] if len(data['runs']) > 0 else None,
                'run_1': data['runs'][1] if len(data['runs']) > 1 else None,
                'run_2': data['runs'][2] if len(data['runs']) > 2 else None,
                'run_3': data['runs'][3] if len(data['runs']) > 3 else None,
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(SCRIPT_DIR, 'diversity_results_with_std.csv'), index=False)
        print(df)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cases = list(all_results.keys())
        means = [all_results[c]['mean'] for c in cases]
        stds = [all_results[c]['std'] for c in cases]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        bars = ax.bar(range(len(cases)), means, yerr=stds, color=colors[:len(cases)], alpha=0.8, capsize=5)
        
        ax.set_xlabel('Number of Training Plates', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title('Test Accuracy vs Plate Diversity (Mean ± Std over 4 runs)', fontsize=14)
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels([f'{c}' for c in cases])
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            ax.text(bar.get_x() + bar.get_width()/2., mean + std + 0.5, 
                   f'{mean:.1f}±{std:.1f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, 'diversity_plot_with_std.png'), dpi=150)
    
    if args.run_all:
        run_all_experiments()
    elif args.case is not None:
        if args.run_subset not in [0, 1, 2, 3]:
            raise ValueError("--run_subset must be 0, 1, 2, or 3")
        run_experiment(args.case, args.run_subset)
    else:
        print("Please specify --case (1-4) and --run_subset (0-3), or use --run_all to run all 16 experiments")


if __name__ == '__main__':
    main()