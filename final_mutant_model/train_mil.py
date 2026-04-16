#!/usr/bin/env python3
"""
MIL training with class-bucket sampling and configurable attention heads.
- Training: 9 crops from 9 DIFFERENT images per class per epoch
- Val/Test: 9 crops from center + 3x3 neighborhood (same image)
- Warmup (6 epochs) + Cosine Annealing decay
- Configurable attention heads (default: 8, recommended: 4-8)

Attention Heads Recommendation:
- 4-8 heads: SAFE for this dataset size (~8K training images)
- 20 heads: AGGRESSIVE - may overfit with limited data
- Each head adds ~165K parameters to the attention layer
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from mil_model import (
    AttentionMILModel,
    ClassBucketDataset,
    SingleImageDataset,
    get_gene_from_path,
    extract_well_from_filename,
)


ALL_PLATES: list[str] = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def weighted_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    """Weighted focal loss with class weights."""
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return (focal * weights).mean()


def attention_entropy_loss(attn_weights: torch.Tensor) -> torch.Tensor:
    """Entropy regularization to prevent attention collapse."""
    entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=1).mean()
    return entropy


def get_image_paths_for_plate(
    plate: str,
    base_dir: str,
    plate_maps: dict[str, dict[str, str]]
) -> list[str]:
    """Get valid image paths for a plate."""
    import glob
    
    plate_dir = os.path.join(base_dir, plate)
    if not os.path.exists(plate_dir):
        return []
    
    paths: list[str] = []
    for pattern in ['*.tif', '*.tiff', 'png']:
        paths.extend(glob.glob(os.path.join(plate_dir, '**', pattern), recursive=True))
    
    valid_paths: list[str] = []
    for path in paths:
        well = extract_well_from_filename(os.path.basename(path))
        if well and well in plate_maps.get(plate, {}):
            valid_paths.append(path)
    
    return valid_paths


def create_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int = 6) -> SequentialLR:
    """Create warmup + cosine annealing scheduler."""
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )


def train_single_fold(
    test_plate: str,
    args: argparse.Namespace,
    script_dir: str,
    base_dir: str,
    plate_maps: dict[str, dict[str, str]],
    gene_to_idx: dict[str, int],
    num_classes: int
) -> dict[str, Any] | None:
    """Train a single fold."""
    OUTPUT_DIR = os.path.join(script_dir, f'fold_{test_plate}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training fold: test_plate={test_plate}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"{'='*60}")
    
    train_val_plates = [p for p in ALL_PLATES if p != test_plate]
    train_plates = train_val_plates[:4]
    val_plates = train_val_plates[4:]
    
    print(f"Train plates: {train_plates}")
    print(f"Val plates: {val_plates}")
    
    train_paths: list[str] = []
    train_labels: list[int] = []
    val_paths: list[str] = []
    val_labels: list[int] = []
    test_paths: list[str] = []
    test_labels: list[int] = []
    
    for plate in train_plates:
        for path in get_image_paths_for_plate(plate, base_dir, plate_maps):
            train_paths.append(path)
            train_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    for plate in val_plates:
        for path in get_image_paths_for_plate(plate, base_dir, plate_maps):
            val_paths.append(path)
            val_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    for path in get_image_paths_for_plate(test_plate, base_dir, plate_maps):
        test_paths.append(path)
        test_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    val_labels_arr_full = np.array(val_labels)
    val_paths_full = val_paths.copy()
    test_labels_arr_full = np.array(test_labels)
    test_paths_full = test_paths.copy()
    
    val_class_to_images: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(val_labels_arr_full):
        val_class_to_images[label].append(i)
    
    val_paths = []
    val_labels = []
    for label, indices in val_class_to_images.items():
        idx = random.Random(args.seed).choice(indices)
        val_paths.append(val_paths_full[idx])
        val_labels.append(label)
    
    test_class_to_images: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(test_labels_arr_full):
        test_class_to_images[label].append(i)
    
    test_paths = []
    test_labels = []
    for label, indices in test_class_to_images.items():
        idx = random.Random(args.seed).choice(indices)
        test_paths.append(test_paths_full[idx])
        test_labels.append(label)
    
    train_labels_arr = np.array(train_labels)
    val_labels_arr = np.array(val_labels)
    test_labels_arr = np.array(test_labels)
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)} (1 per class), Test: {len(test_paths)} (1 per class)")
    
    if len(train_paths) == 0:
        print(f"ERROR: No training data found for fold {test_plate}. Check --data_root path.")
        return None
    
    class_counts = Counter(train_labels_arr)
    total = len(train_labels_arr)
    
    class_weights_list: list[float] = []
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        if count == 0:
            print(f"WARNING: Class {i} has 0 samples in training set. Using weight=1.0")
            class_weights_list.append(1.0)
        else:
            class_weights_list.append(total / (num_classes * count))
    
    class_weights = torch.tensor(class_weights_list, device=DEVICE)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"\nInitializing datasets...")
    train_dataset = ClassBucketDataset(
        train_paths, train_labels_arr,
        augment=True, seed=args.seed, num_crops_per_class=9
    )
    val_dataset = SingleImageDataset(val_paths, val_labels_arr, augment=False, seed=args.seed)
    test_dataset = SingleImageDataset(test_paths, test_labels_arr, augment=False, seed=args.seed)
    
    train_dataset.set_epoch(0)
    val_dataset.set_epoch(0)
    test_dataset.set_epoch(0)
    
    num_workers = 0 if os.name == 'nt' else 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    print(f"\nTraining config:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Warmup epochs: {args.warmup_epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Crops per class per epoch: {train_dataset.num_crops_per_class}")
    print(f"  - Batches per epoch: {len(train_loader)}")
    print(f"  - Epochs to exhaust: {train_dataset.epochs_to_exhaust}")
    
    model = AttentionMILModel(num_classes=num_classes, num_heads=args.num_heads)
    model = model.to(DEVICE)
    
    print(f"Attention Heads: {args.num_heads} (head_dim = 1280 / 4 = {1280 // 4})")
    print(f"Total attention params: ~{args.num_heads * 165120:,}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} total params, {trainable_params:,} trainable")
    
    backbone_params = [p for n, p in model.named_parameters() if 'attention_pool' not in n and 'classifier' not in n]
    attention_params = [p for n, p in model.named_parameters() if 'attention_pool' in n or 'classifier' in n]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': attention_params, 'lr': args.lr}
    ], weight_decay=0.01)
    
    scheduler = create_scheduler(optimizer, args.epochs, args.warmup_epochs)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f'training_metrics_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc', 'lr', 'epoch_time'])
    
    best_metrics: dict[str, float] = {
        'val_auc': 0.0,
        'val_acc': 0.0,
        'val_loss': float('inf')
    }
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    epoch_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_time_start = time.time()
        train_dataset.set_epoch(epoch)
        model.train()
        run_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)):
            images = images.to(DEVICE)
            labels_dev = labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs, attn_weights = model(images, return_attention=True)
            
            main_loss = weighted_focal_loss(outputs, labels_dev, class_weights[labels_dev])
            ent_loss = attention_entropy_loss(attn_weights)
            loss = main_loss + 0.01 * ent_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            run_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels_dev.size(0)
            correct += predicted.eq(labels_dev).sum().item()
        
        scheduler.step()
        
        train_acc = 100. * correct / total
        avg_train_loss = run_loss / len(train_loader)
        
        model.eval()
        all_preds: list[int] = []
        all_probs: list[np.ndarray] = []
        all_labels: list[int] = []
        val_total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation', leave=False):
                images = images.to(DEVICE)
                labels_dev = labels.to(DEVICE)
                outputs, _ = model(images, return_attention=True)
                val_loss = weighted_focal_loss(outputs, labels_dev, class_weights[labels_dev])
                val_total_loss += val_loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels_dev.cpu().numpy())
        
        avg_val_loss = val_total_loss / len(val_loader)
        val_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_auc_score
        all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        val_auc = roc_auc_score(all_labels_bin, np.array(all_probs), average='macro')
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_time_start
        
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}, "
              f"LR={current_lr:.2e}, Time={epoch_time:.1f}s")
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, val_auc, current_lr, epoch_time])
        
        if val_auc > best_metrics['val_auc']:
            best_metrics['val_auc'] = val_auc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, 
                      os.path.join(OUTPUT_DIR, 'best_model_auc.pth'))
        
        if val_acc > best_metrics['val_acc']:
            best_metrics['val_acc'] = val_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, 
                      os.path.join(OUTPUT_DIR, 'best_model_acc.pth'))
        
        if avg_val_loss < best_metrics['val_loss']:
            best_metrics['val_loss'] = avg_val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, 
                      os.path.join(OUTPUT_DIR, 'best_model_loss.pth'))
        
        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, 
                      os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
    
    total_time = time.time() - epoch_start_time
    
    print("\nTesting...")
    
    best_model_path = os.path.join(OUTPUT_DIR, 'best_model_auc.pth')
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(OUTPUT_DIR, 'best_model_acc.pth')
    if not os.path.exists(best_model_path):
        best_model_path = os.path.join(OUTPUT_DIR, 'best_model_loss.pth')
    
    if not os.path.exists(best_model_path):
        print(f"ERROR: No best model found in {OUTPUT_DIR}")
        return None
    
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing', leave=False):
            images = images.to(DEVICE)
            outputs, _ = model(images, return_attention=True)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    test_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    test_auc = roc_auc_score(test_labels_bin, np.array(all_probs), average='macro')
    test_ap = average_precision_score(test_labels_bin, np.array(all_probs), average='macro')
    
    print(f"\nTest Results:")
    print(f"  Test Acc: {test_acc:.2f}%")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Test AP: {test_ap:.4f}")
    
    results: dict[str, Any] = {
        'timestamp': timestamp,
        'config': {
            'epochs': args.epochs,
            'warmup_epochs': args.warmup_epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'test_plate': test_plate,
            'num_heads': args.num_heads,
            'epochs_to_exhaust': train_dataset.epochs_to_exhaust,
        },
        'results': {
            'best_val_auc': float(best_metrics['val_auc']),
            'best_val_acc': float(best_metrics['val_acc']),
            'best_val_loss': float(best_metrics['val_loss']),
            'test_acc': float(test_acc),
            'test_auc': float(test_auc),
            'test_ap': float(test_ap)
        },
        'timing': {
            'total_time': total_time,
            'time_per_epoch': total_time / args.epochs,
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='MIL training with 20 attention heads (gold standard)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (default: 200)')
    parser.add_argument('--warmup_epochs', type=int, default=6, help='Warmup epochs (default: 6)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads (default: 8, recommended: 4-8)')
    parser.add_argument('--test_plate', type=str, default='P6', help='Test plate (default: P6)')
    parser.add_argument('--data_root', type=str, default=None, help='Path to folder containing P1-P6 plate folders')
    parser.add_argument('--run_all_folds', action='store_true', help='Run all 6 folds with skip for done folds')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = args.data_root if args.data_root else os.path.dirname(SCRIPT_DIR)
    
    with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
        plate_data: dict = json.load(f)
    
    plate_maps: dict[str, dict[str, str]] = {}
    for plate in ALL_PLATES:
        plate_maps[plate] = {}
        if plate in plate_data:
            for row, wells in plate_data[plate].items():
                for col, info in wells.items():
                    well = f"{row}{int(col):02d}"
                    plate_maps[plate][well] = info['id']
    
    all_genes = sorted(set(
        label
        for pm in plate_maps.values()
        for label in pm.values()
    ))
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
    num_classes = len(all_genes)
    print(f"Classes: {num_classes}")
    
    if args.run_all_folds:
        all_results: list[dict | None] = []
        for test_plate in ALL_PLATES:
            fold_dir = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
            
            best_model_auc = os.path.join(fold_dir, 'best_model_auc.pth')
            best_model_acc = os.path.join(fold_dir, 'best_model_acc.pth')
            best_model_loss = os.path.join(fold_dir, 'best_model_loss.pth')
            
            if os.path.exists(best_model_auc) or os.path.exists(best_model_acc) or os.path.exists(best_model_loss):
                print(f"\nSkipping {test_plate}: already trained (model exists)")
                results_file = os.path.join(fold_dir, 'training_results.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        existing_result = json.load(f)
                        all_results.append(existing_result)
                continue
            
            print(f"\n{'='*60}")
            print(f"Training fold: {test_plate}")
            print(f"{'='*60}")
            result = train_single_fold(test_plate, args, SCRIPT_DIR, BASE_DIR, plate_maps, gene_to_idx, num_classes)
            all_results.append(result)
        
        with open(os.path.join(SCRIPT_DIR, 'all_folds_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        print("\nAll folds completed!")
    else:
        train_single_fold(args.test_plate, args, SCRIPT_DIR, BASE_DIR, plate_maps, gene_to_idx, num_classes)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
