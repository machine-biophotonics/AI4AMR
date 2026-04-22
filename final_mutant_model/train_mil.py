#!/usr/bin/env python3
"""
MIL training with cycle-based crop extraction + neighbors
Training: 9 crops (center + 3x3 neighborhood)
Validation/Test: single center crop
Supports --run_all_folds for cross-validation
"""

import argparse
import sys
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
import os
import glob
import json
import re
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import random
from tqdm import tqdm
import csv
from datetime import datetime
from collections import Counter
import multiprocessing

from mil_model import AttentionMILModel, MultiCropDataset, get_gene_from_path, extract_well_from_filename
from bag_mix import create_bag_mixer, add_bagmix_args, add_psemix_args
from mammoth_import import create_mammoth, add_mammoth_args

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--grid_size', type=int, default=12, help='Grid size for sampling positions (fixed)')
parser.add_argument('--crop_size', type=int, default=224, help='Crop size for each patch')
parser.add_argument('--neighborhood', type=int, default=5, choices=[3, 5, 7, 9], help='Neighborhood size: 3 (3x3), 5 (5x5), 7 (7x7), 9 (9x9)')
parser.add_argument('--test_plate', type=str, default='P6')
parser.add_argument('--data_root', type=str, default=None, help='Path to folder containing P1-P6 plate folders')
parser.add_argument('--run_all_folds', action='store_true', help='Run all 6 folds')
parser.add_argument('--checkpoint_type', type=str, default='auc', choices=['auc', 'acc', 'loss'], 
                    help='Which checkpoint to use for testing: auc (best AUC), acc (best accuracy), loss (lowest loss)')
parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint file (e.g., checkpoint_epoch_50.pth)')
parser.add_argument('--resume_epoch', type=int, default=None, help='Resume from specific epoch number')
parser = add_bagmix_args(parser)
parser = add_psemix_args(parser)
parser = add_mammoth_args(parser)
args = parser.parse_args()

# Set num_workers based on OS
if sys.platform.startswith('win'):
    NUM_WORKERS = 4  # Try 4 workers on Windows
else:
    NUM_WORKERS = 4

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if args.data_root:
    BASE_DIR = args.data_root
else:
    BASE_DIR = os.path.dirname(SCRIPT_DIR)

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
    return label

all_genes = sorted(set(extract_gene(label) for pm in plate_maps.values() for label in pm.values()))
gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
num_classes = len(all_genes)
print(f"Classes: {num_classes}")

all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

def get_image_paths_for_plate(plate):
    plate_dir = os.path.join(BASE_DIR, plate)
    if not os.path.exists(plate_dir):
        return []
    paths = []
    for pattern in ['*.tif', '*.tiff', '*.png']:
        paths.extend(glob.glob(os.path.join(plate_dir, '**', pattern), recursive=True))
    valid_paths = []
    for path in paths:
        well = extract_well_from_filename(os.path.basename(path))
        if well and well in plate_maps.get(plate, {}):
            valid_paths.append(path)
    return valid_paths

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()

DEBUG = os.environ.get('DEBUG_PSEMIX', '0') == '1'

def weighted_focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0):
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return (focal * weights).mean()

def soft_label_loss(logits: torch.Tensor, labels_a: torch.Tensor, labels_b: torch.Tensor, 
                 mix_ratios: torch.Tensor, weights: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Soft label loss for PseMix.
    loss = lambda * focal_loss(pred, label_a) + (1 - lambda) * focal_loss(pred, label_b)
    """
    if DEBUG:
        print(f"[DEBUG soft_label_loss] logits: {logits.shape}, labels_a: {labels_a.shape if hasattr(labels_a, 'shape') else type(labels_a)}")
    
    if isinstance(labels_a, torch.Tensor):
        labels_a = labels_a.clone().view(-1).long().to(logits.device)
    if isinstance(labels_b, torch.Tensor):
        labels_b = labels_b.clone().view(-1).long().to(logits.device)
    if isinstance(mix_ratios, torch.Tensor):
        mix_ratios = mix_ratios.clone().view(-1).float().to(logits.device)
    
    if DEBUG:
        print(f"[DEBUG soft_label_loss] after prep - labels_a: {labels_a}, labels_b: {labels_b}")
    
    ce_loss_a = nn.functional.cross_entropy(logits, labels_a, reduction='none')
    ce_loss_b = nn.functional.cross_entropy(logits, labels_b, reduction='none')
    
    pt_a = torch.exp(-ce_loss_a)
    pt_b = torch.exp(-ce_loss_b)
    focal_a = alpha * (1 - pt_a) ** gamma * ce_loss_a
    focal_b = alpha * (1 - pt_b) ** gamma * ce_loss_b
    
    weights_a = weights[labels_a]
    weights_b = weights[labels_b]
    
    loss_a = (focal_a * weights_a).sum() / (weights_a.sum() + 1e-8)
    loss_b = (focal_b * weights_b).sum() / (weights_b.sum() + 1e-8)
    
    lam = mix_ratios.mean().clamp(0, 1)
    loss = lam * loss_a + (1 - lam) * loss_b
    
    if DEBUG:
        print(f"[DEBUG soft_label_loss] loss: {loss.item():.4f}")
    
    return loss

def attention_entropy_loss(attn_weights):
    entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=1).mean()
    return entropy

def worker_init_fn(worker_id, seed=42):
    """Module-level worker init function for multiprocessing compatibility"""
    random.seed(seed + worker_id)


def train_single_fold(test_plate):
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training fold: test_plate={test_plate}")
    print(f"{'='*60}")
    
    train_val_plates = [p for p in all_plates if p != test_plate]
    train_plates = train_val_plates[:4]
    val_plates = train_val_plates[4:]
    
    print(f"Train plates: {train_plates}")
    print(f"Val plates: {val_plates}")
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    for plate in train_plates:
        for path in get_image_paths_for_plate(plate):
            train_paths.append(path)
            train_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    for plate in val_plates:
        for path in get_image_paths_for_plate(plate):
            val_paths.append(path)
            val_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    for plate in [test_plate]:
        for path in get_image_paths_for_plate(plate):
            test_paths.append(path)
            test_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    class_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor([total / (num_classes * class_counts[i]) for i in range(num_classes)], device=device)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    train_dataset = MultiCropDataset(train_paths, train_labels, plate_maps, 
                                  crop_size=args.crop_size, grid_size=args.grid_size,
                                  neighborhood=args.neighborhood,
                                  augment=True, seed=SEED)
    val_dataset = MultiCropDataset(val_paths, val_labels, plate_maps,
                                  crop_size=args.crop_size, grid_size=args.grid_size,
                                  neighborhood=args.neighborhood,
                                  augment=False, seed=SEED)
    test_dataset = MultiCropDataset(test_paths, test_labels, plate_maps,
                                  crop_size=args.crop_size, grid_size=args.grid_size,
                                  neighborhood=args.neighborhood,
                                  augment=False, seed=SEED)
    
    train_dataset.set_epoch(0)
    val_dataset.set_epoch(0)
    test_dataset.set_epoch(0)
    
    # Windows Python 3.14: MUST use 0 workers due to multiprocessing pickling changes
    if sys.platform.startswith('win'):
        effective_workers = 0
        print(f"Using {effective_workers} workers (Windows Python 3.14 - multiprocessing spawn required)")
    else:
        effective_workers = NUM_WORKERS
        print(f"Using {effective_workers} workers")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=effective_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=effective_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=effective_workers, pin_memory=True)
    
    # Calculate crops per image based on neighborhood
    crops_per_image = args.neighborhood * args.neighborhood
    neighborhood_label = f"{args.neighborhood}x{args.neighborhood}"
    
    print(f"Crops per image: {crops_per_image} ({neighborhood_label} neighborhood)")
    if args.use_psemix:
        print(f"PseMix: {args.psemix_mode}, n_pseb={args.psemix_n_pseb}, n_pheno={args.psemix_n_pheno}")
        bag_mixer = create_bag_mixer(
            mode=args.psemix_mode,
            use_psemix=True,
            n_pseb=args.psemix_n_pseb,
            n_pheno=args.psemix_n_pheno,
            alpha=args.psemix_alpha,
            prob_mixup=args.psemix_prob
        )
    else:
        print(f"BagMix: {args.bag_mix}")
        bag_mixer = create_bag_mixer(
            mode=args.bag_mix,
            mix_ratio=args.bag_mix_ratio,
            subset_size=args.bag_mix_subset_size,
dropout_ratio=args.bag_mix_dropout,
            alpha=args.bag_mix_alpha
        )
    
    # Create MAMMOTH if enabled
    mammoth = None
    if getattr(args, 'use_mammoth', False):
        # mammMOTH as drop-in replacement for linear layer (1280 -> 512)
        mammoth = create_mammoth(args, input_dim=1280, embed_dim=512)
        print(f"Using MAMMOTH: {args.mammoth_num_experts} experts, {args.mammoth_num_slots} slots")
    
    model = AttentionMILModel(num_classes=num_classes, num_heads=args.num_heads, mammoth=mammoth)
    model = model.to(device)
    
    backbone_params = [p for n, p in model.named_parameters() if 'attention_pool' not in n and 'classifier' not in n]
    attention_params = [p for n, p in model.named_parameters() if 'attention_pool' in n or 'classifier' in n]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': attention_params, 'lr': args.lr}
    ], weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 0
    if args.resume:
        resume_path = os.path.join(OUTPUT_DIR, args.resume)
        if os.path.exists(resume_path):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming from epoch {checkpoint['epoch']}")
        else:
            print(f"WARNING: Resume checkpoint not found: {resume_path}")
    
    if args.resume_epoch:
        resume_path = os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{args.resume_epoch}.pth')
        if os.path.exists(resume_path):
            print(f"Resuming from epoch {args.resume_epoch}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            print(f"WARNING: Resume checkpoint not found: {resume_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f'training_metrics_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc', 'backbone_lr', 'classifier_lr'])
    
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    print("Training...")
    use_psemix = getattr(args, 'use_psemix', False)
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_dataset.set_epoch(epoch)
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            if use_psemix and args.bag_mix_prob > 0 and random.random() < args.bag_mix_prob:
                with torch.no_grad():
                    features = model.extract_features(images)
                
                mixed_features, labels_a, labels_b, mix_ratios = bag_mixer.apply_psemix_to_batch(features, labels)
                
                if DEBUG:
                    print(f"[DEBUG LOOP] features: {features.shape}, mixed: {mixed_features.shape}")
                    print(f"[DEBUG LOOP] labels_a: {labels_a.shape}, labels_b: {labels_b.shape}")
                
                optimizer.zero_grad()
                ret = model.forward_with_features(mixed_features, return_attention=False)
                if isinstance(ret, tuple):
                    outputs, attn_w = ret
                else:
                    outputs = ret
                
                if DEBUG:
                    print(f"[DEBUG LOOP] type(ret)={type(ret)}, outputs.shape={outputs.shape}")
                
                main_loss = soft_label_loss(outputs, labels_a, labels_b, mix_ratios, class_weights)
            else:
                if args.bag_mix != 'none' and random.random() < args.bag_mix_prob:
                    batch_size = images.shape[0]
                    aug_images = []
                    for i in range(batch_size):
                        aug_crops = bag_mixer(images[i])
                        aug_images.append(aug_crops)
                    images = torch.stack(aug_images)
                
                optimizer.zero_grad()
                
                outputs, attn_weights = model(images, return_attention=True)
                
                main_loss = weighted_focal_loss(outputs, labels, class_weights[labels])
            
            loss = main_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            run_loss += main_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        train_acc = 100. * correct / total
        avg_train_loss = run_loss / len(train_loader)
        
        model.eval()
        val_loss_total = 0.0
        all_preds, all_probs, all_labels = [], [], []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validating', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images, return_attention=True)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_loss = weighted_focal_loss(outputs, labels, class_weights[labels])
                val_loss_total += val_loss.item()
        
        val_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        val_auc = roc_auc_score(all_labels_bin, np.array(all_probs), average='macro')
        avg_val_loss = val_loss_total / len(val_loader)
        
        backbone_lr = optimizer.param_groups[0]['lr']
        classifier_lr = optimizer.param_groups[1]['lr']
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}, Backbone LR={backbone_lr:.2e}, Classifier LR={classifier_lr:.2e}, Time={time.time()-epoch_start:.1f}s")
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, val_auc, backbone_lr, classifier_lr])
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_auc.pth'))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_acc.pth'))
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_loss.pth'))
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'val_acc': best_val_acc,
                'val_loss': best_val_loss
            }, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("Testing...")
    if args.checkpoint_type == 'auc':
        checkpoint_file = 'best_model.pth'
    elif args.checkpoint_type == 'acc':
        checkpoint_file = 'best_model_acc.pth'
    else:  # loss
        checkpoint_file = 'best_model_loss.pth'
    
    print(f"Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, checkpoint_file), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing', leave=False):
            images = images.to(device)
            outputs, _ = model(images, return_attention=True)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    test_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    test_auc = roc_auc_score(test_labels_bin, np.array(all_probs), average='macro')
    test_ap = average_precision_score(test_labels_bin, np.array(all_probs), average='macro')
    
    print(f"Test Acc: {test_acc:.2f}%, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
    results = {
        'timestamp': timestamp,
        'config': {'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr, 'test_plate': test_plate},
        'results': {'best_val_auc': float(best_val_auc), 'test_acc': float(test_acc), 'test_auc': float(test_auc), 'test_ap': float(test_ap)}
    }
    
    with open(os.path.join(OUTPUT_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    if args.run_all_folds:
        for test_plate in all_plates:
            fold_dir = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
            
            # Check for any checkpoint files to skip trained folds
            checkpoints = [
                os.path.join(fold_dir, 'best_model.pth'),
                os.path.join(fold_dir, 'best_model_acc.pth'),
                os.path.join(fold_dir, 'best_model_auc.pth'),
                os.path.join(fold_dir, 'best_model_loss.pth'),
            ]
            
            if any(os.path.exists(cp) for cp in checkpoints):
                print(f"\nSkipping {test_plate}: already trained (checkpoint exists)")
                continue
            
            train_single_fold(test_plate)
        
        print("All folds completed!")
    else:
        train_single_fold(args.test_plate)
    
    print("Done!")
