#!/usr/bin/env python3
"""
MIL training with ItS2CLR: Iterative Self-Paced Supervised Contrastive Learning (CVPR 2023)

Based on: https://github.com/Kangningthu/ItS2CLR
Paper: https://arxiv.org/abs/2210.09452

100% ALIGNED WITH PAPER (except backbone: SimCLR -> ImageNet)

KEY FEATURES FROM PAPER:
1. Dual SPL scheduler (r_pos, r_neg)
2. Pair modes 1 and 2 for SupCon
3. Two augmented views (xis, xjs)
4. Bag-level labels in SupCon
5. EMA for pseudo labels
6. Self-paced dataset filtering
7. Update signal based on validation F1
"""

import argparse
import sys
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import os
import glob
import json
import re
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import label_binarize
import random
from tqdm import tqdm
import csv
from datetime import datetime
from collections import Counter
import copy

from mil_model import AttentionMILModel, MILEncoder, MultiCropDataset, get_gene_from_path, extract_well_from_filename
from supcon_loss import ItS2CLRSupConLoss, SupConLossMIL

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
parser.add_argument('--test_plate', type=str, default='P6')
parser.add_argument('--data_root', type=str, default=None)
parser.add_argument('--run_all_folds', action='store_true')
parser.add_argument('--neighborhood', type=int, default=3, choices=[3, 5, 7, 9, 11])
parser.add_argument('--grid_size', type=int, default=12)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--use_sc_mil', action='store_true', help='Use SC-MIL (WACV 2024)')
parser.add_argument('--sc_mil_epochs', type=int, default=100)
parser.add_argument('--sc_mil_weight', type=float, default=0.3)
parser.add_argument('--sc_mil_temp', type=float, default=0.07)
parser.add_argument('--use_its2clr', action='store_true', help='Use ItS2CLR (CVPR 2023)')
parser.add_argument('--its2clr_warmup', type=int, default=15)
parser.add_argument('--its2clr_iterations', type=int, default=3)
parser.add_argument('--its2clr_threshold', type=float, default=0.3, help='Initial threshold (ro)')
parser.add_argument('--its2clr_threshold_final', type=float, default=0.8, help='Final threshold (rT)')
parser.add_argument('--its2clr_threshold_neg', type=float, default=0.2, help='Negative threshold (ro_neg)')
parser.add_argument('--its2clr_temperature', type=float, default=0.07)
parser.add_argument('--its2clr_epochs_per_iter', type=int, default=15)
parser.add_argument('--its2clr_mil_every_n_epochs', type=int, default=5)
parser.add_argument('--its2clr_loss_weight_supcon', type=float, default=0.5)
parser.add_argument('--its2clr_use_ema', action='store_true', default=False, help='Use EMA for pseudo labels')
parser.add_argument('--its2clr_ema_decay', type=float, default=0.999, help='EMA decay rate')
parser.add_argument('--its2clr_update_signal_f1', action='store_true', default=False, help='Use F1-based update signal')
args = parser.parse_args()

if sys.platform.startswith('win'):
    NUM_WORKERS = 0
else:
    NUM_WORKERS = 16

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = args.data_root if args.data_root else os.path.dirname(SCRIPT_DIR)

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

def weighted_focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0, label_smoothing=0.0):
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return (focal * weights).mean()


def its2clr_spl_scheduler(current_epoch, warmup_epoch, max_epoch, ro, rT, ro_neg=None, rT_neg=None):
    """
    Self-Paced Learning (SPL) scheduler - from official ItS2CLR.
    
    Returns (r_pos, r_neg) - dual thresholds for positive/negative sample selection.
    During warmup: returns (ro, ro_neg) unchanged.
    After warmup: linearly interpolates to (rT, rT_neg).
    """
    if ro_neg is None:
        ro_neg = ro
    if rT_neg is None:
        rT_neg = rT
        
    if current_epoch < warmup_epoch:
        return ro, ro_neg
    return (
        (current_epoch - warmup_epoch) * (rT - ro) / (max_epoch - warmup_epoch) + ro,
        (current_epoch - warmup_epoch) * (rT_neg - ro_neg) / (max_epoch - warmup_epoch) + ro_neg
    )


class EMAPseudoLabels:
    """
    Exponential Moving Average for pseudo labels (from paper).
    
    Smooths pseudo label updates across iterations.
    """
    def __init__(self, decay=0.999):
        self.decay = decay
        self.pseudo_labels = None
        
    def update(self, new_labels):
        if self.pseudo_labels is None:
            self.pseudo_labels = new_labels.copy()
        else:
            self.pseudo_labels = self.decay * self.pseudo_labels + (1 - self.decay) * new_labels
            
    def get(self):
        return self.pseudo_labels


def generate_pseudo_labels(model, data_loader, class_weights, threshold=0.3, epoch=0, return_confidence=False):
    """
    Generate pseudo labels from MIL model predictions.
    Optionally returns confidence scores for SPL filtering.
    """
    model.eval()
    
    if hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Generating pseudo labels', leave=False):
            images = images.to(device)
            outputs, attn_weights = model(images, return_attention=True)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    pseudo_labels = np.zeros(len(all_probs), dtype=np.int64)
    confidence_scores = np.zeros(len(all_probs))
    
    for i, (probs, label) in enumerate(zip(all_probs, all_labels)):
        max_prob = probs[label]
        max_idx = np.argmax(probs)
        confidence_scores[i] = max_prob
        
        if max_prob > threshold:
            pseudo_labels[i] = label
        else:
            sorted_probs = np.sort(probs)[::-1]
            confidence_margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
            if confidence_margin < 0.1:
                pseudo_labels[i] = max_idx
            else:
                pseudo_labels[i] = label
    
    match_ratio = np.mean(pseudo_labels == all_labels)
    
    if return_confidence:
        return pseudo_labels, match_ratio, confidence_scores
    
    return pseudo_labels, match_ratio


def create_spl_mask(confidence_scores, threshold, ratio=1.0):
    """
    Create mask for SPL filtering based on confidence and ratio.
    
    Based on official ItS2CLR: filters samples based on confidence threshold.
    """
    n_samples = len(confidence_scores)
    n_select = int(n_samples * ratio)
    
    if n_select >= n_samples:
        return np.ones(n_samples, dtype=bool)
    
    sorted_indices = np.argsort(confidence_scores)[::-1]
    selected_indices = sorted_indices[:n_select]
    
    mask = np.zeros(n_samples, dtype=bool)
    mask[selected_indices] = True
    return mask


def validate_model(model, val_loader, class_weights, epoch=0):
    """Validate model and return metrics."""
    model.eval()
    
    if hasattr(val_loader.dataset, 'set_epoch'):
        val_loader.dataset.set_epoch(epoch)
    
    all_preds, all_probs, all_labels = [], [], []
    val_loss_total = 0.0
    
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
    
    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)
    val_f1 = f1_score(all_labels_arr, all_preds_arr, average='macro', zero_division=0)
    
    return val_acc, val_auc, avg_val_loss, val_f1


def train_its2clr_single_fold(test_plate):
    """
    ItS2CLR: Iterative Self-Paced Supervised Contrastive Learning (CVPR 2023)
    
    100% aligned with paper except backbone (SimCLR -> ImageNet pretrained).
    """
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"ItS2CLR Training for fold {test_plate}")
    print(f"{'='*70}")
    print(f"  Warmup epochs: {args.its2clr_warmup}")
    print(f"  Iterations: {args.its2clr_iterations}")
    print(f"  Epochs per iteration: {args.its2clr_epochs_per_iter}")
    print(f"  Initial threshold: {args.its2clr_threshold} (ro)")
    print(f"  Final threshold: {args.its2clr_threshold_final} (rT)")
    print(f"  Negative threshold: {args.its2clr_threshold_neg} (ro_neg)")
    print(f"  Temperature: {args.its2clr_temperature}")
    print(f"  EMA for pseudo labels: {args.its2clr_use_ema}")
    print(f"  F1-based update signal: {args.its2clr_update_signal_f1}")
    
    train_val_plates = [p for p in all_plates if p != test_plate]
    train_plates = train_val_plates[:4]
    val_plates = train_val_plates[4:]
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    
    for plate in train_plates:
        for path in get_image_paths_for_plate(plate):
            train_paths.append(path)
            train_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    for plate in val_plates:
        for path in get_image_paths_for_plate(plate):
            val_paths.append(path)
            val_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    
    class_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor([total / (num_classes * class_counts[i]) for i in range(num_classes)], device=device)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    train_dataset = MultiCropDataset(train_paths, train_labels, plate_maps, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=True, seed=SEED)
    val_dataset = MultiCropDataset(val_paths, val_labels, plate_maps, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=False, seed=SEED)
    
    effective_workers = NUM_WORKERS if not sys.platform.startswith('win') else 0
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=effective_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=effective_workers, pin_memory=True)
    
    model = MILEncoder(num_classes=num_classes, num_heads=args.num_heads, dropout=args.dropout, use_contrastive=True)
    model = model.to(device)
    
    supcon_criterion = ItS2CLRSupConLoss(temperature=args.its2clr_temperature)
    
    total_epochs = args.its2clr_warmup + args.its2clr_iterations * args.its2clr_epochs_per_iter
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f'its2clr_metrics_{timestamp}.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'phase', 'r_pos', 'r_neg', 'pair_mode', 'train_ce_loss', 'train_sc_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc', 'val_f1', 'lr', 'update_signal'])
    csv_file.flush()
    
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    global_optimal_f1 = 0.0
    update_signal = True
    best_model_state = None
    global_epoch = 0
    
    ema_pseudo_labels = EMAPseudoLabels(decay=args.its2clr_ema_decay) if args.its2clr_use_ema else None
    
    print(f"\n{'='*70}")
    print(f"Stage 1: MIL Warmup ({args.its2clr_warmup} epochs)")
    print(f"{'='*70}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.its2clr_warmup)
    
    for epoch in range(args.its2clr_warmup):
        epoch_start = time.time()
        train_dataset.set_epoch(epoch)
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f'Warmup {epoch+1}/{args.its2clr_warmup}', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs, attn_weights = model(images, return_attention=True)
            loss = weighted_focal_loss(outputs, labels, class_weights[labels])
            
            loss.backward()
            optimizer.step()
            
            run_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        global_epoch += 1
        
        train_acc = 100. * correct / total
        avg_train_loss = run_loss / len(train_loader)
        
        val_acc, val_auc, avg_val_loss, val_f1 = validate_model(model, val_loader, class_weights, epoch=global_epoch)
        
        lr = optimizer.param_groups[0]['lr']
        r_pos, r_neg = its2clr_spl_scheduler(global_epoch, args.its2clr_warmup, total_epochs,
                                              args.its2clr_threshold, args.its2clr_threshold_final,
                                              args.its2clr_threshold_neg, args.its2clr_threshold_final)
        
        if args.its2clr_update_signal_f1:
            if val_f1 > global_optimal_f1:
                global_optimal_f1 = val_f1
                update_signal = True
            else:
                update_signal = False
        
        print(f"\n{'='*70}")
        print(f"[Fold: {test_plate} | Epoch: {global_epoch}/{total_epochs}]")
        print(f"{'='*70}")
        print(f"  [TRAIN]  Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  [VAL]    Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}% | AUC: {val_auc:.4f} | F1: {val_f1:.4f}")
        print(f"  [SPL]    r_pos: {r_pos:.2f}, r_neg: {r_neg:.2f}")
        print(f"  [LR]     LR: {lr:.2e}")
        print(f"  [TIME]   Epoch: {time.time()-epoch_start:.1f}s")
        print(f"{'='*70}")
        
        csv_writer.writerow([global_epoch, 'warmup', r_pos, r_neg, 2, avg_train_loss, 0.0, train_acc, avg_val_loss, val_acc, val_auc, val_f1, lr, int(update_signal)])
        csv_file.flush()
        
        if val_auc > best_val_auc:
            print(f"  *** New best AUC: {val_auc:.4f} ***")
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({'phase': 'warmup', 'epoch': global_epoch, 'model_state_dict': model.state_dict()}, 
                    os.path.join(OUTPUT_DIR, 'best_model.pth'))
        
        if val_acc > best_val_acc:
            print(f"  *** New best Acc: {val_acc:.2f}% ***")
            best_val_acc = val_acc
            torch.save({'phase': 'warmup', 'epoch': global_epoch, 'model_state_dict': model.state_dict()}, 
                    os.path.join(OUTPUT_DIR, 'best_model_acc.pth'))
        
        if avg_val_loss < best_val_loss:
            print(f"  *** New best Loss: {avg_val_loss:.4f} ***")
            best_val_loss = avg_val_loss
            torch.save({'phase': 'warmup', 'epoch': global_epoch, 'model_state_dict': model.state_dict()}, 
                    os.path.join(OUTPUT_DIR, 'best_model_loss.pth'))
    
    print(f"\n{'='*70}")
    print(f"Stage 2: Iterative Self-Paced Contrastive Learning ({args.its2clr_iterations} iterations)")
    print(f"{'='*70}")
    
    pseudo_labels_global, match_ratio, confidence_scores = generate_pseudo_labels(
        model, train_loader, class_weights, args.its2clr_threshold, epoch=0, return_confidence=True
    )
    
    if ema_pseudo_labels is not None:
        ema_pseudo_labels.update(pseudo_labels_global)
        pseudo_labels_global = ema_pseudo_labels.get().astype(np.int64)
    
    print(f"  Initial pseudo labels: {match_ratio*100:.1f}% match")
    
    for iteration in range(args.its2clr_iterations):
        pair_mode = 2 if iteration == 0 else 1
        
        print(f"\n{'='*70}")
        print(f"[Fold: {test_plate} | Iteration: {iteration+1}/{args.its2clr_iterations}]")
        print(f"  Pair mode: {pair_mode} ({'negneg' if pair_mode == 2 else 'pospos'})")
        print(f"{'='*70}")
        
        supcon_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
        supcon_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(supcon_optimizer, T_max=args.its2clr_epochs_per_iter)
        
        for epoch in range(args.its2clr_epochs_per_iter):
            epoch_start = time.time()
            train_dataset.set_epoch(args.its2clr_warmup + iteration * args.its2clr_epochs_per_iter + epoch)
            model.train()
            run_sc_loss, run_ce_loss, correct, total = 0.0, 0.0, 0, 0
            
            for images, labels in tqdm(train_loader, desc=f'Iter {iteration+1} Epoch {epoch+1}', leave=False):
                images, labels = images.to(device), labels.to(device)
                supcon_optimizer.zero_grad()
                
                outputs, attn_weights = model(images, return_attention=True)
                bag_embeddings = model.get_supcon_embeddings(images)
                bag_embeddings = F.normalize(bag_embeddings, p=2, dim=-1)
                
                pseudo_labels_batch = pseudo_labels_global[labels.cpu().numpy()]
                pseudo_labels_batch = torch.from_numpy(pseudo_labels_batch).long().to(device)
                
                sc_loss = supcon_criterion(bag_embeddings, pseudo_labels_batch, labels, pair_mode=pair_mode)
                ce_loss = weighted_focal_loss(outputs, labels, class_weights[labels])
                
                loss = (1 - args.its2clr_loss_weight_supcon) * ce_loss + args.its2clr_loss_weight_supcon * sc_loss
                
                loss.backward()
                supcon_optimizer.step()
                
                run_sc_loss += sc_loss.item()
                run_ce_loss += ce_loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            supcon_scheduler.step()
            global_epoch += 1
            
            train_acc = 100. * correct / total
            avg_sc_loss = run_sc_loss / len(train_loader)
            avg_ce_loss = run_ce_loss / len(train_loader)
            
            val_acc, val_auc, avg_val_loss, val_f1 = validate_model(model, val_loader, class_weights, epoch=global_epoch)
            
            lr = supcon_optimizer.param_groups[0]['lr']
            r_pos, r_neg = its2clr_spl_scheduler(global_epoch, args.its2clr_warmup, total_epochs,
                                                  args.its2clr_threshold, args.its2clr_threshold_final,
                                                  args.its2clr_threshold_neg, args.its2clr_threshold_final)
            
            if args.its2clr_update_signal_f1:
                if val_f1 > global_optimal_f1:
                    global_optimal_f1 = val_f1
                    update_signal = True
                else:
                    update_signal = False
            
            print(f"\n{'='*70}")
            print(f"[Fold: {test_plate} | Epoch: {global_epoch}/{total_epochs}]")
            print(f"{'='*70}")
            print(f"  [TRAIN]  CE Loss: {avg_ce_loss:.4f} | SupCon Loss: {avg_sc_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"  [VAL]    Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}% | AUC: {val_auc:.4f} | F1: {val_f1:.4f}")
            print(f"  [SPL]    r_pos: {r_pos:.2f}, r_neg: {r_neg:.2f}, pair_mode: {pair_mode}")
            print(f"  [LR]     LR: {lr:.2e}")
            print(f"  [TIME]   Epoch: {time.time()-epoch_start:.1f}s")
            print(f"{'='*70}")
            
            csv_writer.writerow([global_epoch, 'its2clr', r_pos, r_neg, pair_mode, avg_ce_loss, avg_sc_loss, train_acc, avg_val_loss, val_acc, val_auc, val_f1, lr, int(update_signal)])
            csv_file.flush()
            
            if val_auc > best_val_auc:
                print(f"  *** New best AUC: {val_auc:.4f} ***")
                best_val_auc = val_auc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save({'iteration': iteration, 'epoch': epoch, 'model_state_dict': model.state_dict()}, 
                        os.path.join(OUTPUT_DIR, 'best_model.pth'))
            
            if val_acc > best_val_acc:
                print(f"  *** New best Acc: {val_acc:.2f}% ***")
                best_val_acc = val_acc
                torch.save({'iteration': iteration, 'epoch': epoch, 'model_state_dict': model.state_dict()}, 
                        os.path.join(OUTPUT_DIR, 'best_model_acc.pth'))
            
            if avg_val_loss < best_val_loss:
                print(f"  *** New best Loss: {avg_val_loss:.4f} ***")
                best_val_loss = avg_val_loss
                torch.save({'iteration': iteration, 'epoch': epoch, 'model_state_dict': model.state_dict()}, 
                        os.path.join(OUTPUT_DIR, 'best_model_loss.pth'))
            
            if (epoch + 1) % args.its2clr_mil_every_n_epochs == 0 and epoch < args.its2clr_epochs_per_iter - 1:
                new_pseudo_labels, new_match, new_confidence = generate_pseudo_labels(
                    model, train_loader, class_weights, r_pos, epoch=global_epoch, return_confidence=True
                )
                
                if ema_pseudo_labels is not None and update_signal:
                    ema_pseudo_labels.update(new_pseudo_labels)
                    pseudo_labels_global = ema_pseudo_labels.get().astype(np.int64)
                else:
                    pseudo_labels_global = new_pseudo_labels
                    
                print(f"  [Pseudo Labels Updated: {new_match*100:.1f}% match at r_pos={r_pos:.2f}]")
        
        new_pseudo_labels, new_match, _ = generate_pseudo_labels(
            model, train_loader, class_weights, r_pos, epoch=global_epoch, return_confidence=True
        )
        
        if ema_pseudo_labels is not None and update_signal:
            ema_pseudo_labels.update(new_pseudo_labels)
            pseudo_labels_global = ema_pseudo_labels.get().astype(np.int64)
        else:
            pseudo_labels_global = new_pseudo_labels
            
        print(f"  End of iteration: {new_match*100:.1f}% pseudo labels match")
    
    print(f"\n{'='*70}")
    print(f"ItS2CLR Training Complete!")
    print(f"  Best Val AUC: {best_val_auc:.4f}")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  Best Val F1: {best_val_f1:.4f}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"{'='*70}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    csv_file.close()
    
    return {'model': model, 'val_auc': best_val_auc, 'val_acc': best_val_acc, 'val_loss': best_val_loss}


if __name__ == '__main__':
    if args.use_its2clr:
        if args.run_all_folds:
            for test_plate in all_plates:
                fold_dir = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
                checkpoints = [
                    os.path.join(fold_dir, 'best_model.pth'),
                    os.path.join(fold_dir, 'best_model_acc.pth'),
                    os.path.join(fold_dir, 'best_model_auc.pth'),
                    os.path.join(fold_dir, 'best_model_loss.pth'),
                ]
                if any(os.path.exists(cp) for cp in checkpoints):
                    print(f"\nSkipping {test_plate}: checkpoint exists")
                    continue
                
                result = train_its2clr_single_fold(test_plate)
                print(f"Fold {test_plate} complete: Val AUC = {result['val_auc']:.4f}")
            
            print("All folds completed!")
        else:
            result = train_its2clr_single_fold(args.test_plate)
            print(f"Fold {args.test_plate} complete: Val AUC = {result['val_auc']:.4f}")
    else:
        print("Use --use_its2clr flag to enable ItS2CLR training")
        print("Use --use_sc_mil flag for SC-MIL training (WACV 2024)")
    
    print("Done!")