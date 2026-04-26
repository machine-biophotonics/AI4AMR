#!/usr/bin/env python3
"""
MIL training with cycle-based crop extraction + neighbors
Training: configurable crops (3x3, 5x5, 7x7, 9x9, 11x11 neighborhood)
Validation/Test: single center crop
Supports --run_all_folds for cross-validation

Key improvements:
- Single forward pass per batch (no double backbone calls)
- Token dropout for regularization (ASMIL-style)
- EMA anchor for attention stabilization (ASMIL-style)
- AEM (Attention Entropy Maximization) regularization
- Per-level validation accuracy (gene, family, pathway)
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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
import random
from tqdm import tqdm
import csv
from datetime import datetime
from collections import Counter

from mil_model import AttentionMILModel, MILEncoder, MultiCropDataset, get_gene_from_path, extract_well_from_filename
from supcon_loss import SupConLoss, SupConLossMIL
from hierarchical_supcon_loss import HierarchicalSupConLossMIL

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
parser.add_argument('--data_root', type=str, default=None, help='Path to folder containing P1-P6 plate folders')
parser.add_argument('--run_all_folds', action='store_true', help='Run all 6 folds')
parser.add_argument('--neighborhood', type=int, default=3, choices=[3, 5, 7, 9, 11],
                    help='Neighborhood size: 3=(3x3=9 crops), 5=(5x5=25 crops), 7=(7x7=49 crops)')
parser.add_argument('--grid_size', type=int, default=12, help='Grid size for crop positions')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for classifier (default 0.5)')
parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay (default 0.05)')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing (default 0.1)')
parser.add_argument('--use_contrastive', action='store_true', help='Use patch-level contrastive pre-training')
parser.add_argument('--use_sc_mil', action='store_true', help='Use SC-MIL (recommended)')
parser.add_argument('--sc_mil_epochs', type=int, default=100, help='SC-MIL epochs (default 100)')
parser.add_argument('--sc_mil_weight', type=float, default=0.3, help='SC-MIL contrastive weight')
parser.add_argument('--sc_mil_temp', type=float, default=0.07, help='SC-MIL temperature')
parser.add_argument('--use_hierarchical', action='store_true', help='Use hierarchical contrastive (Gene/Family/Pathway)')
parser.add_argument('--hierarchical_weights', type=str, default='1.0,0.5,0.2,0.1',
                    help='Weights: guide,gene,family,pathway (default 1.0,0.5,0.2,0.1)')
parser.add_argument('--aem_weight', type=float, default=0.01,
                    help='AEM (Attention Entropy Maximization) weight for regularization (default 0.01)')
parser.add_argument('--aem_anneal', action='store_true', default=True,
                    help='Cosine anneal AEM weight (default True)')
parser.add_argument('--ema_momentum', type=float, default=0.99,
                    help='EMA momentum for anchor attention (default 0.99)')
parser.add_argument('--token_dropout_p', type=float, default=0.3,
                    help='Token dropout probability (default 0.3, set 0 to disable)')
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

def weighted_focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0, label_smoothing=0.0):
    ce_loss = F.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return (focal * weights).mean()

def aem_loss(attn_weights):
    """Attention Entropy Maximization - penalize over-concentrated attention.
    
    Maximizes entropy to prevent the model from focusing on only a few instances.
    AEM weight is annealed (cosine schedule) to start gentle and grow.
    """
    entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=1).mean()
    return -entropy

def kl_divergence_anchor(attn_weights, anchor_attn):
    """KL divergence between current attention and EMA anchor.
    
    Stabilizes attention dynamics by encouraging the online model to match
    the smoothed EMA anchor distribution. Returns 0 if anchor not yet initialized.
    """
    if anchor_attn is None:
        return torch.tensor(0.0, device=attn_weights.device)
    attn_mean = attn_weights.mean(dim=0)
    safe_attn = attn_mean.clamp(min=1e-8)
    safe_anchor = anchor_attn.clamp(min=1e-8)
    safe_anchor = safe_anchor / safe_anchor.sum(dim=-1, keepdim=True)
    kl = safe_attn * (safe_attn / safe_anchor + 1e-8).log()
    return kl.sum(dim=-1).mean()

def worker_init_fn(worker_id, seed=42):
    random.seed(seed + worker_id)


def compute_hierarchy_predictions(model, val_loader, mappings_path):
    import json
    
    with open(mappings_path, 'r') as f:
        mapping_data = json.load(f)
    gene_mappings = mapping_data.get('mappings', {})
    families = mapping_data.get('families', {})
    pathways = mapping_data.get('pathways', {})
    
    all_preds = {'guide': [], 'gene': [], 'family': [], 'pathway': []}
    all_labels = {'guide': [], 'gene': [], 'family': [], 'pathway': []}
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs, _, _ = model(images, return_attention=True, return_crop_embeddings=True)
            _, predicted = outputs.max(1)
            
            for pred, label_idx in zip(predicted.cpu().numpy(), labels.numpy()):
                guide_name = all_genes[label_idx]
                pred_guide = all_genes[pred]
                
                all_labels['guide'].append(guide_name)
                all_preds['guide'].append(pred_guide)
                
                gene_name = guide_name
                if '_' in gene_name:
                    suffix = gene_name.rsplit('_', 1)[1]
                    if suffix in ('1', '2', '3', 'a', 'b', 'c'):
                        gene_name = gene_name.rsplit('_', 1)[0]
                
                pred_gene = pred_guide
                if '_' in pred_gene:
                    suffix = pred_gene.rsplit('_', 1)[1]
                    if suffix in ('1', '2', '3', 'a', 'b', 'c'):
                        pred_gene = pred_gene.rsplit('_', 1)[0]
                
                all_labels['gene'].append(gene_name)
                all_preds['gene'].append(pred_gene)
                
                family = gene_mappings.get(gene_name, {}).get('family', 'Unknown')
                pred_family = gene_mappings.get(pred_gene, {}).get('family', 'Unknown')
                all_labels['family'].append(family)
                all_preds['family'].append(pred_family)
                
                pathway = gene_mappings.get(gene_name, {}).get('pathway', 'Unknown')
                pred_pathway = gene_mappings.get(pred_gene, {}).get('pathway', 'Unknown')
                all_labels['pathway'].append(pathway)
                all_preds['pathway'].append(pred_pathway)
    
    acc = {}
    for level in ['guide', 'gene', 'family', 'pathway']:
        correct = sum(1 for p, l in zip(all_preds[level], all_labels[level]) if p == l)
        acc[level] = correct / len(all_labels[level]) if all_labels[level] else 0.0
    
    return acc


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
    
    train_dataset = MultiCropDataset(train_paths, train_labels, plate_maps, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=True, seed=SEED)
    val_dataset = MultiCropDataset(val_paths, val_labels, plate_maps, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=False, seed=SEED)
    test_dataset = MultiCropDataset(test_paths, test_labels, plate_maps, neighborhood=args.neighborhood, grid_size=args.grid_size, augment=False, seed=SEED)
    
    train_dataset.set_epoch(0)
    val_dataset.set_epoch(0)
    test_dataset.set_epoch(0)
    
    if sys.platform.startswith('win'):
        effective_workers = 0
    else:
        effective_workers = NUM_WORKERS
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=effective_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=effective_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=effective_workers, pin_memory=True)
    
    print(f"Crops per image: {args.neighborhood}x{args.neighborhood}={args.neighborhood**2} crops")
    
    if args.use_sc_mil:
        print(f"Using MILEncoder with SC-MIL + single forward pass...")
        print(f"  Token dropout p={args.token_dropout_p}")
        print(f"  EMA momentum={args.ema_momentum}")
        print(f"  AEM weight={args.aem_weight} (annealed: {args.aem_anneal})")
        model = MILEncoder(
            num_classes=num_classes,
            num_heads=args.num_heads,
            dropout=args.dropout,
            use_contrastive=True,
            token_dropout_p=args.token_dropout_p
        )
    else:
        model = AttentionMILModel(num_classes=num_classes, num_heads=args.num_heads, dropout=args.dropout)
    model = model.to(device)
    
    backbone_params = [p for n, p in model.named_parameters() if 'attention_pool' not in n and 'classifier' not in n]
    attention_params = [p for n, p in model.named_parameters() if 'attention_pool' in n or 'classifier' in n]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': attention_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f'training_metrics_{timestamp}.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    if args.use_sc_mil:
        if args.use_hierarchical:
            csv_writer.writerow([
                'epoch', 'train_ce_loss', 'train_sc_loss', 'train_aem_loss', 'train_acc',
                'val_ce_loss', 'val_acc', 'val_auc',
                'val_acc_guide', 'val_acc_gene', 'val_acc_family', 'val_acc_pathway',
                'sc_loss_guide', 'sc_loss_gene', 'sc_loss_family', 'sc_loss_pathway',
                'lr', 'time'
            ])
        else:
            csv_writer.writerow([
                'epoch', 'train_ce_loss', 'train_sc_loss', 'train_aem_loss', 'train_acc',
                'val_ce_loss', 'val_acc', 'val_auc', 'lr', 'time'
            ])
    else:
        csv_writer.writerow([
            'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc', 'backbone_lr', 'classifier_lr'
        ])
    csv_file.flush()
    
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    if args.use_sc_mil:
        print(f"\n{'='*60}")
        print(f"SC-MIL Training: Single Forward Pass per Batch")
        print(f"  Epochs: {args.sc_mil_epochs}, Temp: {args.sc_mil_temp}")
        print(f"  Contrastive weight: {args.sc_mil_weight}")
        print(f"  AEM weight: {args.aem_weight} (annealed: {args.aem_anneal})")
        print(f"  EMA momentum: {args.ema_momentum}")
        print(f"  Token dropout: {args.token_dropout_p}")
        print(f"{'='*60}")
        
        if args.use_hierarchical:
            w_guide, w_gene, w_family, w_pathway = map(float, args.hierarchical_weights.split(','))
            hierarchical_weights = {'guide': w_guide, 'gene': w_gene, 'family': w_family, 'pathway': w_pathway}
            print(f"Hierarchical SupCon weights: {hierarchical_weights}")
        
        mappings_path = os.path.join(SCRIPT_DIR, 'hierarchical_mappings.json')
        
        sc_mil_params = [p for n, p in model.named_parameters()]
        sc_mil_optimizer = torch.optim.AdamW(sc_mil_params, lr=args.lr, weight_decay=args.weight_decay)
        sc_mil_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sc_mil_optimizer, T_max=args.sc_mil_epochs)
        aem_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            torch.optim.Adam([torch.tensor(1.0)], lr=args.aem_weight),
            T_max=args.sc_mil_epochs
        ) if args.aem_anneal else None
        
        if args.use_hierarchical:
            sc_criterion = HierarchicalSupConLossMIL(
                temperature=args.sc_mil_temp,
                weights=hierarchical_weights,
                mappings_path=mappings_path
            )
        else:
            sc_criterion = SupConLossMIL(temperature=args.sc_mil_temp)
        
        model.ema_anchor.ema_momentum = args.ema_momentum
        
        for epoch in range(args.sc_mil_epochs):
            epoch_start = time.time()
            train_dataset.set_epoch(epoch)
            model.train()
            
            run_ce_loss, run_sc_loss, run_aem_loss = 0.0, 0.0, 0.0
            correct, total = 0, 0
            
            if args.use_hierarchical:
                run_sc_guide, run_sc_gene, run_sc_family, run_sc_pathway = 0.0, 0.0, 0.0, 0.0
            
            for images, labels in tqdm(train_loader, desc=f'SC-MIL Epoch {epoch}', leave=False):
                images, labels = images.to(device), labels.to(device)
                sc_mil_optimizer.zero_grad()
                
                outputs, attn_weights, crop_emb = model(
                    images, return_attention=True, return_crop_embeddings=True
                )
                
                crop_emb = model.token_dropout(crop_emb)
                crop_emb = F.normalize(crop_emb, p=2, dim=-1)
                
                if args.use_hierarchical:
                    gene_names = [all_genes[l] for l in labels.cpu().numpy()]
                    sc_loss, sc_metrics = sc_criterion(crop_emb, gene_names)
                    run_sc_guide += sc_metrics['loss_guide']
                    run_sc_gene += sc_metrics['loss_gene']
                    run_sc_family += sc_metrics['loss_family']
                    run_sc_pathway += sc_metrics['loss_pathway']
                else:
                    sc_loss = sc_criterion(crop_emb, labels)
                
                ce_loss = weighted_focal_loss(outputs, labels, class_weights[labels])
                
                current_aem_weight = args.aem_weight
                if args.aem_anneal:
                    progress = epoch / args.sc_mil_epochs
                    current_aem_weight = args.aem_weight * (0.5 + 0.5 * (1 + np.cos(np.pi * progress)))
                
                aem = aem_loss(attn_weights)
                kl_anchor = kl_divergence_anchor(attn_weights, model.ema_anchor.get_anchor())
                
                loss = (1 - args.sc_mil_weight) * ce_loss + args.sc_mil_weight * sc_loss
                loss = loss + current_aem_weight * aem + 0.1 * kl_anchor
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                sc_mil_optimizer.step()
                
                model.ema_anchor.update(attn_weights)
                sc_mil_scheduler.step()
                
                run_ce_loss += ce_loss.item()
                run_sc_loss += sc_loss.item()
                run_aem_loss += aem.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * correct / total
            avg_ce = run_ce_loss / len(train_loader)
            avg_sc = run_sc_loss / len(train_loader)
            avg_aem = run_aem_loss / len(train_loader)
            
            backbone_lr = sc_mil_optimizer.param_groups[0]['lr']
            
            model.eval()
            val_ce_loss = 0.0
            val_correct, val_total = 0, 0
            all_val_preds, all_val_probs, all_val_labels = [], [], []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs, _, _ = model(images, return_attention=True, return_crop_embeddings=True)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = outputs.max(1)
                    all_val_preds.extend(predicted.cpu().numpy())
                    all_val_probs.extend(probs.cpu().numpy())
                    all_val_labels.extend(labels.cpu().numpy())
                    val_loss = weighted_focal_loss(outputs, labels, class_weights[labels])
                    val_ce_loss += val_loss.item()
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = 100. * val_correct / val_total
            all_val_labels_bin = label_binarize(all_val_labels, classes=list(range(num_classes)))
            val_auc = roc_auc_score(all_val_labels_bin, np.array(all_val_probs), average='macro')
            avg_val_ce_loss = val_ce_loss / len(val_loader)
            
            if args.use_hierarchical:
                hier_acc = compute_hierarchy_predictions(model, val_loader, mappings_path)
                avg_sc_guide = run_sc_guide / len(train_loader)
                avg_sc_gene = run_sc_gene / len(train_loader)
                avg_sc_family = run_sc_family / len(train_loader)
                avg_sc_pathway = run_sc_pathway / len(train_loader)
                
                print(f"\n{'='*70}")
                print(f"[Fold: {test_plate} | SC-MIL Epoch: {epoch+1}/{args.sc_mil_epochs}]")
                print(f"{'='*70}")
                print(f"  [TRAIN] CE Loss: {avg_ce:.4f} | SC Loss: {avg_sc:.4f} | AEM: {avg_aem:.4f} | Acc: {train_acc:.2f}%")
                print(f"  [SC]    Guide: {avg_sc_guide:.4f} | Gene: {avg_sc_gene:.4f} | Family: {avg_sc_family:.4f} | Pathway: {avg_sc_pathway:.4f}")
                print(f"  [VAL]   CE Loss: {avg_val_ce_loss:.4f} | Acc: {val_acc:.2f}% | AUC: {val_auc:.4f}")
                print(f"  [VAL-LVL] Guide: {hier_acc['guide']*100:.1f}% | Gene: {hier_acc['gene']*100:.1f}% | Family: {hier_acc['family']*100:.1f}% | Pathway: {hier_acc['pathway']*100:.1f}%")
                print(f"  [LR]    {backbone_lr:.2e} | [TIME] {time.time()-epoch_start:.1f}s")
                print(f"{'='*70}")
                
                csv_writer.writerow([
                    epoch+1, f'{avg_ce:.6f}', f'{avg_sc:.6f}', f'{avg_aem:.6f}', f'{train_acc:.2f}',
                    f'{avg_val_ce_loss:.6f}', f'{val_acc:.2f}', f'{val_auc:.4f}',
                    f"{hier_acc['guide']*100:.2f}", f"{hier_acc['gene']*100:.2f}", f"{hier_acc['family']*100:.2f}", f"{hier_acc['pathway']*100:.2f}",
                    f'{avg_sc_guide:.6f}', f'{avg_sc_gene:.6f}', f'{avg_sc_family:.6f}', f'{avg_sc_pathway:.6f}',
                    f'{backbone_lr:.2e}', f'{time.time()-epoch_start:.1f}'
                ])
            else:
                print(f"\n{'='*70}")
                print(f"[Fold: {test_plate} | SC-MIL Epoch: {epoch+1}/{args.sc_mil_epochs}]")
                print(f"{'='*70}")
                print(f"  [TRAIN] CE Loss: {avg_ce:.4f} | SC Loss: {avg_sc:.4f} | AEM: {avg_aem:.4f} | Acc: {train_acc:.2f}%")
                print(f"  [VAL]   CE Loss: {avg_val_ce_loss:.4f} | Acc: {val_acc:.2f}% | AUC: {val_auc:.4f}")
                print(f"  [LR]    {backbone_lr:.2e} | [TIME] {time.time()-epoch_start:.1f}s")
                print(f"{'='*70}")
                
                csv_writer.writerow([
                    epoch+1, f'{avg_ce:.6f}', f'{avg_sc:.6f}', f'{avg_aem:.6f}', f'{train_acc:.2f}',
                    f'{avg_val_ce_loss:.6f}', f'{val_acc:.2f}', f'{val_auc:.4f}',
                    f'{backbone_lr:.2e}', f'{time.time()-epoch_start:.1f}'
                ])
            
            csv_file.flush()
            
            if val_auc > best_val_auc:
                print(f"  *** New best AUC: {val_auc:.4f} (prev: {best_val_auc:.4f}) ***")
                best_val_auc = val_auc
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model.pth'))
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_auc.pth'))
            
            if val_acc > best_val_acc:
                print(f"  *** New best Acc: {val_acc:.2f}% (prev: {best_val_acc:.2f}%) ***")
                best_val_acc = val_acc
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_acc.pth'))
            
            if avg_val_ce_loss < best_val_loss:
                print(f"  *** New best Loss: {avg_val_ce_loss:.4f} (prev: {best_val_loss:.4f}) ***")
                best_val_loss = avg_val_ce_loss
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_loss.pth'))
        
        print(f"SC-MIL training complete!")
        epoch = args.sc_mil_epochs
    else:
        print("Training...")
        epoch = None
    
    if epoch is None:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            train_dataset.set_epoch(epoch)
            model.train()
            run_loss, correct, total = 0.0, 0, 0
            
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                outputs, attn_weights = model(images, return_attention=True)
                
                main_loss = weighted_focal_loss(outputs, labels, class_weights[labels], label_smoothing=args.label_smoothing)
                ent_loss = aem_loss(attn_weights) if args.aem_weight > 0 else torch.tensor(0.0)
                loss = main_loss + args.aem_weight * ent_loss
                
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
                    val_loss = weighted_focal_loss(outputs, labels, class_weights[labels], label_smoothing=args.label_smoothing)
                    val_loss_total += val_loss.item()
            
            val_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
            all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
            val_auc = roc_auc_score(all_labels_bin, np.array(all_probs), average='macro')
            avg_val_loss = val_loss_total / len(val_loader)
            
            backbone_lr = optimizer.param_groups[0]['lr']
            classifier_lr = optimizer.param_groups[1]['lr']
            
            print(f"\n{'='*70}")
            print(f"[Fold: {test_plate} | Epoch: {epoch+1}/{args.epochs}]")
            print(f"{'='*70}")
            print(f"  [TRAIN] Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"  [VAL]   Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}% | AUC: {val_auc:.4f}")
            print(f"  [LR]    Backbone: {backbone_lr:.2e} | Classifier: {classifier_lr:.2e}")
            print(f"  [TIME]  {time.time()-epoch_start:.1f}s")
            print(f"{'='*70}")
            
            if val_auc > best_val_auc:
                print(f"  *** New best AUC: {val_auc:.4f} (previous: {best_val_auc:.4f}) ***")
                best_val_auc = val_auc
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model.pth'))
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_auc.pth'))
            
            if val_acc > best_val_acc:
                print(f"  *** New best Acc: {val_acc:.2f}% (previous: {best_val_acc:.2f}%) ***")
                best_val_acc = val_acc
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_acc.pth'))
            
            if avg_val_loss < best_val_loss:
                print(f"  *** New best Loss: {avg_val_loss:.4f} (previous: {best_val_loss:.4f}) ***")
                best_val_loss = avg_val_loss
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_loss.pth'))
            
            if (epoch + 1) % 10 == 0:
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
            
            csv_writer.writerow([
                epoch+1, f'{avg_train_loss:.6f}', f'{train_acc:.2f}',
                f'{avg_val_loss:.6f}', f'{val_acc:.2f}', f'{val_auc:.4f}',
                f'{backbone_lr:.2e}', f'{classifier_lr:.2e}'
            ])
            csv_file.flush()
    
    print("Testing...")
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing', leave=False):
            images = images.to(device)
            outputs, _, _ = model(images, return_attention=True, return_crop_embeddings=True)
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
        'config': {
            'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr,
            'test_plate': test_plate, 'dropout': args.dropout, 'weight_decay': args.weight_decay,
            'neighborhood': args.neighborhood, 'use_sc_mil': args.use_sc_mil,
            'use_hierarchical': args.use_hierarchical, 'aem_weight': args.aem_weight,
            'ema_momentum': args.ema_momentum, 'token_dropout_p': args.token_dropout_p,
            'sc_mil_weight': args.sc_mil_weight
        },
        'results': {
            'best_val_auc': float(best_val_auc),
            'test_acc': float(test_acc),
            'test_auc': float(test_auc),
            'test_ap': float(test_ap)
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    csv_file.close()
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
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
                print(f"\nSkipping {test_plate}: already trained (checkpoint exists)")
                continue
            
            train_single_fold(test_plate)
        
        print("All folds completed!")
    else:
        train_single_fold(args.test_plate)
    
    print("Done!")