#!/usr/bin/env python3
# Must be set before any imports to suppress inductor SM warning
import warnings
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")

import os
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN,CPP"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_CUDNN_DETERMINISTIC"] = "1"

"""
MIL training with cycle-based crop extraction + neighbors
Training: configurable crops (3x3, 5x5, 7x7, 9x9, 11x11 neighborhood)
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
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
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
from functools import partial

from mil_model import AttentionMILModel, MILEncoder, MultiCropDataset, get_gene_from_path, extract_well_from_filename
from supcon_loss import SupConLoss, SupConLossMIL

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Disable inductor max_autotune_gemm at runtime to avoid SM warning on small GPUs
import torch._inductor.config as inductor_config
inductor_config.max_autotune_gemm = False
inductor_config.max_autotune_gemm_backends = "ATEN,CPP"

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
parser.add_argument('--grid_size', type=int, default=12,
                    help='Grid size for crop positions')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate for classifier (default 0.5 for stronger regularization)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                    help='Weight decay (default 0.05 for stronger regularization)')
parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='Label smoothing (default 0.1, helps with small datasets)')
parser.add_argument('--use_contrastive', action='store_true',
                    help='Use patch-level contrastive pre-training')
parser.add_argument('--use_sc_mil', action='store_true',
                    help='Use SC-MIL: supervised contrastive + classification joint training (recommended)')
parser.add_argument('--sc_mil_epochs', type=int, default=100,
                    help='Epochs for SC-MIL joint training (default 100)')
parser.add_argument('--sc_mil_weight', type=float, default=0.3,
                    help='Weight for SC-MIL contrastive loss vs classification (0.1-1.0)')
parser.add_argument('--sc_mil_temp', type=float, default=0.07,
                    help='Temperature for SC-MIL contrastive loss')
parser.add_argument('--warmup_epochs', type=int, default=None,
                    help='Warmup epochs (default: 5%% of epochs, i.e. 10 for 200)')
parser.add_argument('--checkpoint_every', type=int, default=1,
                    help='Save checkpoint every N epochs (default: 1)')
args = parser.parse_args()

if args.warmup_epochs is None:
    args.warmup_epochs = int(args.epochs * 0.05)

# Set num_workers based on OS
if sys.platform.startswith('win'):
    NUM_WORKERS = 0  # Windows Python 3.14: multiprocessing spawn required
else:
    NUM_WORKERS = 16

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

def weighted_focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0, label_smoothing=0.0):
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return (focal * weights).mean()

def attention_entropy_loss(attn_weights):
    entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=1).mean()
    return entropy

def worker_init_fn(worker_id, seed=42):
    """Module-level worker init function for multiprocessing compatibility"""
    import random
    import numpy as np
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


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
    
    # Windows Python 3.14: MUST use 0 workers due to multiprocessing pickling changes
    if sys.platform.startswith('win'):
        effective_workers = 0
        print(f"Using {effective_workers} workers (Windows Python 3.14 - multiprocessing spawn required)")
    else:
        effective_workers = NUM_WORKERS
        print(f"Using {effective_workers} workers")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=effective_workers, pin_memory=True, drop_last=True,
                              worker_init_fn=partial(worker_init_fn, seed=SEED))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=effective_workers, pin_memory=True,
                            worker_init_fn=partial(worker_init_fn, seed=SEED))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=effective_workers, pin_memory=True,
                             worker_init_fn=partial(worker_init_fn, seed=SEED))
    
    print(f"Crops per image: {args.neighborhood}x{args.neighborhood}={args.neighborhood**2} crops")
    
    # Model selection based on flags
    if args.use_sc_mil:
        print(f"Using MILEncoder with SC-MIL supervised contrastive...")
        model = MILEncoder(num_classes=num_classes, num_heads=args.num_heads, dropout=args.dropout, use_contrastive=True)
    else:
        model = AttentionMILModel(num_classes=num_classes, num_heads=args.num_heads, dropout=args.dropout)
    model = model.to(device)
    
    backbone_params = [p for n, p in model.named_parameters() if 'attention_pool' not in n and 'classifier' not in n]
    attention_params = [p for n, p in model.named_parameters() if 'attention_pool' in n or 'classifier' in n]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': attention_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay, fused=True if torch.cuda.is_available() else False)

    # AMP scaler for mixed precision training
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, scheduler])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f'training_metrics_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc', 'backbone_lr', 'classifier_lr'])
    
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    # Stage 1: Patch-Level SimCLR Pre-training (proven in papers)
    if args.use_contrastive:
        print(f"\n{'='*60}")
        print(f"Stage 1: Patch-Level SimCLR Pre-training for {args.contrastive_epochs} epochs...")
        print(f"Contrastive batch size: {args.contrastive_batch_size}")
        print(f"{'='*60}")
        
        # Create two augmented views for each image
        crop_dataset_v1 = MultiCropDataset(train_paths, train_labels, plate_maps, neighborhood=1, grid_size=args.grid_size, augment=True, seed=SEED)
        crop_dataset_v2 = MultiCropDataset(train_paths, train_labels, plate_maps, neighborhood=1, grid_size=args.grid_size, augment=True, seed=SEED+1)
        
        # Set initial epoch for both
        crop_dataset_v1.set_epoch(0)
        crop_dataset_v2.set_epoch(0)
        
        # Higher batch size for contrastive (more negatives = better learning)
        crop_loader_v1 = DataLoader(crop_dataset_v1, batch_size=args.contrastive_batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        crop_loader_v2 = DataLoader(crop_dataset_v2, batch_size=args.contrastive_batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        
        # Train encoder + projection head
        contrastive_params = [p for n, p in model.named_parameters() if 'contrastive_head' in n or 'head_proj' in n or 'backbone' in n]
        contrastive_optimizer = torch.optim.Adam(contrastive_params, lr=args.lr, fused=True if torch.cuda.is_available() else False)
        contrastive_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(contrastive_optimizer, T_max=args.contrastive_epochs)
        contrastive_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        for epoch in range(args.contrastive_epochs):
            model.train()
            run_loss = 0.0
            n_batches = 0
            
            iter_v2 = iter(crop_loader_v2)
            for images_v1, _ in tqdm(crop_loader_v1, desc=f'Contrastive Epoch {epoch}', leave=False):
                try:
                    images_v2, _ = next(iter_v2)
                except StopIteration:
                    iter_v2 = iter(crop_loader_v2)
                    images_v2, _ = next(iter_v2)
                
                images_v1 = images_v1.to(device)
                images_v2 = images_v2.to(device)
                contrastive_optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    # Get features for both views
                    feat_v1 = model.get_projected_features(images_v1)
                    feat_v2 = model.get_projected_features(images_v2)
                    
                    # Normalize
                    feat_v1 = F.normalize(feat_v1, dim=1)
                    feat_v2 = F.normalize(feat_v2, dim=1)
                    
                    # SimCLR: z1*z2 for positives, z1*z_all for negatives
                    batch_size = feat_v1.shape[0]
                    temp = args.contrastive_temp
                    
                    # Compute similarity matrix
                    z1 = feat_v1 / temp
                    z2 = feat_v2 / temp
                    
                    # Similarity between matching pairs
                    sim_pos = torch.sum(z1 * z2, dim=1)
                    
                    # All pairs (including negatives)
                    similarity = torch.matmul(z1, z2.T)
                    
                    # Labels: diagonal (matching indices)
                    labels = torch.arange(batch_size, device=feat_v1.device)
                    
                    # Compute InfoNCE loss
                    loss = F.cross_entropy(similarity, labels)
                
                contrastive_scaler.scale(loss).backward()
                contrastive_scaler.step(contrastive_optimizer)
                contrastive_scaler.update()
                
                run_loss += loss.item()
                n_batches += 1
            
            contrastive_scheduler.step()
            avg_loss = run_loss / max(n_batches, 1)
            print(f"Contrastive Epoch {epoch}: Loss={avg_loss:.4f}")
        
        print(f"Stage 1 complete! Now training MIL classifier...")
        train_dataset.set_epoch(0)
    
    # SC-MIL: Supervised Bag-Level Contrastive + Classification Joint Training
    if args.use_sc_mil:
        print(f"\n{'='*60}")
        print(f"SC-MIL: Supervised Bag-Level Contrastive Joint Training")
        print(f"SC-MIL epochs: {args.sc_mil_epochs}, Temp: {args.sc_mil_temp}")
        print(f"Contrastive weight: {args.sc_mil_weight}")
        print(f"{'='*60}")
        
        # Batch size for SC-MIL
        effective_batch_size = args.batch_size  # Use full batch size (16)
        print(f"Using batch size: {effective_batch_size}")
        
        # Recreate data loaders with smaller batch size
        train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, num_workers=effective_workers, pin_memory=True, drop_last=True)
        
        # Train encoder + attention + classifier jointly
        sc_mil_params = [p for n, p in model.named_parameters()]
        sc_mil_optimizer = torch.optim.AdamW(sc_mil_params, lr=args.lr, weight_decay=args.weight_decay, fused=True if torch.cuda.is_available() else False)
        sc_mil_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sc_mil_optimizer, T_max=args.sc_mil_epochs)
        if args.warmup_epochs > 0:
            sc_mil_warmup = torch.optim.lr_scheduler.LinearLR(
                sc_mil_optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
            )
            sc_mil_scheduler = torch.optim.lr_scheduler.ChainedScheduler([sc_mil_warmup, sc_mil_scheduler])
        sc_mil_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        # Create CSV file for SC-MIL metrics
        timestamp_sc_mil = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path_sc_mil = os.path.join(OUTPUT_DIR, f"training_sc_mil_{timestamp_sc_mil}.csv")
        with open(csv_path_sc_mil, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_ce_loss', 'train_sc_loss', 'train_acc', 'val_ce_loss', 'val_acc', 'val_auc', 'lr'])
        
        for epoch in range(args.sc_mil_epochs):
            epoch_start = time.time()
            train_dataset.set_epoch(epoch)
            model.train()
            run_cl_loss, run_ce_loss, correct, total = 0.0, 0.0, 0, 0
            
            for images, labels in tqdm(train_loader, desc=f'SC-MIL Epoch {epoch}', leave=False):
                images, labels = images.to(device), labels.to(device)
                sc_mil_optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    # Get bag embeddings (after attention pooling)
                    outputs, attn_weights, crop_embeddings = model(images, return_attention=True, return_crop_embeddings=True)
                    bag_embeddings = F.normalize(crop_embeddings, p=2, dim=-1)
                    # Use official SupConLoss from supcon_loss.py
                    sc_criterion = SupConLoss(temperature=args.sc_mil_temp)
                    sc_loss = sc_criterion(bag_embeddings, labels)
                    
                    # Classification loss
                    ce_loss = weighted_focal_loss(outputs, labels, class_weights[labels])
                    
                    # Combined loss
                    loss = (1 - args.sc_mil_weight) * ce_loss + args.sc_mil_weight * sc_loss
                
                sc_mil_scaler.scale(loss).backward()
                sc_mil_scaler.step(sc_mil_optimizer)
                sc_mil_scaler.update()
                
                run_cl_loss += sc_loss.item()
                run_ce_loss += ce_loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            sc_mil_scheduler.step()
            
            train_acc = 100. * correct / total
            avg_cl_loss = run_cl_loss / len(train_loader)
            avg_ce_loss = run_ce_loss / len(train_loader)
            
            # VALIDATION after each SC-MIL epoch
            model.eval()
            val_cl_loss, val_ce_loss = 0.0, 0.0
            val_correct, val_total = 0, 0
            all_val_preds, all_val_probs, all_val_labels = [], [], []
            
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                for images, labels in tqdm(val_loader, desc='Validating', leave=False):
                    images, labels = images.to(device), labels.to(device)
                    outputs, _ = model(images, return_attention=True)
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
            
            print(f"SC-MIL Epoch {epoch}: CE Loss={avg_ce_loss:.4f}, SupCon Loss={avg_cl_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}, Time={time.time()-epoch_start:.1f}s")
            
            # Save checkpoint every epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'checkpoint_epoch.pth'))
            
            # Save metrics to CSV
            with open(csv_path_sc_mil, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_ce_loss, avg_cl_loss, train_acc, avg_val_ce_loss, val_acc, val_auc, sc_mil_optimizer.param_groups[0]['lr']])
            
            # Save best model based on validation AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model.pth'))
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_auc.pth'))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_acc.pth'))
            
            if avg_val_ce_loss < best_val_loss:
                best_val_loss = avg_val_ce_loss
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model_loss.pth'))
        
        print(f"SC-MIL training complete!")
        # Skip standard training, go directly to evaluation
        epoch = args.sc_mil_epochs  # Mark as complete
    
    else:
        print("Training...")
        epoch = None  # Means standard training
    
    # Standard or SC-MIL training loop
    if epoch is None:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            train_dataset.set_epoch(epoch)
            model.train()
            run_loss, correct, total = 0.0, 0, 0
            
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs, attn_weights = model(images, return_attention=True)
                    
                    main_loss = weighted_focal_loss(outputs, labels, class_weights[labels], label_smoothing=args.label_smoothing)
                    ent_loss = attention_entropy_loss(attn_weights)
                    loss = main_loss + 0.01 * ent_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
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
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
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
        
        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'checkpoint_epoch.pth'))
    
    print("Testing...")
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
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
        'config': {'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr, 'test_plate': test_plate, 'dropout': args.dropout, 'weight_decay': args.weight_decay, 'neighborhood': args.neighborhood},
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






