#!/usr/bin/env python3
"""
MIL Training Script for CRISPRi Reference Plate Imaging

This script trains a Multiple Instance Learning (MIL) model with attention pooling
for classifying CRISPRi guide experiments from plate-based images.

Architecture:
- Backbone: EfficientNet-B0 (ImageNet pretrained)
- Pooling: Gated Multi-head Attention (4 heads)
- Crops: 3x3 neighborhood (9 crops per image)
- Feature Dim: 1280

Training Pipeline:
- Stage 1: Patch-level SimCLR contrastive pre-training (optional, controlled by --contrastive_epochs)
  - neighborhood=1 (single crop), InfoNCE loss, learns generic features
  - Skip with: --contrastive_epochs 0

- Stage 2: SC-MIL supervised contrastive + classification joint training (default)
  - neighborhood=3 (9 crops = bag), SupCon + Focal CE loss
  - Disable with: --no_sc_mil

Model Selection:
- MILEncoder: Used for Stage 1 (needs contrastive projection head)
- AttentionMILModel: Only when BOTH stages disabled

Arguments:
--epochs           : Total training epochs (default: 200)
--batch_size       : Batch size (default: 16)
--lr              : Learning rate (default: 1e-4)
--num_heads        : Attention heads (default: 4)
--seed            : Random seed (default: 42)
--test_plate       : Test plate P1-P6 (default: P6)
--neighborhood    : Crop neighborhood 3/5/7/9/11 (default: 3)
--dropout        : Dropout rate (default: 0.5)
--weight_decay   : Weight decay (default: 0.05)
--label_smoothing: Label smoothing (default: 0.1)

--contrastive_epochs : Epochs for Stage 1, 0 to skip (default: 50)
--contrastive_batch_size: Batch size for Stage 1 (default: 128)
--contrastive_temp  : Temperature for SimCLR (default: 0.1)

--sc_mil         : Enable SC-MIL (default: enabled)
--no_sc_mil      : Disable SC-MIL, use standard
--sc_mil_epochs  : Epochs for SC-MIL (default: 200)
--sc_mil_weight  : Weight for contrastive loss (default: 0.3)
--sc_mil_temp   : Temperature for SupCon (default: 0.07)

Usage:
    python train_mil.py --test_plate P6
    python train_mil.py --test_plate P6 --contrastive_epochs 0
    python train_mil.py --test_plate P6 --no_sc_mil
    python train_mil.py --run_all_folds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import csv
import random

# Local imports
from mil_model import AttentionMILModel, MILEncoder, MultiCropDataset, get_gene_from_path, extract_well_from_filename
from supcon_loss import SupConLoss

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

SEED: int = 42
"""Random seed for reproducibility"""

ALL_PLATES: list[str] = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
"""Available plate directories"""

DEFAULT_CONFIG: dict[str, Any] = {
    'epochs': 200,
    'batch_size': 16,
    'lr': 1e-4,
    'num_heads': 4,
    'dropout': 0.5,
    'weight_decay': 0.05,
    'label_smoothing': 0.1,
    'neighborhood': 3,
    'grid_size': 12,
    'use_contrastive': True,
    'use_sc_mil': True,
    'sc_mil_epochs': 200,
    'sc_mil_weight': 0.3,
    'sc_mil_temp': 0.07,
    'contrastive_epochs': 50,
    'contrastive_batch_size': 128,
    'contrastive_temp': 0.1,
}

# ============================================================================
# SETUP
# ============================================================================

def setup_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get available compute device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Focal Loss for handling class imbalance.
    
    Args:
        logits: Model predictions (before softmax)
        targets: Ground truth labels
        alpha: Weighting factor for class balance
        gamma: Focusing parameter for hard examples
    
    Returns:
        Scalar loss value
    """
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def weighted_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Weighted Focal Loss with class weights and optional label smoothing.
    
    Args:
        logits: Model predictions (before softmax)
        targets: Ground truth labels (indices)
        weights: Class weights tensor (same device as targets)
        alpha: Weighting factor for class balance
        gamma: Focusing parameter for hard examples
        label_smoothing: Label smoothing factor (0.0 = no smoothing)
    
    Returns:
        Scalar loss value
    """
    ce_loss = nn.functional.cross_entropy(
        logits, targets, reduction='none', label_smoothing=label_smoothing
    )
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return (focal * weights).mean()


def attention_entropy_loss(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Attention entropy loss for encouraging focused attention maps.
    
    Encourages the model to attend to fewer, more informative crops
    by penalizing uniform attention distributions.
    
    Args:
        attn_weights: Attention weights from MIL pooling (batch_size, num_heads, num_crops)
    
    Returns:
        Scalar entropy loss
    """
    # Ensure valid inputs to avoid log(0)
    attn_safe = attn_weights + 1e-8
    entropy = -(attn_safe * torch.log(attn_safe)).sum(dim=-1).mean()
    return entropy


# ============================================================================
# DATA LOADING
# ============================================================================

def load_plate_config(script_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load plate-well-to-gene mapping from JSON config."""
    config_path = script_dir / 'plate_well_id_path.json'
    with open(config_path, 'r') as f:
        return json.load(f)


def build_plate_maps(plate_data: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, str]]:
    """
    Build plate-to-well mapping dictionary.
    
    Args:
        plate_data: Raw plate data from JSON
    
    Returns:
        Nested dict: plate -> well_id -> gene_id
    """
    plate_maps: dict[str, dict[str, str]] = {}
    for plate in ALL_PLATES:
        plate_maps[plate] = {}
        for row, wells in plate_data[plate].items():
            for col, info in wells.items():
                # Format: "A1" -> "A01"
                well_id = f"{row}{int(col):02d}"
                plate_maps[plate][well_id] = info['id']
    return plate_maps


def get_image_paths(
    plate_dir: Path,
    plate: str,
    plate_maps: dict[str, dict[str, str]]
) -> list[str]:
    """
    Get all valid image paths for a plate.
    
    Args:
        plate_dir: Base directory containing plate folders
        plate: Plate identifier (e.g., 'P1')
        plate_maps: Well mapping dictionary
    
    Returns:
        List of valid image file paths
    """
    plate_path = plate_dir / plate
    if not plate_path.exists():
        return []
    
    # Find all image files (support multiple formats)
    patterns = ['*.tif', '*.tiff', '*.png']
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(plate_path.glob(f'**/{pattern}'))
    
    # Filter to valid wells
    valid_paths = []
    for path in paths:
        well = extract_well_from_filename(path.name)
        if well and well in plate_maps.get(plate, {}):
            valid_paths.append(str(path))
    
    return valid_paths


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute inverse frequency class weights for imbalanced data.
    
    Args:
        labels: Array of class labels
        num_classes: Total number of classes
        device: Target compute device
    
    Returns:
        Tensor of class weights
    """
    counts = Counter(labels)
    total = len(labels)
    weights = torch.tensor([
        total / (num_classes * counts.get(i, 1)) for i in range(num_classes)
    ], device=device)
    # Normalize so weights sum to num_classes
    return weights / weights.sum() * num_classes


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def worker_init_fn(worker_id: int, seed: int = 42) -> None:
    """
    Initialize worker random state for DataLoader multiprocessing.
    
    Args:
        worker_id: DataLoader worker ID
        seed: Base seed value
    """
    random.seed(seed + worker_id)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    label_smoothing: float = 0.0
) -> tuple[float, float, float, list[np.ndarray], list[np.ndarray], list[int]]:
    """
    Run validation/test evaluation.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        class_weights: Class weights tensor
        device: Compute device
        label_smoothing: Label smoothing value
    
    Returns:
        Tuple of (accuracy, AUC, loss, predictions, probabilities, labels)
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Validating', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(images, return_attention=True)
            probs = torch.softmax(outputs, dim=1)
            
            loss = weighted_focal_loss(outputs, labels, class_weights[labels], label_smoothing=label_smoothing)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    all_labels = np.array(all_labels)
    all_probs_np = np.array(all_probs)
    
    accuracy = 100.0 * np.mean(np.array(all_preds) == all_labels)
    
    # Handle multi-class AUC
    if len(all_labels) > 0:
        labels_bin = label_binarize(all_labels, classes=list(range(len(class_weights))))
        auc = roc_auc_score(labels_bin, all_probs_np, average='macro')
    else:
        auc = 0.0
    
    avg_loss = total_loss / len(data_loader)
    return accuracy, auc, avg_loss, all_preds, all_probs_np, all_labels


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    metrics: dict[str, float]
) -> None:
    """
    Save model checkpoint with metadata.
    
    Args:
        path: Save path
        epoch: Current epoch number (1-indexed)
        model: PyTorch model
        metrics: Dict of validation metrics
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        **metrics
    }, path)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_single_fold(
    test_plate: str,
    args: argparse.Namespace,
    device: torch.device,
    script_dir: Path,
    base_dir: Path,
    plate_maps: dict[str, dict[str, str]],
    gene_to_idx: dict[str, int],
    num_classes: int
) -> dict[str, Any]:
    """
    Train a single fold (test plate held out).
    
    Args:
        test_plate: Plate to use as test set
        args: Training arguments
        device: Compute device
        script_dir: Script directory
        base_dir: Data directory
        plate_maps: Well mapping
        gene_to_idx: Gene name to index mapping
        num_classes: Number of classes
    
    Returns:
        Dict of training results
    """
    # Setup directories
    output_dir = script_dir / f'fold_{test_plate}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training fold: test_plate={test_plate}")
    print(f"{'='*60}")
    
    # Split plates: 4 train, 1 val, 1 test
    train_val_plates = [p for p in ALL_PLATES if p != test_plate]
    train_plates = train_val_plates[:4]
    val_plates = train_val_plates[4:5]
    
    print(f"Train plates: {train_plates}")
    print(f"Val plates: {val_plates}")
    
    # Load data
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    for plate in train_plates:
        for path in get_image_paths(base_dir, plate, plate_maps):
            train_paths.append(path)
            train_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    for plate in val_plates:
        for path in get_image_paths(base_dir, plate, plate_maps):
            val_paths.append(path)
            val_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    for plate in [test_plate]:
        for path in get_image_paths(base_dir, plate, plate_maps):
            test_paths.append(path)
            test_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Class weights (computed on training set)
    class_weights = compute_class_weights(train_labels, num_classes, device)
    
    # Create datasets
    train_dataset = MultiCropDataset(
        train_paths, train_labels, plate_maps,
        neighborhood=args.neighborhood,
        grid_size=args.grid_size,
        augment=True,
        seed=SEED
    )
    val_dataset = MultiCropDataset(
        val_paths, val_labels, plate_maps,
        neighborhood=args.neighborhood,
        grid_size=args.grid_size,
        augment=False,
        seed=SEED
    )
    test_dataset = MultiCropDataset(
        test_paths, test_labels, plate_maps,
        neighborhood=args.neighborhood,
        grid_size=args.grid_size,
        augment=False,
        seed=SEED
    )
    
    # Set initial epoch
    train_dataset.set_epoch(0)
    val_dataset.set_epoch(0)
    test_dataset.set_epoch(0)
    
    # Determine workers
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 8
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=worker_init_fn,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=worker_init_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=worker_init_fn,
        pin_memory=True
    )
    
    print(f"Crops per image: {args.neighborhood}x{args.neighborhood}={args.neighborhood**2}")
    
    # Create model
    # Need MILEncoder for both SC-MIL and Stage 1 contrastive pre-training
    # Use AttentionMILModel only when both stages are disabled
    use_mil_encoder = args.use_sc_mil or args.contrastive_epochs > 0
    
    if use_mil_encoder:
        model = MILEncoder(
            num_classes=num_classes,
            num_heads=args.num_heads,
            dropout=args.dropout,
            use_contrastive=True
        )
    else:
        model = AttentionMILModel(
            num_classes=num_classes,
            num_heads=args.num_heads,
            dropout=args.dropout
        )
    model = model.to(device)
    
    # Optimizer setup
    backbone_params = [
        p for n, p in model.named_parameters()
        if 'attention_pool' not in n and 'classifier' not in n
    ]
    attention_params = [
        p for n, p in model.named_parameters()
        if 'attention_pool' in n or 'classifier' in n
    ]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': attention_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    # Initialize best metrics
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    # =========================================================================
    # STAGE 1: Contrastive Pre-training (optional)
    # =========================================================================
    # Stage 1 runs if contrastive_epochs > 0
    if args.contrastive_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Stage 1: Patch-Level SimCLR Pre-training")
        print(f"Epochs: {args.contrastive_epochs}, Batch size: {args.contrastive_batch_size}")
        print(f"{'='*60}")
        
        # Create contrastive datasets (single crop)
        cont_dataset_1 = MultiCropDataset(
            train_paths, train_labels, plate_maps,
            neighborhood=1, grid_size=args.grid_size,
            augment=True, seed=SEED
        )
        cont_dataset_2 = MultiCropDataset(
            train_paths, train_labels, plate_maps,
            neighborhood=1, grid_size=args.grid_size,
            augment=True, seed=SEED + 1
        )
        
        # Initialize epoch centers for contrastive datasets
        cont_dataset_1.set_epoch(0)
        cont_dataset_2.set_epoch(0)
        
        cont_loader_1 = DataLoader(
            cont_dataset_1, batch_size=args.contrastive_batch_size,
            shuffle=True, num_workers=0,
            worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True
        )
        cont_loader_2 = DataLoader(
            cont_dataset_2, batch_size=args.contrastive_batch_size,
            shuffle=True, num_workers=0,
            worker_init_fn=worker_init_fn, pin_memory=True, drop_last=True
        )
        
        # CSV logging for Stage 1
        timestamp_c1 = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path_c1 = output_dir / f'training_contrastive_{timestamp_c1}.csv'
        csv_file_c1 = open(csv_path_c1, 'w', newline='')
        csv_writer_c1 = csv.writer(csv_file_c1)
        csv_writer_c1.writerow(['epoch', 'loss', 'lr'])
        csv_file_c1.flush()
        
        # Contrastive optimizer
        cont_params = [
            p for n, p in model.named_parameters()
            if 'contrastive_head' in n or 'head_proj' in n or 'backbone' in n
        ]
        cont_optimizer = torch.optim.Adam(cont_params, lr=args.lr)
        cont_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            cont_optimizer, T_max=args.contrastive_epochs
        )
        
        for epoch in range(args.contrastive_epochs):
            model.train()
            run_loss = 0.0
            n_batches = 0
            
            iter_2 = iter(cont_loader_2)
            for images_v1, _ in tqdm(cont_loader_1, desc=f'Contrastive {epoch}', leave=False):
                try:
                    images_v2, _ = next(iter_2)
                except StopIteration:
                    iter_2 = iter(cont_loader_2)
                    images_v2, _ = next(iter_2)
                
                images_v1 = images_v1.to(device)
                images_v2 = images_v2.to(device)
                cont_optimizer.zero_grad()
                
                # Get features
                feat_v1 = model.get_projected_features(images_v1)
                feat_v2 = model.get_projected_features(images_v2)
                
                # L2 normalize
                feat_v1 = F.normalize(feat_v1, dim=1)
                feat_v2 = F.normalize(feat_v2, dim=1)
                
                # InfoNCE loss
                batch_size = feat_v1.shape[0]
                temp = args.contrastive_temp
                similarity = torch.matmul(feat_v1 / temp, feat_v2.T)
                labels = torch.arange(batch_size, device=device)
                
                loss = F.cross_entropy(similarity, labels)
                loss.backward()
                cont_optimizer.step()
                
                run_loss += loss.item()
                n_batches += 1
            
            cont_scheduler.step()
            avg_loss = run_loss / max(n_batches, 1)
            lr = cont_optimizer.param_groups[0]['lr']
            
            # Save to CSV
            csv_writer_c1.writerow([epoch + 1, avg_loss, lr])
            csv_file_c1.flush()
            
            print(f"Epoch {epoch}: Loss={avg_loss:.4f}, LR={lr:.2e}")
        
        csv_file_c1.close()
        print("Stage 1 complete!")
        train_dataset.set_epoch(0)
    
    # =========================================================================
    # STAGE 2: SC-MIL Training (or Standard)
    # =========================================================================
    if args.use_sc_mil and args.sc_mil_epochs > 0:
        _train_sc_mil(
            model, train_loader, val_loader, test_loader,
            class_weights, optimizer, device, args,
            output_dir, best_val_auc, best_val_acc, best_val_loss
        )
    else:
        _train_standard(
            model, train_loader, val_loader, test_loader,
            class_weights, optimizer, scheduler, device, args,
            output_dir
        )
    
    # =========================================================================
    # Final Evaluation
    # =========================================================================
    print("\nTesting...")
    checkpoint = torch.load(output_dir / 'best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_acc, test_auc, test_ap, _, all_probs, all_labels = _evaluate_full(
        model, test_loader, class_weights, device
    )
    
    print(f"Test Acc: {test_acc:.2f}%, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
    return {
        'test_plate': test_plate,
        'best_val_auc': float(best_val_auc),
        'test_acc': float(test_acc),
        'test_auc': float(test_auc),
        'test_ap': float(test_ap)
    }


def _train_sc_mil(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
    best_val_auc: float,
    best_val_acc: float,
    best_val_loss: float
) -> None:
    """SC-MIL training with supervised contrastive loss."""
    
    print(f"\n{'='*60}")
    print(f"SC-MIL: Supervised Contrastive + Classification")
    print(f"Epochs: {args.sc_mil_epochs}, Weight: {args.sc_mil_weight}")
    print(f"{'='*60}")
    
    # SC-MIL optimizer (train all parameters)
    sc_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        sc_optimizer, T_max=args.sc_mil_epochs
    )
    
    # CSV logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f'training_sc_mil_{timestamp}.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_ce', 'train_cl', 'train_acc',
                     'val_ce', 'val_acc', 'val_auc', 'lr'])
    csv_file.flush()
    
    best_auc, best_acc, best_loss = best_val_auc, best_val_acc, best_val_loss
    
    for epoch in range(args.sc_mil_epochs):
        epoch_start = time.time()
        model.train()
        
        train_ce, train_cl, correct, total = 0.0, 0.0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            sc_optimizer.zero_grad()
            
            # Forward pass with all outputs
            outputs, attn, bag_emb = model(images, return_attention=True, return_crop_embeddings=True)
            
            # L2 normalize for SupCon
            bag_emb = F.normalize(bag_emb, p=2, dim=-1)
            
            # Supervised contrastive loss
            sc_criterion = SupConLoss(temperature=args.sc_mil_temp)
            sc_loss = sc_criterion(bag_emb, labels)
            
            # Classification loss
            ce_loss = weighted_focal_loss(outputs, labels, class_weights[labels])
            
            # Combined loss
            loss = (1 - args.sc_mil_weight) * ce_loss + args.sc_mil_weight * sc_loss
            loss.backward()
            sc_optimizer.step()
            
            train_ce += ce_loss.item()
            train_cl += sc_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        sc_scheduler.step()
        
        train_acc = 100.0 * correct / total
        avg_ce = train_ce / len(train_loader)
        avg_cl = train_cl / len(train_loader)
        lr = sc_optimizer.param_groups[0]['lr']
        
        # Validation
        val_acc, val_auc, val_loss, _, _, _ = evaluate_model(
            model, val_loader, class_weights, device
        )
        
        # Log to CSV
        csv_writer.writerow([epoch + 1, avg_ce, avg_cl, train_acc,
                          val_loss, val_acc, val_auc, lr])
        csv_file.flush()
        
        print(f"\n{'='*60}")
        print(f"Epoch: {epoch+1}/{args.sc_mil_epochs}")
        print(f"TRAIN - CE: {avg_ce:.4f}, Cl: {avg_cl:.4f}, Acc: {train_acc:.2f}%")
        print(f"VAL   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.4f}")
        print(f"LR: {lr:.2e}, Time: {time.time()-epoch_start:.1f}s")
        print(f"{'='*60}")
        
        # Save best models
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(output_dir / 'best_model.pth', epoch + 1, model,
                          {'best_val_auc': val_auc, 'best_val_acc': val_acc, 'best_val_loss': val_loss})
            save_checkpoint(output_dir / 'best_model_auc.pth', epoch + 1, model,
                          {'best_val_auc': val_auc, 'best_val_acc': val_acc, 'best_val_loss': val_loss})
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(output_dir / 'best_model_acc.pth', epoch + 1, model,
                          {'best_val_auc': val_auc, 'best_val_acc': val_acc, 'best_val_loss': val_loss})
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(output_dir / 'best_model_loss.pth', epoch + 1, model,
                          {'best_val_auc': val_auc, 'best_val_acc': val_acc, 'best_val_loss': val_loss})
    
    csv_file.close()


def _train_standard(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path
) -> None:
    """Standard training without contrastive loss."""
    
    print(f"\n{'='*60}")
    print(f"Standard Training: {args.epochs} epochs")
    print(f"{'='*60}")
    
    # CSV logging for standard training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f'training_standard_{timestamp}.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc', 'lr'])
    csv_file.flush()
    
    best_auc, best_acc, best_loss = 0.0, 0.0, float('inf')
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        
        train_loss = 0.0
        correct, total = 0, 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs, attn = model(images, return_attention=True)
            
            main_loss = weighted_focal_loss(outputs, labels, class_weights[labels],
                                    label_smoothing=args.label_smoothing)
            ent_loss = attention_entropy_loss(attn)
            loss = main_loss + 0.01 * ent_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += main_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        train_acc = 100.0 * correct / total
        avg_loss = train_loss / len(train_loader)
        
        # Validation
        val_acc, val_auc, val_loss, _, _, _ = evaluate_model(
            model, val_loader, class_weights, device,
            label_smoothing=args.label_smoothing
        )
        
        lr = optimizer.param_groups[0]['lr']
        
        # Save to CSV
        csv_writer.writerow([epoch + 1, avg_loss, train_acc, val_loss, val_acc, val_auc, lr])
        csv_file.flush()
        
        print(f"\nEpoch: {epoch+1}/{args.epochs}")
        print(f"TRAIN - Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"VAL   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.4f}")
        print(f"LR: {lr:.2e}, Time: {time.time()-epoch_start:.1f}s")
        
        # Save best
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(output_dir / 'best_model.pth', epoch + 1, model,
                          {'best_val_auc': val_auc, 'best_val_acc': val_acc, 'best_val_loss': val_loss})
            save_checkpoint(output_dir / 'best_model_auc.pth', epoch + 1, model,
                          {'best_val_auc': val_auc, 'best_val_acc': val_acc, 'best_val_loss': val_loss})
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(output_dir / 'best_model_acc.pth', epoch + 1, model,
                          {'best_val_auc': val_auc, 'best_val_acc': val_acc, 'best_val_loss': val_loss})
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(output_dir / 'best_model_loss.pth', epoch + 1, model,
                          {'best_val_auc': val_auc, 'best_val_acc': val_acc, 'best_val_loss': val_loss})
    
    csv_file.close()


def _evaluate_full(
    model: nn.Module,
    data_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device
) -> tuple[float, float, float, list, list, list]:
    """Full evaluation with all metrics."""
    model.eval()
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(images, return_attention=True)
            probs = torch.softmax(outputs, dim=1)
            
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = 100.0 * np.mean(np.array(all_preds) == all_labels)
    
    labels_bin = label_binarize(all_labels, classes=list(range(len(class_weights))))
    auc = roc_auc_score(labels_bin, all_probs, average='macro')
    ap = average_precision_score(labels_bin, all_probs, average='macro')
    
    avg_loss = total_loss / len(data_loader)
    return accuracy, auc, ap, all_preds, all_probs, all_labels


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='MIL Training for CRISPRi')
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
    
    # Stage 1: Contrastive pre-training
    parser.add_argument('--contrastive_epochs', type=int, default=50,
                        help='Epochs for Stage 1 (0 to skip)')
    parser.add_argument('--contrastive_batch_size', type=int, default=128)
    parser.add_argument('--contrastive_temp', type=float, default=0.1,
                        help='Temperature for SimCLR loss')
    
    # Stage 2: SC-MIL
    parser.add_argument('--sc_mil', action='store_true', default=True,
                        help='Use SC-MIL (default: enabled)')
    parser.add_argument('--no_sc_mil', action='store_true',
                        help='Disable SC-MIL, use standard training')
    parser.add_argument('--sc_mil_epochs', type=int, default=200,
                        help='SC-MIL epochs')
    parser.add_argument('--sc_mil_weight', type=float, default=0.3,
                        help='Weight for contrastive loss in SC-MIL')
    parser.add_argument('--sc_mil_temp', type=float, default=0.07,
                        help='Temperature for SupCon loss')
    args = parser.parse_args()
    
    # Handle toggles
    args.use_sc_mil = not args.no_sc_mil
    
    # Print configuration
    print(f"\n{'='*60}")
    print("CONFIGURATION:")
    print(f"  test_plate: {args.test_plate}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  lr: {args.lr}")
    print(f"  neighborhood: {args.neighborhood}")
    print(f"  dropout: {args.dropout}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  Stage 1 (Contrastive): epochs={args.contrastive_epochs}")
    print(f"  Stage 2 (SC-MIL): use_sc_mil={args.use_sc_mil}, epochs={args.sc_mil_epochs}")
    print(f"{'='*60}\n")
    
    # Setup
    setup_seeds(args.seed)
    device = get_device()
    script_dir = Path(__file__).parent.resolve()
    
    # base_dir: look next to script_dir (siblings P1-P6), OR use explicit data_root
    # Structure: /workspace/P1/, /workspace/P2/, ..., /workspace/final_mutant_model/
    base_dir = Path(args.data_root) if args.data_root else script_dir.parent
    
    # Debug info
    print(f"Script dir: {script_dir}")
    print(f"Data dir: {base_dir}")
    print(f"P1 exists: {(base_dir / 'P1').exists()}")
    
    # Load configuration
    plate_data = load_plate_config(script_dir)
    plate_maps = build_plate_maps(plate_data)
    
    # Build gene mapping
    all_genes = sorted(set(
        gene for pm in plate_maps.values() for gene in pm.values()
    ))
    gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
    num_classes = len(all_genes)
    print(f"Classes: {num_classes}")
    
    # Run training
    if args.run_all_folds:
        for test_plate in ALL_PLATES:
            fold_dir = script_dir / f'fold_{test_plate}'
            checkpoints = [
                fold_dir / 'best_model.pth',
                fold_dir / 'best_model_acc.pth',
                fold_dir / 'best_model_auc.pth',
                fold_dir / 'best_model_loss.pth',
            ]
            if any(cp.exists() for cp in checkpoints):
                print(f"\nSkipping {test_plate}: already trained")
                continue
            
            train_single_fold(test_plate, args, device, script_dir, base_dir,
                        plate_maps, gene_to_idx, num_classes)
        
        print("All folds complete!")
    else:
        train_single_fold(args.test_plate, args, device, script_dir, base_dir,
                       plate_maps, gene_to_idx, num_classes)
    
    print("Done!")


if __name__ == '__main__':
    main()