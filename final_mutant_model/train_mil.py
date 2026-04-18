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

# Try to import torchmil for VAEABMIL
try:
    from torchmil.models import VAEABMIL
    from torchmil.nn import VariationalAutoEncoderMIL, AttentionPool
    TORCHMIL_AVAILABLE = True
except ImportError:
    TORCHMIL_AVAILABLE = False
    print("Warning: torchmil not available, VAEABMIL will not work")

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
parser.add_argument('--warmup_epochs', type=int, default=0, help='Cosine warmup epochs (0 = no warmup)')
parser.add_argument('--num_workers', type=int, default=16, help='Number of workers (default: 16)')
parser.add_argument('--optimizer', type=str, default='adamax', choices=['adam', 'adamax'], help='Optimizer to use')
parser.add_argument('--use_attention', action='store_true', default=True, help='Use attention pooling (disable for GMP only)')
parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension in classification head')
parser.add_argument('--model_type', type=str, default='custom', choices=['custom', 'vaeabmil', 'abmil', 'dsmil'], help='Model architecture to use')
args = parser.parse_args()

# Set num_workers based on args
NUM_WORKERS = args.num_workers

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
    if label_smoothing > 0:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    else:
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return (focal * weights).mean()

def attention_entropy_loss(attn_weights):
    entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=1).mean()
    return entropy

def diversity_loss(attn_weights):
    """Encourage different heads to attend to different instances"""
    # attn_weights: (B, N, H) where H = num_heads
    # We want each head to have different attention patterns
    # Measure diversity as negative correlation between heads
    B, N, H = attn_weights.shape
    avg_attn = attn_weights.mean(dim=1)  # (B, H)
    # Encourage low correlation between heads
    corr = torch.corrcoef(avg_attn.T + 1e-8)
    off_diag = corr - torch.eye(H, device=corr.device) * corr.diag()
    return off_diag.abs().mean()

def token_dropout(attn_weights, dropout_rate, training=True):
    """Apply dropout to attention tokens during training"""
    if not training or dropout_rate <= 0:
        return attn_weights
    B, N, H = attn_weights.shape
    mask = torch.rand(B, N, 1, device=attn_weights.device) > dropout_rate
    return attn_weights * mask.float()

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
    
    train_dataset = MultiCropDataset(train_paths, train_labels, plate_maps, augment=True, seed=SEED)
    val_dataset = MultiCropDataset(val_paths, val_labels, plate_maps, augment=False, seed=SEED)
    test_dataset = MultiCropDataset(test_paths, test_labels, plate_maps, augment=False, seed=SEED)
    
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
    
    print(f"Crops per image: 25 (center + 5x5 neighbors)")
    
    # Create model based on model_type
    if args.model_type == 'vaeabmil':
        if not TORCHMIL_AVAILABLE:
            raise ValueError("torchmil not installed. Install with: pip install torchmil")
        
        from torchmil.nn import VariationalAutoEncoderMIL, AttentionPool
        
        # Create EfficientNet backbone
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )
        
        # VAE feature extractor
        vae_ext = VariationalAutoEncoderMIL(
            input_shape=(1280,),
            layer_sizes=[256, 128],
            activations=['relu', None]
        )
        
        # Attention pooling
        attn_pool = AttentionPool(in_dim=128, att_dim=128, act='tanh', gated=True)
        
        # Multi-class classifier
        classifier = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Full VAEABMIL wrapper for multi-class
        class VAEABMILMultiClass(nn.Module):
            def __init__(self, backbone, vae_ext, attn_pool, classifier):
                super().__init__()
                self.backbone = backbone
                self.vae_ext = vae_ext
                self.attn_pool = attn_pool
                self.classifier = classifier
            
            def forward(self, x, return_attention=False):
                batch_size, num_crops = x.shape[:2]
                
                # Extract features
                x = x.view(batch_size * num_crops, *x.shape[2:])
                x = self.backbone(x)
                x = x.view(batch_size, num_crops, -1)  # (B, N, 1280)
                
                # VAE encoding: get latent representations
                z = self.vae_ext(x)  # (B, N, 1, 128)
                z = z.squeeze(2)  # (B, N, 128)
                
                # Attention pooling on latent space
                pooled, attn = self.attn_pool(z, return_att=True)
                
                # Multi-class classification
                logits = self.classifier(pooled)
                
                if return_attention:
                    return logits, attn
                return logits
            
            def get_optimizer_groups(self, lr_backbone, lr_mil):
                return [
                    {'params': self.backbone.parameters(), 'lr': lr_backbone},
                    {'params': list(self.vae_ext.parameters()) + list(self.attn_pool.parameters()) + list(self.classifier.parameters()), 'lr': lr_mil}
                ]
        
        model = VAEABMILMultiClass(backbone, vae_ext, attn_pool, classifier).to(device)
        optimizer_groups = model.get_optimizer_groups(args.lr * 0.1, args.lr)
        
        if args.optimizer == 'adamax':
            optimizer = torch.optim.Adamax(optimizer_groups, weight_decay=0.01)
        else:
            optimizer = torch.optim.Adam(optimizer_groups, weight_decay=0.01)
        
        def train_step(model, images, labels, optimizer):
            model.train()
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            return loss, outputs
        
        def eval_step(model, images, labels=None):
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
            return outputs, probs, predicted
        
        train_func = train_step
        eval_func = eval_step
        
        print(f"Using VAEABMIL (manual) multi-class model with EfficientNet-B0 backbone")
        
    elif args.model_type == 'abmil':
        if not TORCHMIL_AVAILABLE:
            raise ValueError("torchmil not installed")
        
        from torchmil.models import ABMIL
        from torchmil.nn import AttentionPool
        
        # Backbone
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        feature_extractor = nn.Sequential(
            base_model.features,
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
        )
        
        attn_pool = AttentionPool(in_dim=1280, att_dim=128, act='tanh', gated=True)
        mil_model = ABMIL(feat_ext=attn_pool)
        
        # Multi-class classifier with 6-layer head (same as custom model)
        classifier = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        class ABMILWithBackbone(nn.Module):
            def __init__(self, backbone, mil_model, classifier):
                super().__init__()
                self.backbone = backbone
                self.mil_model = mil_model
                self.classifier = classifier
            
            def forward(self, x, return_attention=False):
                batch_size, num_crops = x.shape[:2]
                x = x.view(batch_size * num_crops, *x.shape[2:])
                x = self.backbone(x)
                x = x.view(batch_size, num_crops, -1)
                # ABMIL returns (output, attn) - output is attention-pooled features
                pooled_features, attn = self.mil_model(x, return_att=True)
                # Apply classifier
                output = self.classifier(pooled_features)
                if return_attention:
                    return output, attn
                return output
            
            def get_optimizer_groups(self, lr_backbone, lr_mil):
                return [
                    {'params': self.backbone.parameters(), 'lr': lr_backbone},
                    {'params': list(self.mil_model.parameters()) + list(self.classifier.parameters()), 'lr': lr_mil}
                ]
        
        model = ABMILWithBackbone(feature_extractor, mil_model, classifier).to(device)
        optimizer_groups = model.get_optimizer_groups(args.lr * 0.1, args.lr)
        
        if args.optimizer == 'adamax':
            optimizer = torch.optim.Adamax(optimizer_groups, weight_decay=0.01)
        else:
            optimizer = torch.optim.Adam(optimizer_groups, weight_decay=0.01)
        
        def train_step(model, images, labels, optimizer):
            model.train()
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            return loss, outputs
        
        def eval_step(model, images, labels=None):
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
            return outputs, probs, predicted
        
        train_func = train_step
        eval_func = eval_step
        print(f"Using ABMIL model with EfficientNet-B0 backbone")
        
    else:
        # Custom model (existing)
        model = AttentionMILModel(
            num_classes=num_classes, 
            num_heads=args.num_heads,
            use_attention=args.use_attention,
            hidden_dim=args.hidden_dim
        )
        model = model.to(device)
        
        # Differential learning rates
        if args.use_attention:
            backbone_params = [p for n, p in model.named_parameters() if 'attention_pool' not in n and 'classifier' not in n]
            attention_params = [p for n, p in model.named_parameters() if 'attention_pool' in n or 'classifier' in n]
            param_groups = [
                {'params': backbone_params, 'lr': args.lr * 0.1},
                {'params': attention_params, 'lr': args.lr}
            ]
        else:
            backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
            classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n]
            param_groups = [
                {'params': backbone_params, 'lr': args.lr * 0.1},
                {'params': classifier_params, 'lr': args.lr}
            ]
        
        # Adamax or Adam optimizer
        if args.optimizer == 'adamax':
            optimizer = torch.optim.Adamax(param_groups, weight_decay=0.01)
        else:
            optimizer = torch.optim.Adam(param_groups, weight_decay=0.01)
        
        def train_step(model, images, labels, optimizer):
            model.train()
            optimizer.zero_grad()
            outputs, _ = model(images, return_attention=True)
            loss = focal_loss(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            return loss, outputs
        
        def eval_step(model, images, labels=None):
            model.eval()
            with torch.no_grad():
                outputs, _ = model(images, return_attention=True)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
            return outputs, probs, predicted
        
        train_func = train_step
        eval_func = eval_step
        print(f"Using custom model: optimizer={args.optimizer.upper()}, attention={args.use_attention}")
    
    # Cosine warmup then cosine annealing
    # Warmup: start at 0.1x LR, linearly ramp to full LR over warmup_epochs
    # Then cosine decay from full LR to near-zero
    def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup: start at 0.1 (10% of LR), ramp to 1.0 at end of warmup
                return 0.1 + 0.9 * (epoch / max(1, warmup_epochs))
            else:
                # Cosine decay from 1.0 to 0
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = warmup_cosine_scheduler(optimizer, args.warmup_epochs, args.epochs)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f'training_metrics_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc', 'lr'])
    
    best_val_auc = 0.0
    
    print("Training...")
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_dataset.set_epoch(epoch)
        
        run_loss, correct, total = 0.0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            images, labels = images.to(device), labels.to(device)
            
            loss, outputs = train_func(model, images, labels, optimizer)
            
            run_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        train_acc = 100. * correct / total
        avg_train_loss = run_loss / len(train_loader)
        
        # Validation
        all_preds, all_probs, all_labels = [], [], []
        val_loss_total = 0.0
        
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, probs, predicted = eval_func(model, images, labels)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            val_loss_total += focal_loss(outputs, labels).item()
        
        val_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        val_auc = roc_auc_score(all_labels_bin, np.array(all_probs), average='macro')
        avg_val_loss = val_loss_total / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}, LR={current_lr:.2e}, Time={time.time()-epoch_start:.1f}s")
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, val_auc, current_lr])
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model.pth'))
        
        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("Testing...")
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs, probs, predicted = eval_func(model, images)
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
