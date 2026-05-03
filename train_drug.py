#!/usr/bin/env python3
"""
Drug Data Training - MIL with SC-MIL
Uses trial_daniel/data folder structure: Plate_1-6/Compound_dose/Well*.tiff
"""

import os
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN,CPP"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_CUDNN_DETERMINISTIC"] = "1"

import warnings
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")

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
from torch.utils.data import Dataset, DataLoader
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
from functools import partial

import torch._inductor.config as inductor_config
inductor_config.max_autotune_gemm = False
inductor_config.max_autotune_gemm_backends = "ATEN,CPP"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class AttentionPooling(nn.Module):
    """Gated attention MIL pooling"""
    def __init__(self, in_features, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = 256
        
        self.V = nn.Linear(in_features, self.hidden_dim)
        self.U = nn.Linear(in_features, self.hidden_dim)
        self.w = nn.Linear(self.hidden_dim, num_heads)
    
    def forward(self, x, temperature=0.5):
        A = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        attn_weights = self.w(A)
        attn_weights = torch.softmax(attn_weights / temperature, dim=1)
        pooled = torch.einsum('bnh,bnf->bhf', attn_weights, x)
        return pooled, attn_weights


class ContrastiveEncoder(nn.Module):
    """Encoder for contrastive learning"""
    def __init__(self, feature_dim=1280, projection_dim=256):
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x):
        return self.projection_head(x)
    
    def get_embedding(self, x):
        with torch.no_grad():
            return F.normalize(x, dim=1)


class MILEncoder(nn.Module):
    """MIL encoder with optional contrastive head"""
    def __init__(self, num_classes, num_heads=4, attention_temp=0.5, dropout=0.2, use_contrastive=False, projection_dim=256):
        super().__init__()
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.feature_dim = 1280
        self.use_contrastive = use_contrastive
        
        self.attention_pool = AttentionPooling(self.feature_dim, num_heads)
        self.attention_temp = attention_temp
        
        self.head_proj = nn.Linear(self.feature_dim * num_heads, self.feature_dim)
        
        if use_contrastive:
            self.contrastive_head = ContrastiveEncoder(self.feature_dim, projection_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x, return_attention=False, return_embedding=False, return_crop_embeddings=False, return_pooled_embeddings=False, return_instance_logits=False):
        batch_size, num_crops = x.shape[:2]
        
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        crop_embeddings = x.view(batch_size, num_crops, -1)
        
        pooled, attn_weights = self.attention_pool(crop_embeddings, temperature=self.attention_temp)
        pooled = pooled.reshape(batch_size, -1)
        pooled = self.head_proj(pooled)
        
        if return_embedding and self.use_contrastive:
            embedding = self.contrastive_head.get_embedding(pooled)
            if return_attention:
                return embedding, attn_weights
            return embedding
        
        output = self.classifier(pooled)
        
        results = [output]
        if return_attention:
            results.append(attn_weights)
        if return_crop_embeddings:
            results.append(crop_embeddings)
        if return_pooled_embeddings:
            results.append(pooled)
        if return_instance_logits:
            instance_logits = self.classifier(crop_embeddings)
            results.append(instance_logits)
        
        return results[0] if len(results) == 1 else tuple(results)
    
    def get_projected_features(self, x):
        if len(x.shape) == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            x = self.backbone(x)
            x = x.view(B, N, -1)
            pooled, _ = self.attention_pool(x, temperature=self.attention_temp)
            pooled = pooled.reshape(B, -1)
        else:
            x = self.backbone(x)
            pooled = x
        
        pooled = self.head_proj(pooled)
        return pooled
    
    def get_mil_embeddings(self, x):
        batch_size, num_crops = x.shape[:2]
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)
        
        pooled, _ = self.attention_pool(x, temperature=self.attention_temp)
        pooled = pooled.reshape(batch_size, -1)
        pooled = self.head_proj(pooled)
        
        return pooled


class DrugDataset(Dataset):
    """Dataset for drug data - loads all images per well"""
    
    def __init__(self, image_paths, labels, crop_size=224, neighborhood=5, augment=True, seed=42, epoch=0):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.neighborhood = neighborhood
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from PIL import Image
        
        if augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        self.grid_size = 12
        self._setup_positions()
        
        jitter_range = 20
        self.jitter_range = jitter_range
        
        print(f"DrugDataset: {len(self.image_paths)} images, {neighborhood}x{neighborhood} crops")
    
    def _setup_positions(self):
        self.stride = 40
        half_n = self.neighborhood // 2
        
        positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                left = j * self.stride
                top = i * self.stride
                positions.append((left, top))
        
        self.positions = positions
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        rng = random.Random(self.seed + epoch)
        shuffled = self.positions.copy()
        rng.shuffle(shuffled)
        
        num_images = len(self.image_paths)
        self.epoch_centers = {}
        for idx in range(num_images):
            assigned_idx = (idx + epoch) % len(shuffled)
            self.epoch_centers[idx] = shuffled[assigned_idx]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        center_left, center_top = self.epoch_centers[idx]
        
        jitter_range = self.jitter_range
        crops_list = []
        half_n = self.neighborhood // 2
        
        for di in range(-half_n, half_n + 1):
            for dj in range(-half_n, half_n + 1):
                if self.augment:
                    jitter_x = random.randint(-jitter_range, jitter_range)
                    jitter_y = random.randint(-jitter_range, jitter_range)
                else:
                    jitter_x = jitter_y = 0
                
                left = center_left + dj * self.stride + jitter_x
                top = center_top + di * self.stride + jitter_y
                
                w, h = image.size
                left = max(0, min(left, w - self.crop_size))
                top = max(0, min(top, h - self.crop_size))
                
                crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
                crop = np.array(crop)
                crop = self.transform(image=crop)['image']
                crops_list.append(crop)
        
        num_crops = self.neighborhood * self.neighborhood
        if self.augment:
            perm = list(range(num_crops))
            random.shuffle(perm)
            crops_list = [crops_list[i] for i in perm]
        
        crops = torch.stack(crops_list)
        
        return crops, self.labels[idx]


class SingleImageDataset(Dataset):
    """Simple dataset for single image per well"""
    
    def __init__(self, image_paths, labels, crop_size=224, grid_size=12, neighborhood=3, augment=True, seed=42, epoch=0):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.neighborhood = neighborhood
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from PIL import Image
        
        if augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
        half_n = neighborhood // 2
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if left + crop_size <= w and top + crop_size <= h:
                    can_left = left - half_n * stride >= 0
                    can_right = left + half_n * stride + crop_size <= w
                    can_top = top - half_n * stride >= 0
                    can_bottom = top + half_n * stride + crop_size <= h
                    if can_left and can_right and can_top and can_bottom:
                        positions.append((left, top))
        
        self.positions = positions
        print(f"SingleImageDataset: {len(positions)} positions, {neighborhood}x{neighborhood} crops/image")
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        num_pos = len(self.positions)
        num_images = len(self.image_paths)
        
        if not self.augment:
            center_left = (self.image_size - self.crop_size) // 2
            center_top = (self.image_size - self.crop_size) // 2
            self.epoch_centers = {i: (center_left, center_top) for i in range(num_images)}
            return
        
        cycle = epoch // num_pos
        pos_in_cycle = epoch % num_pos
        rng = random.Random(self.seed + cycle)
        shuffled = self.positions.copy()
        rng.shuffle(shuffled)
        
        self.epoch_centers = {}
        for idx in range(num_images):
            assigned_idx = (idx + pos_in_cycle) % num_pos
            self.epoch_centers[idx] = shuffled[assigned_idx]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        center_left, center_top = self.epoch_centers[idx]
        
        jitter_range = self.stride // 4
        crops_list = []
        half_n = self.neighborhood // 2
        
        for di in range(-half_n, half_n + 1):
            for dj in range(-half_n, half_n + 1):
                if self.augment:
                    jitter_x = random.randint(-jitter_range, jitter_range)
                    jitter_y = random.randint(-jitter_range, jitter_range)
                else:
                    jitter_x = jitter_y = 0
                
                left = center_left + dj * self.stride + jitter_x
                top = center_top + di * self.stride + jitter_y
                left = max(0, min(left, self.image_size - self.crop_size))
                top = max(0, min(top, self.image_size - self.crop_size))
                
                crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
                crop = np.array(crop)
                crop = self.transform(image=crop)['image']
                crops_list.append(crop)
        
        num_crops = self.neighborhood * self.neighborhood
        if self.augment:
            perm = list(range(num_crops))
            random.shuffle(perm)
            crops_list = [crops_list[i] for i in perm]
        
        crops = torch.stack(crops_list)
        
        return crops, self.labels[idx]


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        labels = labels.view(-1, 1)
        batch_size = embeddings.shape[0]
        
        embeddings = F.normalize(embeddings, dim=1)
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        mask = torch.eq(labels, labels.T).float()
        mask_diag = torch.eye(batch_size, device=mask.device)
        mask = mask - mask_diag
        
        similarity = similarity - similarity.diag().unsqueeze(1) * 100
        
        exp_sim = torch.exp(similarity)
        mask_pos = mask * exp_sim
        denom = mask_pos.sum(dim=1) + (1 - mask).sum(dim=1) * 0
        
        num_pos = mask.sum(dim=1)
        loss = -torch.log((mask_pos.sum(dim=1) + 1e-8) / (denom + 1e-8))
        loss = loss.mean()
        
        return loss


def weighted_focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0, label_smoothing=0.0):
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return (focal * weights).mean()


def attention_entropy_loss(attn_weights):
    entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=1).mean()
    return entropy


def worker_init_fn(worker_id, seed=42):
    import random
    import numpy as np
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def extract_compound_dose(folder_name):
    return folder_name


def get_image_paths_for_plate(plate_dir, plate_name):
    if not os.path.exists(plate_dir):
        return []
    paths = []
    for compound_folder in os.listdir(plate_dir):
        compound_path = os.path.join(plate_dir, compound_folder)
        if not os.path.isdir(compound_path):
            continue
        for pattern in ['*.tif', '*.tiff', '*.png']:
            paths.extend(glob.glob(os.path.join(compound_path, pattern)))
    return paths, compound_folder


def train_single_fold(test_plate, data_root, args):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, f'drug_fold_{test_plate}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training fold: test_plate={test_plate}")
    print(f"{'='*60}")
    
    all_plates = ['Plate_1', 'Plate_2', 'Plate_3', 'Plate_4', 'Plate_5', 'Plate_6']
    train_val_plates = [p for p in all_plates if p != test_plate]
    train_plates = train_val_plates[:4]
    val_plates = train_val_plates[4:]
    
    print(f"Train plates: {train_plates}")
    print(f"Val plates: {val_plates}")
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    all_labels_set = set()
    
    for plate in train_plates:
        plate_dir = os.path.join(data_root, plate)
        if not os.path.exists(plate_dir):
            continue
        for compound in os.listdir(plate_dir):
            compound_path = os.path.join(plate_dir, compound)
            if not os.path.isdir(compound_path):
                continue
            for f in glob.glob(os.path.join(compound_path, '*.tif')):
                train_paths.append(f)
                train_labels.append(compound)
                all_labels_set.add(compound)
    
    for plate in val_plates:
        plate_dir = os.path.join(data_root, plate)
        if not os.path.exists(plate_dir):
            continue
        for compound in os.listdir(plate_dir):
            compound_path = os.path.join(plate_dir, compound)
            if not os.path.isdir(compound_path):
                continue
            for f in glob.glob(os.path.join(compound_path, '*.tif')):
                val_paths.append(f)
                val_labels.append(compound)
                all_labels_set.add(compound)
    
    for plate in [test_plate]:
        plate_dir = os.path.join(data_root, plate)
        if not os.path.exists(plate_dir):
            continue
        for compound in os.listdir(plate_dir):
            compound_path = os.path.join(plate_dir, compound)
            if not os.path.isdir(compound_path):
                continue
            for f in glob.glob(os.path.join(compound_path, '*.tif')):
                test_paths.append(f)
                test_labels.append(compound)
                all_labels_set.add(compound)
    
    all_labels = sorted(all_labels_set)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    num_classes = len(all_labels)
    print(f"Classes: {num_classes}")
    
    train_labels = np.array([label_to_idx[l] for l in train_labels])
    val_labels = np.array([label_to_idx[l] for l in val_labels])
    test_labels = np.array([label_to_idx[l] for l in test_labels])
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    class_counts = Counter(train_labels)
    total = len(train_labels)
    class_weights = torch.tensor([total / (num_classes * class_counts[i]) for i in range(num_classes)], device=device)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    train_dataset = DrugDataset(train_paths, train_labels, neighborhood=args.neighborhood, augment=True, seed=SEED)
    val_dataset = DrugDataset(val_paths, val_labels, neighborhood=args.neighborhood, augment=False, seed=SEED)
    test_dataset = DrugDataset(test_paths, test_labels, neighborhood=args.neighborhood, augment=False, seed=SEED)
    
    train_dataset.set_epoch(0)
    val_dataset.set_epoch(0)
    test_dataset.set_epoch(0)
    
    effective_workers = 0 if sys.platform.startswith('win') else 16
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=effective_workers, pin_memory=True, drop_last=True,
                              worker_init_fn=partial(worker_init_fn, seed=SEED))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=effective_workers, pin_memory=True,
                            worker_init_fn=partial(worker_init_fn, seed=SEED))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=effective_workers, pin_memory=True,
                             worker_init_fn=partial(worker_init_fn, seed=SEED))
    
    print(f"Crops per image: {args.neighborhood}x{args.neighborhood}={args.neighborhood**2} crops")
    
    model = MILEncoder(num_classes=num_classes, num_heads=args.num_heads, dropout=args.dropout, use_contrastive=True)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True if torch.cuda.is_available() else False)
    
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(warmup_scheduler, scheduler)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f'training_metrics_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc', 'lr'])
    
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
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
        all_preds, all_probs, all_labels_list = [], [], []
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for images, labels in tqdm(val_loader, desc='Validating', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images, return_attention=True)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())
                val_loss = weighted_focal_loss(outputs, labels, class_weights[labels], label_smoothing=args.label_smoothing)
                val_loss_total += val_loss.item()
        
        val_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels_list))
        all_labels_bin = label_binarize(all_labels_list, classes=list(range(num_classes)))
        val_auc = roc_auc_score(all_labels_bin, np.array(all_probs), average='macro')
        avg_val_loss = val_loss_total / len(val_loader)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}, LR={lr:.2e}, Time={time.time()-epoch_start:.1f}s")
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, train_acc, avg_val_loss, val_acc, val_auc, lr])
        
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
    
    all_preds, all_probs, all_labels_list = [], [], []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for images, labels in tqdm(test_loader, desc='Testing', leave=False):
            images = images.to(device)
            outputs, _ = model(images, return_attention=True)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels_list.extend(labels.numpy())
    
    test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels_list))
    test_labels_bin = label_binarize(all_labels_list, classes=list(range(num_classes)))
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
    
    return best_val_auc, test_acc, test_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_plate', type=str, default='Plate_6')
    parser.add_argument('--data_root', type=str, default='trial_daniel/data', help='Path to drug data folder')
    parser.add_argument('--run_all_folds', action='store_true', default=True, help='Run all 6 folds')
    parser.add_argument('--neighborhood', type=int, default=5, choices=[3, 5, 7, 9, 11])
    parser.add_argument('--grid_size', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--checkpoint_every', type=int, default=10)
    args = parser.parse_args()
    
    if args.run_all_folds:
        all_plates = ['Plate_1', 'Plate_2', 'Plate_3', 'Plate_4', 'Plate_5', 'Plate_6']
        for test_plate in all_plates:
            train_single_fold(test_plate, args.data_root, args)
        print("All folds completed!")
    else:
        train_single_fold(args.test_plate, args.data_root, args)
    
    print("Done!")
