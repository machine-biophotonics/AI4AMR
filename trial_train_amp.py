"""
Trial Training Script with All Optimizations
=============================================
- Mixed Precision Training (AMP) - ~2x faster
- Optimal batch_size=8 (based on GPU memory test)
- Fixed Grid Dataset (144 crops per image)
- Proper augmentations (flips, rotation, color jitter)
- Class weights + label smoothing
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torchvision.models as models
from torchvision.transforms import (
    ToTensor, RandomHorizontalFlip, RandomVerticalFlip, 
    CenterCrop, Normalize, Compose, ColorJitter, RandomRotation
)
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import json
import re
import numpy as np
from collections import Counter
import time
import random
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  # Enable for fixed input sizes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # OPTIMIZATION: Use Channels Last memory format for faster convolutions (RTX cards)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Load plate data
with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

all_labels = sorted(set(label for pm in plate_maps.values() for label in pm.values()))
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(all_labels)
print(f"Number of classes: {num_classes}")

def extract_well_from_filename(filename: str):
    match = re.search(r'Well(\w\d+)_', filename)
    if match:
        return match.group(1)
    return None

def get_label_from_path(img_path: str):
    dirname = os.path.dirname(img_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        return plate_maps[plate][well]
    return None


class FixedGridDataset(Dataset):
    """Fixed Grid Dataset - ALL crops from entire image (no edge margin).
    
    Augmentation pipeline:
    - RandomVerticalFlip (p=0.5)
    - RandomHorizontalFlip (p=0.5)
    - RandomRotation (90 degrees)
    - ColorJitter (brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1)
    - ToTensor
    - Normalize
    """
    def __init__(self, image_paths, transform=None, get_labels=True, augment=True):
        self.image_paths = image_paths
        self.transform = transform
        self.get_labels = get_labels
        self.augment = augment
        self.patch_size = 224
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.n_crops_w = w // self.patch_size
        self.n_crops_h = h // self.patch_size
        self.n_crops = self.n_crops_w * self.n_crops_h
        
        self.augment_transform = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=90),
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
        ]) if augment else None
        
        print(f"Fixed Grid: {self.n_crops_w}x{self.n_crops_h} = {self.n_crops} crops/image (augment={augment})")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        patches = []
        for i in range(self.n_crops_h):
            for j in range(self.n_crops_w):
                left = j * self.patch_size
                top = i * self.patch_size
                patch = image.crop((left, top, left + self.patch_size, top + self.patch_size))
                
                if self.augment_transform:
                    patch = self.augment_transform(patch)
                
                if self.transform:
                    patch = self.transform(patch)
                patches.append(patch)
        
        patches_tensor = torch.stack(patches)
        
        if self.get_labels:
            label_str = get_label_from_path(img_path)
            if label_str is None:
                label_str = "Unknown"
            label = label_to_idx[label_str]
            return patches_tensor, label, img_path
        return patches_tensor, 0


# Get image paths
train_paths = []
for p in ['P1', 'P2', 'P3', 'P4']:
    train_paths.extend(glob.glob(os.path.join(BASE_DIR, p, '*.tif')))

val_paths = glob.glob(os.path.join(BASE_DIR, 'P5', '*.tif'))
test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

# Compute class weights
train_labels_list = []
for f in train_paths:
    label = get_label_from_path(f)
    if label:
        train_labels_list.append(label_to_idx[label])

class_counts = Counter(train_labels_list)
total_samples = len(train_labels_list)
class_weights = torch.tensor([
    total_samples / (num_classes * class_counts[i]) if class_counts[i] > 0 else 0 
    for i in range(num_classes)
], dtype=torch.float32).to(device)

# Transforms
train_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_data = FixedGridDataset(train_paths, transform=train_transform, get_labels=True, augment=True)
val_data = FixedGridDataset(val_paths, transform=val_transform, get_labels=True, augment=False)

# OPTIMIZED: batch_size=1 for stable training (144 samples/batch)
# With 144 crops per image, batch_size=1 means 1 image = 144 patches
# Use gradient accumulation to reach effective batch of 1152
BATCH_SIZE = 1  # 1 image per batch = 144 patches
GRAD_ACCUM_STEPS = 8  # Effective batch = 144 * 8 = 1152
N_CROPS = train_data.n_crops  # 144
EFFECTIVE_BATCH = BATCH_SIZE * N_CROPS  # 1152

train_loader = DataLoader(
    train_data, 
    batch_size=BATCH_SIZE,
    shuffle=True, 
    num_workers=8, 
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_data, 
    batch_size=BATCH_SIZE,
    shuffle=False, 
    num_workers=8, 
    pin_memory=True,
    persistent_workers=True
)

print(f"\nTraining configuration:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Crops per image: {N_CROPS}")
print(f"  Effective batch: {EFFECTIVE_BATCH} samples")
print(f"  Batches per epoch: {len(train_loader)}")

# Model
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = 1280
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, num_classes)
)
model = model.to(device)

# Keep reference to uncompiled model for saving
model_for_save = model

# OPTIMIZATION: Compile model for ~30% speedup (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    print("\n[OPTIMIZATION] Compiling model with torch.compile (reduce-overhead mode)...")
    model = torch.compile(model, mode='reduce-overhead')
    print("[OPTIMIZATION] Model compiled successfully!")

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

# AMP SCALER for mixed precision training
scaler = torch.cuda.amp.GradScaler()

NUM_EPOCHS = 5  # Quick trial
best_val_acc = 0.0

def train_epoch(model, loader, criterion, optimizer, scaler, device, grad_accum_steps=4):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    optimizer.zero_grad()
    
    batch_idx = 0
    for images, labels, _ in pbar:
        batch_idx += 1
        images = images.to(device)  # [B, 144, 3, 224, 224]
        labels = labels.to(device)  # [B] - one label per image
        
        # Reshape: [B, 144, C, H, W] -> [B*144, C, H, W]
        B, N, C, H, W = images.shape
        images = images.view(B * N, C, H, W)
        
        # Replicate labels to match number of crops: [B] -> [B*144]
        labels = labels.repeat(N)  # Repeat each label N times
        
        # Mixed precision forward
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) / grad_accum_steps
        
        # Scaled backward
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if batch_idx % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item() * grad_accum_steps:.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    # Final step if not divisible by grad_accum_steps
    if batch_idx % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            B, N, C, H, W = images.shape
            images = images.view(B * N, C, H, W)
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Aggregate predictions at Well level (average across crops)
            outputs_reshaped = outputs.view(B, N, -1)  # [B, 144, num_classes]
            well_preds = outputs_reshaped.mean(dim=1)   # [B, num_classes]
            
            _, predicted = well_preds.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
    
    return total_loss / len(loader), 100. * correct / total, all_preds, all_labels, all_paths

print(f"\n{'='*60}")
print("STARTING TRIAL TRAINING WITH AMP")
print(f"{'='*60}")

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, grad_accum_steps=GRAD_ACCUM_STEPS)
    val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
    
    epoch_time = time.time() - start_time
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    print(f"Time elapsed: {epoch_time/60:.1f} min")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_for_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, os.path.join(BASE_DIR, 'trial_best_model.pth'))
        print(f"  -> Saved best model (val_acc: {val_acc:.2f}%)")

total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"TRIAL COMPLETE")
print(f"{'='*60}")
print(f"Total time: {total_time/60:.1f} minutes")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"Throughput: {(len(train_paths) * N_CROPS) * NUM_EPOCHS / total_time:.1f} samples/sec")
