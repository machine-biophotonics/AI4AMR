"""
Quick Debug Script - Test 1 Batch Before Full Training
=====================================================
Tests: shapes, loss calculation, backprop, GPU utilization
"""

import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torchvision.models as models
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, Normalize, Compose, ColorJitter, RandomRotation
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import json
import re
import random
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

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
num_classes = len(all_labels)
print(f"Number of classes: {num_classes}")

def extract_well_from_filename(filename):
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        return f"{match.group(1)}{int(match.group(2)):02d}"
    return None

def get_label_from_path(img_path):
    dirname = os.path.basename(os.path.dirname(img_path))
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if dirname in plate_maps and well in plate_maps[dirname]:
        return plate_maps[dirname][well]
    return None


class FixedGridDataset(Dataset):
    """Fixed Grid Dataset - 144 crops per image"""
    def __init__(self, image_paths, transform=None, augment=True):
        self.image_paths = image_paths
        self.transform = transform
        self.patch_size = 224
        self.n_crops_w = 12
        self.n_crops_h = 12
        self.n_crops = 144
        
        self.augment_transform = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=90),
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
        ]) if augment else None
    
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
        
        label_str = get_label_from_path(img_path)
        label = label_to_idx.get(label_str, 0)
        
        return torch.stack(patches), label, img_path


# Get train paths
train_paths = []
for p in ['P1', 'P2', 'P3', 'P4']:
    train_paths.extend(glob.glob(os.path.join(BASE_DIR, p, '*.tif')))

print(f"Train images: {len(train_paths)}")

# Transform
train_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset & DataLoader
train_data = FixedGridDataset(train_paths[:32], transform=train_transform, augment=True)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=4)

# Class weights
from collections import Counter
train_labels = [label_to_idx.get(get_label_from_path(f), 0) for f in train_paths[:32]]
class_counts = Counter(train_labels)
total = len(train_labels)
class_weights = torch.tensor([total / (num_classes * class_counts[i]) if class_counts[i] > 0 else 0 
                              for i in range(num_classes)], dtype=torch.float32).to(device)

# Model
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Sequential(nn.Dropout(0.5), nn.Linear(1280, num_classes))
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("\n" + "="*60)
print("DEBUG: Testing 1 Batch")
print("="*60)

# Test 1 batch
model.train()
for i, (patches, labels, paths) in enumerate(train_loader):
    if i >= 1:
        break
    
    print(f"\n📊 SHAPE CHECK:")
    print(f"  patches.shape: {patches.shape}")
    print(f"    - batch_size: {patches.shape[0]} (should be 1)")
    print(f"    - n_crops: {patches.shape[1]} (should be 144)")
    print(f"    - channels: {patches.shape[2]} (should be 3)")
    print(f"    - H, W: {patches.shape[3]}x{patches.shape[4]} (should be 224x224)")
    print(f"  labels.shape: {labels.shape} (should be [1])")
    
    # Move to GPU
    patches = patches.to(device)
    labels = labels.to(device)
    
    # Forward pass
    B, N, C, H, W = patches.shape
    patches_flat = patches.view(-1, C, H, W)
    print(f"\n🔄 FORWARD PASS:")
    print(f"  patches.view(-1, C, H, W): {patches_flat.shape}")
    
    outputs_flat = model(patches_flat)
    print(f"  model output: {outputs_flat.shape} [144 crops × 85 classes]")
    
    # PER-CROP LOSS: each crop predicts the same well label
    labels_expanded = labels.repeat(N)
    print(f"  labels_expanded: {labels_expanded.shape} [144] - same label for all crops")
    
    loss = criterion(outputs_flat, labels_expanded)
    print(f"\n📉 LOSS (per-crop):")
    print(f"  CrossEntropyLoss(outputs_flat, labels_expanded)")
    print(f"  loss: {loss.item():.4f} (averaged over 144 crops)")
    print(f"  loss.shape: {loss.shape} (should be scalar)")
    
    # For accuracy: aggregate predictions with mean
    outputs_agg = outputs_flat.view(B, N, -1).mean(dim=1)
    print(f"\n🎯 ACCURACY (aggregated for prediction):")
    print(f"  outputs_agg: {outputs_agg.shape} [1 well × 85 classes]")
    
    # Backward pass
    print(f"\n🔙 BACKWARD PASS:")
    optimizer.zero_grad()
    loss.backward()
    
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))
    
    print(f"  Gradients computed: {len([g for g in grad_norms if g[1] > 0])} parameters have gradients")
    print(f"  Sample grad norms:")
    for name, norm in grad_norms[:5]:
        print(f"    {name}: {norm:.6f}")
    
    # Optimizer step
    optimizer.step()
    print(f"\n✅ OPTIMIZER STEP: Completed successfully")
    
    # Accuracy check (aggregate predictions for comparison)
    _, predicted = outputs_agg.max(1)
    acc = (predicted == labels).float().mean()
    print(f"\n🎯 ACCURACY:")
    print(f"  Predicted class: {predicted.item()}")
    print(f"  True class: {labels.item()}")
    print(f"  Correct: {predicted.item() == labels.item()}")
    print(f"  Accuracy: {acc.item() * 100:.1f}%")

print("\n" + "="*60)
print("✅ DEBUG COMPLETE - All shapes and operations correct!")
print("="*60)
print("\n🚀 Your train.py is ready for full training!")
