"""
Quick Debug Script - Test 1 Batch Before Full Training
=======================================================
Tests: shapes, loss calculation, backprop
"""

import os
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn, optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob
import json
import re
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Config (AI4AB Style)
CONFIG = {
    'crop_size': 500,
    'resize_size': 256,
    'n_crops': 9,
    'batch_size': 16,
}

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
print(f"Classes: {num_classes}")

def get_label_from_path(img_path):
    dirname = os.path.basename(os.path.dirname(img_path))
    filename = os.path.basename(img_path)
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        well = f"{match.group(1)}{int(match.group(2)):02d}"
        if dirname in plate_maps and well in plate_maps[dirname]:
            return plate_maps[dirname][well]
    return None


class AvgPoolCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, n_crops=9):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        self.backbone.classifier[1] = nn.Identity()  # Keep Dropout, replace Linear
        self.avg_pool_2d = nn.AdaptiveAvgPool2d(output_size=1)
        self.avg_pool_1d = nn.AvgPool1d(kernel_size=n_crops)
        self.fc_final = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        bs, ncrops, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        features = self.backbone.features(x)
        features = self.avg_pool_2d(features).view(bs, ncrops, -1)
        feat_vec = self.avg_pool_1d(features.permute(0, 2, 1)).view(bs, -1)
        logits = self.fc_final(feat_vec)
        return logits, feat_vec


class SimpleDataset(Dataset):
    """AI4AB-style dataset with augmentations (like training script)."""
    def __init__(self, paths, train=True):
        from torchvision import transforms as T
        self.paths = paths[:32]
        self.n_crops = CONFIG['n_crops']
        self.resize_size = CONFIG['resize_size']
        
        # Augmentations (AI4AB style) - applied to FULL image before cropping
        if train:
            self.augment = T.Compose([
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=90),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
            ])
        else:
            self.augment = None
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        from torchvision import transforms as T
        img = Image.open(self.paths[idx]).convert('RGB')
        w, h = img.size
        
        # Step 1: Apply augmentations to FULL image (AI4AB style)
        if self.augment is not None:
            img = self.augment(img)
        
        # Step 2: Extract 3x3 crops from center
        crops = []
        grid = int(np.sqrt(self.n_crops))
        step = min(w, h) // (grid + 1)
        start = step
        
        for i in range(grid):
            for j in range(grid):
                left = start + j * step
                top = start + i * step
                crop = img.crop((left, top, left + CONFIG['crop_size'], top + CONFIG['crop_size']))
                crop = crop.resize((self.resize_size, self.resize_size))
                crop = T.ToTensor()(crop)
                crop = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(crop)
                crops.append(crop)
        
        label = label_to_idx.get(get_label_from_path(self.paths[idx]), 0)
        return torch.stack(crops), label


# Load data
train_paths = []
for p in ['P1', 'P2', 'P3', 'P4']:
    train_paths.extend(glob.glob(os.path.join(BASE_DIR, p, '*.tif')))

dataset = SimpleDataset(train_paths, train=True)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

# Model
model = AvgPoolCNN(num_classes=num_classes, pretrained=True, n_crops=CONFIG['n_crops'])
model = model.to(device)

# Class weights
from collections import Counter
labels = [label_to_idx.get(get_label_from_path(f), 0) for f in train_paths[:32]]
counts = Counter(labels)
total = len(labels)
weights = torch.tensor([total / (num_classes * counts[i]) if counts[i] > 0 else 0 
                         for i in range(num_classes)], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("\n" + "="*60)
print("DEBUG: Testing 1 Batch")
print("="*60)

# Test 1 batch
model.train()
for i, (crops, labels) in enumerate(loader):
    if i >= 1:
        break
    
    print(f"\n📊 SHAPES:")
    print(f"  crops: {crops.shape} [B={crops.shape[0]}, n_crops={crops.shape[1]}, C=3, H={crops.shape[3]}, W={crops.shape[4]}]")
    print(f"  labels: {labels.shape}")
    
    crops = crops.to(device)
    labels = labels.to(device)
    
    print(f"\n🔄 FORWARD:")
    logits, feat_vec = model(crops)
    print(f"  logits: {logits.shape} [B, num_classes]")
    print(f"  feat_vec: {feat_vec.shape} [B, 1280]")
    
    print(f"\n📉 LOSS:")
    loss = criterion(logits, labels)
    print(f"  loss: {loss.item():.4f}")
    
    print(f"\n🔙 BACKWARD:")
    optimizer.zero_grad()
    loss.backward()
    
    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  Parameters with gradients: {grad_count}")
    
    optimizer.step()
    print(f"  Optimizer step: ✅")
    
    acc = (logits.argmax(1) == labels).float().mean()
    print(f"\n🎯 ACCURACY: {acc.item() * 100:.1f}%")
    print(f"  Predicted: {logits.argmax(1).cpu().tolist()}")
    print(f"  True: {labels.cpu().tolist()}")

print("\n" + "="*60)
print("✅ DEBUG COMPLETE - All shapes and operations correct!")
print("="*60)
print("\n🚀 Ready to run: python train_ai4ab.py")
