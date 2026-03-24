"""
Batch size and performance testing script
Test different configurations to find fastest implementation
"""

import os
import sys
import time
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize
from PIL import Image
import glob
import random
import json
from tqdm import tqdm

# Set seeds
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = "/home/student/Desktop/2025_12_19 CRISPRi Reference Plate Imaging"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data info
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

def get_label_from_path(path):
    filename = os.path.basename(path)
    for well, label in plate_maps[os.path.basename(os.path.dirname(path))].items():
        if well in filename:
            return label
    return None

# Simple dataset
class QuickDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = get_label_from_path(self.paths[idx])
        label_idx = label_to_idx.get(label, 0)
        return img, label_idx

# Get train paths
train_paths = []
for p in ['P1']:
    train_paths.extend(glob.glob(os.path.join(BASE_DIR, p, '*.tif')))

print(f"Total images: {len(train_paths)}")

transform = ToTensor()

# Test configurations
configs = [
    {"batch_size": 64, "num_workers": 4},
    {"batch_size": 128, "num_workers": 4},
    {"batch_size": 256, "num_workers": 4},
    {"batch_size": 512, "num_workers": 4},
    {"batch_size": 64, "num_workers": 8},
    {"batch_size": 128, "num_workers": 8},
]

# Create model
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(1280, num_classes)
)
model = model.to(device)
model.eval()

print("\n" + "="*60)
print("PERFORMANCE TESTING")
print("="*60)

results = []

for config in configs:
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    
    print(f"\nTesting: batch_size={batch_size}, num_workers={num_workers}")
    
    # Create dataset with limited samples for quick test
    subset_paths = train_paths[:500]  # Use 500 images for quick test
    
    dataset = QuickDataset(subset_paths, transform=transform)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Warm up
    model.train()
    for i, (imgs, labels) in enumerate(loader):
        if i >= 2:
            break
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.amp.autocast('cuda'):
            _ = model(imgs)
    
    torch.cuda.synchronize()
    
    # Timed run
    model.train()
    start_time = time.time()
    
    batches_processed = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            _ = model(imgs)
        
        batches_processed += 1
        if batches_processed >= 20:  # Process 20 batches
            break
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    images_per_sec = (batches_processed * batch_size) / elapsed
    
    result = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "time": elapsed,
        "images_per_sec": images_per_sec
    }
    results.append(result)
    
    print(f"  Time: {elapsed:.2f}s, Images/sec: {images_per_sec:.1f}")

# Summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Batch Size':<12} {'Workers':<10} {'Images/sec':<15} {'Time (20 batches)':<20}")
print("-"*60)
for r in sorted(results, key=lambda x: -x["images_per_sec"]):
    print(f"{r['batch_size']:<12} {r['num_workers']:<10} {r['images_per_sec']:<15.1f} {r['time']:<20.2f}")

best = max(results, key=lambda x: x["images_per_sec"])
print(f"\nBest config: batch_size={best['batch_size']}, num_workers={best['num_workers']}")
print(f"Best speed: {best['images_per_sec']:.1f} images/sec")