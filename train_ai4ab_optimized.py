"""
AI4AB-Style Training Script - OPTIMIZED VERSION
================================================
Speed optimizations:
1. Mixed Precision (AMP) - 2-3x speedup
2. torch.compile() - 1.5-2x speedup
3. Increased batch size - better GPU utilization
4. Channels-last memory format - faster convolutions
5. Optimized DataLoader settings
"""

import argparse
import os
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn, optim
import torchvision.models as models
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob
import json
import re
import random
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='Resume from best_model.pth')
parser.add_argument('--no-compile', action='store_true', help='Skip torch.compile (debug mode)')
args = parser.parse_args()

# ============================================
# CONFIGURATION - OPTIMIZED
# ============================================
CONFIG = {
    'crop_size': 500,        # Size of each crop (pixels)
    'resize_size': 256,      # Resize to 256x256 (AI4AB default)
    'n_crops': 9,            # 3x3 grid = 9 crops per image
    'batch_size': 32,        # OPTIMIZED: Increased from 16
    'epochs': 250,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'num_workers': 16,        # OPTIMIZED: Use more workers (36 CPUs available)
    'use_amp': True,         # OPTIMIZED: Mixed precision
    'use_compile': True,     # OPTIMIZED: torch.compile
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
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(all_labels)
print(f"Classes: {num_classes}")

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

# ============================================
# MODEL (AI4AB-Style AvgPoolCNN)
# ============================================
class AvgPoolCNN(nn.Module):
    def __init__(self, num_classes=2, num_channels=3, pretrained=True, n_crops=9):
        super().__init__()
        
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        self.backbone.classifier[1] = nn.Identity()
        
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

# ============================================
# DATASET (AI4AB-Style with Random Crops)
# ============================================
class CenterCropMultiChannel:
    def __init__(self, crop_size, n_crops):
        self.crop_size = crop_size
        self.n_crops = n_crops
    
    def __call__(self, img):
        w, h = img.size
        crops = []
        
        grid_size = int(np.sqrt(self.n_crops))
        step_w = (w - self.crop_size) // (grid_size - 1) if grid_size > 1 else w - self.crop_size
        step_h = (h - self.crop_size) // (grid_size - 1) if grid_size > 1 else h - self.crop_size
        
        start_w = (w - self.crop_size - step_w * (grid_size - 1)) // 2
        start_h = (h - self.crop_size - step_h * (grid_size - 1)) // 2
        
        for i in range(grid_size):
            for j in range(grid_size):
                left = start_w + j * step_w
                top = start_h + i * step_h
                crop = img.crop((left, top, left + self.crop_size, top + self.crop_size))
                crops.append(crop)
        
        return crops


class ImageDatasetAI4AB(Dataset):
    def __init__(self, image_paths, train=True):
        self.image_paths = image_paths
        self.train = train
        self.n_crops = CONFIG['n_crops']
        
        self.augment = T.Compose([
            T.RandomVerticalFlip(p=0.5),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=90),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
        ]) if train else None
        
        self.crop_extractor = CenterCropMultiChannel(
            crop_size=CONFIG['crop_size'],
            n_crops=self.n_crops
        )
        
        self.resize_transform = T.Resize((CONFIG['resize_size'], CONFIG['resize_size']))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.augment is not None:
            image = self.augment(image)
        
        crops = self.crop_extractor(image)
        
        crops_tensor = torch.stack([
            self.normalize(self.to_tensor(crop.resize((CONFIG['resize_size'], CONFIG['resize_size']))))
            for crop in crops
        ])
        
        label_str = get_label_from_path(img_path)
        label = label_to_idx.get(label_str, 0)
        
        return crops_tensor, label


# ============================================
# DATASET PREPARATION
# ============================================
train_paths, val_paths, test_paths = [], [], []

for plate in ['P1', 'P2', 'P3', 'P4']:
    train_paths.extend(glob.glob(os.path.join(BASE_DIR, plate, '*.tif')))
val_paths = glob.glob(os.path.join(BASE_DIR, 'P5', '*.tif'))
test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

train_dataset = ImageDatasetAI4AB(train_paths, train=True)
val_dataset = ImageDatasetAI4AB(val_paths, train=False)
test_dataset = ImageDatasetAI4AB(test_paths, train=False)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True, 
    num_workers=CONFIG['num_workers'], 
    pin_memory=True, 
    drop_last=True,
    persistent_workers=True, 
    prefetch_factor=4,
    worker_init_fn=seed_worker, 
    generator=g
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=False, 
    num_workers=CONFIG['num_workers'], 
    pin_memory=True,
    persistent_workers=True, 
    prefetch_factor=4,
    worker_init_fn=seed_worker, 
    generator=g
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=False, 
    num_workers=CONFIG['num_workers'], 
    pin_memory=True,
    worker_init_fn=seed_worker, 
    generator=g
)

# ============================================
# CLASS WEIGHTS
# ============================================
train_labels = [label_to_idx.get(get_label_from_path(f), 0) for f in train_paths]
class_counts = Counter(train_labels)
total = len(train_labels)
class_weights = torch.tensor([
    total / (num_classes * class_counts[i]) if class_counts[i] > 0 else 0 
    for i in range(num_classes)
], dtype=torch.float32).to(device)
print(f"Class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}")

# ============================================
# MODEL, LOSS, OPTIMIZER
# ============================================
model = AvgPoolCNN(num_classes=num_classes, pretrained=True, n_crops=CONFIG['n_crops'])
model = model.to(device)

# OPTIMIZED: Use channels_last for faster convolutions
model = model.to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)

# ============================================
# COMPILE MODEL (OPTIMIZATION)
# ============================================
if CONFIG['use_compile'] and not args.no_compile:
    print("\nCompiling model with torch.compile()...")
    compile_start = time.time()
    model = torch.compile(model, mode="reduce-overhead")
    print(f"Compilation done in {time.time() - compile_start:.1f}s")

# ============================================
# AMP SCALER (OPTIMIZATION)
# ============================================
use_amp = CONFIG['use_amp'] and torch.cuda.is_available()
scaler = torch.amp.GradScaler('cuda') if use_amp else None
if use_amp:
    print("AMP (Mixed Precision) enabled")

# ============================================
# TRAINING FUNCTIONS
# ============================================
def train_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    total_loss, correct, total = 0, 0, 0
    epoch_start = time.time()
    
    pbar = tqdm(loader, desc="Training")
    for crops, labels in pbar:
        crops = crops.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                logits, _ = model(crops)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(crops)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    epoch_time = time.time() - epoch_start
    samples_per_sec = total / epoch_time
    
    return total_loss / len(loader), 100. * correct / total, samples_per_sec


def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for crops, labels in tqdm(loader, desc="Evaluating"):
            crops = crops.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(crops)
                    loss = criterion(logits, labels)
            else:
                logits, _ = model(crops)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), 100. * correct / total


def get_predictions(model, loader, device, use_amp=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for crops, labels in tqdm(loader, desc="Predicting"):
            crops = crops.to(device, non_blocking=True)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(crops)
            else:
                logits, _ = model(crops)
            
            probs = torch.softmax(logits, dim=1)
            
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.append(probs.cpu())
    
    return np.array(all_preds), np.array(all_labels), torch.cat(all_probs).numpy()


# ============================================
# TRAINING LOOP
# ============================================
best_val_acc = 0
train_losses, train_accs, val_losses, val_accs = [], [], [], []
total_train_time = 0

print(f"\n{'='*60}")
print(f"AI4AB-STYLE TRAINING - OPTIMIZED")
print(f"Crops: {CONFIG['n_crops']} per image, Size: {CONFIG['resize_size']}x{CONFIG['resize_size']}")
print(f"Batch: {CONFIG['batch_size']}, LR: {CONFIG['lr']}")
print(f"AMP: {use_amp}, Compile: {CONFIG['use_compile'] and not args.no_compile}")
print(f"{'='*60}\n")

for epoch in range(CONFIG['epochs']):
    epoch_start = time.time()
    print(f"Epoch {epoch+1}/{CONFIG['epochs']} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    train_loss, train_acc, samples_per_sec = train_epoch(
        model, train_loader, criterion, optimizer, device, scaler, use_amp
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)
    
    scheduler.step()
    
    epoch_time = time.time() - epoch_start
    total_train_time += epoch_time
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Train: {train_loss:.4f} / {train_acc:.1f}% | Val: {val_loss:.4f} / {val_acc:.1f}% | Time: {epoch_time:.1f}s | Speed: {samples_per_sec:.1f} img/s")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Handle torch.compile model saving - extract underlying module state_dict
        if hasattr(model, '_orig_mod'):
            model_state = model._orig_mod.state_dict()
        else:
            model_state = model.state_dict()
        
        torch.save({
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
            'num_classes': num_classes,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'epoch': epoch + 1,
        }, os.path.join(BASE_DIR, 'best_model_ai4ab_optimized.pth'))
        print(f"  -> Saved best model ({val_acc:.1f}%)")

print(f"\nTotal training time: {total_train_time/60:.1f} minutes")
print(f"Average speed: {len(train_paths) * CONFIG['epochs'] / total_train_time:.1f} images/second")

# ============================================
# TEST EVALUATION
# ============================================
print(f"\n{'='*60}")
print("TEST EVALUATION")
print(f"{'='*60}")

# Load best model - create fresh model for loading (don't load into compiled model)
eval_model = AvgPoolCNN(num_classes=num_classes, pretrained=False, n_crops=CONFIG['n_crops'])
eval_model = eval_model.to(device)

checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model_ai4ab_optimized.pth'), map_location=device, weights_only=True)
eval_model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_acc = evaluate(eval_model, test_loader, criterion, device, use_amp)
print(f"Test Accuracy: {test_acc:.1f}%")

# ============================================
# PLOTS & METRICS
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_losses, label='Train')
axes[0].plot(val_losses, label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss Curves')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(train_accs, label='Train')
axes[1].plot(val_accs, label='Val')
axes[1].axhline(y=test_acc, color='r', linestyle='--', label=f'Test: {test_acc:.1f}%')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Accuracy Curves')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'training_plots_ai4ab_optimized.png'), dpi=150)
plt.close()

# ROC & PR Curves
test_preds, test_labels, test_probs = get_predictions(eval_model, test_loader, device, use_amp)
test_labels_bin = label_binarize(test_labels, classes=list(range(num_classes)))
classes_with_samples = [i for i in range(num_classes) if test_labels_bin[:, i].sum() > 0]

fpr, tpr, roc_auc, precision, recall, ap = {}, {}, {}, {}, {}, {}
for i in classes_with_samples:
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], test_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    precision[i], recall[i], _ = precision_recall_curve(test_labels_bin[:, i], test_probs[:, i])
    ap[i] = average_precision_score(test_labels_bin[:, i], test_probs[:, i])

print(f"\nMean ROC AUC: {np.mean([roc_auc[i] for i in classes_with_samples]):.4f}")
print(f"Mean AP: {np.mean([ap[i] for i in classes_with_samples]):.4f}")

sorted_auc = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 5 by AUC:")
for i, val in sorted_auc[:5]:
    print(f"  {idx_to_label[i]}: {val:.4f}")

print(f"\nBottom 5 by AUC:")
for i, val in sorted_auc[-5:]:
    print(f"  {idx_to_label[i]}: {val:.4f}")

import csv
csv_path = os.path.join(BASE_DIR, 'class_metrics_ai4ab_optimized.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['class', 'idx', 'auc', 'ap', 'samples'])
    for i in classes_with_samples:
        writer.writerow([idx_to_label[i], i, f'{roc_auc.get(i, 0):.4f}', 
                       f'{ap.get(i, 0):.4f}', int(test_labels_bin[:, i].sum())])

print(f"\n{'='*60}")
print("DONE!")
print(f"Model: best_model_ai4ab_optimized.pth")
print(f"Plots: training_plots_ai4ab_optimized.png")
print(f"Metrics: class_metrics_ai4ab_optimized.csv")
print(f"{'='*60}")
