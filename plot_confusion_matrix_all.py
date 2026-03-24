"""
Three Confusion Matrix plots with proper aggregation (same architecture as train.py):
1. Individual: 97 classes but WT grouped (NC_* + WT* → WT)
2. Numbers together: gene_1, gene_2, gene_3 → gene (keeps WT separate)
3. Letters together: first 3 letters grouped (rplA, rplC → rpl)

Aggregation: Image (patches avg) → Point (images avg) → confusion matrix

EXACT SAME architecture as train.py:
- EfficientNet_B0 with pretrained weights
- Dropout(p=0.5) -> Linear(1280, num_classes)
- Same transforms as validation (train_transform)
- 10 patches per image
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torchvision.transforms import v2
from torchvision.transforms import ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, RandomErasing, Normalize, Compose
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import json
import re
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def bin_wt(label):
    if label is None:
        return None
    if label.startswith('NC_') or label.startswith('WT'):
        return 'WT'
    return label


def bin_numbers(label):
    if label is None:
        return None
    base = label
    if base.endswith('_1') or base.endswith('_2') or base.endswith('_3'):
        base = base[:-2]
    return base


def bin_letters(label):
    if label is None:
        return None
    if label.startswith('NC_') or label.startswith('WT'):
        return 'WT'
    base = label
    if base.endswith('_1') or base.endswith('_2') or base.endswith('_3'):
        base = base[:-2]
    if len(base) > 3 and base[:3].isalpha():
        base = base[:3]
    return base


def extract_well_from_filename(filename):
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        row = match.group(1)
        col = int(match.group(2))
        return f"{row}{col:02d}"
    return None


def extract_point_from_filename(filename):
    match = re.search(r'Point(\w)(\d+)_', filename)
    if match:
        row = match.group(1)
        col = int(match.group(2))
        return f"{row}{col:02d}"
    return None


class RandomCenterCrop:
    def __init__(self, size: int, edge_margin: int = 200):
        self.size = size
        self.edge_margin = edge_margin
    
    def __call__(self, img):
        w, h = img.size
        center_w_start = self.edge_margin
        center_w_end = w - self.edge_margin
        center_h_start = self.edge_margin
        center_h_end = h - self.edge_margin
        
        max_top = center_h_end - self.size
        max_left = center_w_end - self.size
        
        if max_top <= 0 or max_left <= 0:
            left = (w - self.size) // 2
            top = (h - self.size) // 2
        else:
            top = random.randint(center_h_start, max_top)
            left = random.randint(center_w_start, max_left)
        
        return img.crop((left, top, left + self.size, top + self.size))


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


def get_label_from_path(img_path):
    dirname = os.path.basename(os.path.dirname(img_path))
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if dirname in plate_maps and well in plate_maps[dirname]:
        return plate_maps[dirname][well]
    return None


class MultiPatchDataset(Dataset):
    def __init__(self, image_paths, transform, n_patches=10):
        self.image_paths = image_paths
        self.transform = transform
        self.n_patches = n_patches
        self.patch_size = 224
        self.edge_margin = 200
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        patches = []
        center_w_start = self.edge_margin
        center_w_end = w - self.edge_margin
        center_h_start = self.edge_margin
        center_h_end = h - self.edge_margin
        
        center_w = center_w_end - center_w_start
        center_h = center_h_end - center_h_start
        
        for _ in range(self.n_patches):
            left = int(np.random.uniform(0, max(1, center_w - self.patch_size)))
            top = int(np.random.uniform(0, max(1, center_h - self.patch_size)))
            left = center_w_start + left
            top = center_h_start + top
            
            patch = image.crop((left, top, left + self.patch_size, top + self.patch_size))
            
            if self.transform:
                patch = self.transform(patch)
            patches.append(patch)
        
        return torch.stack(patches), img_path


def worker_init_fn(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


train_transform = Compose([
    RandomCenterCrop(224, edge_margin=200),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotation(degrees=15),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    ToTensor(),
    RandomErasing(p=0.3, scale=(0.02, 0.15)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    print("Loading model...")
    checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model.pth'), map_location=device)
    
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = 1280
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))
    print(f"Found {len(test_paths)} test images")
    
    test_dataset = MultiPatchDataset(test_paths, transform=train_transform, n_patches=10)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    print("Step 1: Getting predictions for each image (average patches)...")
    image_probs = {}
    image_labels = {}
    
    with torch.no_grad():
        for patches, paths in tqdm(test_loader, desc="Image predictions"):
            batch_size, n_patches, C, H, W = patches.shape
            patches = patches.view(-1, C, H, W).to(device)
            
            outputs = model(patches)
            outputs = outputs.view(batch_size, n_patches, -1)
            probs = torch.softmax(outputs, dim=2)
            avg_probs = probs.mean(dim=1)
            
            for i, path in enumerate(paths):
                true_label = get_label_from_path(path)
                if true_label and true_label in label_to_idx:
                    image_probs[path] = avg_probs[i].cpu().numpy()
                    image_labels[path] = true_label
    
    print(f"  Got predictions for {len(image_probs)} images")
    
    print("\nStep 2: Aggregating predictions by Well (average all images per well)...")
    well_probs = defaultdict(list)
    well_labels = {}
    
    for path, probs in image_probs.items():
        filename = os.path.basename(path)
        well = extract_well_from_filename(filename)
        if well:
            well_probs[well].append(probs)
            well_labels[well] = image_labels[path]
    
    well_avg_probs = {}
    for well, prob_list in well_probs.items():
        well_avg_probs[well] = np.mean(prob_list, axis=0)
    
    print(f"  Aggregated into {len(well_avg_probs)} wells")
    
    all_true_labels = []
    all_pred_probs = []
    
    for well, avg_probs in well_avg_probs.items():
        all_true_labels.append(well_labels[well])
        all_pred_probs.append(avg_probs)
    
    all_pred_probs = np.array(all_pred_probs)
    all_pred_arr = np.array([idx_to_label[p.argmax()] for p in all_pred_probs])
    all_true_arr = np.array(all_true_labels)
    
    print(f"\nTotal point-level predictions: {len(all_true_arr)}")
    print(f"Unique true labels: {len(set(all_true_arr))}")
    print(f"Unique predicted labels: {len(set(all_pred_arr))}")
    
    unique_labels = sorted(set(all_true_arr) | set(all_pred_arr))
    
    def plot_confusion_matrix(true_labels, pred_labels, classes, title, filename, annot=True):
        idx_map = {c: i for i, c in enumerate(classes)}
        true_idx = np.array([idx_map.get(l, -1) for l in true_labels])
        pred_idx = np.array([idx_map.get(l, -1) for l in pred_labels])
        
        valid_mask = (true_idx >= 0) & (pred_idx >= 0)
        true_idx = true_idx[valid_mask]
        pred_idx = pred_idx[valid_mask]
        
        n = len(classes)
        cm = confusion_matrix(true_idx, pred_idx, labels=list(range(n)))
        
        overall_acc = (true_idx == pred_idx).mean() * 100
        
        fig, ax = plt.subplots(figsize=(max(18, n*0.45), max(16, n*0.4)))
        
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt='.2f' if annot else '',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            square=True,
            ax=ax,
            vmin=0,
            vmax=1,
            annot_kws={'fontsize': 7 if n > 30 else 9},
            cbar_kws={'label': 'Recall', 'shrink': 0.8}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'{title}\nOverall Accuracy: {overall_acc:.1f}% | {n} Classes', fontsize=14)
        
        rot = 90 if n > 30 else 45
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rot, ha='right' if rot == 45 else 'center', fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, filename), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {BASE_DIR}/{filename}")
        
        return cm, overall_acc
    
    print("\n" + "="*60)
    print("1. INDIVIDUAL CONFUSION MATRIX (WT grouped)")
    print("="*60)
    classes_indiv = sorted([c for c in set([bin_wt(l) for l in unique_labels]) if c is not None], key=lambda x: (0 if x == 'WT' else 1, x))
    cm_indiv, acc_indiv = plot_confusion_matrix(
        [bin_wt(l) for l in all_true_arr],
        [bin_wt(l) for l in all_pred_arr],
        classes_indiv,
        'Confusion Matrix (Individual Classes, WT Grouped)\nAggregated by Point (images avg → patches avg)',
        'confusion_matrix_individual.png',
        annot=False
    )
    
    print("\n" + "="*60)
    print("2. NUMBERS TOGETHER CONFUSION MATRIX")
    print("="*60)
    classes_nums = sorted([c for c in set([bin_numbers(l) for l in unique_labels]) if c is not None])
    cm_nums, acc_nums = plot_confusion_matrix(
        [bin_numbers(l) for l in all_true_arr],
        [bin_numbers(l) for l in all_pred_arr],
        classes_nums,
        'Confusion Matrix (Numbers Together: gene_1/2/3 → gene)\nAggregated by Point',
        'confusion_matrix_numbers.png',
        annot=True
    )
    
    print("\n" + "="*60)
    print("3. LETTERS TOGETHER CONFUSION MATRIX")
    print("="*60)
    classes_letters = sorted([c for c in set([bin_letters(l) for l in unique_labels]) if c is not None])
    cm_letters, acc_letters = plot_confusion_matrix(
        [bin_letters(l) for l in all_true_arr],
        [bin_letters(l) for l in all_pred_arr],
        classes_letters,
        'Confusion Matrix (Letters Together: rplA/rplC → rpl)\nAggregated by Point',
        'confusion_matrix_letters.png',
        annot=True
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"1. Individual (WT grouped): {len(classes_indiv)} classes, Accuracy: {acc_indiv:.1f}%")
    print(f"2. Numbers together: {len(classes_nums)} classes, Accuracy: {acc_nums:.1f}%")
    print(f"3. Letters together: {len(classes_letters)} classes, Accuracy: {acc_letters:.1f}%")
    
    print("\n" + "="*60)
    print("Per-well predictions (sample):")
    print("="*60)
    for i, (well, label) in enumerate(zip(well_avg_probs.keys(), all_true_labels)):
        pred = all_pred_arr[i]
        correct = "✓" if pred == label else "✗"
        print(f"  {well}: {label} → {pred} {correct}")
        if i >= 19:
            print(f"  ... and {len(well_avg_probs) - 20} more wells")
            break


if __name__ == "__main__":
    main()
