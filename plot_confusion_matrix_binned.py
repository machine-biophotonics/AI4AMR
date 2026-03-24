"""
Binned Confusion Matrix - bins by gene family (same prefix, ignores _1/_2/_3 and letter suffix).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import json
import re
import random
from tqdm import tqdm
from torchvision.transforms import ToTensor, Normalize, Compose
from sklearn.metrics import confusion_matrix
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def bin_label(label):
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


class MultiPatchDataset(Dataset):
    def __init__(self, image_paths, transform=None, n_patches=10):
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


def extract_well_from_filename(filename):
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        row = match.group(1)
        col = int(match.group(2))
        return f"{row}{col:02d}"
    return None


def load_plate_data():
    with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
        plate_data = json.load(f)
    
    plate_maps = {}
    for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        plate_maps[plate] = {}
        for row, wells in plate_data[plate].items():
            for col, info in wells.items():
                well = f"{row}{int(col):02d}"
                plate_maps[plate][well] = info['id']
    return plate_maps


def get_label_from_path(img_path, plate_maps):
    dirname = os.path.basename(os.path.dirname(img_path))
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if dirname in plate_maps and well in plate_maps[dirname]:
        return plate_maps[dirname][well]
    return None


def worker_init_fn(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def main():
    print("Loading model...")
    checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model.pth'), map_location=device)
    
    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = checkpoint['idx_to_label']
    num_classes = checkpoint['num_classes']
    
    print(f"Loaded model with {num_classes} classes")
    
    model = models.efficientnet_b0(weights=None)
    in_features = 1280
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    plate_maps = load_plate_data()
    
    test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))
    print(f"Found {len(test_paths)} test images")
    
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = MultiPatchDataset(test_paths, transform=transform, n_patches=10)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False, 
        num_workers=8,
        worker_init_fn=worker_init_fn
    )
    
    print("Getting predictions...")
    all_true_binned = []
    all_pred_binned = []
    
    with torch.no_grad():
        for patches, paths in tqdm(test_loader, desc="Predicting"):
            batch_size, n_patches, C, H, W = patches.shape
            patches = patches.view(-1, C, H, W).to(device)
            
            outputs = model(patches)
            outputs = outputs.view(batch_size, n_patches, -1).mean(dim=1)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            for i, path in enumerate(paths):
                true_label = get_label_from_path(path, plate_maps)
                if true_label and true_label in label_to_idx:
                    true_idx = label_to_idx[true_label]
                    true_name = idx_to_label[true_idx]
                    pred_idx = predicted[i].item()
                    pred_name = idx_to_label[pred_idx]
                    
                    true_binned = bin_label(true_name)
                    pred_binned = bin_label(pred_name)
                    
                    all_true_binned.append(true_binned)
                    all_pred_binned.append(pred_binned)
    
    all_binned_labels = sorted(set(all_true_binned + all_pred_binned))
    BINNED_TO_IDX = {c: i for i, c in enumerate(all_binned_labels)}
    BINNED_CLASSES = all_binned_labels
    
    print(f"\nBinned classes ({len(BINNED_CLASSES)}): {BINNED_CLASSES}")
    
    all_true_arr = np.array([BINNED_TO_IDX[l] for l in all_true_binned])
    all_pred_arr = np.array([BINNED_TO_IDX[l] for l in all_pred_binned])
    
    print(f"Total predictions: {len(all_true_arr)}")
    
    n_binned = len(BINNED_CLASSES)
    cm = confusion_matrix(all_true_arr, all_pred_arr, labels=list(range(n_binned)))
    
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
    overall_accuracy = (all_true_arr == all_pred_arr).mean() * 100
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=BINNED_CLASSES,
        yticklabels=BINNED_CLASSES,
        square=True,
        ax=ax,
        vmin=0,
        vmax=1,
        annot_kws={'fontsize': 9},
        cbar_kws={'label': 'Recall', 'shrink': 0.8}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title(
        f'Binned Confusion Matrix (Gene Family Level)\n'
        f'Overall Accuracy: {overall_accuracy:.1f}% | {n_binned} Classes',
        fontsize=16
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix_binned.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {BASE_DIR}/confusion_matrix_binned.png")
    
    print("\nPer-class accuracy:")
    for i, c in enumerate(BINNED_CLASSES):
        print(f"  {c}: {accuracy_per_class[i]*100:.1f}%")


if __name__ == "__main__":
    main()
