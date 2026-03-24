"""
Confusion Matrix Visualization for CRISPRi Classification.
Generates a heatmap showing model performance across all gene classes.
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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


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
    all_labels = checkpoint['all_labels']
    
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
    all_preds = []
    all_labels_list = []
    all_probs = []
    
    with torch.no_grad():
        for patches, paths in tqdm(test_loader, desc="Predicting"):
            batch_size, n_patches, C, H, W = patches.shape
            patches = patches.view(-1, C, H, W).to(device)
            
            outputs = model(patches)
            outputs = outputs.view(batch_size, n_patches, -1).mean(dim=1)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            for i, path in enumerate(paths):
                label = get_label_from_path(path, plate_maps)
                if label and label in label_to_idx:
                    all_preds.append(predicted[i].item())
                    all_labels_list.append(label_to_idx[label])
                    all_probs.append(probs[i].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels_arr = np.array(all_labels_list)
    all_probs = np.array(all_probs)
    
    print(f"\nTotal predictions: {len(all_preds)}")
    
    cm = confusion_matrix(all_labels_arr, all_preds, labels=list(range(num_classes)))
    
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)
    print(f"\nPer-class accuracy (recall):")
    print(f"  Min: {accuracy_per_class.min()*100:.1f}%")
    print(f"  Max: {accuracy_per_class.max()*100:.1f}%")
    print(f"  Mean: {accuracy_per_class.mean()*100:.1f}%")
    
    overall_accuracy = (all_preds == all_labels_arr).mean() * 100
    print(f"\nOverall accuracy: {overall_accuracy:.2f}%")
    
    label_names = [idx_to_label[i] for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=(24, 22))
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt='.2f',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        square=True,
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Recall (True Positive Rate)', 'shrink': 0.8}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title(
        f'Confusion Matrix (Normalized by Row/True Label)\n'
        f'Overall Accuracy: {overall_accuracy:.1f}% | {num_classes} Classes',
        fontsize=16
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {BASE_DIR}/confusion_matrix.png")
    
    top_confused_idx = np.argsort(-cm.sum(axis=1) - cm.sum(axis=0) + 2*np.diagonal(cm))[:40]
    top_labels = [label_names[i] for i in top_confused_idx]
    
    cm_subset = cm[np.ix_(top_confused_idx, top_confused_idx)]
    cm_subset_normalized = cm_subset.astype('float') / cm_subset.sum(axis=1)[:, np.newaxis]
    cm_subset_normalized = np.nan_to_num(cm_subset_normalized)
    
    fig2, ax2 = plt.subplots(figsize=(16, 14))
    
    sns.heatmap(
        cm_subset_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=top_labels,
        yticklabels=top_labels,
        square=True,
        ax=ax2,
        vmin=0,
        vmax=1,
        annot_kws={'fontsize': 9},
        cbar_kws={'label': 'Recall', 'shrink': 0.8}
    )
    
    ax2.set_xlabel('Predicted Label', fontsize=14)
    ax2.set_ylabel('True Label', fontsize=14)
    ax2.set_title(
        f'Confusion Matrix - Top 40 Confused Classes\n'
        f'(Classes with most total predictions or errors)',
        fontsize=14
    )
    
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix_top_confused.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {BASE_DIR}/confusion_matrix_top_confused.png")
    
    sorted_indices = np.argsort(accuracy_per_class)
    worst_classes = [(label_names[i], accuracy_per_class[i]) for i in sorted_indices[:10]]
    best_classes = [(label_names[i], accuracy_per_class[i]) for i in sorted_indices[-10:][::-1]]
    
    print("\n" + "="*50)
    print("TOP 10 BEST PERFORMING CLASSES:")
    print("="*50)
    for name, acc in best_classes:
        print(f"  {name}: {acc*100:.1f}%")
    
    print("\n" + "="*50)
    print("TOP 10 WORST PERFORMING CLASSES:")
    print("="*50)
    for name, acc in worst_classes:
        print(f"  {name}: {acc*100:.1f}%")
    
    fig3, ax3 = plt.subplots(figsize=(20, 6))
    
    colors = ['red' if acc < 0.5 else 'orange' if acc < 0.7 else 'green' for acc in accuracy_per_class]
    bars = ax3.bar(label_names, accuracy_per_class * 100, color=colors)
    
    ax3.axhline(y=overall_accuracy, color='blue', linestyle='--', linewidth=2, label=f'Overall: {overall_accuracy:.1f}%')
    ax3.axhline(y=100/num_classes, color='gray', linestyle=':', linewidth=1, label=f'Random: {100/num_classes:.1f}%')
    
    ax3.set_xlabel('Gene Class', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title('Per-Class Accuracy (Recall)', fontsize=14)
    ax3.set_ylim(0, 105)
    
    ax3.set_xticklabels(label_names, rotation=90, fontsize=7)
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'per_class_accuracy.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {BASE_DIR}/per_class_accuracy.png")
    
    unique_labels = sorted(set(all_labels_list))
    report = classification_report(all_labels_arr, all_preds, labels=unique_labels, target_names=unique_labels, output_dict=True, zero_division=0)
    
    report_df_path = os.path.join(BASE_DIR, 'classification_report.csv')
    with open(report_df_path, 'w') as f:
        f.write("class,precision,recall,f1-score,support\n")
        for label in unique_labels:
            if label in report:
                r = report[label]
                f.write(f"{label},{r['precision']:.4f},{r['recall']:.4f},{r['f1-score']:.4f},{r['support']}\n")
    print(f"Saved: {report_df_path}")
    
    print("\n" + "="*50)
    print("ALL OUTPUTS SAVED:")
    print("="*50)
    print(f"  1. {BASE_DIR}/confusion_matrix.png")
    print(f"  2. {BASE_DIR}/confusion_matrix_top_confused.png")
    print(f"  3. {BASE_DIR}/per_class_accuracy.png")
    print(f"  4. {BASE_DIR}/classification_report.csv")


if __name__ == "__main__":
    main()
