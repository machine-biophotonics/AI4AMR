import argparse
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torchvision
from torchvision import models
from torchvision.transforms import ToTensor, Normalize, Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomChoice, ColorJitter, Lambda
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
import json
import re
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from typing import Optional
import random
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.set_num_threads(16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--n_crops', type=int, default=144, help='Number of crops per image')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
args = parser.parse_args()

with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']


def extract_well_from_filename(filename: str) -> Optional[str]:
    match = re.search(r'Well(\w\d+)_', filename)
    if match:
        return match.group(1)
    return None


all_labels = sorted(set(label for pm in plate_maps.values() for label in pm.values()))
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(all_labels)
print(f"Number of classes: {num_classes}")


def get_label_from_path(img_path: str) -> Optional[str]:
    dirname = os.path.dirname(img_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        return plate_maps[plate][well]
    return None


class ShuffledCropSampler:
    def __init__(self, total_crops: int, shuffle: bool = True):
        self.indices = list(range(total_crops))
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.indices)
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


class MixedCropDataset(Dataset):
    def __init__(self, image_paths: list, labels: list, 
                 crop_size: int = 224, grid_size: int = 12,
                 n_crops_per_image: int = 144, augment: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        
        self.total_w = w - crop_size
        self.total_h = h - crop_size
        self.step_w = self.total_w / (grid_size - 1) if grid_size > 1 else 0
        self.step_h = self.total_h / (grid_size - 1) if grid_size > 1 else 0
        
        self.crop_positions = []
        for img_idx in range(len(image_paths)):
            for i in range(grid_size):
                for j in range(grid_size):
                    left = int(j * self.step_w)
                    top = int(i * self.step_h)
                    self.crop_positions.append((img_idx, left, top))
        
        self.n_crops_per_image = min(n_crops_per_image, grid_size * grid_size)
        
        if self.n_crops_per_image < grid_size * grid_size:
            if self.n_crops_per_image == 9:
                center_start = grid_size // 2 - 1
                center_end = grid_size // 2 + 2
                indices = []
                for i in range(center_start, center_end):
                    for j in range(center_start, center_end):
                        indices.append(i * grid_size + j)
            else:
                indices = random.sample(range(grid_size * grid_size), self.n_crops_per_image)
            
            self.crop_positions = []
            for img_idx in range(len(image_paths)):
                for pos_idx in indices:
                    i = pos_idx // grid_size
                    j = pos_idx % grid_size
                    left = int(j * self.step_w)
                    top = int(i * self.step_h)
                    self.crop_positions.append((img_idx, left, top))
        
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.flip_h = RandomHorizontalFlip(p=0.5)
        self.flip_v = RandomVerticalFlip(p=0.5)
        self.rotate_90 = RandomChoice([
            Lambda(lambda x: x.rotate(0)),
            Lambda(lambda x: x.rotate(90)),
            Lambda(lambda x: x.rotate(180)),
            Lambda(lambda x: x.rotate(270)),
        ])
        self.color = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    
    def __len__(self):
        return len(self.crop_positions)
    
    def __getitem__(self, idx):
        img_idx, left, top = self.crop_positions[idx]
        
        img = Image.open(self.image_paths[img_idx]).convert('RGB')
        crop = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        
        if self.augment:
            crop = self.flip_h(crop)
            crop = self.flip_v(crop)
            crop = self.rotate_90(crop)
            crop = self.color(crop)
        
        crop = self.to_tensor(crop)
        crop = self.normalize(crop)
        
        # Store image info for tracking which crop comes from which image
        return crop, self.labels[img_idx], (img_idx, left, top, os.path.basename(self.image_paths[img_idx]))


train_paths = []
for p in ['P1', 'P2', 'P3', 'P4']:
    train_paths.extend(glob.glob(os.path.join(BASE_DIR, p, '*.tif')))

val_paths = glob.glob(os.path.join(BASE_DIR, 'P5', '*.tif'))
test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")


def get_label_for_path(img_path):
    label_str = get_label_from_path(img_path)
    if label_str is None:
        label_str = "Unknown"
    return label_to_idx.get(label_str, 0)

test_labels = [get_label_for_path(p) for p in test_paths]

test_data = MixedCropDataset(
    test_paths, test_labels,
    crop_size=224, grid_size=12,
    n_crops_per_image=args.n_crops, augment=False
)

print(f"Test: {len(test_data)} crops ({args.n_crops} crops/image)")


def worker_init_fn(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def collate_fn(batch):
    crops = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    return crops, labels, metadata

test_loader = DataLoader(
    test_data, 
    batch_size=args.batch_size,
    sampler=ShuffledCropSampler(len(test_data), shuffle=False),
    shuffle=False, 
    num_workers=4, 
    pin_memory=True, 
    prefetch_factor=2, 
    persistent_workers=False,
    worker_init_fn=worker_init_fn,
    collate_fn=collate_fn
)


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x


model = EfficientNetClassifier(num_classes=num_classes)
model = model.to(device)

checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model.pth'), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from best_model.pth")

if 'val_acc' in checkpoint:
    print(f"Original validation accuracy: {checkpoint['val_acc']:.2f}%")

criterion = nn.CrossEntropyLoss()


def evaluate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for crops, labels, _ in pbar:
            crops = crops.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                gene_logits = model(crops)
                loss = criterion(gene_logits, labels)
            
            running_loss += loss.item()
            _, predicted = gene_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total


def get_all_predictions_and_labels(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    metadata = []
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Predicting", leave=False)
    with torch.no_grad():
        for crops_batch, labels, meta in pbar:
            crops_batch = crops_batch.to(device, non_blocking=True)
            labels_dev = labels.to(device, non_blocking=True)
            gene_logits = model(crops_batch)
            probs = torch.softmax(gene_logits, dim=1)
            _, predicted = gene_logits.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels_dev).sum().item()
            
            batch_size = predicted.size(0)
            for i in range(batch_size):
                pred = predicted[i].item()
                lbl = labels[i].item() if hasattr(labels[i], 'item') else labels[i].cpu().numpy()
                prob = probs[i].cpu().numpy()
                
                meta_info = meta[i]
                if isinstance(meta_info, tuple) and len(meta_info) == 4:
                    img_idx, left, top, filename = meta_info
                    meta_info = {
                        'img_idx': int(img_idx),
                        'left': int(left),
                        'top': int(top),
                        'filename': str(filename)
                    }
                
                all_preds.append(pred)
                all_labels.append(lbl)
                all_probs.append(prob)
                metadata.append(meta_info)
    
    test_acc = 100. * correct / total
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), metadata, test_acc


test_preds, test_labels_arr, test_probs, image_metadata, test_acc = get_all_predictions_and_labels(model, test_loader, device)
print(f"\nTest Accuracy: {test_acc:.2f}%")

test_labels_bin = label_binarize(test_labels_arr, classes=list(range(num_classes)))
classes_with_samples = [i for i in range(test_labels_bin.shape[1]) if test_labels_bin[:, i].sum() > 0]

fpr = {}
tpr = {}
roc_auc = {}
precision = {}
recall = {}
ap = {}

for i in tqdm(classes_with_samples, desc="Computing metrics"):
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], test_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    precision[i], recall[i], _ = precision_recall_curve(test_labels_bin[:, i], test_probs[:, i])
    ap[i] = average_precision_score(test_labels_bin[:, i], test_probs[:, i])

mean_roc_auc = np.mean([roc_auc[i] for i in classes_with_samples])
mean_ap = np.mean([ap[i] for i in classes_with_samples])

print(f"\n=== Test Results ===")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Mean ROC AUC: {mean_roc_auc:.4f}")
print(f"Mean Average Precision: {mean_ap:.4f}")

sorted_by_auc = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 5 Classes by AUC:")
for i, val in sorted_by_auc[:5]:
    print(f"  {idx_to_label[i]}: {val:.4f}")
print(f"Bottom 5 Classes by AUC:")
for i, val in sorted_by_auc[-5:]:
    print(f"  {idx_to_label[i]}: {val:.4f}")

output_dir = os.path.join(BASE_DIR, 'eval_results')
os.makedirs(output_dir, exist_ok=True)

# Save per-crop predictions
np.save(os.path.join(output_dir, 'test_preds.npy'), test_preds)
np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels_arr)
np.save(os.path.join(output_dir, 'test_probs.npy'), test_probs)
print(f"\nSaved per-crop predictions to {output_dir}/")

# Save mapping: which crop came from which image with well info
def extract_well_plate_from_filename(filename):
    match = re.search(r'Well(\w\d+)_', filename)
    if match:
        well = match.group(1)
        row = well[0]
        col = int(well[1:])
        return well, row, col
    return None, None, None

crop_to_image_mapping = {}
for idx, meta in enumerate(image_metadata):
    if isinstance(meta, dict):
        filename = meta.get('filename', '')
    else:
        filename = str(meta)
    
    well, row, col = extract_well_plate_from_filename(filename)
    
    crop_to_image_mapping[idx] = {
        'filename': filename,
        'well': well,
        'row': row,
        'col': col,
        'img_idx': meta.get('img_idx', -1) if isinstance(meta, dict) else -1,
        'crop_position': (meta.get('left', -1), meta.get('top', -1)) if isinstance(meta, dict) else (-1, -1)
    }

with open(os.path.join(output_dir, 'crop_to_image_mapping.json'), 'w') as f:
    json.dump(crop_to_image_mapping, f, indent=2)
print(f'Saved crop-to-image mapping ({len(crop_to_image_mapping)} crops) to eval_results/crop_to_image_mapping.json')

import csv
import collections
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(test_labels_arr, test_preds)
np.save(os.path.join(output_dir, 'confusion_matrix.npy'), cm)
print(f"Saved confusion matrix to {output_dir}/confusion_matrix.npy")

with open(os.path.join(output_dir, 'idx_to_label.json'), 'w') as f:
    json.dump(idx_to_label, f)

# Aggregate predictions by well for majority voting (run after ROC/PR curves)
agg_results = {}  # well -> {'plate': plate, 'filename': fname, 'preds': [list of class indices], 'labels': label list}
current_image_labels = test_labels_arr.tolist()
