#!/usr/bin/env python3
"""
MIL training with cycle-based crop extraction + neighbors
Training: 9 crops (center + 3x3 neighborhood)
Validation/Test: single center crop
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
import os
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

from mil_model import AttentionMILModel, MultiCropDataset, get_gene_from_path, extract_well_from_filename

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--test_plate', type=str, default='P6')
args = parser.parse_args()

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

def extract_gene(label):
    return label

all_genes = sorted(set(extract_gene(label) for pm in plate_maps.values() for label in pm.values()))
gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
num_classes = len(all_genes)
print(f"Classes: {num_classes}")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, f'fold_{args.test_plate}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
train_val_plates = [p for p in all_plates if p != args.test_plate]
train_plates = train_val_plates[:4]
val_plates = train_val_plates[4:]

def get_image_paths_for_plate(plate):
    plate_dir = os.path.join(BASE_DIR, plate)
    if not os.path.exists(plate_dir):
        return []
    paths = []
    for pattern in ['*.tif', '*.tiff', '*.png']:
        paths.extend(glob.glob(os.path.join(plate_dir, '**', pattern), recursive=True))
    valid_paths = []
    for path in paths:
        well = extract_well_from_filename(os.path.basename(path))
        if well and well in plate_maps.get(plate, {}):
            valid_paths.append(path)
    return valid_paths

train_paths, train_labels = [], []
val_paths, val_labels = [], []
test_paths, test_labels = [], []

for plate in train_plates:
    for path in get_image_paths_for_plate(plate):
        train_paths.append(path)
        train_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])

for plate in val_plates:
    for path in get_image_paths_for_plate(plate):
        val_paths.append(path)
        val_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])

for plate in [args.test_plate]:
    for path in get_image_paths_for_plate(plate):
        test_paths.append(path)
        test_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

class_counts = Counter(train_labels)
total = len(train_labels)
class_weights = torch.tensor([total / (num_classes * class_counts[i]) for i in range(num_classes)], device=device)
class_weights = class_weights / class_weights.sum() * num_classes

def get_combined_weights(plates, labels):
    plate_counts = Counter(plates)
    n_plates = len(plate_counts)
    domain_weights = {p: 1.0 / np.sqrt(c) for p, c in plate_counts.items()}
    dom_sum = sum(domain_weights.values())
    domain_weights = {k: v / dom_sum * n_plates for k, v in domain_weights.items()}
    
    weights = []
    for plate, label in zip(plates, labels):
        class_w = class_weights[label].item()
        domain_w = domain_weights.get(plate, 1.0)
        weights.append(class_w * domain_w)
    weights = np.array(weights)
    weights = weights / weights.mean()
    return torch.tensor(weights, device=device)

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()

def weighted_focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0):
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return (focal * weights).mean()

def attention_entropy_loss(attn_weights):
    """Entropy regularization to prevent attention collapse"""
    # attn_weights: (B, N, H)
    entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=1).mean()
    return entropy

train_dataset = MultiCropDataset(train_paths, train_labels, augment=True, seed=SEED)
val_dataset = MultiCropDataset(val_paths, val_labels, augment=False, seed=SEED)
test_dataset = MultiCropDataset(test_paths, test_labels, augment=False, seed=SEED)

train_dataset.set_epoch(0)
val_dataset.set_epoch(0)
test_dataset.set_epoch(0)

def worker_init_fn(worker_id):
    random.seed(SEED + worker_id)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True)

print(f"Crops per image: 9 (center + 8 neighbors)")

model = AttentionMILModel(num_classes=num_classes)
model = model.to(device)

backbone_params = [p for n, p in model.named_parameters() if 'attention_pool' not in n and 'classifier' not in n]
attention_params = [p for n, p in model.named_parameters() if 'attention_pool' in n or 'classifier' in n]

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': args.lr * 0.1},
    {'params': attention_params, 'lr': args.lr}
], weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = os.path.join(OUTPUT_DIR, f'training_metrics_{timestamp}.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc', 'lr'])

best_val_auc = 0.0

print("Training...")
for epoch in range(args.epochs):
    train_dataset.set_epoch(epoch)
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs, attn_weights = model(images, return_attention=True)
        
        # Main loss + entropy regularization
        main_loss = weighted_focal_loss(outputs, labels, class_weights[labels])
        ent_loss = attention_entropy_loss(attn_weights)
        loss = main_loss + 0.01 * ent_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        run_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    scheduler.step()
    
    train_acc = 100. * correct / total
    avg_train_loss = run_loss / len(train_loader)
    
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images, return_attention=True)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    val_auc = roc_auc_score(all_labels_bin, np.array(all_probs), average='macro')
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}, LR={current_lr:.2e}")
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_train_loss, train_acc, 0, val_acc, val_auc, current_lr])
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'best_model.pth'))

print("Testing...")
checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs, _ = model(images, return_attention=True)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
test_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
test_auc = roc_auc_score(test_labels_bin, np.array(all_probs), average='macro')
test_ap = average_precision_score(test_labels_bin, np.array(all_probs), average='macro')

print(f"Test Acc: {test_acc:.2f}%, Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")

results = {
    'timestamp': timestamp,
    'config': {'epochs': args.epochs, 'batch_size': args.batch_size, 'lr': args.lr, 'test_plate': args.test_plate},
    'results': {'best_val_auc': float(best_val_auc), 'test_acc': float(test_acc), 'test_auc': float(test_auc), 'test_ap': float(test_ap)}
}

with open(os.path.join(OUTPUT_DIR, 'training_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("Done!")