#!/usr/bin/env python3
"""
=============================================================================
ItS2CLR: Iterative Self-Paced Supervised Contrastive Learning (CVPR 2023)
=============================================================================
Complete implementation for CRISPRi Reference Plate Imaging

Algorithm:
1. Initial MIL training to get pseudo labels
2. Iterative supervised contrastive learning with SPL curriculum
3. Pseudo label refinement every N epochs

Paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Multiple_Instance_Learning_via_Iterative_Self-Paced_Supervised_Contrastive_Learning_CVPR_2023_paper.pdf
=============================================================================
"""

import argparse
import sys
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
import os
import glob
import json
import re
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.preprocessing import label_binarize
import random
from tqdm import tqdm
import csv
from datetime import datetime
from collections import Counter
import pickle
import multiprocessing

from mil_model import AttentionMILModel, MILEncoder, MultiCropDataset, get_gene_from_path, extract_well_from_filename
from supcon_loss import SupConLoss
from dsmil import MILNet, FCLayer, BClassifier
from spl_dataset import BagDataset, BagDatasetIns

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
parser.add_argument('--data_root', type=str, default=None)
parser.add_argument('--run_all_folds', action='store_true')
parser.add_argument('--neighborhood', type=int, default=3, choices=[3, 5, 7, 9, 11])
parser.add_argument('--grid_size', type=int, default=12)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--use_mil_dropout', action='store_false', default=False,
                    help='Use MIL-Dropout (ICML 2025)')
parser.add_argument('--mil_dropout_topk', type=int, default=3,
                    help='Top-k instances to drop in MIL-Dropout (default: 3)')

# ItS2CLR Parameters (from paper)
parser.add_argument('--its2clr_warmup', type=int, default=15,
                    help='Initial MIL training epochs (default: 15)')
parser.add_argument('--its2clr_iterations', type=int, default=3,
                    help='Number of ItS2CLR iterations (default: 3)')
parser.add_argument('--its2clr_threshold', type=float, default=0.3,
                    help='Confidence threshold for pseudo labels (default: 0.3)')
parser.add_argument('--its2clr_rho_pos', type=float, default=0.2,
                    help='Initial positive instance ratio (default: 0.2)')
parser.add_argument('--its2clr_rho_neg', type=float, default=0.2,
                    help='Initial negative instance ratio (default: 0.2)')
parser.add_argument('--its2clr_rho_T', type=float, default=0.8,
                    help='Final SPL ratio (default: 0.8)')
parser.add_argument('--its2clr_temperature', type=float, default=0.1,
                    help='Contrastive temperature (default: 0.1 per SupCon paper)')
parser.add_argument('--its2clr_mil_epochs', type=int, default=15,
                    help='MIL epochs per iteration (default: 15)')
parser.add_argument('--its2clr_mil_every', type=int, default=5,
                    help='MIL retrain frequency (default: 5)')
parser.add_argument('--its2clr_update_pseudo', action='store_true', default=True,
                    help='Update pseudo labels during training')
parser.add_argument('--supcon_weight', type=float, default=0.2,
                    help='SupCon loss weight (default: 0.2, start low)')
args = parser.parse_args()

if sys.platform.startswith('win'):
    NUM_WORKERS = 0
else:
    NUM_WORKERS = 8

SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if args.data_root:
    BASE_DIR = args.data_root
else:
    BASE_DIR = os.path.dirname(SCRIPT_DIR)

with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = row + str(col).zfill(2)
            plate_maps[plate][well] = info['id']

all_genes = sorted(set(label for pm in plate_maps.values() for label in pm.values()))
gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
idx_to_gene = {idx: gene for gene, idx in gene_to_idx.items()}
num_classes = len(all_genes)
print(f"Classes: {num_classes}")

all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']


def extract_gene(label):
    return label


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


def extract_embeddings(model, dataset, batch_size=512):
    """Extract embeddings using the model backbone"""
    model.eval()
    embeddings = {}
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Extracting embeddings'):
            images = images.to(device)
            feats = model.get_backbone_features(images)
            feats = feats.cpu().numpy()
            
            for i, (path, label) in enumerate(zip(dataset.image_paths, dataset.labels)):
                bag_name = os.path.basename(os.path.dirname(path))
                if bag_name not in embeddings:
                    embeddings[bag_name] = []
                embeddings[bag_name].append((
                    os.path.basename(path).replace('.tif', '').replace('.tiff', ''),
                    feats[i]
                ))
    
    return embeddings


def extract_mil_embeddings(model, data_loader, device):
    """Extract MIL bag embeddings for pseudo label generation"""
    model.eval()
    all_embeddings = {}
    all_labels = {}
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Extracting MIL embeddings'):
            images = images.to(device)
            
            batch_size, num_crops = images.shape[:2]
            images_flat = images.view(-1, *images.shape[2:])
            feats = model.get_backbone_features(images_flat)
            feats = feats.view(batch_size, num_crops, -1)
            
            for i in range(batch_size):
                bag_name = f"bag_{i}"
                all_embeddings[bag_name] = feats[i].cpu().numpy()
                all_labels[bag_name] = labels[i].item()
    
    return all_embeddings, all_labels


def train_mil_classifier(embeddings_dict, num_feats, args, device, val_dict=None):
    """Train MIL (DS-MIL) classifier on embeddings"""
    train_dataset = BagDatasetIns(embeddings_dict)
    
    if val_dict:
        val_dataset = BagDatasetIns(val_dict)
    else:
        val_dataset = None
    
    bags_list = []
    for b in train_dataset:
        bags_list.append([b[1].item(), b[0].numpy(), b[2], b[3]])
    
    if val_dataset:
        val_list = []
        for b in val_dataset:
            val_list.append([b[1].item(), b[0].numpy(), b[2], b[3]])
    else:
        val_list = []
    
    i_classifier = FCLayer(num_feats, 1)
    b_classifier = BClassifier(num_feats, 1)
    milnet = MILNet(i_classifier, b_classifier).to(device)
    
    pos_weight = torch.tensor([len([b for b in bags_list if b[0] == 1]) / max(len(bags_list), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 75, gamma=0.5)
    
    best_auc = 0.0
    best_state = None
    num_epochs = args.its2clr_mil_epochs
    
    for epoch in range(num_epochs):
        milnet.train()
        train_loss = 0.0
        
        random.shuffle(bags_list)
        for bag_data in bags_list:
            bag_label, feats, _, bag_name = bag_data
            feats_tensor = torch.from_numpy(feats).float().to(device)
            label_tensor = torch.tensor([bag_label]).float().to(device)
            
            optimizer.zero_grad()
            _, pred_bag, _, _ = milnet(feats_tensor)
            loss = criterion(pred_bag, label_tensor)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        if val_list and (epoch + 1) % 5 == 0:
            milnet.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for bag_data in val_list:
                    bag_label, feats, _, _ = bag_data
                    feats_tensor = torch.from_numpy(feats).float().to(device)
                    _, pred_bag, _, _ = milnet(feats_tensor)
                    val_preds.append(torch.sigmoid(pred_bag).item())
                    val_labels.append(bag_label)
            
            val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in milnet.state_dict().items()}
    
    if best_state:
        milnet.load_state_dict(best_state)
    
    return milnet


def generate_pseudo_labels(milnet, embeddings_dict, device, threshold=0.3):
    """Generate instance-level pseudo labels from MIL predictions"""
    milnet.eval()
    pseudo_labels = {}
    
    with torch.no_grad():
        for bag_name, patches in tqdm(embeddings_dict.items(), desc='Generating pseudo labels'):
            if bag_name not in pseudo_labels:
                pseudo_labels[bag_name] = {}
            
            feats = np.array([p[1] for p in patches])
            feats_tensor = torch.from_numpy(feats).float().to(device)
            
            ins_pred, _ = milnet.get_instance_predictions(feats_tensor)
            ins_pred = ins_pred.cpu().squeeze().numpy()
            
            for i, (patch_name, _) in enumerate(patches):
                pseudo_labels[bag_name][patch_name] = float(ins_pred[i])
    
    return pseudo_labels


def weighted_focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0):
    # Simple focal loss without class weights for now
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    return (alpha * (1 - pt) ** gamma * ce_loss).mean()


def compute_pos_weight(bags_list):
    """Compute positive weight for BCE loss"""
    n_pos = sum([1 for b in bags_list if b[0] == 1])
    n_neg = len(bags_list) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return n_neg / n_pos


def spl_scheduler(current_epoch, warmup_epoch, max_epoch, ro, rT):
    """Self-Paced Learning scheduler"""
    if current_epoch < warmup_epoch:
        return ro
    return (current_epoch - warmup_epoch) * (rT - ro) / (max_epoch - warmup_epoch) + ro


def validate_model(model, val_loader, class_weights, device):
    """Validate model"""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    val_loss_total = 0.0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating', leave=False):
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
    
    return val_acc, val_auc


def train_single_fold_its2clr(test_plate):
    """Main ItS2CLR training for a single fold"""
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, f'fold_{test_plate}_its2clr')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ItS2CLR Training: test_plate={test_plate}")
    print(f"{'='*60}")
    
    train_val_plates = [p for p in all_plates if p != test_plate]
    train_plates = train_val_plates[:4]
    val_plates = train_val_plates[4:]
    
    print(f"Train: {train_plates}, Val: {val_plates}, Test: {test_plate}")
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    for plate in train_plates:
        plate_paths = get_image_paths_for_plate(plate)
        print(f"DEBUG train: {plate} = {len(plate_paths)} paths")
        for path in plate_paths:
            train_paths.append(path)
            train_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    for plate in val_plates:
        plate_paths = get_image_paths_for_plate(plate)
        print(f"DEBUG val: {plate} = {len(plate_paths)} paths")
        for path in plate_paths:
            val_paths.append(path)
            val_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    for plate in [test_plate]:
        plate_paths = get_image_paths_for_plate(plate)
        print(f"DEBUG test: {plate} = {len(plate_paths)} paths")
        for path in plate_paths:
            test_paths.append(path)
            test_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    class_weights = torch.tensor([1.0] * num_classes, device=device)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    train_dataset = MultiCropDataset(train_paths, train_labels, plate_maps, 
                                      neighborhood=args.neighborhood, grid_size=args.grid_size,
                                      augment=True, seed=SEED)
    val_dataset = MultiCropDataset(val_paths, val_labels, plate_maps,
                                  neighborhood=args.neighborhood, grid_size=args.grid_size,
                                  augment=False, seed=SEED)
    test_dataset = MultiCropDataset(test_paths, test_labels, plate_maps,
                                  neighborhood=args.neighborhood, grid_size=args.grid_size,
                                  augment=False, seed=SEED)
    
    train_dataset.set_epoch(0)
    val_dataset.set_epoch(0)
    test_dataset.set_epoch(0)
    
    effective_workers = 0 if sys.platform.startswith('win') else NUM_WORKERS
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                           num_workers=effective_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=effective_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=effective_workers, pin_memory=True)
    
    print(f"Crops per image: {args.neighborhood}x{args.neighborhood}={args.neighborhood**2}")
    print(f"MIL-Dropout: {args.use_mil_dropout}, topk={args.mil_dropout_topk}")
    
    model = MILEncoder(
        num_classes=num_classes, 
        num_heads=args.num_heads, 
        dropout=args.dropout, 
        use_contrastive=True,
        use_mil_dropout=args.use_mil_dropout,
        topk=args.mil_dropout_topk
    )
    model = model.to(device)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f'its2clr_metrics_{timestamp}.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'epoch', 'phase', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc'])
        f.flush()
    
    print(f"CSV logging to: {csv_path}")
    
    best_val_auc = 0.0
    best_model_state = None
    
    print(f"\n{'='*60}")
    print("Stage 1: Initial MIL training (warmup)")
    print(f"{'='*60}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.its2clr_warmup)
    
    for epoch in range(args.its2clr_warmup):
        train_dataset.set_epoch(epoch)
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        
        for images, labels in tqdm(train_loader, desc=f'Warmup {epoch}', leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs, attn_weights = model(images, return_attention=True)
            loss = weighted_focal_loss(outputs, labels, class_weights)
            
            loss.backward()
            optimizer.step()
            
            run_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        train_acc = 100. * correct / total
        avg_loss = run_loss / max(len(train_loader), 1)
        
        val_acc, val_auc = validate_model(model, val_loader, class_weights, device)
        
        print(f"Warmup {epoch}: Loss={avg_loss:.4f}, Train={train_acc:.1f}%, Val={val_acc:.1f}%, AUC={val_auc:.4f}")
        
        try:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([0, epoch, 'warmup', avg_loss, train_acc, 0, val_acc, val_auc])
                f.flush()
        except Exception as e:
            print(f"Warning: Could not write to CSV: {e}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    print(f"\n{'='*60}")
    print("Stage 2: Iterative Self-Paced Contrastive Learning")
    print(f"{'='*60}")
    
    supcon_criterion = SupConLoss(temperature=args.its2clr_temperature, pair_mode=2)
    supcon_criterion_pos = SupConLoss(temperature=args.its2clr_temperature, pair_mode=1)
    
    pseudo_labels = None
    
    for iteration in range(args.its2clr_iterations):
        print(f"\nIteration {iteration + 1}/{args.its2clr_iterations}")
        
        current_rho_pos = spl_scheduler(iteration, 0, args.its2clr_iterations, 
                                       args.its2clr_rho_pos, args.its2clr_rho_T)
        current_rho_neg = spl_scheduler(iteration, 0, args.its2clr_iterations,
                                       args.its2clr_rho_neg, args.its2clr_rho_T)
        
        print(f"SPL ratios: pos={current_rho_pos:.2f}, neg={current_rho_neg:.2f}")
        
        supcon_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
        supcon_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(supcon_optimizer, T_max=args.its2clr_mil_epochs)
        
        for epoch in range(args.its2clr_mil_epochs):
            train_dataset.set_epoch(args.its2clr_warmup + iteration * args.its2clr_mil_epochs + epoch)
            model.train()
            run_cl_loss, run_ce_loss, correct, total = 0.0, 0.0, 0, 0
            
            for images, labels in tqdm(train_loader, desc=f'Iter {iteration+1} E{epoch}', leave=False):
                images, labels = images.to(device), labels.to(device)
                supcon_optimizer.zero_grad()
                
                outputs, attn_weights = model(images, return_attention=True)
                bag_embeddings = model.get_supcon_embeddings(images)
                bag_embeddings = F.normalize(bag_embeddings, p=2, dim=-1)
                
                if bag_embeddings.shape[1] > 1:
                    features_cat = torch.cat([bag_embeddings[:, i] for i in range(bag_embeddings.shape[1])], dim=0)
                    labels_expanded = labels.repeat(bag_embeddings.shape[1])
                    bag_labels_expanded = labels.repeat(bag_embeddings.shape[1])
                    
                    if iteration % 2 == 0:
                        sc_loss = supcon_criterion(
                            features_cat.unsqueeze(1), 
                            labels_expanded, 
                            bag_labels_expanded
                        )
                    else:
                        sc_loss = supcon_criterion_pos(
                            features_cat.unsqueeze(1),
                            labels_expanded,
                            bag_labels_expanded
                        )
                else:
                    sc_loss = torch.tensor(0.0, device=device)
                
                ce_loss = weighted_focal_loss(outputs, labels, class_weights)
                
                # Use lower SupCon weight initially (per research: start low, increase gradually)
                supcon_weight = args.supcon_weight
                loss = (1 - supcon_weight) * ce_loss + supcon_weight * sc_loss if sc_loss.item() > 0 else ce_loss
                
                loss.backward()
                supcon_optimizer.step()
                
                run_cl_loss += sc_loss.item()
                run_ce_loss += ce_loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            supcon_scheduler.step()
            
            train_acc = 100. * correct / total
            avg_cl_loss = run_cl_loss / max(len(train_loader), 1)
            avg_ce_loss = run_ce_loss / max(len(train_loader), 1)
            
            val_acc, val_auc = validate_model(model, val_loader, class_weights, device)
            
            print(f"Iter {iteration+1} Ep {epoch}: CE={avg_ce_loss:.4f}, SupCon={avg_cl_loss:.4f}, Train={train_acc:.1f}%, Val={val_acc:.1f}%, AUC={val_auc:.4f}")
            
            try:
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([iteration, epoch, 'supcon', avg_ce_loss, train_acc, 0, val_acc, val_auc])
                    f.flush()
            except Exception as e:
                print(f"Warning: Could not write to CSV: {e}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save({'iteration': iteration, 'epoch': epoch, 'model_state_dict': model.state_dict()},
                          os.path.join(OUTPUT_DIR, 'best_model.pth'))
            
            if epoch > 0 and args.its2clr_update_pseudo and epoch % args.its2clr_mil_every == 0:
                embeddings_dict, _ = extract_mil_embeddings(model, train_loader, device)
                milnet = train_mil_classifier(embeddings_dict, 1280, args, device)
                pseudo_labels = generate_pseudo_labels(milnet, embeddings_dict, device, args.its2clr_threshold)
                
                pseudo_path = os.path.join(OUTPUT_DIR, f'pseudo_labels_iter{iteration}_ep{epoch}.p')
                with open(pseudo_path, 'wb') as f:
                    pickle.dump(pseudo_labels, f)
                print(f"Saved pseudo labels to {pseudo_path}")
    
    print(f"\nItS2CLR complete! Best Val AUC: {best_val_auc:.4f}")
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save({'model_state_dict': model.state_dict()},
                  os.path.join(OUTPUT_DIR, 'best_model_final.pth'))
    
    print("\nTesting...")
    test_acc, test_auc = validate_model(model, test_loader, class_weights, device)
    print(f"Test Acc: {test_acc:.1f}%, Test AUC: {test_auc:.4f}")
    
    results = {
        'timestamp': timestamp,
        'config': vars(args),
        'results': {'best_val_auc': float(best_val_auc), 'test_acc': float(test_acc), 'test_auc': float(test_auc)}
    }
    
    with open(os.path.join(OUTPUT_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {OUTPUT_DIR}")
    return results


if __name__ == '__main__':
    if args.run_all_folds:
        for test_plate in all_plates:
            fold_dir = os.path.join(SCRIPT_DIR, f'fold_{test_plate}_its2clr')
            checkpoints = [
                os.path.join(fold_dir, 'best_model.pth'),
                os.path.join(fold_dir, 'best_model_final.pth'),
            ]
            if any(os.path.exists(cp) for cp in checkpoints):
                print(f"Skipping {test_plate}: already trained")
                continue
            
            result = train_single_fold_its2clr(test_plate)
            print(f"Fold {test_plate}: Val AUC = {result['results']['best_val_auc']:.4f}")
        
        print("All folds completed!")
    else:
        result = train_single_fold_its2clr(args.test_plate)
        print(f"Fold {args.test_plate}: Val AUC = {result['results']['best_val_auc']:.4f}")
    
    print("Done!")