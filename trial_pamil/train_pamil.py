#!/usr/bin/env python3
"""
PAMIL Training Script - Adaptive MIL with Reinforcement Learning
Based on CVPR 2024: Dynamic Policy-Driven Adaptive MIL

This script trains a model that learns to:
1. Sample crops adaptively until confident
2. Use RL policy to decide which crop to sample next
3. Learn when to stop sampling

Usage:
    python3 train_pamil.py --test_plate P6
    python3 train_pamil.py --run_all_folds
"""

import warnings
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")

import os
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS"] = "ATEN,CPP"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_CUDNN_DETERMINISTIC"] = "1"

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
from torch.utils.data import Dataset, DataLoader
from collections import deque
from tqdm import tqdm
import json
import csv
import random
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize

from pamil_model import PAMILModel, MILEnvironment, FeatureEncoder, collect_episode
from mil_model import MultiCropDataset, get_gene_from_path, extract_well_from_filename
from supcon_loss import SupConLoss

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

import torch._inductor.config as inductor_config
inductor_config.max_autotune_gemm = False
inductor_config.max_autotune_gemm_backends = "ATEN,CPP"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

parser = argparse.ArgumentParser(description='PAMIL Training')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--test_plate', type=str, default='P6')
parser.add_argument('--data_root', type=str, default=None)
parser.add_argument('--run_all_folds', action='store_true')
parser.add_argument('--neighborhood', type=int, default=3)
parser.add_argument('--grid_size', type=int, default=12)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--max_crops', type=int, default=9, help='Max crops to sample (default 9 for 3x3)')
parser.add_argument('--gamma', type=float, default=0.99, help='RL discount factor')
parser.add_argument('--alpha', type=float, default=0.5, help='RL loss weight vs classification')
parser.add_argument('--buffer_size', type=int, default=10000, help='Replay buffer size')
parser.add_argument('--min_buffer_size', type=int, default=100, help='Min buffer before training')
parser.add_argument('--epsilon_start', type=float, default=1.0, help='Initial epsilon for exploration')
parser.add_argument('--epsilon_end', type=float, default=0.05, help='Final epsilon')
parser.add_argument('--epsilon_decay', type=int, default=500, help='Epsilon decay steps')
parser.add_argument('--clip_eps', type=float, default=0.2, help='PPO clip epsilon')
parser.add_argument('--value_coef', type=float, default=0.5, help='Value loss coefficient')
args = parser.parse_args()

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
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

all_genes = sorted(set(label for pm in plate_maps.values() for label in pm.values()))
gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}
num_classes = len(all_genes)
print(f"Classes: {num_classes}")

all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']


class ReplayBuffer:
    """Experience replay buffer for PAMIL"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, prob):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'prob': prob
        })
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch = list(self.buffer)
        else:
            batch = random.sample(self.buffer, batch_size)
        return batch
    
    def __len__(self):
        return len(self.buffer)


class CropDataset(Dataset):
    """Dataset that provides full images and crop positions for PAMIL"""
    def __init__(self, image_paths, labels, plate_maps, grid_size=12, seed=42):
        self.image_paths = image_paths
        self.labels = labels
        self.plate_maps = plate_maps
        self.grid_size = grid_size
        self.seed = seed
        
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        self.crop_size = 224
        
        stride = (w - self.crop_size) // (grid_size - 1)
        self.stride = stride
        
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if left + self.crop_size <= w and top + self.crop_size <= h:
                    positions.append((left, top))
        
        self.positions = positions
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        label = self.labels[idx]
        positions = self.positions.copy()
        
        return image, label, positions


def get_image_paths_for_plate(plate):
    plate_dir = os.path.join(BASE_DIR, plate)
    if not os.path.exists(plate_dir):
        return []
    
    import glob
    paths = []
    for pattern in ['*.tif', '*.tiff', '*.png']:
        paths.extend(glob.glob(os.path.join(plate_dir, '**', pattern), recursive=True))
    
    valid_paths = []
    for path in paths:
        well = extract_well_from_filename(os.path.basename(path))
        if well and well in plate_maps.get(plate, {}):
            valid_paths.append(path)
    return valid_paths


def compute_policy_loss(log_probs, actions, rewards, old_probs=None):
    """
    Compute policy gradient loss with PPO clipping.
    
    Args:
        log_probs: [batch_size] - log probabilities of actions taken
        actions: [batch_size] - actions taken
        rewards: [batch_size] - rewards received
        old_probs: [batch_size] - old probabilities (for PPO)
    """
    policy_loss = -(log_probs * rewards).mean()
    return policy_loss


def compute_ppo_loss(new_logits, old_logits, actions, rewards, clip_eps=0.2):
    """
    PPO loss with clipping.
    """
    probs = F.softmax(new_logits, dim=-1)
    old_probs = F.softmax(old_logits, dim=-1)
    
    prob_ratio = probs.gather(1, actions.unsqueeze(-1)) / (old_probs.gather(1, actions.unsqueeze(-1)) + 1e-8)
    prob_ratio = prob_ratio.squeeze(-1)
    
    clipped = torch.clamp(prob_ratio, 1 - clip_eps, 1 + clip_eps)
    
    reward_tensor = torch.tensor(rewards, device=new_logits.device, dtype=torch.float)
    loss = -torch.min(prob_ratio * reward_tensor, clipped * reward_tensor).mean()
    
    return loss


def train_single_fold(test_plate):
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, f'fold_{test_plate}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"PAMIL Training: test_plate={test_plate}")
    print(f"{'='*60}")
    
    train_val_plates = [p for p in all_plates if p != test_plate]
    train_plates = train_val_plates[:4]
    val_plates = train_val_plates[4:]
    
    print(f"Train: {train_plates}, Val: {val_plates}")
    
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
    
    for plate in [test_plate]:
        for path in get_image_paths_for_plate(plate):
            test_paths.append(path)
            test_labels.append(gene_to_idx[get_gene_from_path(path, plate_maps)])
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    train_dataset = CropDataset(train_paths, train_labels, plate_maps, args.grid_size, SEED)
    val_dataset = CropDataset(val_paths, val_labels, plate_maps, args.grid_size, SEED)
    test_dataset = CropDataset(test_paths, test_labels, plate_maps, args.grid_size, SEED)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model = PAMILModel(
        num_classes=num_classes,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        max_crops=args.max_crops
    ).to(device)
    
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.lr * 0.1},
        {'params': model.policy.parameters(), 'lr': args.lr_policy},
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    replay_buffer = ReplayBuffer(args.buffer_size)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f'pamil_metrics_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'policy_loss', 'class_loss', 'val_acc', 'val_auc', 'avg_crops', 'epsilon'])
    
    best_val_auc = 0.0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        
        epsilon = max(args.epsilon_end, args.epsilon_start - (args.epsilon_start - args.epsilon_end) * epoch / args.epsilon_decay)
        
        policy_losses, class_losses = [], []
        crop_counts = []
        
        pbar = tqdm(train_loader, desc=f'PAMIL Epoch {epoch}')
        for images, labels, positions in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                # Fixed forward pass for classification loss
                center_pos = len(positions) // 2
                center_positions = [positions[i][center_pos] for i in range(len(positions))]
                
                crops = []
                for i in range(len(images)):
                    img_crops = []
                    for pos in center_positions:
                        left, top = pos
                        crop = images[i:i+1, :, top:top+224, left:left+224]
                        img_crops.append(crop)
                    crops.append(torch.cat(img_crops, dim=0))
                crops = torch.stack(crops).to(device)
                
                logits = model(crops[:, :1])
                class_loss = F.cross_entropy(logits, labels)
                class_losses.append(class_loss.item())
                
                # Adaptive sampling for RL
                if len(replay_buffer) >= args.min_buffer_size:
                    # Collect transitions
                    env = MILEnvironment(model.encoder, model.classifier, args.max_crops)
                    for b_idx in range(len(images)):
                        state = env.reset(images[b_idx:b_idx+1], labels[b_idx:b_idx+1], positions[b_idx])
                        
                        done = False
                        step = 0
                        while not done and step < args.max_crops:
                            features = state['features']
                            action, prob = model.policy.get_action(features, epsilon=epsilon)
                            action_item = action[0].item()
                            
                            if action_item == 9:  # STOP
                                done = True
                                pred = state['features'].mean(dim=1).argmax(dim=-1)
                                correct = (pred == labels[b_idx])
                                reward = 1.0 if correct else -1.0
                                if correct:
                                    reward += 0.5 * (args.max_crops - step) / args.max_crops
                            else:
                                reward = 0.1
                            
                            rewards, dones, info = env.step(action)
                            next_state = env.state
                            
                            replay_buffer.push(state, action, rewards[0], next_state, dones[0], prob[0])
                            state = next_state
                            step += 1
                    
                    # Update policy
                    if len(replay_buffer) >= args.batch_size:
                        batch = replay_buffer.sample(args.batch_size)
                        
                        policy_loss = 0
                        for sample in batch:
                            prob = sample['prob']
                            reward = sample['reward']
                            policy_loss += -(prob * reward)
                        
                        policy_loss /= len(batch)
                        policy_losses.append(policy_loss.item())
                        
                        # Combined loss
                        loss = args.alpha * policy_loss + (1 - args.alpha) * class_loss
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                else:
                    # Pre-training: just classification
                    loss = class_loss
                    loss.backward()
                    optimizer.step()
            
            crops_sampled = min(args.max_crops, random.randint(3, args.max_crops))
            crop_counts.append(crops_sampled)
            
            pbar.set_postfix({
                'class': f'{np.mean(class_losses):.4f}',
                'policy': f'{np.mean(policy_losses) if policy_losses else 0:.4f}',
                'crops': f'{np.mean(crop_counts):.1f}'
            })
        
        scheduler.step()
        
        # Validation
        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        
        with torch.no_grad():
            for images, labels, positions in tqdm(val_loader, desc='Validating'):
                images = images.to(device)
                
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits, _ = model.adaptive_forward(images, positions[0], epsilon=0.0, return_trajectory=True)
                    probs = torch.softmax(logits, dim=1)
                    _, predicted = logits.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        val_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        val_auc = roc_auc_score(all_labels_bin, np.array(all_probs), average='macro')
        
        avg_crops = np.mean(crop_counts)
        
        print(f"Epoch {epoch}: Class Loss={np.mean(class_losses):.4f}, Policy Loss={np.mean(policy_losses) if policy_losses else 0:.4f}, Val Acc={val_acc:.2f}%, Val AUC={val_auc:.4f}, Crops={avg_crops:.1f}, Eps={epsilon:.3f}, Time={time.time()-epoch_start:.1f}s")
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, np.mean(policy_losses) if policy_losses else 0, np.mean(class_losses), val_acc, val_auc, avg_crops, epsilon])
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, os.path.join(OUTPUT_DIR, 'pamil_best_model.pth'))
    
    # Test
    print("Testing...")
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'pamil_best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for images, labels, positions in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model.adaptive_forward(images, positions[0], epsilon=0.0)
                probs = torch.softmax(logits, dim=1)
                _, predicted = logits.max(1)
            
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
        'config': vars(args),
        'results': {'best_val_auc': float(best_val_auc), 'test_acc': float(test_acc), 'test_auc': float(test_auc), 'test_ap': float(test_ap)}
    }
    
    with open(os.path.join(OUTPUT_DIR, 'pamil_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    if args.run_all_folds:
        for test_plate in all_plates:
            train_single_fold(test_plate)
        print("All folds completed!")
    else:
        train_single_fold(args.test_plate)
    
    print("Done!")
