#!/usr/bin/env python3
"""Train Conditional Flow Matching on bacterial images (185 classes).

Usage:
    python3 train_flow.py --epochs 100 --batch_size 64

At each epoch, generates 5 sample images per class for a subset of classes.
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import csv

from mil_model import FlowCropDataset, load_labels
from flow_model import FlowUNet, compute_flow_loss, sample

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--val_split', type=float, default=0.05)
parser.add_argument('--block_channels', type=str, default='32,64,128,256')
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--save_interval', type=int, default=10)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

run_suffix = args.run_name or f'flow'
OUTPUT_DIR = args.output_dir or os.path.join(
    SCRIPT_DIR,
    f'flow_run_{run_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=OUTPUT_DIR)

print("=" * 60)
print(f"Flow Matching Training")
print(f"Output: {OUTPUT_DIR}")
print("=" * 60)

print("\n[1/5] Loading data ...")
image_list, class_names, label_to_idx = load_labels(PROJECT_ROOT, SCRIPT_DIR)
num_classes = len(class_names)
print(f"  {len(image_list)} images, {num_classes} classes")

n_val = max(1, int(len(image_list) * args.val_split))
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(image_list))
val_items = [image_list[i] for i in perm[:n_val]]
train_items = [image_list[i] for i in perm[n_val:]]
print(f"  Train: {len(train_items)}, Val: {len(val_items)}")

train_ds = FlowCropDataset(train_items, augment=True)
val_ds = FlowCropDataset(val_items, augment=False)

from torch.utils.data import WeightedRandomSampler

# Class-balanced sampling: each class gets equal total weight
class_counts = np.bincount([cid for _, cid in train_items], minlength=num_classes)
weights_per_class = 1.0 / class_counts.astype(np.float64)
sample_weights = np.array([weights_per_class[cid] for _, cid in train_items], dtype=np.float64)
train_sampler = WeightedRandomSampler(sample_weights, len(train_items), replacement=True)

train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, sampler=train_sampler,
    num_workers=args.num_workers, pin_memory=True, drop_last=True,
    persistent_workers=True, prefetch_factor=4,
)
val_loader = DataLoader(
    val_ds, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True,
    persistent_workers=True, prefetch_factor=4,
)

print("\n[2/5] Building model ...")
block_channels = tuple(int(x) for x in args.block_channels.split(','))
model = FlowUNet(
    in_channels=1,
    sample_size=224,
    block_out_channels=block_channels,
    layers_per_block=2,
    num_class_embeds=num_classes,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Params: {n_params:,}")

def add_weight_decay(model, wd=0.05):
    decay, no_decay = [], []
    skip = {'bias', 'norm', 'embed'}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in skip):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': wd},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

param_groups = add_weight_decay(model, args.weight_decay)
optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

total_steps = len(train_loader) * args.epochs
warmup_steps = len(train_loader) * 5

def lr_schedule(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
scaler = torch.amp.GradScaler('cuda', enabled=True)

start_epoch = 0
if args.resume:
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"  Resumed epoch {ckpt['epoch']}")

# Visualization: pick first 10 classes for generation samples
vis_classes = list(range(min(10, num_classes)))
vis_names = [class_names[i] for i in vis_classes]
print(f"  Viz classes: {vis_names}")
vis_ids = torch.tensor(vis_classes, device=device)

print("\n[3/5] Training ...")
best_val_loss = float('inf')
amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and
    hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
) else torch.float16
print(f"  AMP: {amp_dtype}")

metrics_path = os.path.join(OUTPUT_DIR, 'metrics.csv')
with open(metrics_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'time_s'])

for epoch in range(start_epoch, args.epochs):
    train_ds.set_epoch(epoch)
    val_ds.set_epoch(epoch)

    model.train()
    train_loss = 0.0
    train_steps = 0
    t0 = time.time()

    pbar = tqdm(train_loader, desc=f"E{epoch+1:03d}", leave=False)
    for imgs, class_ids in pbar:
        imgs = imgs.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype):
            loss = compute_flow_loss(model, imgs, class_labels=class_ids)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()
        train_steps += 1
        pbar.set_postfix(loss=loss.item())

    train_loss /= max(1, train_steps)
    epoch_time = time.time() - t0
    writer.add_scalar('train/loss', train_loss, epoch)

    model.eval()
    val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for imgs, class_ids in tqdm(val_loader, desc=f"E{epoch+1:03d} val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss = compute_flow_loss(model, imgs, class_labels=class_ids)
            val_loss += loss.item()
            val_steps += 1
    val_loss /= max(1, val_steps)
    writer.add_scalar('val/loss', val_loss, epoch)

    lr_now = optimizer.param_groups[0]['lr']
    print(f"  E{epoch+1:03d} train={train_loss:.6f} val={val_loss:.6f} ({epoch_time:.0f}s)")

    with open(metrics_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([epoch+1, f'{train_loss:.6f}', f'{val_loss:.6f}',
                    f'{lr_now:.2e}', f'{epoch_time:.0f}'])

    # Generate 5 samples per class for visualization classes
    if epoch % 1 == 0:
        model.eval()
        with torch.no_grad():
            n_per = 5
            n_viz = len(vis_classes)
            all_samples = []
            for ci in range(n_viz):
                cid = vis_ids[ci:ci+1].repeat(n_per)
                samps = sample(model, n_per, num_steps=args.num_steps,
                               class_labels=cid, device=device)
                all_samples.append(samps.cpu())

            fig, axes = plt.subplots(n_viz, n_per + 1,
                                     figsize=((n_per + 1) * 1.5, n_viz * 1.5))
            for ci in range(n_viz):
                axes[ci, 0].text(0.5, 0.5, vis_names[ci].replace('_', '\n'),
                                 ha='center', va='center', fontsize=6,
                                 transform=axes[ci, 0].transAxes)
                axes[ci, 0].axis('off')
                for si in range(n_per):
                    img = all_samples[ci][si]
                    img_01 = (img * 0.5 + 0.5).clamp(0, 1)
                    axes[ci, si + 1].imshow(img_01.squeeze(), cmap='gray', vmin=0, vmax=1)
                    axes[ci, si + 1].axis('off')
            plt.suptitle(f'Epoch {epoch+1}: Generated samples', fontsize=10)
            plt.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, f'samples_{epoch+1:03d}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close(fig)

    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss

    if (epoch + 1) % args.save_interval == 0 or is_best or epoch == args.epochs - 1:
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(args),
        }
        if is_best:
            torch.save(ckpt, os.path.join(OUTPUT_DIR, 'flow_best.pth'))
            print(f"  -> Best (val={val_loss:.6f})")
        if (epoch + 1) % args.save_interval == 0:
            torch.save(ckpt, os.path.join(OUTPUT_DIR, f'flow_{epoch+1:03d}.pth'))

print(f"\nDone. Best val: {best_val_loss:.6f}")
print(f"Outputs: {OUTPUT_DIR}")
writer.close()
