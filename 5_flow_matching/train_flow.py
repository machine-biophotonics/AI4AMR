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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import csv

from mil_model import FlowCropDataset, load_labels, extract_plate_from_path
from flow_model import FlowUNet, FreqFlowUNet
from flow_model import compute_flow_loss, sample



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
parser.add_argument('--freq_flow', action='store_true', default=False,
                    help='Use FreqFlow two-branch architecture (Ren et al., CVPR 2026)')
parser.add_argument('--freq_filter_D', type=float, default=8.0,
                    help='FreqFlow Gaussian filter cutoff D')
parser.add_argument('--freq_loss_weight', type=float, default=0.25,
                    help='FreqFlow frequency branch loss weight (paper: 1/4)')
parser.add_argument('--freq_block_channels', type=str, default='32,64,128,256',
                    help='FreqFlow frequency branch block channels')
parser.add_argument('--delta_fm', action='store_true', default=False,
                    help='Use Contrastive Flow Matching (DeltaFM) loss (Stoica et al., ICCV 2025)')
parser.add_argument('--delta_fm_lambda', type=float, default=0.05,
                    help='Contrastive loss weight lambda for DeltaFM (default: 0.05)')
parser.add_argument('--train_plates', type=str, default='P1,P2,P3,P4',
                    help='Comma-separated plates for training')
parser.add_argument('--val_plate', type=str, default='P5',
                    help='Plate for validation')
parser.add_argument('--test_plate', type=str, default='P6',
                    help='Plate for testing (excluded from training/validation)')
parser.add_argument('--unsupervised', action='store_true', default=False,
                    help='Null label: all images get class 0, num_classes=1 (purely unconditional)')

args = parser.parse_args()

if args.freq_flow and args.batch_size == 64:
    args.batch_size = 32

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

if args.unsupervised:
    image_list = [(p, 0) for p, _ in image_list]
    num_classes = 1
    class_names = ['null']

train_plates = set(args.train_plates.split(','))
val_plate = args.val_plate
test_plate = args.test_plate

train_items = [x for x in image_list if extract_plate_from_path(x[0]) in train_plates]
val_items   = [x for x in image_list if extract_plate_from_path(x[0]) == val_plate]
test_items  = [x for x in image_list if extract_plate_from_path(x[0]) == test_plate]
print(f"  Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)} (test excluded from training)")

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
freq_block_channels = tuple(int(x) for x in args.freq_block_channels.split(',')) if args.freq_flow else block_channels

if args.freq_flow:
    print(f"  FreqFlow: batch_size={args.batch_size}")
    model = FreqFlowUNet(
        in_channels=1,
        sample_size=224,
        block_out_channels=block_channels,
        freq_block_out_channels=freq_block_channels,
        layers_per_block=2,
        num_class_embeds=num_classes,
        freq_filter_D=args.freq_filter_D,
    ).to(device)
    print(f"  FreqFlow: freq_branch={freq_block_channels}")
else:
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

# Visualization
if args.unsupervised:
    n_viz = min(50, len(val_items))
    vis_classes = [0] * n_viz
    drug_viz = mutant_viz = list(range(n_viz))
    print(f"  Unconditional viz: {n_viz} samples")
else:
    drug_viz = sorted([i for i, n in enumerate(class_names) if n.endswith('_2x')])
    drug_viz += [i for i, n in enumerate(class_names) if n == 'control']
    mutant_viz = sorted([i for i, n in enumerate(class_names) if n.endswith('_1') and n not in ('NC_1', 'WT NC_1')])
    mutant_viz += [i for i, n in enumerate(class_names) if n in ('NC_1', 'WT NC_1')]
    vis_classes = drug_viz + mutant_viz
    print(f"  Drug viz: {len(drug_viz)} classes, Mutant viz: {len(mutant_viz)} classes, Total: {len(vis_classes)}")


print("\n[3/5] Training ...")
best_val_loss = float('inf')
amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and
    hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported()
) else torch.float16
print(f"  AMP: {amp_dtype}")

metrics_path = os.path.join(OUTPUT_DIR, 'metrics.csv')
with open(metrics_path, 'w', newline='') as f:
    w = csv.writer(f)
    csv_header = ['epoch', 'train_loss', 'val_loss',
                  'train_spatial', 'val_spatial', 'train_freq', 'val_freq',
                  'train_neg', 'val_neg', 'lr', 'time_s']
    w.writerow(csv_header)

for epoch in range(start_epoch, args.epochs):
    train_ds.set_epoch(epoch)
    val_ds.set_epoch(epoch)

    model.train()
    train_loss = 0.0
    train_steps = 0
    train_spatial = 0.0
    train_freq = 0.0
    train_neg = 0.0
    t0 = time.time()

    pbar = tqdm(train_loader, desc=f"E{epoch+1:03d}", leave=False)
    for imgs, class_ids in pbar:
        imgs = imgs.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)

        delta_lambda = args.delta_fm_lambda if args.delta_fm else 0.0
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            loss, comp = compute_flow_loss(
                model, imgs, class_labels=class_ids,
                freq_flow=args.freq_flow,
                freq_filter_D=args.freq_filter_D,
                freq_loss_weight=args.freq_loss_weight,
                delta_fm_lambda=delta_lambda,
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()
        train_spatial += comp.get('spatial', 0.0)
        train_freq += comp.get('freq', 0.0)
        train_neg += comp.get('neg', 0.0)
        train_steps += 1
        pbar.set_postfix(loss=loss.item())

    train_loss /= max(1, train_steps)
    train_spatial /= max(1, train_steps)
    train_freq /= max(1, train_steps)
    train_neg /= max(1, train_steps)
    epoch_time = time.time() - t0
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/spatial', train_spatial, epoch)
    writer.add_scalar('train/freq', train_freq, epoch)
    writer.add_scalar('train/neg', train_neg, epoch)

    model.eval()
    val_loss = 0.0
    val_steps = 0
    val_spatial = 0.0
    val_freq = 0.0
    val_neg = 0.0
    with torch.no_grad():
        for imgs, class_ids in tqdm(val_loader, desc=f"E{epoch+1:03d} val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            delta_lambda = args.delta_fm_lambda if args.delta_fm else 0.0
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss, comp = compute_flow_loss(
                    model, imgs, class_labels=class_ids,
                    freq_flow=args.freq_flow,
                    freq_filter_D=args.freq_filter_D,
                    freq_loss_weight=args.freq_loss_weight,
                    delta_fm_lambda=delta_lambda,
                )
            val_loss += loss.item()
            val_spatial += comp.get('spatial', 0.0)
            val_freq += comp.get('freq', 0.0)
            val_neg += comp.get('neg', 0.0)
            val_steps += 1
    val_loss /= max(1, val_steps)
    val_spatial /= max(1, val_steps)
    val_freq /= max(1, val_steps)
    val_neg /= max(1, val_steps)
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/spatial', val_spatial, epoch)
    writer.add_scalar('val/freq', val_freq, epoch)
    writer.add_scalar('val/neg', val_neg, epoch)

    lr_now = optimizer.param_groups[0]['lr']
    print(f"  E{epoch+1:03d} train={train_loss:.6f} val={val_loss:.6f} "
          f"(spat={val_spatial:.4f} freq={val_freq:.4f} neg={val_neg:.4f}) ({epoch_time:.0f}s)")

    with open(metrics_path, 'a', newline='') as f:
        w = csv.writer(f)
        row = [epoch+1, f'{train_loss:.6f}', f'{val_loss:.6f}',
               f'{train_spatial:.4f}', f'{val_spatial:.4f}',
               f'{train_freq:.4f}', f'{val_freq:.4f}',
               f'{train_neg:.4f}', f'{val_neg:.4f}',
               f'{lr_now:.2e}', f'{epoch_time:.0f}']
        w.writerow(row)

    # Generate samples
    if epoch % 1 == 0:
        model.eval()
        with torch.no_grad():
            if args.unsupervised:
                samp = sample(model, n_viz, num_steps=args.num_steps,
                              class_labels=torch.zeros(n_viz, dtype=torch.long, device=device),
                              freq_flow=args.freq_flow)
                all_samples = [samp[i:i+1].cpu() for i in range(n_viz)]
                n_cols = min(n_viz, 10)
                n_rows = (n_viz + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.5, n_rows * 0.5))
                axes = axes.flatten() if n_rows > 1 else [axes]
                for i in range(n_cols * n_rows):
                    ax = axes[i]
                    if i < n_viz:
                        img = (all_samples[i] * 0.5 + 0.5).clamp(0, 1)
                        ax.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)
                    ax.set_xticks([]); ax.set_yticks([])
                plt.suptitle(f'Epoch {epoch+1}: Unconditional samples', fontsize=7, y=0.98)
            else:
                all_samples = []
                for ci in vis_classes:
                    cid = torch.tensor([ci], device=device)
                    samp = sample(model, 1, num_steps=args.num_steps,
                                  class_labels=cid, device=device,
                                  freq_flow=args.freq_flow)
                    all_samples.append(samp.cpu())

                n_drugs, n_mutants = len(drug_viz), len(mutant_viz)
                n_cols = max(n_drugs, n_mutants)

                fig = plt.figure(figsize=(n_cols * 0.45, 5))
                gs = GridSpec(2, n_cols, figure=fig, hspace=0.35, wspace=0.02,
                              height_ratios=[1, 1])

                for i in range(n_cols):
                    ax = fig.add_subplot(gs[0, i])
                    if i < n_drugs:
                        img = all_samples[i]
                        img_01 = (img * 0.5 + 0.5).clamp(0, 1)
                        ax.imshow(img_01.squeeze(), cmap='gray', vmin=0, vmax=1)
                        ax.set_xlabel(class_names[drug_viz[i]].replace('_', ' '), fontsize=3)
                    ax.set_xticks([]); ax.set_yticks([])

                    ax = fig.add_subplot(gs[1, i])
                    if i < n_mutants:
                        img = all_samples[n_drugs + i]
                        img_01 = (img * 0.5 + 0.5).clamp(0, 1)
                        ax.imshow(img_01.squeeze(), cmap='gray', vmin=0, vmax=1)
                        ax.set_xlabel(class_names[mutant_viz[i]].replace('_', ' '), fontsize=3)
                    ax.set_xticks([]); ax.set_yticks([])

                plt.suptitle(f'Epoch {epoch+1}: All classes (drugs 2x top, mutants 1 bottom)', fontsize=7, y=0.98)

            plt.tight_layout()
            fig.savefig(os.path.join(OUTPUT_DIR, f'samples_{epoch+1:03d}.png'),
                       dpi=200, bbox_inches='tight')
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
