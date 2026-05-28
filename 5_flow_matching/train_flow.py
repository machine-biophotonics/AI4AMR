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

from mil_model import FlowCropDataset, load_labels
from flow_model import FlowUNet, FreqFlowUNet, SemanticPrototype, AuxProjectionHead, CoralProjection
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
parser.add_argument('--aux_path', action='store_true', default=False,
                    help='Use AuxPath-FM semantic prototype path (arXiv:2605.06364)')
parser.add_argument('--aux_path_weight', type=float, default=0.01,
                    help='Prototype supervision weight for AuxPath-FM')
parser.add_argument('--aux_ce_weight', type=float, default=0.01,
                    help='Weight for auxiliary CE head on bottleneck features (default 0.01, auto-enabled with --aux_path)')
parser.add_argument('--coral', action='store_true', default=False,
                    help='Enable CORAL supervised contrastive loss on bottleneck (NeurIPS 2025)')
parser.add_argument('--coral_weight', type=float, default=0.1,
                    help='CORAL SupCon loss weight (default 0.1)')
parser.add_argument('--coral_temperature', type=float, default=0.1,
                    help='CORAL SupCon temperature (default 0.1)')

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

prototype = None
if args.aux_path:
    prototype = SemanticPrototype(num_classes=num_classes).to(device)
    n_proto = sum(p.numel() for p in prototype.parameters())
    print(f"  AuxPath-FM prototype: {n_proto:,} params (class-specific path encoding)")

aux_ce_head = None
if args.aux_ce_weight > 0.0:
    aux_ce_head = AuxProjectionHead(bottleneck_dim=256, num_classes=num_classes).to(device)
    n_ce = sum(p.numel() for p in aux_ce_head.parameters())
    print(f"  Aux CE head: {n_ce:,} params (weight={args.aux_ce_weight})")

coral_proj = None
if args.coral:
    coral_proj = CoralProjection(bottleneck_dim=256, latent_dim=128).to(device)
    n_coral = sum(p.numel() for p in coral_proj.parameters())
    print(f"  CORAL projection: {n_coral:,} params (weight={args.coral_weight}, temp={args.coral_temperature})")

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
if prototype is not None:
    param_groups.append({'params': prototype.parameters(), 'weight_decay': 0.0})
if aux_ce_head is not None:
    param_groups.append({'params': aux_ce_head.parameters(), 'weight_decay': 0.0})
if coral_proj is not None:
    param_groups.append({'params': coral_proj.parameters(), 'weight_decay': 0.0})
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
    if prototype is not None and 'prototype_state_dict' in ckpt:
        prototype.load_state_dict(ckpt['prototype_state_dict'])
    if aux_ce_head is not None and 'aux_ce_state_dict' in ckpt:
        aux_ce_head.load_state_dict(ckpt['aux_ce_state_dict'])
    if coral_proj is not None and 'coral_state_dict' in ckpt:
        coral_proj.load_state_dict(ckpt['coral_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"  Resumed epoch {ckpt['epoch']}")

# Visualization: all 22 antibiotics (2x) + control, all 28 mutants (1) + 2 controls
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
                  'train_neg', 'val_neg', 'train_aux', 'val_aux',
                  'train_ce', 'val_ce', 'train_coral', 'val_coral', 'lr', 'time_s']
    w.writerow(csv_header)

for epoch in range(start_epoch, args.epochs):
    train_ds.set_epoch(epoch)
    val_ds.set_epoch(epoch)

    model.train()
    prototype.train() if prototype is not None else None
    aux_ce_head.train() if aux_ce_head is not None else None
    coral_proj.train() if coral_proj is not None else None
    train_loss = 0.0
    train_steps = 0
    train_spatial = 0.0
    train_freq = 0.0
    train_neg = 0.0
    train_aux = 0.0
    train_ce = 0.0
    train_coral = 0.0
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
                aux_path=args.aux_path,
                prototype=prototype,
                aux_path_weight=args.aux_path_weight,
                aux_ce_head=aux_ce_head,
                aux_ce_weight=args.aux_ce_weight,
                coral_proj=coral_proj,
                coral_weight=args.coral_weight if args.coral else 0.0,
                coral_temperature=args.coral_temperature,
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if prototype is not None:
            torch.nn.utils.clip_grad_norm_(prototype.parameters(), 1.0)
        if aux_ce_head is not None:
            torch.nn.utils.clip_grad_norm_(aux_ce_head.parameters(), 1.0)
        if coral_proj is not None:
            torch.nn.utils.clip_grad_norm_(coral_proj.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()
        train_spatial += comp.get('spatial', 0.0)
        train_freq += comp.get('freq', 0.0)
        train_neg += comp.get('neg', 0.0)
        train_aux += comp.get('aux', 0.0)
        train_ce += comp.get('ce', 0.0)
        train_coral += comp.get('coral', 0.0)
        train_steps += 1
        pbar.set_postfix(loss=loss.item())

    train_loss /= max(1, train_steps)
    train_spatial /= max(1, train_steps)
    train_freq /= max(1, train_steps)
    train_neg /= max(1, train_steps)
    train_aux /= max(1, train_steps)
    train_ce /= max(1, train_steps)
    train_coral /= max(1, train_steps)
    epoch_time = time.time() - t0
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/spatial', train_spatial, epoch)
    writer.add_scalar('train/freq', train_freq, epoch)
    writer.add_scalar('train/neg', train_neg, epoch)
    writer.add_scalar('train/aux', train_aux, epoch)
    writer.add_scalar('train/ce', train_ce, epoch)
    writer.add_scalar('train/coral', train_coral, epoch)

    model.eval()
    prototype.eval() if prototype is not None else None
    aux_ce_head.eval() if aux_ce_head is not None else None
    coral_proj.eval() if coral_proj is not None else None
    val_loss = 0.0
    val_steps = 0
    val_spatial = 0.0
    val_freq = 0.0
    val_neg = 0.0
    val_aux = 0.0
    val_ce = 0.0
    val_coral = 0.0
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
                aux_path=args.aux_path,
                prototype=prototype,
                aux_path_weight=args.aux_path_weight,
                aux_ce_head=aux_ce_head,
                aux_ce_weight=args.aux_ce_weight,
                coral_proj=coral_proj,
                coral_weight=args.coral_weight if args.coral else 0.0,
                coral_temperature=args.coral_temperature,
            )
            val_loss += loss.item()
            val_spatial += comp.get('spatial', 0.0)
            val_freq += comp.get('freq', 0.0)
            val_neg += comp.get('neg', 0.0)
            val_aux += comp.get('aux', 0.0)
            val_ce += comp.get('ce', 0.0)
            val_coral += comp.get('coral', 0.0)
            val_steps += 1
    val_loss /= max(1, val_steps)
    val_spatial /= max(1, val_steps)
    val_freq /= max(1, val_steps)
    val_neg /= max(1, val_steps)
    val_aux /= max(1, val_steps)
    val_ce /= max(1, val_steps)
    val_coral /= max(1, val_steps)
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/spatial', val_spatial, epoch)
    writer.add_scalar('val/freq', val_freq, epoch)
    writer.add_scalar('val/neg', val_neg, epoch)
    writer.add_scalar('val/aux', val_aux, epoch)
    writer.add_scalar('val/ce', val_ce, epoch)
    writer.add_scalar('val/coral', val_coral, epoch)

    lr_now = optimizer.param_groups[0]['lr']
    print(f"  E{epoch+1:03d} train={train_loss:.6f} val={val_loss:.6f} "
          f"(spat={val_spatial:.4f} freq={val_freq:.4f} neg={val_neg:.4f}"
          f" aux={val_aux:.4f} ce={val_ce:.4f} coral={val_coral:.4f}) ({epoch_time:.0f}s)")

    with open(metrics_path, 'a', newline='') as f:
        w = csv.writer(f)
        row = [epoch+1, f'{train_loss:.6f}', f'{val_loss:.6f}',
               f'{train_spatial:.4f}', f'{val_spatial:.4f}',
               f'{train_freq:.4f}', f'{val_freq:.4f}',
               f'{train_neg:.4f}', f'{val_neg:.4f}',
               f'{train_aux:.4f}', f'{val_aux:.4f}',
               f'{train_ce:.4f}', f'{val_ce:.4f}',
               f'{train_coral:.4f}', f'{val_coral:.4f}',
               f'{lr_now:.2e}', f'{epoch_time:.0f}']
        w.writerow(row)

    # Generate 1 sample per class in 2-row table: drugs top, mutants bottom
    if epoch % 1 == 0:
        model.eval()
        with torch.no_grad():
            all_samples = []
            for ci in vis_classes:
                cid = torch.tensor([ci], device=device)
                samp = sample(model, 1, num_steps=args.num_steps,
                              class_labels=cid, device=device,
                              freq_flow=args.freq_flow,
                              aux_path=args.aux_path,
                              prototype=prototype)
                all_samples.append(samp.cpu())

            n_drugs, n_mutants = len(drug_viz), len(mutant_viz)
            n_cols = max(n_drugs, n_mutants)

            fig = plt.figure(figsize=(n_cols * 0.45, 5))
            gs = GridSpec(2, n_cols, figure=fig, hspace=0.35, wspace=0.02,
                          height_ratios=[1, 1])

            for i in range(n_cols):
                # Drug row
                ax = fig.add_subplot(gs[0, i])
                if i < n_drugs:
                    img = all_samples[i]
                    img_01 = (img * 0.5 + 0.5).clamp(0, 1)
                    ax.imshow(img_01.squeeze(), cmap='gray', vmin=0, vmax=1)
                    ax.set_xlabel(class_names[drug_viz[i]].replace('_', ' '), fontsize=3)
                ax.set_xticks([])
                ax.set_yticks([])

                # Mutant row
                ax = fig.add_subplot(gs[1, i])
                if i < n_mutants:
                    img = all_samples[n_drugs + i]
                    img_01 = (img * 0.5 + 0.5).clamp(0, 1)
                    ax.imshow(img_01.squeeze(), cmap='gray', vmin=0, vmax=1)
                    ax.set_xlabel(class_names[mutant_viz[i]].replace('_', ' '), fontsize=3)
                ax.set_xticks([])
                ax.set_yticks([])

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
            'prototype_state_dict': prototype.state_dict() if prototype is not None else None,
            'aux_ce_state_dict': aux_ce_head.state_dict() if aux_ce_head is not None else None,
            'coral_state_dict': coral_proj.state_dict() if coral_proj is not None else None,
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
