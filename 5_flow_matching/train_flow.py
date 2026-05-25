#!/usr/bin/env python3
"""Train Conditional Flow Matching with DFM + contrastive + aux CE.

Usage:
    python3 train_flow.py --model_type unet --epochs 100
    python3 train_flow.py --model_type unet --aux_weight 0.1 --contrastive_weight 0.1
    python3 train_flow.py --model_type unet --delta_fm_weight 0.5 --lognormal

Features:
    - DFM contrastive flow matching (velocity repulsion, Stoica et al., ICCV 2025)
    - Auxiliary linear probe classification (DDAE-style)
    - Supervised Contrastive bottleneck loss (CORAL-style, NeurIPS 2025)
    - DiT transformer backbone (Peebles & Xie, 2023)
    - Lognormal timestep sampling (fine-detail bias)
    - Classifier-free guidance (CFG)
    - Midpoint ODE solver
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
from datetime import datetime
import csv

from mil_model import FlowCropDataset, load_labels
from flow_model import FlowUNet, AuxProjectionHead, ContrastiveProjection, compute_flow_loss, sample as unet_sample
from dit_model import DiT, build_dit

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
parser.add_argument('--val_split', type=float, default=0.05,
                    help='Fraction for validation (ignored if --train_plates is set)')
parser.add_argument('--train_plates', type=str, default=None,
                    help='Comma-separated plates for training, e.g. P1,P2,P3,P4')
parser.add_argument('--val_plate', type=str, default=None,
                    help='Single plate for validation, e.g. P5')
parser.add_argument('--test_plate', type=str, default=None,
                    help='Single plate held out for testing, e.g. P6')
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--save_interval', type=int, default=10)

# Model type
parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'dit'],
                    help='Backbone: unet (diffusers UNet2DModel) or dit (transformer)')
parser.add_argument('--dit_size', type=str, default='S', choices=['S', 'B'],
                    help='DiT model size: S=33M, B=131M (default S)')
parser.add_argument('--block_channels', type=str, default='32,64,128,256',
                    help='UNet block channels (ignored for dit)')

# DFM
parser.add_argument('--delta_fm_weight', type=float, default=0.05,
                    help='DFM repulsive weight l (0=disabled, default 0.05)')

# Lognormal timestep sampling
parser.add_argument('--lognormal', action='store_true', default=False,
                    help='Use log-normal timestep sampling (REPA-style, finer details)')
parser.add_argument('--lognormal_mean', type=float, default=-1.0,
                    help='LogNormal mean (default -1.0)')
parser.add_argument('--lognormal_std', type=float, default=1.0,
                    help='LogNormal std (default 1.0)')

# CFG
parser.add_argument('--label_dropout', type=float, default=0.1,
                    help='Probability to drop class label for CFG training')
parser.add_argument('--cfg_scale', type=float, default=0.0,
                    help='CFG scale (0=disabled, 1.5 recommended)')
parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'midpoint'])
parser.add_argument('--null_label', type=int, default=-1)
parser.add_argument('--clip_grad_norm', type=float, default=5.0,
                    help='Gradient clipping max norm (default 5.0)')

# Auxiliary 185-way classification on bottleneck features (DDAE / CORAL-style)
parser.add_argument('--aux_weight', type=float, default=0.0,
                    help='Weight for auxiliary 185-class CE loss on bottleneck (0=disabled, default 0.0)')

# Supervised Contrastive Loss (SupCon on bottleneck, CORAL-style)
parser.add_argument('--contrastive_weight', type=float, default=0.0,
                    help='Weight for Supervised Contrastive loss (0=disabled, default 0.0)')
parser.add_argument('--contrastive_temperature', type=float, default=0.1,
                    help='Temperature for SupCon loss (default 0.1)')
parser.add_argument('--contrastive_proj_dim', type=int, default=128,
                    help='Projection dimension for contrastive head (default 128)')

parser.add_argument('--fine_tune', type=str, default=None,
                    help='Checkpoint path to fine-tune from')
parser.add_argument('--finetune_lr', type=float, default=5e-5,
                    help='Learning rate for fine-tuning (default 5e-5)')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

run_suffix = args.run_name or f'{args.model_type}'
OUTPUT_DIR = args.output_dir or os.path.join(
    SCRIPT_DIR,
    f'flow_run_{run_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=OUTPUT_DIR)

print("=" * 60)
print(f"Flow Matching Training — backbone={args.model_type}")
print(f"  DFM={args.delta_fm_weight}, aux={args.aux_weight}, "
      f"contrastive={args.contrastive_weight}, lognormal={args.lognormal}")
if args.train_plates:
    print(f"  Plates — train: {args.train_plates}", end='')
    if args.val_plate:
        print(f", val: {args.val_plate}", end='')
    if args.test_plate:
        print(f", test: {args.test_plate} (held out)", end='')
    print()
print(f"Output: {OUTPUT_DIR}")
print("=" * 60)

print("\n[1/5] Loading data ...")
image_list, class_names, label_to_idx = load_labels(PROJECT_ROOT, SCRIPT_DIR)
num_classes = len(class_names)
if args.null_label < 0:
    args.null_label = num_classes  # use last valid embedding slot for null
print(f"  {len(image_list)} images, {num_classes} classes (null_label={args.null_label})")

def _plate_from_path(path: str) -> str:
    import re
    m = re.search(r'/[Pp](\d)/', path)
    return f'P{m.group(1)}' if m else ''

# Plate-based or random split
use_plate_split = args.train_plates is not None
if use_plate_split:
    train_plates = [p.strip() for p in args.train_plates.split(',')]
    val_plate = args.val_plate or f'P{len(train_plates)+1}'
    test_plate = args.test_plate

    train_items = [(p, c) for p, c in image_list if _plate_from_path(p) in train_plates]
    val_items = [(p, c) for p, c in image_list if _plate_from_path(p) == val_plate]
    test_items = [(p, c) for p, c in image_list if _plate_from_path(p) == test_plate] if test_plate else []

    print(f"  Train: {len(train_items)} ({', '.join(train_plates)})")
    print(f"  Val:   {len(val_items)} ({val_plate})")
    if test_items:
        print(f"  Test:  {len(test_items)} ({test_plate}) — held out")
else:
    n_val = max(1, int(len(image_list) * args.val_split))
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(image_list))
    val_items = [image_list[i] for i in perm[:n_val]]
    train_items = [image_list[i] for i in perm[n_val:]]
    test_items = []
    print(f"  Train: {len(train_items)}, Val: {len(val_items)} (random split)")

train_ds = FlowCropDataset(train_items, augment=True)
val_ds = FlowCropDataset(val_items, augment=False)

from torch.utils.data import WeightedRandomSampler
class_counts = np.bincount([cid for _, cid in train_items], minlength=num_classes)
weights_per_class = np.where(class_counts == 0, 0, 1.0 / class_counts.astype(np.float64))
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
if args.model_type == 'unet':
    block_channels = tuple(int(x) for x in args.block_channels.split(','))
    model = FlowUNet(
        in_channels=1, sample_size=224,
        block_out_channels=block_channels,
        layers_per_block=2, num_class_embeds=num_classes,
    ).to(device)
    model.unet = torch.compile(model.unet, mode="reduce-overhead")
    print("  UNet compiled with torch.compile (reduce-overhead)")
    sample_fn = unet_sample
else:
    model = build_dit(
        args.dit_size,
        in_channels=1, img_size=224, patch_size=16,
        num_classes=num_classes,
    ).to(device)
    sample_fn = unet_sample  # same interface: sample(model, ...)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Params: {n_params:,}")

# Auxiliary 185-way classifier on bottleneck (DDAE / CORAL style)
feat_dim = block_channels[-1] if args.model_type == 'unet' else 256
aux_head = AuxProjectionHead(feat_dim=feat_dim, num_classes=num_classes).to(device) if args.aux_weight > 0 else None
if aux_head:
    print(f"  AuxProjectionHead: {sum(p.numel() for p in aux_head.parameters()):,} params (185-way)")

contrastive_projector = ContrastiveProjection(feat_dim=feat_dim, proj_dim=args.contrastive_proj_dim).to(device) if args.contrastive_weight > 0 else None
if contrastive_projector:
    print(f"  ContrastiveProjection: {sum(p.numel() for p in contrastive_projector.parameters()):,} params "
          f"(feat={feat_dim} -> proj={args.contrastive_proj_dim}, t={args.contrastive_temperature})")

# Identify control class indices: drug control + NC_1..NC_6 + WT NC_1..WT NC_6
control_keywords = {'control', 'NC_1', 'NC_2', 'NC_3', 'NC_4', 'NC_5', 'NC_6',
                    'WT NC_1', 'WT NC_2', 'WT NC_3', 'WT NC_4', 'WT NC_5', 'WT NC_6'}
control_indices = {label_to_idx[name] for name in control_keywords if name in label_to_idx}
if contrastive_projector:
    print(f"  Control indices: {sorted(control_indices)} ({len(control_indices)} classes)")

if args.delta_fm_weight > 0:
    print(f"  DFM: l={args.delta_fm_weight}")
if args.lognormal:
    print(f"  Lognormal t-sampling: mean={args.lognormal_mean}, std={args.lognormal_std}")
print(f"  CFG: label_dropout={args.label_dropout}, cfg_scale={args.cfg_scale}")
print(f"  Solver: {args.solver}")

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
if aux_head:
    param_groups.append({'params': aux_head.parameters(), 'weight_decay': 0.0})
if contrastive_projector:
    param_groups.append({'params': contrastive_projector.parameters(), 'weight_decay': 0.0})

effective_lr = args.finetune_lr if args.fine_tune else args.lr
optimizer = torch.optim.AdamW(param_groups, lr=effective_lr, betas=(0.9, 0.95))

total_steps = len(train_loader) * args.epochs
warmup_steps = len(train_loader) * 5

def lr_schedule(step):
    if step < warmup_steps:
        return max(1e-8, step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
scaler = torch.amp.GradScaler('cuda', enabled=True)

start_epoch = 0
if args.resume:
    ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"  Resumed epoch {ckpt['epoch']}")
elif args.fine_tune:
    ckpt = torch.load(args.fine_tune, map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    # Strip _orig_mod prefix from torch.compile checkpoints
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd)
    start_epoch = 0  # restart from epoch 0
    print(f"  Fine-tuning from checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt.get('val_loss', 'N/A'):.6f})")

TARGET_NAMES = [
    'Ciprofloxacin_2x', 'Rifampicin_2x', 'Kanamycin_2x', 'Colistin_2x', 'Trimethoprim_2x',
    'gyrA_1', 'rpoB_1', 'rpsL_1', 'lpxC_1', 'folA_1',
]
vis_names = [n for n in TARGET_NAMES if n in label_to_idx]
vis_classes = [label_to_idx[n] for n in vis_names]
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
    w.writerow(['epoch', 'train_flow_loss', 'train_delta_fm_repulsive',
                'train_cls_loss', 'train_cls_acc', 'train_contrastive_loss', 'train_total_loss',
                'val_loss', 'val_flow_loss', 'val_aux_loss', 'val_aux_acc', 'val_delta_fm',
                'lr', 'grad_norm', 'time_s'])

for epoch in range(start_epoch, args.epochs):
    train_ds.set_epoch(epoch)
    val_ds.set_epoch(epoch)

    model.train()

    train_flow_loss = 0.0
    train_delta_fm = 0.0
    train_cls = 0.0
    train_cls_acc = 0.0
    train_contrastive = 0.0
    train_grad_norm = 0.0
    train_steps = 0
    t0 = time.time()

    pbar = tqdm(train_loader, desc=f"E{epoch+1:03d}", leave=False)
    for imgs, class_ids in pbar:
        imgs = imgs.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype):
            loss, info = compute_flow_loss(
                model, imgs, class_labels=class_ids,
                delta_fm_weight=args.delta_fm_weight,
                label_dropout_prob=args.label_dropout,
                null_label=args.null_label,
                lognormal_sampling=args.lognormal,
                aux_head=aux_head,
                aux_weight=args.aux_weight,
                num_classes=num_classes,
                contrastive_weight=args.contrastive_weight,
                contrastive_projector=contrastive_projector,
                contrastive_temperature=args.contrastive_temperature,
                control_indices=control_indices if args.contrastive_weight > 0 else None,
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_flow_loss += info.get('flow_loss', loss.item())
        train_delta_fm += info.get('delta_fm_repulsive', 0.0)
        train_cls += info.get('aux_loss', 0.0)
        train_cls_acc += info.get('aux_acc', 0.0)
        train_contrastive += info.get('contrastive_loss', 0.0)
        gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        info['grad_norm'] = gn
        train_grad_norm += gn
        if (train_steps + 1) % 50 == 0:
            print(f"    grad_norm={gn:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
        train_steps += 1
        pbar.set_postfix(loss=f"{info['loss']:.6f}")

    train_flow_loss /= max(1, train_steps)
    train_delta_fm /= max(1, train_steps)
    train_cls /= max(1, train_steps)
    train_cls_acc /= max(1, train_steps)
    train_contrastive /= max(1, train_steps)
    train_grad_norm /= max(1, train_steps)
    train_total_loss = train_flow_loss - args.delta_fm_weight * train_delta_fm + args.aux_weight * train_cls + args.contrastive_weight * train_contrastive
    epoch_time = time.time() - t0
    writer.add_scalar('train/flow_loss', train_flow_loss, epoch)
    writer.add_scalar('train/delta_fm_repulsive', train_delta_fm, epoch)
    writer.add_scalar('train/aux_loss', train_cls, epoch)
    writer.add_scalar('train/aux_acc', train_cls_acc, epoch)
    writer.add_scalar('train/contrastive_loss', train_contrastive, epoch)
    writer.add_scalar('train/total_loss', train_total_loss, epoch)
    writer.add_scalar('train/grad_norm', train_grad_norm, epoch)

    # Per-timestep bucket losses
    for key in info:
        if key.startswith('flow_loss_t_'):
            writer.add_scalar(f'train/{key}', info[key], epoch)

    model.eval()
    val_loss = 0.0
    val_flow = 0.0
    val_cls = 0.0
    val_cls_acc = 0.0
    val_delta = 0.0
    val_steps = 0
    with torch.no_grad():
        for imgs, class_ids in tqdm(val_loader, desc=f"E{epoch+1:03d} val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                vloss, vinfo = compute_flow_loss(
                    model, imgs, class_labels=class_ids,
                    delta_fm_weight=args.delta_fm_weight,
                    label_dropout_prob=0.0,
                    lognormal_sampling=args.lognormal,
                    aux_head=aux_head,
                    aux_weight=args.aux_weight,
                    num_classes=num_classes,
                    null_label=args.null_label,
                    contrastive_weight=args.contrastive_weight,
                    contrastive_projector=contrastive_projector,
                    contrastive_temperature=args.contrastive_temperature,
                    control_indices=control_indices if args.contrastive_weight > 0 else None,
                )
            val_loss += vloss.item()
            val_flow += vinfo.get('flow_loss', 0.0)
            val_cls += vinfo.get('aux_loss', 0.0)
            val_cls_acc += vinfo.get('aux_acc', 0.0)
            val_delta += vinfo.get('delta_fm_repulsive', 0.0)
            val_steps += 1
    val_loss /= max(1, val_steps)
    val_flow /= max(1, val_steps)
    val_cls /= max(1, val_steps)
    val_cls_acc /= max(1, val_steps)
    val_delta /= max(1, val_steps)
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/flow_loss', val_flow, epoch)
    writer.add_scalar('val/aux_loss', val_cls, epoch)
    writer.add_scalar('val/aux_acc', val_cls_acc, epoch)
    writer.add_scalar('val/delta_fm', val_delta, epoch)

    lr_now = optimizer.param_groups[0]['lr']
    aux_str = f"aux={train_cls:.6f} acc={train_cls_acc:.3f}" if args.aux_weight > 0 else "aux=-"
    contrastive_str = f"con={train_contrastive:.6f}" if args.contrastive_weight > 0 else ""
    val_aux_str = f"vaux={val_cls:.6f} vacc={val_cls_acc:.3f}" if args.aux_weight > 0 else ""
    print(f"  E{epoch+1:03d} flow={train_flow_loss:.6f} DFM={train_delta_fm:.6f} "
          f"{aux_str} {contrastive_str} val={val_loss:.6f} "
          f"{val_aux_str} grad={train_grad_norm:.4f} ({epoch_time:.0f}s)")

    with open(metrics_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([epoch+1, f'{train_flow_loss:.6f}', f'{train_delta_fm:.6f}',
                    f'{train_cls:.6f}', f'{train_cls_acc:.4f}',
                    f'{train_contrastive:.6f}', f'{train_total_loss:.6f}',
                    f'{val_loss:.6f}', f'{val_flow:.6f}', f'{val_cls:.6f}', f'{val_cls_acc:.4f}', f'{val_delta:.6f}',
                    f'{lr_now:.2e}', f'{train_grad_norm:.4f}', f'{epoch_time:.0f}'])

    if epoch > 0:
        model.eval()
        with torch.no_grad():
            n_per = 2
            n_viz = len(vis_classes)
            all_samples = []
            for ci in range(n_viz):
                cid = vis_ids[ci:ci+1].repeat(n_per)
                samps = sample_fn(model, n_per, num_steps=args.num_steps,
                                  class_labels=cid,
                                  cfg_scale=args.cfg_scale, null_label=args.null_label,
                                  solver=args.solver)
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
            'train_loss': train_total_loss,
            'val_loss': val_loss,
            'args': vars(args),
        }
        if aux_head:
            ckpt['aux_head_state_dict'] = aux_head.state_dict()
        if contrastive_projector:
            ckpt['contrastive_projector_state_dict'] = contrastive_projector.state_dict()
        if is_best:
            torch.save(ckpt, os.path.join(OUTPUT_DIR, 'flow_best.pth'))
            print(f"  -> Best (val={val_loss:.6f})")
        if (epoch + 1) % args.save_interval == 0:
            torch.save(ckpt, os.path.join(OUTPUT_DIR, f'flow_{epoch+1:03d}.pth'))

if test_items:
    test_path = os.path.join(OUTPUT_DIR, 'test_items.pkl')
    import pickle
    with open(test_path, 'wb') as f:
        pickle.dump(test_items, f)
    print(f"  Test items saved: {test_path}")

print(f"\nDone. Best val: {best_val_loss:.6f}")
print(f"Outputs: {OUTPUT_DIR}")
writer.close()
