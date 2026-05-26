#!/usr/bin/env python3
"""CellFlux training — exact reproduction of CellFlux (ICML 2025).

Flow matching from control→perturbed images, conditioned on perturbation embeddings.

Usage:
    python3 train_cellflux.py --epochs 200 --batch_size 32
    python3 train_cellflux.py --test_plate P6 --epochs 200
"""
import os, sys, warnings, time, json, math
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import csv

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from cellflux_model import UNetModel
from cellflux_dataset import build_datasets

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=2,
                    help='Per-GPU batch size. GPU has 24GB; bs=2 fits with checkpoint.')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--model_channels', type=int, default=128)
parser.add_argument('--num_res_blocks', type=int, default=2)
parser.add_argument('--channel_mult', type=str, default='2,2,2')
parser.add_argument('--attention_resolutions', type=str, default='2')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--condition_dim', type=int, default=512,
                    help='Dimension of learned perturbation embeddings')
parser.add_argument('--class_drop_prob', type=float, default=0.2,
                    help='Probability to drop conditioning (CFG)')
parser.add_argument('--val_split', type=float, default=0.2)
parser.add_argument('--grad_accum', type=int, default=8,
                    help='Gradient accumulation steps (effective BS = batch_size * grad_accum)')
parser.add_argument('--test_plate', type=str, default=None)
parser.add_argument('--skewed_timesteps', action='store_true', default=True,
                    help='Use skewed timestep sampling (EDM-style)')
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=4)
args = parser.parse_args()

args.channel_mult = tuple(int(x) for x in args.channel_mult.split(','))
args.attention_resolutions = tuple(int(x) for x in args.attention_resolutions.split(','))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"cellflux_{TIMESTAMP}"
OUTPUT_DIR = args.output_dir or os.path.join(SCRIPT_DIR, RUN_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

writer = SummaryWriter(log_dir=OUTPUT_DIR)

print("=" * 60)
print(f"CellFlux Training — {RUN_NAME}")
print(f"  model_channels={args.model_channels}, num_res_blocks={args.num_res_blocks}")
print(f"  channel_mult={args.channel_mult}, condition_dim={args.condition_dim}")
print(f"  lr={args.lr}, class_drop_prob={args.class_drop_prob}")
print(f"  test_plate={args.test_plate}, grad_accum={args.grad_accum}")
print(f"  skewed_timesteps={args.skewed_timesteps}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 60)

# ─── Build Datasets ──────────────────────────────────────────────────────
print("\n[1/5] Building paired datasets (CellFlux-style control→perturbed) ...")
train_ds, val_ds, test_ds, num_pert_classes, class_names, pert2cond = build_datasets(
    PROJECT_ROOT, SCRIPT_DIR,
    test_plate=args.test_plate,
    val_split=args.val_split,
    seed=SEED,
)
print(f"  Train: {len(train_ds)} paired samples")
print(f"  Val:   {len(val_ds)} paired samples")
if test_ds:
    print(f"  Test:  {len(test_ds)} paired samples (plate {args.test_plate})")
print(f"  {num_pert_classes} perturbation classes, {len(class_names)} total classes")

# Weighted sampler for class balance
train_labels = np.array([train_ds.pert2cond[c] for _, c in train_ds.perturbed_items])
class_counts = np.bincount(train_labels, minlength=num_pert_classes)
weights_per_class = np.where(class_counts == 0, 0, 1.0 / class_counts.astype(np.float64))
sample_weights = np.array([weights_per_class[l] for l in train_labels], dtype=np.float64)
train_sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

# ─── Model + Embedding ───────────────────────────────────────────────────
print("\n[2/5] Building CellFlux UNetModel ...")
model = UNetModel(
    in_channels=1, model_channels=args.model_channels, out_channels=1,
    num_res_blocks=args.num_res_blocks, channel_mult=args.channel_mult,
    attention_resolutions=args.attention_resolutions, dropout=args.dropout,
    condition_dim=args.condition_dim,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model: {n_params:,} params")

# Learned perturbation embedding (no pre-computed embeddings available)
pert_embedding = nn.Embedding(num_pert_classes, args.condition_dim).to(device)
print(f"  Pert embedding: {sum(p.numel() for p in pert_embedding.parameters()):,} params ({num_pert_classes} × {args.condition_dim})")

# ─── Optimizer + LR Schedule ─────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(pert_embedding.parameters()),
    lr=args.lr, betas=(0.9, 0.95),
)

# Linear LR schedule: decay from lr to ~0 over epochs (CellFlux-style)
def lr_lambda(epoch):
    progress = epoch / args.epochs
    return max(1e-8 / args.lr, 1.0 - progress)

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print(f"  Optimizer: AdamW (lr={args.lr}, betas=(0.9, 0.95))")
print(f"  LR schedule: linear decay to ~0 over {args.epochs} epochs")
print(f"  Effective batch size: {args.batch_size * args.grad_accum} ({args.batch_size} × {args.grad_accum})")

# ─── Training ────────────────────────────────────────────────────────────
print("\n[3/5] Training ...")
scaler = torch.cuda.amp.GradScaler()

def skewed_timestep_sample(num_samples, device):
    """EDM-style lognormal timestep sampling."""
    P_mean, P_std = -1.2, 1.2
    rnd_normal = torch.randn(num_samples, device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    t = 1 / (1 + sigma)
    return t.clamp(0.0001, 1.0)


def condot_path(x_0, x_1, t):
    """Conditional OT path: linear interpolation.
    x_t = (1-t)*x_0 + t*x_1
    dx_t = x_1 - x_0
    """
    t_exp = t.view(-1, 1, 1, 1)
    x_t = (1 - t_exp) * x_0 + t_exp * x_1
    dx_t = x_1 - x_0
    return x_t, dx_t


best_val_loss = float('inf')
metrics_path = os.path.join(OUTPUT_DIR, 'metrics.csv')
with open(metrics_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'time_s'])

optimizer.zero_grad()

for epoch in range(1, args.epochs + 1):
    epoch_start = time.time()
    model.train()
    pert_embedding.train()

    train_losses = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

    for step, (x_ctrl, x_trt, cond) in enumerate(pbar):
        x_ctrl = x_ctrl.to(device)
        x_trt = x_trt.to(device)
        cond = cond.to(device)
        B = x_ctrl.shape[0]

        if args.skewed_timesteps:
            t = skewed_timestep_sample(B, device)
        else:
            t = torch.rand(B, device=device)

        x_0 = x_ctrl
        x_1 = x_trt
        x_t, u_t = condot_path(x_0, x_1, t)

        # Condition embedding (CellFlux: embedding_matrix → mol_embed_transform)
        cond_emb = pert_embedding(cond)  # (B, condition_dim)

        # Label dropout for CFG (CellFlux: if rand < class_drop_prob → empty conditioning)
        if torch.rand(1).item() < args.class_drop_prob:
            extra = {}
        else:
            extra = {"concat_conditioning": cond_emb}

        with torch.cuda.amp.autocast():
            v_pred = model(x_t, t, extra=extra)
            loss = F.mse_loss(v_pred, u_t)

        loss_scaled = loss / args.grad_accum
        scaler.scale(loss_scaled).backward()

        if (step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(pert_embedding.parameters()), 10.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_losses.append(loss.item())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # ─── Validation ─────────────────────────────────────────────────────
    model.eval()
    pert_embedding.eval()
    val_losses = []

    with torch.no_grad():
        for x_ctrl, x_trt, cond in val_loader:
            x_ctrl = x_ctrl.to(device)
            x_trt = x_trt.to(device)
            cond = cond.to(device)
            B = x_ctrl.shape[0]

            t = torch.rand(B, device=device)
            x_0 = x_ctrl
            x_1 = x_trt
            x_t, u_t = condot_path(x_0, x_1, t)

            cond_emb = pert_embedding(cond)
            extra = {"concat_conditioning": cond_emb}

            with torch.cuda.amp.autocast():
                v_pred = model(x_t, t, extra=extra)
                val_loss = F.mse_loss(v_pred, u_t)

            val_losses.append(val_loss.item())

    t_epoch = time.time() - epoch_start

    train_loss_avg = float(np.mean(train_losses))
    val_loss_avg = float(np.mean(val_losses))
    current_lr = optimizer.param_groups[0]['lr']

    # Logging
    with open(metrics_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([epoch, f"{train_loss_avg:.6f}", f"{val_loss_avg:.6f}",
                    f"{current_lr:.6f}", f"{t_epoch:.0f}"])

    writer.add_scalar('loss/train', train_loss_avg, epoch)
    writer.add_scalar('loss/val', val_loss_avg, epoch)
    writer.add_scalar('train/lr', current_lr, epoch)

    print(f"Epoch {epoch:3d} | train_loss={train_loss_avg:.4f} | val_loss={val_loss_avg:.4f} | lr={current_lr:.2e} | {t_epoch:.0f}s")

    # Save best model
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        ckpt = {
            'epoch': epoch, 'model': model.state_dict(),
            'pert_embedding': pert_embedding.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss_avg,
            'args': vars(args),
        }
        torch.save(ckpt, os.path.join(OUTPUT_DIR, 'best_model.pth'))
        print(f"  → Saved best model (val_loss={val_loss_avg:.4f})")

    # Save latest checkpoint
    torch.save({
        'epoch': epoch, 'model': model.state_dict(),
        'pert_embedding': pert_embedding.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(OUTPUT_DIR, 'latest.pth'))

    lr_scheduler.step()

print(f"\nDone! Best val_loss={best_val_loss:.4f}")
print(f"Results in {OUTPUT_DIR}")
