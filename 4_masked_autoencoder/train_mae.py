#!/usr/bin/env python3
"""Train Masked Autoencoder on single crops from all plates (drug + mutant).

Usage:
    python3 train_mae.py
    python3 train_mae.py --epochs 300 --batch_size 64 --model tiny
    python3 train_mae.py --run_name mae_vit_small --model small --lr 1.5e-4
"""
import os, sys, warnings, glob, json, re, time
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import math
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from mil_model import MAECropDataset
from mae_model import mae_vit_tiny, mae_vit_small, mae_vit_base

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1.5e-4)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--warmup_epochs', type=int, default=10)
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--model', type=str, default='small', choices=['tiny', 'small', 'base'])
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--val_split', type=float, default=0.1)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--log_interval', type=int, default=50)
parser.add_argument('--save_interval', type=int, default=50)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--grid_size', type=int, default=0,
                    help='0=random crop (default), >0=enables fixed grid positions')
parser.add_argument('--use_fg_loss', action='store_true',
                    help='AttG-style loss: exp(fg_weight/temperature) weighting')
parser.add_argument('--fg_temperature', type=float, default=0.5,
                    help='Temperature for AttG-style loss (lower = higher FG amplification)')
parser.add_argument('--use_fg_masking', action='store_true',
                    help='Foreground-biased masking: mask 95% FG, 67% BG patches')
parser.add_argument('--disable_norm_pix_loss', action='store_true',
                    help='Disable per-patch normalization in loss target')
args = parser.parse_args()

run_suffix = args.run_name or args.model
if args.disable_norm_pix_loss:
    run_suffix += '_nonorm'
if args.use_fg_loss:
    run_suffix += '_fg'
if args.use_fg_masking:
    run_suffix += '_fgmask'
OUTPUT_DIR = args.output_dir or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f'mae_run_{run_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=OUTPUT_DIR)

print("=" * 60)
print(f"MAE Training: {args.model}")
print(f"Output: {OUTPUT_DIR}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Data — collect ALL images from ALL plates (drug + mutant)
# ---------------------------------------------------------------------------
print("\n[1/5] Collecting images from all plates ...")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

all_paths = []
for pi in range(1, 7):
    for cond in ('Drugs_Data', 'Mutants_Data'):
        d = os.path.join(PROJECT_ROOT, cond, f'P{pi}')
        if os.path.exists(d):
            for ext in ('*.tif', '*.tiff'):
                all_paths.extend(glob.glob(os.path.join(d, '**', ext), recursive=True))
all_paths = sorted(set(all_paths))
print(f"  Found {len(all_paths)} images total")

# Train/val split
n_val = max(1, int(len(all_paths) * args.val_split))
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(all_paths))
val_paths = [all_paths[i] for i in perm[:n_val]]
train_paths = [all_paths[i] for i in perm[n_val:]]
print(f"  Train: {len(train_paths)}, Val: {len(val_paths)}")

train_dataset = MAECropDataset(train_paths, augment=True, seed=SEED)
val_dataset = MAECropDataset(val_paths, augment=False, seed=SEED)

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=True, drop_last=True,
    persistent_workers=True, prefetch_factor=4,
)
val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True,
    persistent_workers=True, prefetch_factor=4,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
print("\n[2/5] Building model ...")
model_map = {'tiny': mae_vit_tiny, 'small': mae_vit_small, 'base': mae_vit_base}
mae = model_map[args.model](
    in_chans=1, mask_ratio=args.mask_ratio,
    norm_pix_loss=not args.disable_norm_pix_loss,
    use_fg_loss=args.use_fg_loss,
    use_fg_masking=args.use_fg_masking,
    fg_temperature=args.fg_temperature,
)
mae.to(device)
n_params = sum(p.numel() for p in mae.parameters())
n_enc = sum(p.numel() for p in mae.encoder.parameters())
n_dec = sum(p.numel() for p in mae.decoder.parameters())
print(f"  Total params: {n_params:,}  Encoder: {n_enc:,}  Decoder: {n_dec:,}")

# ---------------------------------------------------------------------------
# Optimizer + scheduler
# ---------------------------------------------------------------------------
print("\n[3/5] Setting up optimizer ...")

def add_weight_decay(model, weight_decay=0.05, skip_list=None):
    """Separate params into wd/no_wd groups. Excludes biases, norms, embeddings."""
    if skip_list is None:
        skip_list = {'bias', 'norm', 'cls_token', 'mask_token', 'pos_embed'}
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

param_groups = add_weight_decay(mae, args.weight_decay)
optimizer = torch.optim.AdamW(
    param_groups, lr=args.lr, betas=(0.9, 0.95)
)

total_steps = len(train_loader) * args.epochs
warmup_steps = len(train_loader) * args.warmup_epochs

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
    mae.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    print(f"  Resumed from epoch {ckpt['epoch']}")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
print("\n[4/5] Training ...")
best_val_loss = float('inf')
amp_dtype = torch.bfloat16 if torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported() else torch.float16
print(f"  AMP dtype: {amp_dtype}")

# Metrics CSV
metrics_path = os.path.join(OUTPUT_DIR, 'metrics.csv')
with open(metrics_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['epoch', 'train_loss', 'val_loss', 'val_psnr', 'val_ssim', 'grad_norm', 'lr', 'epoch_time_s'])

for epoch in range(start_epoch, args.epochs):
    train_dataset.set_epoch(epoch)
    val_dataset.set_epoch(epoch)

    mae.train()
    train_loss = 0
    train_steps = 0
    grad_norm_sum = 0
    t0 = time.time()

    pbar = tqdm(train_loader, desc=f"E{epoch+1:03d} train", leave=False)
    for step, imgs in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype):
            output = mae(imgs)
            loss = output['loss']

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()
        train_steps += 1
        grad_norm_sum += gn.item()

        pbar.set_postfix(loss=loss.item(), gn=gn.item())

        if step % args.log_interval == 0:
            lr_now = optimizer.param_groups[0]['lr']
            writer.add_scalar('train/loss_step', loss.item(),
                              epoch * len(train_loader) + step)
            writer.add_scalar('train/grad_norm_step', gn.item(),
                              epoch * len(train_loader) + step)
            writer.add_scalar('train/lr', lr_now,
                              epoch * len(train_loader) + step)

    train_loss /= max(1, train_steps)
    avg_grad_norm = grad_norm_sum / max(1, train_steps)
    epoch_time = time.time() - t0
    writer.add_scalar('train/loss_epoch', train_loss, epoch)
    writer.add_scalar('train/grad_norm_epoch', avg_grad_norm, epoch)
    tqdm.write(f"  E{epoch+1:03d} train_loss={train_loss:.4f} gn={avg_grad_norm:.4f} ({epoch_time:.0f}s)")

    # Validation
    mae.eval()
    val_loss = 0
    val_steps = 0
    psnr_sum = 0
    ssim_sum = 0
    save_recon = True

    val_pbar = tqdm(val_loader, desc=f"E{epoch+1:03d} val", leave=False)
    with torch.no_grad():
        for imgs in val_pbar:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                output = mae(imgs)
                loss = output['loss']

            val_loss += loss.item()
            val_steps += 1

            # PSNR/SSIM on pixel-space reconstruction
            recon_pixel = output['recon_pixel']  # (B, C, H, W) in [-1, 1]
            target_pixel = output['target_pixel']  # (B, C, H, W) in [-1, 1]
            r, t_ = recon_pixel.float(), target_pixel.float()
            # Convert from [-1, 1] to [0, 1] for metrics
            r_01 = (r + 1) / 2
            t_01 = (t_ + 1) / 2
            r_01 = r_01.clamp(0, 1)
            t_01 = t_01.clamp(0, 1)
            psnr_sum += peak_signal_noise_ratio(r_01, t_01, data_range=1.0).item()
            ssim_sum += structural_similarity_index_measure(r_01, t_01, data_range=1.0).item()

            val_pbar.set_postfix(loss=loss.item(), psnr=f"{psnr_sum/val_steps:.1f}")

            if save_recon:
                save_recon = False
                recon_pixel = output['recon_pixel'].cpu()
                n_show = min(8, imgs.shape[0])
                imgs_01 = (imgs.cpu() * 0.5 + 0.5).clamp(0, 1)
                mask = output['mask'].cpu()

                fig, axes = plt.subplots(4, n_show, figsize=(n_show * 2, 8))
                for i in range(n_show):
                    orig_01 = imgs_01[i]
                    rec_01 = (recon_pixel[i] * 0.5 + 0.5).clamp(0, 1)

                    img_patches = mae.patchify(imgs[i:i+1]).cpu()
                    img_patches_01 = (img_patches + 1) / 2
                    m_patches = mask[i]
                    masked_patches = img_patches_01 * (1 - m_patches).unsqueeze(-1)
                    masked_img = mae.unpatchify(masked_patches)
                    masked_img = masked_img[0].clamp(0, 1)

                    axes[0, i].imshow(orig_01.squeeze(), cmap='gray', vmin=0, vmax=1)
                    axes[0, i].set_title('Original', fontsize=7)
                    axes[0, i].axis('off')

                    axes[1, i].imshow(masked_img.squeeze(), cmap='gray', vmin=0, vmax=1)
                    axes[1, i].set_title('Masked', fontsize=7)
                    axes[1, i].axis('off')

                    axes[2, i].imshow(rec_01.squeeze(), cmap='gray', vmin=0, vmax=1)
                    axes[2, i].set_title('Reconstruction', fontsize=7)
                    axes[2, i].axis('off')

                    diff = (rec_01 - orig_01).abs().squeeze()
                    axes[3, i].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
                    axes[3, i].set_title(f'Error (max={diff.max():.2f})', fontsize=7)
                    axes[3, i].axis('off')

                config_parts = []
                if args.use_fg_loss:
                    config_parts.append('FG loss')
                if args.use_fg_masking:
                    config_parts.append('FG masking')
                if args.disable_norm_pix_loss:
                    config_parts.append('no norm')
                if not config_parts:
                    config_parts.append('baseline')
                plt.suptitle(f'MAE Epoch {epoch+1}: {"+".join(config_parts)}', fontsize=9)
                plt.tight_layout()
                recon_path = os.path.join(OUTPUT_DIR, f'recon_epoch_{epoch+1:03d}.png')
                fig.savefig(recon_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

    val_loss /= max(1, val_steps)
    val_psnr = psnr_sum / max(1, val_steps)
    val_ssim = ssim_sum / max(1, val_steps)
    lr_now = optimizer.param_groups[0]['lr']

    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/psnr', val_psnr, epoch)
    writer.add_scalar('val/ssim', val_ssim, epoch)

    # Append to CSV
    with open(metrics_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([epoch + 1, f'{train_loss:.6f}', f'{val_loss:.6f}',
                    f'{val_psnr:.4f}', f'{val_ssim:.4f}',
                    f'{avg_grad_norm:.4f}', f'{lr_now:.2e}', f'{epoch_time:.0f}'])

    tqdm.write(f"  E{epoch+1:03d} val_loss={val_loss:.4f}  PSNR={val_psnr:.2f}  SSIM={val_ssim:.4f}")

    # Save checkpoint
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss

    if (epoch + 1) % args.save_interval == 0 or is_best or epoch == args.epochs - 1:
        ckpt = {
            'epoch': epoch,
            'model_state_dict': mae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'args': vars(args),
        }
        if is_best:
            torch.save(ckpt, os.path.join(OUTPUT_DIR, 'mae_best.pth'))
            tqdm.write(f"  -> New best model (val_loss={val_loss:.4f}, PSNR={val_psnr:.2f})")
        if (epoch + 1) % args.save_interval == 0:
            torch.save(ckpt, os.path.join(OUTPUT_DIR, f'mae_epoch_{epoch+1:03d}.pth'))

# ---------------------------------------------------------------------------
# Final
# ---------------------------------------------------------------------------
print("\n[5/5] Done")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Metrics saved to: {metrics_path}")
print(f"Outputs: {OUTPUT_DIR}")
print(f"{'=' * 60}")
writer.close()
