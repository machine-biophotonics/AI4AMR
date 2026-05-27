#!/usr/bin/env python3
"""Train SCFM (Sumba et al., arXiv:2605.07676, 2026).

Exact paper implementation:
  - Shared UNet predicts μ_ϕ(x_t, t) = E[x₀|x_t] (posterior mean)
  - Velocity derived: v = (x_t - μ_ϕ)/t  (Eq 11)
  - Endpoint VAE encoder (Eq 14)
  - Exogenous regularization R_ε (Eq 15)
  - VaDE-style GMM prior
  - Total loss: L_SCFM = L_VFM + β·L_KL + L_rec + R_ε  (Eq 16)

Usage:
  python3 train_scfm.py --epochs 200 --use_gmm --unsupervised_gmm --gmm_components 30
"""
import os, sys, warnings, time
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import csv

from mil_model import FlowCropDataset, load_labels
from scfm_model import SCFM, scfm_loss, sample_scfm

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
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

parser.add_argument('--latent_dim', type=int, default=64,
                    help='Structured latent dimension d_z (default: 64)')
parser.add_argument('--exogenous_dim', type=int, default=64,
                    help='Exogenous dimension d_ε for μ_ε head (default: 64)')
parser.add_argument('--beta', type=float, default=0.01,
                    help='KL weight β (β-VAE endpoint, default: 0.01)')
parser.add_argument('--recon_weight', type=float, default=0.1,
                    help='Reconstruction loss weight (default: 0.1)')
parser.add_argument('--kl_anneal', action='store_true', default=False,
                    help='Linearly anneal β from 0 to beta over kl_warmup_epochs')
parser.add_argument('--kl_warmup_epochs', type=int, default=20,
                    help='Epochs for linear KL warmup (default: 20)')
parser.add_argument('--use_gmm', action='store_true', default=False,
                    help='Learnable GMM prior instead of N(0,I)')
parser.add_argument('--gmm_components', type=int, default=30,
                    help='Number of GMM components K (default: 30)')
parser.add_argument('--unsupervised_gmm', action='store_true', default=False,
                    help='VaDE-style GMM (q(c|x) via Bayes rule, no labels)')
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

run_suffix = args.run_name or 'scfm'
OUTPUT_DIR = args.output_dir or os.path.join(
    SCRIPT_DIR,
    f'scfm_run_{run_suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=OUTPUT_DIR)

print("=" * 60)
print("SCFM Training (Sumba et al., arXiv:2605.07676)")
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

print("\n[2/5] Building SCFM model ...")
block_channels = tuple(int(x) for x in args.block_channels.split(','))

model = SCFM(
    in_channels=1,
    sample_size=224,
    block_out_channels=block_channels,
    layers_per_block=2,
    num_class_embeds=num_classes,
    latent_dim=args.latent_dim,
    exogenous_dim=args.exogenous_dim,
    use_gmm=args.use_gmm,
    gmm_components=args.gmm_components,
    unsupervised_gmm=args.unsupervised_gmm,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"  SCFM: latent_dim={args.latent_dim}, d_ε={args.exogenous_dim}")
print(f"  GMM: {'None (N(0,I))' if not args.use_gmm else f'{args.gmm_components}c unsupervised={args.unsupervised_gmm}'}")
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

drug_viz = sorted([i for i, n in enumerate(class_names) if n.endswith('_2x')])
drug_viz += [i for i, n in enumerate(class_names) if n == 'control']
mutant_viz = sorted([i for i, n in enumerate(class_names) if n.endswith('_1') and n not in ('NC_1', 'WT NC_1')])
mutant_viz += [i for i, n in enumerate(class_names) if n in ('NC_1', 'WT NC_1')]
vis_classes = drug_viz + mutant_viz
print(f"  Drug viz: {len(drug_viz)} classes, Mutant viz: {len(mutant_viz)} classes, Total: {len(vis_classes)}")
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
    w.writerow(['epoch', 'train_loss', 'val_loss',
                'train_flow', 'val_flow',
                'train_kl', 'val_kl',
                'train_recon', 'val_recon',
                'train_r_eps', 'val_r_eps',
                'lr', 'beta_used', 'time_s'])

for epoch in range(start_epoch, args.epochs):
    train_ds.set_epoch(epoch)
    val_ds.set_epoch(epoch)

    if args.kl_anneal:
        beta_used = args.beta * min(epoch / args.kl_warmup_epochs, 1.0)
    else:
        beta_used = args.beta

    model.train()
    train_loss = 0.0
    train_steps = 0
    train_flow = 0.0
    train_kl = 0.0
    train_recon = 0.0
    train_r_eps = 0.0
    t0 = time.time()

    pbar = tqdm(train_loader, desc=f"E{epoch+1:03d}", leave=False)
    for imgs, class_ids in pbar:
        imgs = imgs.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=amp_dtype):
            loss, comp = scfm_loss(
                model, imgs, class_labels=class_ids,
                beta=beta_used,
                recon_weight=args.recon_weight,
                use_gmm=args.use_gmm,
                unsupervised_gmm=args.unsupervised_gmm,
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()
        train_flow += comp.get('flow', 0.0)
        train_kl += comp.get('kl', 0.0)
        train_recon += comp.get('recon', 0.0)
        train_r_eps += comp.get('r_eps', 0.0)
        train_steps += 1
        pbar.set_postfix(loss=loss.item())

    train_loss /= max(1, train_steps)
    train_flow /= max(1, train_steps)
    train_kl /= max(1, train_steps)
    train_recon /= max(1, train_steps)
    train_r_eps /= max(1, train_steps)
    epoch_time = time.time() - t0

    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/flow', train_flow, epoch)
    writer.add_scalar('train/kl', train_kl, epoch)
    writer.add_scalar('train/recon', train_recon, epoch)
    writer.add_scalar('train/r_eps', train_r_eps, epoch)

    model.eval()
    val_loss = 0.0
    val_steps = 0
    val_flow = 0.0
    val_kl = 0.0
    val_recon = 0.0
    val_r_eps = 0.0
    with torch.no_grad():
        for imgs, class_ids in tqdm(val_loader, desc=f"E{epoch+1:03d} val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss, comp = scfm_loss(
                    model, imgs, class_labels=class_ids,
                    beta=beta_used,
                    recon_weight=args.recon_weight,
                    use_gmm=args.use_gmm,
                    unsupervised_gmm=args.unsupervised_gmm,
                )
            val_loss += loss.item()
            val_flow += comp.get('flow', 0.0)
            val_kl += comp.get('kl', 0.0)
            val_recon += comp.get('recon', 0.0)
            val_r_eps += comp.get('r_eps', 0.0)
            val_steps += 1
    val_loss /= max(1, val_steps)
    val_flow /= max(1, val_steps)
    val_kl /= max(1, val_steps)
    val_recon /= max(1, val_steps)
    val_r_eps /= max(1, val_steps)

    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/flow', val_flow, epoch)
    writer.add_scalar('val/kl', val_kl, epoch)
    writer.add_scalar('val/recon', val_recon, epoch)
    writer.add_scalar('val/r_eps', val_r_eps, epoch)

    lr_now = optimizer.param_groups[0]['lr']

    # GMM health diagnostics
    gmm_str = ""
    if args.use_gmm and args.unsupervised_gmm and hasattr(model, 'gmm'):
        with torch.no_grad():
            q_c_all = []
            for imgs, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                mu_z, _ = model.encode(imgs, class_labels=None, return_all=False)
                q_c_all.append(model.gmm.responsibilities(mu_z))
            if q_c_all:
                q_c_cat = torch.cat(q_c_all, dim=0)
                ent = (-(q_c_cat * torch.log(q_c_cat.clamp(1e-10, 1))).sum(dim=1)).mean().item()
                assignments = q_c_cat.argmax(dim=1)
                active = assignments.unique().numel()
                gmm_str = f" gmm_ent={ent:.3f} active={active}/{model.gmm.n_components}"

    beta_str = f" β={beta_used:.4f}" if args.kl_anneal else ""
    print(f"  E{epoch+1:03d} train={train_loss:.6f} val={val_loss:.6f} "
          f"(flow={val_flow:.4f} kl={val_kl:.4f} recon={val_recon:.4f} "
          f"r_eps={val_r_eps:.4f}{beta_str}{gmm_str}) ({epoch_time:.0f}s)")

    with open(metrics_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([epoch+1, f'{train_loss:.6f}', f'{val_loss:.6f}',
                    f'{train_flow:.4f}', f'{val_flow:.4f}',
                    f'{train_kl:.4f}', f'{val_kl:.4f}',
                    f'{train_recon:.4f}', f'{val_recon:.4f}',
                    f'{train_r_eps:.4f}', f'{val_r_eps:.4f}',
                    f'{lr_now:.2e}', f'{beta_used:.4e}', f'{epoch_time:.0f}'])

    # Generate samples each epoch
    if epoch % 1 == 0:
        model.eval()
        with torch.no_grad():
            all_samples = []
            for ci in vis_classes:
                cid = torch.tensor([ci], device=device)
                samp = sample_scfm(model, 1, num_steps=args.num_steps,
                                   class_labels=cid, device=device)
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
                ax.set_xticks([])
                ax.set_yticks([])

                ax = fig.add_subplot(gs[1, i])
                if i < n_mutants:
                    img = all_samples[n_drugs + i]
                    img_01 = (img * 0.5 + 0.5).clamp(0, 1)
                    ax.imshow(img_01.squeeze(), cmap='gray', vmin=0, vmax=1)
                    ax.set_xlabel(class_names[mutant_viz[i]].replace('_', ' '), fontsize=3)
                ax.set_xticks([])
                ax.set_yticks([])

            plt.suptitle(f'Epoch {epoch+1}: SCFM samples (drugs 2x top, mutants 1 bottom)', fontsize=7, y=0.98)
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
            torch.save(ckpt, os.path.join(OUTPUT_DIR, 'scfm_best.pth'))
            print(f"  -> Best (val={val_loss:.6f})")
        if (epoch + 1) % args.save_interval == 0:
            torch.save(ckpt, os.path.join(OUTPUT_DIR, f'scfm_{epoch+1:03d}.pth'))

print(f"\nDone. Best val: {best_val_loss:.6f}")
print(f"Outputs: {OUTPUT_DIR}")
writer.close()
