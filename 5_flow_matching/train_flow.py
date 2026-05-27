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
from matplotlib.gridspec import GridSpec
from datetime import datetime
import csv

from mil_model import FlowCropDataset, load_labels
from flow_model import FlowUNet, FreqFlowUNet, StructFlowUNet, CombinedFlowUNet
from flow_model import compute_flow_loss, sample, compute_struct_flow_loss, sample_struct
from flow_model import compute_unified_loss, sample_combined



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
parser.add_argument('--struct_flow', action='store_true', default=False,
                    help='Use StructFlow (SCFM) architecture with structured latent')
parser.add_argument('--struct_latent_dim', type=int, default=64,
                    help='Structured latent dimension for StructFlow')
parser.add_argument('--struct_kl_weight', type=float, default=0.001,
                    help='KL divergence weight for StructFlow VAE loss')
parser.add_argument('--struct_recon_weight', type=float, default=0.1,
                    help='Reconstruction loss weight for StructFlow')
parser.add_argument('--supcon', action='store_true', default=False,
                    help='Supervised contrastive loss on μ_z (SupCon, Khosla et al. 2020)')
parser.add_argument('--supcon_weight', type=float, default=0.1,
                    help='SupCon loss weight')
parser.add_argument('--supcon_temperature', type=float, default=0.1,
    help='SupCon temperature parameter')
parser.add_argument('--predict_mu', action='store_true', default=False,
    help='UNet predicts posterior mean μ (SCFM) instead of velocity v')
parser.add_argument('--dual_predict_mu', action='store_true', default=False,
    help='UNet outputs both v (channel 0) and μ (channel 1) — stable class-conditional ΔFM')
parser.add_argument('--exogenous_dim', type=int, default=64,
    help='Exogenous dimension d_ε for μ_ε head (SCFM, default: 64)')
parser.add_argument('--use_gmm', action='store_true', default=False,
    help='Use learnable GMM prior instead of N(0,I) (SCFM only)')
parser.add_argument('--gmm_components', type=int, default=30,
    help='Number of GMM components K (default: 30)')
parser.add_argument('--unsupervised_gmm', action='store_true', default=False,
    help='VaDE-style GMM: q(c|x) via Bayes rule, no labels needed')
parser.add_argument('--mu_anneal', action='store_true', default=False,
    help='Linearly anneal struct_kl_weight from 0 over mu_warmup_epochs (SCFM)')
parser.add_argument('--mu_warmup_epochs', type=int, default=20,
    help='Epochs for linear beta warmup (SCFM, default: 20)')
parser.add_argument('--r_eps_weight', type=float, default=1.0,
    help='Weight for exogenous regularization R_ε (default: 1.0). '
         'Reduce if R_ε dominates flow loss (e.g., large exogenous_dim).')
parser.add_argument('--kmeans_init', action='store_true', default=False,
    help='Initialize GMM means with k-means on encoder outputs after kmeans_init_epoch')
parser.add_argument('--kmeans_init_epoch', type=int, default=3,
    help='Epoch to perform k-means GMM initialization (default: 3)')
parser.add_argument('--cyclical_anneal', action='store_true', default=False,
    help='Use cyclical β annealing instead of monotonic (Fu et al., NAACL 2019)')
parser.add_argument('--num_cycles', type=int, default=4,
    help='Number of β cycles for cyclical annealing (default: 4)')
parser.add_argument('--min_beta_frac', type=float, default=0.0,
    help='Minimum β as fraction of struct_kl_weight during cyclical annealing (default: 0.0)')
parser.add_argument('--anneal_schedule', type=str, default='linear',
    choices=['linear', 'cosine', 'sigmoid'],
    help='Annealing curve shape (default: linear)')
parser.add_argument('--detach_z', action='store_true', default=False,
    help='Stop gradient at z before decode (SCFM paper default). '
         'When False (default), flow gradient reaches encoder.')
args = parser.parse_args()


# ── β annealing function ──────────────────────────────────────────────────
def get_beta(epoch: int, total_epochs: int, kl_weight: float,
             warmup_epochs: int, cyclical: bool = False,
             num_cycles: int = 4, min_frac: float = 0.0,
             schedule: str = 'linear') -> float:
    """Compute β for current epoch.

    Monotonic:  linear 0→kl_weight over warmup_epochs, then clipped.
    Cyclical:   num_cycles repeats of 0→kl_weight→0 (Fu et al. 2019).
    """
    if not cyclical:
        return kl_weight * min(epoch / max(1, warmup_epochs), 1.0)

    # Cyclical annealing
    cycle_len = max(1, total_epochs / max(1, num_cycles))
    t = (epoch % cycle_len) / cycle_len  # position within cycle [0, 1)
    # t < 0.5: ramp up; t >= 0.5: ramp down
    tau = 2 * t if t < 0.5 else 2 * (1 - t)
    tau = max(0.0, min(1.0, tau))
    if schedule == 'cosine':
        tau = 0.5 * (1 - np.cos(np.pi * tau))
    elif schedule == 'sigmoid':
        tau = 1.0 / (1.0 + np.exp(-10 * (tau - 0.5)))
    return kl_weight * (min_frac + (1 - min_frac) * tau)


# ── k-means init function ─────────────────────────────────────────────────
def kmeans_init_gmm(model, loader, device, gmm_components: int, seed: int = 42):
    """Run k-means on encoder μ_z and copy centroids to model.gmm.means."""
    from sklearn.cluster import KMeans
    from tqdm import tqdm
    model.eval()
    mu_z_all = []
    with torch.no_grad():
        for imgs, class_ids in tqdm(loader, desc='k-means init', leave=False):
            imgs = imgs.to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            t_enc = torch.full((imgs.shape[0],), 1.0, device=device)
            if hasattr(model, 'encode'):
                out = model.encode(imgs, t_enc, class_ids)
                mu_z = out[0]  # mu_z is first return value
            else:
                continue
            mu_z_all.append(mu_z.cpu().numpy())
    if not mu_z_all:
        return
    X = np.concatenate(mu_z_all, axis=0)
    kmeans = KMeans(n_clusters=gmm_components, random_state=seed, n_init=3).fit(X)
    centroids = torch.from_numpy(kmeans.cluster_centers_).to(dtype=torch.float32, device=device)
    if hasattr(model, 'gmm') and model.gmm is not None:
        model.gmm.means.data = centroids
        print(f"  -> k-means GMM init: {gmm_components} centroids loaded (inertia={kmeans.inertia_:.2f})")

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

# Unconditional UNet (no class embedding) when using SCFM posterior-mean mode
model_num_classes = None if (args.predict_mu and not args.dual_predict_mu) else num_classes
if model_num_classes is None and args.struct_flow:
    print(f"  Unconditional UNet: class conditioning removed (SCFM mode)")

# Supervised GMM: each class needs its own component
if args.use_gmm and not args.unsupervised_gmm:
    if args.gmm_components != num_classes:
        print(f"  Supervised GMM: overriding gmm_components from {args.gmm_components} to {num_classes}")
        args.gmm_components = num_classes

if args.freq_flow and args.struct_flow:
    print(f"  CombinedFlow: freq_flow + struct_flow, predict_mu={args.predict_mu}")
    model = CombinedFlowUNet(
        in_channels=1,
        sample_size=224,
        block_out_channels=block_channels,
        freq_block_out_channels=freq_block_channels,
        layers_per_block=2,
        num_class_embeds=model_num_classes,
        freq_filter_D=args.freq_filter_D,
        use_freq=True,
        use_struct=True,
        latent_dim=args.struct_latent_dim,
        predict_mu=args.predict_mu,
        exogenous_dim=args.exogenous_dim,
        use_gmm=args.use_gmm,
        gmm_components=args.gmm_components,
        unsupervised_gmm=args.unsupervised_gmm,
        dual_predict_mu=args.dual_predict_mu,
    ).to(device)
    print(f"  CombinedFlow: freq_branch={freq_block_channels}, latent_dim={args.struct_latent_dim}")
elif args.freq_flow:
    print(f"  FreqFlow: batch_size={args.batch_size}")
    model = FreqFlowUNet(
        in_channels=1,
        sample_size=224,
        block_out_channels=block_channels,
        freq_block_out_channels=freq_block_channels,
        layers_per_block=2,
        num_class_embeds=model_num_classes,
        freq_filter_D=args.freq_filter_D,
    ).to(device)
    print(f"  FreqFlow: freq_branch={freq_block_channels}")
elif args.struct_flow:
    print(f"  StructFlow: latent_dim={args.struct_latent_dim}, predict_mu={args.predict_mu}")
    model = StructFlowUNet(
        in_channels=1,
        sample_size=224,
        block_out_channels=block_channels,
        layers_per_block=2,
        num_class_embeds=model_num_classes,
        latent_dim=args.struct_latent_dim,
        predict_mu=args.predict_mu,
        exogenous_dim=args.exogenous_dim,
        use_gmm=args.use_gmm,
        gmm_components=args.gmm_components,
        unsupervised_gmm=args.unsupervised_gmm,
    ).to(device)
    if args.predict_mu:
        print(f"  SCFM mode: exogenous_dim={args.exogenous_dim}, GMM={args.use_gmm}")
    print(f"  StructFlow: kl_weight={args.struct_kl_weight}, recon_weight={args.struct_recon_weight}")
else:
    model = FlowUNet(
        in_channels=1,
        sample_size=224,
        block_out_channels=block_channels,
        layers_per_block=2,
        num_class_embeds=model_num_classes,
    ).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Params: {n_params:,}")

use_unified = isinstance(model, CombinedFlowUNet)
if use_unified:
    print(f"  Unified dispatch: freq={args.freq_flow}, struct={args.struct_flow}, delta_fm={args.delta_fm}")

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

# Visualization: all 22 antibiotics (2x) + control, all 28 mutants (1) + 2 controls
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
    if use_unified:
        csv_header = ['epoch', 'train_loss', 'val_loss',
                      'train_spatial', 'val_spatial', 'train_freq', 'val_freq',
                      'train_kl', 'val_kl', 'train_recon', 'val_recon',
                      'train_r_eps', 'val_r_eps',
                      'train_neg', 'val_neg']
        if args.supcon:
            csv_header += ['train_supcon', 'val_supcon']
        if args.predict_mu or args.dual_predict_mu:
            csv_header += ['beta']
        csv_header += ['lr', 'time_s']
    elif args.struct_flow:
        csv_header = ['epoch', 'train_loss', 'val_loss',
                      'train_flow', 'val_flow',
                      'train_kl', 'val_kl', 'train_recon', 'val_recon',
                      'train_r_eps', 'val_r_eps',
                      'train_neg', 'val_neg']
        if args.supcon:
            csv_header += ['train_supcon', 'val_supcon']
        if args.predict_mu or args.dual_predict_mu:
            csv_header += ['beta']
        csv_header += ['lr', 'time_s']
    else:
        csv_header = ['epoch', 'train_loss', 'val_loss',
                      'train_spatial', 'val_spatial', 'train_freq', 'val_freq',
                      'train_neg', 'val_neg', 'lr', 'time_s']
    w.writerow(csv_header)

for epoch in range(start_epoch, args.epochs):
    train_ds.set_epoch(epoch)
    val_ds.set_epoch(epoch)

    # Beta annealing for SCFM / dual-predict-μ mode
    use_beta = args.predict_mu or args.dual_predict_mu
    if use_beta and args.mu_anneal:
        if args.cyclical_anneal:
            beta_used = get_beta(epoch, args.epochs, args.struct_kl_weight,
                                 args.mu_warmup_epochs, cyclical=True,
                                 num_cycles=args.num_cycles,
                                 min_frac=args.min_beta_frac,
                                 schedule=args.anneal_schedule)
        else:
            beta_used = get_beta(epoch, args.epochs, args.struct_kl_weight,
                                 args.mu_warmup_epochs, cyclical=False)
    elif use_beta:
        beta_used = args.struct_kl_weight
    else:
        beta_used = None

    use_gmm_kl = args.use_gmm and (epoch >= args.kmeans_init_epoch)

    # K-means GMM initialization
    if (args.kmeans_init and args.use_gmm and args.unsupervised_gmm and
            epoch == args.kmeans_init_epoch and hasattr(model, 'gmm') and model.gmm is not None):
        train_ds_clean = FlowCropDataset(train_items, augment=False)
        init_loader = DataLoader(train_ds_clean, batch_size=min(128, args.batch_size * 2),
                                 shuffle=False, num_workers=min(4, args.num_workers),
                                 pin_memory=True, drop_last=False)
        kmeans_init_gmm(model, init_loader, device, args.gmm_components, seed=SEED)

    model.train()
    train_loss = 0.0
    train_steps = 0
    train_spatial = 0.0
    train_freq = 0.0
    train_kl = 0.0
    train_recon = 0.0
    train_neg = 0.0
    train_supcon = 0.0
    train_r_eps = 0.0
    t0 = time.time()

    pbar = tqdm(train_loader, desc=f"E{epoch+1:03d}", leave=False)
    for imgs, class_ids in pbar:
        imgs = imgs.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)

        delta_lambda = args.delta_fm_lambda if args.delta_fm else 0.0
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            if use_unified:
                loss, comp = compute_unified_loss(
                    model, imgs, class_labels=class_ids,
                    use_freq=args.freq_flow, use_struct=args.struct_flow,
                    freq_filter_D=args.freq_filter_D,
                    freq_loss_weight=args.freq_loss_weight,
                    kl_weight=args.struct_kl_weight,
                    recon_weight=args.struct_recon_weight,
                    delta_fm_lambda=delta_lambda,
                    supcon_weight=args.supcon_weight if args.supcon else 0.0,
                    supcon_temperature=args.supcon_temperature,
                    predict_mu=args.predict_mu,
                    beta=beta_used,
                    use_gmm=args.use_gmm,
                    use_gmm_kl=use_gmm_kl,
                    unsupervised_gmm=args.unsupervised_gmm,
                    dual_predict_mu=args.dual_predict_mu,
                    r_eps_weight=args.r_eps_weight,
                    detach_z=args.detach_z,
                )
            elif args.struct_flow:
                loss, comp = compute_struct_flow_loss(
                    model, imgs, class_labels=class_ids,
                    kl_weight=args.struct_kl_weight,
                    recon_weight=args.struct_recon_weight,
                    delta_fm_lambda=delta_lambda,
                    supcon_weight=args.supcon_weight if args.supcon else 0.0,
                    supcon_temperature=args.supcon_temperature,
                    predict_mu=args.predict_mu,
                    beta=beta_used,
                    use_gmm=args.use_gmm,
                    use_gmm_kl=use_gmm_kl,
                    unsupervised_gmm=args.unsupervised_gmm,
                    r_eps_weight=args.r_eps_weight,
                    detach_z=args.detach_z,
                )
            else:
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
        train_spatial += comp.get('spatial', comp.get('flow', 0.0))
        train_freq += comp.get('freq', 0.0)
        train_kl += comp.get('kl', 0.0)
        train_recon += comp.get('recon', 0.0)
        train_neg += comp.get('neg', 0.0)
        train_supcon += comp.get('supcon', 0.0)
        train_r_eps += comp.get('r_eps', 0.0)
        train_steps += 1
        pbar.set_postfix(loss=loss.item())

    train_loss /= max(1, train_steps)
    train_spatial /= max(1, train_steps)
    train_freq /= max(1, train_steps)
    train_kl /= max(1, train_steps)
    train_recon /= max(1, train_steps)
    train_neg /= max(1, train_steps)
    train_supcon /= max(1, train_steps)
    train_r_eps /= max(1, train_steps)
    epoch_time = time.time() - t0
    writer.add_scalar('train/loss', train_loss, epoch)
    if use_unified:
        writer.add_scalar('train/spatial', train_spatial, epoch)
        writer.add_scalar('train/freq', train_freq, epoch)
        writer.add_scalar('train/kl', train_kl, epoch)
        writer.add_scalar('train/recon', train_recon, epoch)
        writer.add_scalar('train/neg', train_neg, epoch)
        if args.supcon:
            writer.add_scalar('train/supcon', train_supcon, epoch)
    elif args.struct_flow:
        writer.add_scalar('train/flow', train_spatial, epoch)
        writer.add_scalar('train/kl', train_kl, epoch)
        writer.add_scalar('train/recon', train_recon, epoch)
        writer.add_scalar('train/neg', train_neg, epoch)
        writer.add_scalar('train/r_eps', train_r_eps, epoch)
        if args.supcon:
            writer.add_scalar('train/supcon', train_supcon, epoch)
    else:
        writer.add_scalar('train/spatial', train_spatial, epoch)
        writer.add_scalar('train/freq', train_freq, epoch)
        writer.add_scalar('train/neg', train_neg, epoch)

    model.eval()
    val_loss = 0.0
    val_steps = 0
    val_spatial = 0.0
    val_freq = 0.0
    val_neg = 0.0
    val_kl = 0.0
    val_recon = 0.0
    val_supcon = 0.0
    val_r_eps = 0.0
    with torch.no_grad():
        for imgs, class_ids in tqdm(val_loader, desc=f"E{epoch+1:03d} val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            delta_lambda = args.delta_fm_lambda if args.delta_fm else 0.0
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                if use_unified:
                    loss, comp = compute_unified_loss(
                        model, imgs, class_labels=class_ids,
                        use_freq=args.freq_flow, use_struct=args.struct_flow,
                        freq_filter_D=args.freq_filter_D,
                        freq_loss_weight=args.freq_loss_weight,
                        kl_weight=args.struct_kl_weight,
                        recon_weight=args.struct_recon_weight,
                        delta_fm_lambda=delta_lambda,
                        supcon_weight=args.supcon_weight if args.supcon else 0.0,
                        supcon_temperature=args.supcon_temperature,
                        predict_mu=args.predict_mu,
                        beta=beta_used,
                        use_gmm=args.use_gmm,
                        use_gmm_kl=use_gmm_kl,
                        unsupervised_gmm=args.unsupervised_gmm,
                        dual_predict_mu=args.dual_predict_mu,
                        r_eps_weight=args.r_eps_weight,
                        detach_z=args.detach_z,
                    )
                elif args.struct_flow:
                    loss, comp = compute_struct_flow_loss(
                        model, imgs, class_labels=class_ids,
                        kl_weight=args.struct_kl_weight,
                        recon_weight=args.struct_recon_weight,
                        delta_fm_lambda=delta_lambda,
                        supcon_weight=args.supcon_weight if args.supcon else 0.0,
                        supcon_temperature=args.supcon_temperature,
                        predict_mu=args.predict_mu,
                        beta=beta_used,
                        use_gmm=args.use_gmm,
                        use_gmm_kl=use_gmm_kl,
                        unsupervised_gmm=args.unsupervised_gmm,
                        r_eps_weight=args.r_eps_weight,
                        detach_z=args.detach_z,
                    )
                else:
                    loss, comp = compute_flow_loss(
                        model, imgs, class_labels=class_ids,
                        freq_flow=args.freq_flow,
                        freq_filter_D=args.freq_filter_D,
                        freq_loss_weight=args.freq_loss_weight,
                        delta_fm_lambda=delta_lambda,
                    )
            val_loss += loss.item()
            val_spatial += comp.get('spatial', comp.get('flow', 0.0))
            val_freq += comp.get('freq', 0.0)
            val_kl += comp.get('kl', 0.0)
            val_recon += comp.get('recon', 0.0)
            val_neg += comp.get('neg', 0.0)
            val_supcon += comp.get('supcon', 0.0)
            val_r_eps += comp.get('r_eps', 0.0)
            val_steps += 1
    val_loss /= max(1, val_steps)
    val_spatial /= max(1, val_steps)
    val_freq /= max(1, val_steps)
    val_neg /= max(1, val_steps)
    val_kl /= max(1, val_steps)
    val_recon /= max(1, val_steps)
    val_supcon /= max(1, val_steps)
    val_r_eps /= max(1, val_steps)
    writer.add_scalar('val/loss', val_loss, epoch)
    if use_unified:
        writer.add_scalar('val/spatial', val_spatial, epoch)
        writer.add_scalar('val/freq', val_freq, epoch)
        writer.add_scalar('val/kl', val_kl, epoch)
        writer.add_scalar('val/recon', val_recon, epoch)
        writer.add_scalar('val/neg', val_neg, epoch)
        if args.supcon:
            writer.add_scalar('val/supcon', val_supcon, epoch)
    elif args.struct_flow:
        writer.add_scalar('val/flow', val_spatial, epoch)
        writer.add_scalar('val/kl', val_kl, epoch)
        writer.add_scalar('val/recon', val_recon, epoch)
        writer.add_scalar('val/neg', val_neg, epoch)
        writer.add_scalar('val/r_eps', val_r_eps, epoch)
        if args.supcon:
            writer.add_scalar('val/supcon', val_supcon, epoch)

    lr_now = optimizer.param_groups[0]['lr']
    use_beta = args.predict_mu or args.dual_predict_mu
    if use_unified:
        supcon_str = f" supcon={val_supcon:.4f}" if args.supcon else ""
        beta_str = f" β={beta_used:.4f}" if use_beta else ""
        print(f"  E{epoch+1:03d} train={train_loss:.6f} val={val_loss:.6f} "
              f"(spat={val_spatial:.4f} freq={val_freq:.4f} kl={val_kl:.4f} "
              f"recon={val_recon:.4f} r_eps={val_r_eps:.4f} neg={val_neg:.4f}{supcon_str}{beta_str}) ({epoch_time:.0f}s)")
    elif args.struct_flow:
        supcon_str = f" supcon={val_supcon:.4f}" if args.supcon else ""
        beta_str = f" β={beta_used:.4f}" if use_beta else ""
        print(f"  E{epoch+1:03d} train={train_loss:.6f} val={val_loss:.6f} "
              f"(flow={val_spatial:.4f} kl={val_kl:.4f} recon={val_recon:.4f} r_eps={val_r_eps:.4f} neg={val_neg:.4f}{supcon_str}{beta_str}) ({epoch_time:.0f}s)")
    else:
        print(f"  E{epoch+1:03d} train={train_loss:.6f} val={val_loss:.6f} "
              f"(spat={val_spatial:.4f} freq={val_freq:.4f} neg={val_neg:.4f}) ({epoch_time:.0f}s)")

    with open(metrics_path, 'a', newline='') as f:
        w = csv.writer(f)
        if use_unified:
            row = [epoch+1, f'{train_loss:.6f}', f'{val_loss:.6f}',
                   f'{train_spatial:.4f}', f'{val_spatial:.4f}',
                   f'{train_freq:.4f}', f'{val_freq:.4f}',
                   f'{train_kl:.4f}', f'{val_kl:.4f}',
                   f'{train_recon:.4f}', f'{val_recon:.4f}',
                   f'{train_r_eps:.4f}', f'{val_r_eps:.4f}',
                   f'{train_neg:.4f}', f'{val_neg:.4f}']
            if args.supcon:
                row += [f'{train_supcon:.4f}', f'{val_supcon:.4f}']
            if args.predict_mu or args.dual_predict_mu:
                row += [f'{beta_used:.4e}']
            row += [f'{lr_now:.2e}', f'{epoch_time:.0f}']
        elif args.struct_flow:
            row = [epoch+1, f'{train_loss:.6f}', f'{val_loss:.6f}',
                   f'{train_spatial:.4f}', f'{val_spatial:.4f}',
                   f'{train_kl:.4f}', f'{val_kl:.4f}',
                   f'{train_recon:.4f}', f'{val_recon:.4f}',
                   f'{train_r_eps:.4f}', f'{val_r_eps:.4f}',
                   f'{train_neg:.4f}', f'{val_neg:.4f}']
            if args.supcon:
                row += [f'{train_supcon:.4f}', f'{val_supcon:.4f}']
            if args.predict_mu or args.dual_predict_mu:
                row += [f'{beta_used:.4e}']
            row += [f'{lr_now:.2e}', f'{epoch_time:.0f}']
        else:
            row = [epoch+1, f'{train_loss:.6f}', f'{val_loss:.6f}',
                   f'{train_spatial:.4f}', f'{val_spatial:.4f}',
                   f'{train_freq:.4f}', f'{val_freq:.4f}',
                   f'{train_neg:.4f}', f'{val_neg:.4f}',
                   f'{lr_now:.2e}', f'{epoch_time:.0f}']
        w.writerow(row)

    # GMM health diagnostics (SCFM mode)
    gmm_diag = None
    if use_beta and args.use_gmm and args.unsupervised_gmm and args.struct_flow \
       and epoch >= args.kmeans_init_epoch:
        with torch.no_grad():
            mu_z_all = []
            for imgs, class_ids in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                class_ids = class_ids.to(device, non_blocking=True)
                t_enc = torch.full((imgs.shape[0],), 1.0, device=device)
                mu_z, _, _ = model.encode(imgs, t_enc, class_ids)
                mu_z_all.append(mu_z)
            if mu_z_all:
                mu_z_cat = torch.cat(mu_z_all, dim=0)
                if hasattr(model, 'gmm') and model.gmm is not None:
                    gmm_diag = model.gmm.diagnostics(mu_z_cat)
                    for k, v in gmm_diag.items():
                        writer.add_scalar(f'gmm/{k}', v, epoch)

    # Append GMM diagnostics to separate CSV
    if gmm_diag is not None:
        gmm_csv = os.path.join(OUTPUT_DIR, 'gmm_diagnostics.csv')
        gmm_cols = ['epoch'] + list(gmm_diag.keys())
        write_header = not os.path.exists(gmm_csv)
        with open(gmm_csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=gmm_cols)
            if write_header:
                w.writeheader()
            row = {'epoch': epoch + 1, **gmm_diag}
            w.writerow(row)

    # Generate samples
    if epoch % 1 == 0:
        model.eval()
        with torch.no_grad():
            if args.predict_mu and not args.dual_predict_mu:
                # Unconditional: 5 random samples (no class labels)
                samp = sample_combined(model, 5, num_steps=args.num_steps,
                                       class_labels=None, device=device,
                                       use_freq=args.freq_flow,
                                       use_struct=args.struct_flow,
                                       predict_mu=args.predict_mu,
                                       dual_predict_mu=args.dual_predict_mu)
                fig, axes = plt.subplots(1, 5, figsize=(5 * 0.9, 2.5))
                for i in range(5):
                    img = samp[i].cpu()
                    img_01 = (img * 0.5 + 0.5).clamp(0, 1)
                    axes[i].imshow(img_01.squeeze(), cmap='gray', vmin=0, vmax=1)
                    axes[i].axis('off')
                plt.suptitle(f'Epoch {epoch+1}: Unconditional samples', fontsize=7, y=0.98)
                plt.tight_layout()
                fig.savefig(os.path.join(OUTPUT_DIR, f'samples_{epoch+1:03d}.png'),
                           dpi=200, bbox_inches='tight')
                plt.close(fig)
            else:
                # Class-conditional: 1 sample per class in 2-row table
                all_samples = []
                for ci in vis_classes:
                    cid = torch.tensor([ci], device=device)
                    if use_unified:
                        samp = sample_combined(model, 1, num_steps=args.num_steps,
                                               class_labels=cid, device=device,
                                               use_freq=args.freq_flow,
                                               use_struct=args.struct_flow,
                                               predict_mu=args.predict_mu,
                                               dual_predict_mu=args.dual_predict_mu)
                    elif args.struct_flow:
                        samp = sample_struct(model, 1, num_steps=args.num_steps,
                                             class_labels=cid, device=device,
                                             predict_mu=args.predict_mu)
                    else:
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
