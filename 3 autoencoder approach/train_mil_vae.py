#!/usr/bin/env python3
import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

"""
Joint MIL + VAE training:
  Default: joint MIL+VAE (single stage, combined loss)
  --two_stage: two-stage (MIL then VAE separately)
  --stage2_only: VAE only (from checkpoint)
"""

import argparse
import time
import json
import csv
import re
import glob
import random
from datetime import datetime
from functools import partial
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from mil_model import MultiCropDataset, extract_well_from_filename, get_gene_from_path
from supcon_loss import SupConLoss
from vae_model import MILVAE

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch._inductor.config as inductor_config
inductor_config.max_autotune_gemm = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def worker_init_fn(worker_id, seed=42):
    import random, numpy as np
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def compute_robust_auc(labels, probs, num_classes):
    import warnings as _w
    _w.filterwarnings('ignore')
    labels_np = np.array(labels)
    probs_np = np.array(probs)
    unique = np.unique(labels_np)
    if len(unique) < 2:
        return float('nan')
    if len(unique) == 2:
        pos_idx = unique[1]
        try:
            return float(roc_auc_score(labels_np, probs_np[:, pos_idx]))
        except Exception:
            return float('nan')
    labels_bin = label_binarize(labels_np, classes=unique)
    probs_f = probs_np[:, unique]
    try:
        return float(roc_auc_score(labels_bin, probs_f, average='macro'))
    except Exception:
        try:
            return float(roc_auc_score(labels_bin, probs_f, average='weighted'))
        except Exception:
            return float('nan')


def weighted_focal_loss(logits, targets, weights, alpha=0.25, gamma=2.0, label_smoothing=0.0):
    ce = F.cross_entropy(logits, targets, reduction='none', label_smoothing=label_smoothing)
    pt = torch.exp(-ce)
    focal = alpha * (1 - pt) ** gamma * ce
    return (focal * weights).mean()


def _gaussian_window(size, sigma):
    gauss = torch.arange(size, dtype=torch.float32) - size // 2
    gauss = torch.exp(-gauss ** 2 / (2 * sigma ** 2))
    return gauss / gauss.sum()


def compute_ssim(img1, img2, window_size=11, sigma=1.5, max_val=2.0):
    window_1d = _gaussian_window(window_size, sigma)
    window_2d = window_1d.outer(window_1d)
    window = window_2d.view(1, 1, window_size, window_size).to(img1.device)
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    pad = window_size // 2
    mu1 = F.conv2d(F.pad(img1, (pad, pad, pad, pad), mode='reflect'), window)
    mu2 = F.conv2d(F.pad(img2, (pad, pad, pad, pad), mode='reflect'), window)
    sigma1_sq = F.conv2d(F.pad(img1 ** 2, (pad, pad, pad, pad), mode='reflect'), window) - mu1 ** 2
    sigma2_sq = F.conv2d(F.pad(img2 ** 2, (pad, pad, pad, pad), mode='reflect'), window) - mu2 ** 2
    sigma12 = F.conv2d(F.pad(img1 * img2, (pad, pad, pad, pad), mode='reflect'), window) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def compute_psnr(img1, img2, max_val=2.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(max_val / torch.sqrt(mse))).item()


def main():
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--test_plate', type=str, default='Plate_6')
    parser.add_argument('--run_all_folds', action='store_true')
    parser.add_argument('--data_mode', type=str, default='mutant', choices=['drug', 'mutant', 'both'])
    parser.add_argument('--drug_no_concentration', action='store_true')
    parser.add_argument('--data_root', type=str, default=None)
    # Model
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='efficientnet_b0')
    parser.add_argument('--pooling', type=str, default='attention')
    parser.add_argument('--pretrained', type=str, default='imagenet')
    parser.add_argument('--neighborhood', type=int, default=3)
    parser.add_argument('--grid_size', type=int, default=12)
    parser.add_argument('--extraction_mode', type=str, default='neighborhood')
    # VAE
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--vae_beta', type=float, default=0.1)
    parser.add_argument('--vae_epochs', type=int, default=100)
    parser.add_argument('--vae_lr', type=float, default=5e-5)
    parser.add_argument('--vae_batch_size', type=int, default=None,
                        help='Batch size for VAE stage (default: 4x batch_size for frozen backbone)')
    parser.add_argument('--stage2_only', action='store_true', help='Skip MIL, load checkpoint and train VAE only')
    parser.add_argument('--mil_checkpoint', type=str, default=None, help='Load MIL checkpoint for stage 2')
    parser.add_argument('--no_pixel_decoder', action='store_true')
    parser.add_argument('--no_feature_decoder', action='store_true')
    parser.add_argument('--vae_checkpoint_every', type=int, default=1,
                        help='Save VAE checkpoint every N epochs (default: 1)')
    parser.add_argument('--mil_checkpoint_every', type=int, default=1,
                        help='Save MIL checkpoint every N epochs (0 = disable, default: 1)')
    # SC-MIL
    parser.add_argument('--sc_mil_weight', type=float, default=0.3)
    parser.add_argument('--sc_mil_temp', type=float, default=0.07)
    parser.add_argument('--contrastive_level', type=str, default='both')
    parser.add_argument('--instance_weight', type=float, default=0.5)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=16)
    # Joint training
    parser.add_argument('--two_stage', action='store_true',
                        help='Use two-stage training (MIL then VAE) instead of joint (default)')
    parser.add_argument('--vae_loss_weight', type=float, default=0.1,
                        help='Weight for VAE loss contribution in joint training')
    # Beta annealing
    parser.add_argument('--beta_anneal', action='store_true',
                        help='Linearly anneal beta from beta_start to vae_beta over beta_warmup_epochs')
    parser.add_argument('--beta_start', type=float, default=0.0,
                        help='Starting beta for annealing (default: 0.0)')
    parser.add_argument('--beta_warmup_epochs', type=int, default=50,
                        help='Epochs over which to anneal beta (default: 50)')
    # Reconstruction metrics (monitoring only, not in loss)
    parser.add_argument('--compute_recon_metrics', action='store_true',
                        help='Compute SSIM, LPIPS, PSNR during validation (requires lpips package)')
    args = parser.parse_args()

    if args.warmup_epochs is None:
        args.warmup_epochs = int(args.epochs * 0.05)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    if args.data_root:
        BASE_DIR = args.data_root
    elif args.data_mode == 'drug':
        BASE_DIR = os.path.join(PROJECT_ROOT, 'Drugs_Data')
    else:
        BASE_DIR = os.path.join(PROJECT_ROOT, 'Mutants_Data')

    IC50_MAPPING_PATH = os.path.join(PROJECT_ROOT, 'final_mutant_model', 'plate_well_ic50_mapping.json')
    MUTANT_MAPPING_PATH = os.path.join(PROJECT_ROOT, 'final_mutant_model', 'plate_well_id_path.json')

    with open(IC50_MAPPING_PATH, 'r') as f:
        ic50_data = json.load(f)
    with open(MUTANT_MAPPING_PATH, 'r') as f:
        mutant_data = json.load(f)

    plate_maps = {}
    for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        plate_maps[plate] = {}
        if args.data_mode in ('drug', 'both') and plate in ic50_data:
            for well, info in ic50_data[plate].items():
                antibiotic = info.get('antibiotic', '')
                ic50_mult = info.get('ic50_multiple', '')
                if antibiotic and ic50_mult:
                    if args.drug_no_concentration:
                        drug_class = antibiotic.replace(' ', '_')
                    else:
                        if ic50_mult == 'control':
                            drug_class = 'control'
                        else:
                            ic50_str = ic50_mult if 'x' in ic50_mult else f"{ic50_mult}x"
                            drug_class = f"{antibiotic.replace(' ', '_')}_{ic50_str}"
                    plate_maps[plate][f"drug_{well}"] = drug_class
        if args.data_mode in ('mutant', 'both') and plate in mutant_data:
            for row, cols in mutant_data[plate].items():
                for col, info in cols.items():
                    if 'id' in info:
                        well = f"{row}{int(col):02d}"
                        plate_maps[plate][f"mutant_{well}"] = info['id']

    all_plates = ['Plate_1', 'Plate_2', 'Plate_3', 'Plate_4', 'Plate_5', 'Plate_6']

    def get_image_paths(plate):
        plate_key = f"P{plate.split('_')[-1]}"
        search_dirs = []
        if args.data_mode in ('drug', 'both'):
            drug_base = os.path.join(PROJECT_ROOT, 'Drugs_Data')
            search_dirs.append((os.path.join(drug_base, plate_key), 'drug'))
        if args.data_mode in ('mutant', 'both'):
            mutant_base = os.path.join(PROJECT_ROOT, 'Mutants_Data')
            search_dirs.append((os.path.join(mutant_base, plate_key), 'mutant'))
        valid = []
        for plate_dir, source_type in search_dirs:
            if not os.path.exists(plate_dir):
                continue
            paths = []
            for pattern in ['*.tif', '*.tiff', '*.png']:
                paths.extend(glob.glob(os.path.join(plate_dir, '**', pattern), recursive=True))
            well_prefix = f"{source_type}_"
            for path in paths:
                well = extract_well_from_filename(os.path.basename(path))
                composite = f"{well_prefix}{well}"
                if composite and composite in plate_maps.get(plate_key, {}):
                    valid.append(path)
        return valid

    def train_single_fold(test_plate):
        data_mode_folder = args.data_mode
        if args.data_mode == 'drug' and args.drug_no_concentration:
            data_mode_folder = 'drug_noconcentration'

        OUTPUT_DIR = os.path.join(SCRIPT_DIR, f'mil_vae_{data_mode_folder}', f'fold_{test_plate}')
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Fold: test_plate={test_plate}, mode={args.data_mode}")
        print(f"{'='*60}")

        test_norm = f"Plate_{test_plate[-1]}" if 'P' in test_plate.upper() and test_plate[-1].isdigit() else test_plate
        train_val_plates = [p for p in all_plates if p != test_norm]
        test_num = int(test_norm.split('_')[1])
        val_num = (test_num - 2) % 6 + 1
        val_plate = f"Plate_{val_num}"
        val_plates = [val_plate] if val_plate in train_val_plates else [train_val_plates[0]]
        train_plates = [p for p in train_val_plates if p not in val_plates][:4]

        print(f"Train: {train_plates}, Val: {val_plates}, Test: {[test_norm]}")

        plate_key_map = {f'Plate_{i}': f'P{i}' for i in range(1, 7)}

        all_classes_set = set()
        for pm in plate_maps.values():
            for lbl in pm.values():
                if lbl:
                    all_classes_set.add(lbl)
        all_classes = sorted(all_classes_set)
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        num_classes = len(all_classes)
        drug_classes = set()
        mutant_classes = set()
        for pm in plate_maps.values():
            for k, v in pm.items():
                if v:
                    if k.startswith('drug_'):
                        drug_classes.add(v)
                    elif k.startswith('mutant_'):
                        mutant_classes.add(v)
        print(f"Classes: {num_classes} total ({len(drug_classes)} drug, {len(mutant_classes)} mutant)")

        def extract_label(path):
            path_lower = path.lower()
            for pn in range(1, 7):
                if f'/p{pn}/' in path_lower:
                    plate_key = f'P{pn}'
                    break
            else:
                return None
            well = extract_well_from_filename(os.path.basename(path))
            if well is None:
                return None
            if '/mutants_data/' in path_lower or '\\mutants_data\\' in path_lower:
                prefix = 'mutant_'
            else:
                prefix = 'drug_'
            cw = f"{prefix}{well}"
            if plate_key in plate_maps and cw in plate_maps[plate_key]:
                return plate_maps[plate_key][cw]
            return None

        train_paths, train_labels = [], []
        val_paths, val_labels = [], []
        test_paths, test_labels = [], []

        for plate in train_plates:
            for p in get_image_paths(plate):
                lbl = extract_label(p)
                if lbl in class_to_idx:
                    train_paths.append(p)
                    train_labels.append(class_to_idx[lbl])
        for plate in val_plates:
            for p in get_image_paths(plate):
                lbl = extract_label(p)
                if lbl in class_to_idx:
                    val_paths.append(p)
                    val_labels.append(class_to_idx[lbl])
        for plate in [test_norm]:
            for p in get_image_paths(plate):
                lbl = extract_label(p)
                if lbl in class_to_idx:
                    test_paths.append(p)
                    test_labels.append(class_to_idx[lbl])

        print(f"Train: {len(train_paths)} imgs, Val: {len(val_paths)}, Test: {len(test_paths)}")

        class_counts = Counter(train_labels)
        total = len(train_labels)
        class_weights = torch.tensor(
            [total / (num_classes * max(class_counts[i], 1)) for i in range(num_classes)],
            device=device
        )
        class_weights = class_weights / class_weights.sum() * num_classes

        use_pixel_decoder = not args.no_pixel_decoder
        use_feature_decoder = not args.no_feature_decoder

        model = MILVAE(
            num_classes=num_classes,
            latent_dim=args.latent_dim,
            beta=args.vae_beta,
            num_heads=args.num_heads,
            dropout=args.dropout,
            use_contrastive=True,
            num_channels=args.num_channels,
            pretrained=args.pretrained,
            backbone=args.backbone,
            pooling=args.pooling,
            img_size=224,
            feature_decoder=use_feature_decoder,
            pixel_decoder=use_pixel_decoder,
        ).to(device)

        NUM_WORKERS = 0 if sys.platform.startswith('win') else args.num_workers

        train_dataset = MultiCropDataset(
            train_paths, train_labels, None,
            neighborhood=args.neighborhood, grid_size=args.grid_size,
            augment=True, seed=SEED, num_channels=args.num_channels,
            extraction_mode=args.extraction_mode
        )
        val_dataset = MultiCropDataset(
            val_paths, val_labels, None,
            neighborhood=args.neighborhood, grid_size=args.grid_size,
            augment=False, seed=SEED, num_channels=args.num_channels,
            extraction_mode=args.extraction_mode
        )
        test_dataset = MultiCropDataset(
            test_paths, test_labels, None,
            neighborhood=args.neighborhood, grid_size=args.grid_size,
            augment=False, seed=SEED, num_channels=args.num_channels,
            extraction_mode=args.extraction_mode
        )

        train_dataset.set_epoch(0)
        val_dataset.set_epoch(0)
        test_dataset.set_epoch(0)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            worker_init_fn=partial(worker_init_fn, seed=SEED), drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            worker_init_fn=partial(worker_init_fn, seed=SEED)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
            worker_init_fn=partial(worker_init_fn, seed=SEED)
        )

        use_amp = torch.cuda.is_available()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_writer = SummaryWriter(log_dir=OUTPUT_DIR)

        # Set up LPIPS if requested
        lpips_fn = None
        if args.compute_recon_metrics:
            try:
                import lpips as _lpips
                lpips_fn = _lpips.LPIPS(net='alex').to(device)
                print("LPIPS loaded for reconstruction metrics")
            except ImportError:
                print("WARNING: lpips not installed, LPIPS metric disabled. Install with: pip install lpips")

        # =====================================================================
        # Determine training path
        # =====================================================================
        is_joint = not args.two_stage and not args.stage2_only

        best_val_auc = 0.0
        best_vae_loss = float('inf')

        # =====================================================================
        # JOINT TRAINING (default): MIL + VAE optimized together
        # =====================================================================
        if is_joint:
            print(f"\n{'='*60}")
            print(f"JOINT TRAINING: MIL + VAE (latent_dim={args.latent_dim}, "
                  f"vae_beta={args.vae_beta}, vae_loss_weight={args.vae_loss_weight})")
            print(f"{'='*60}")

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                fused=torch.cuda.is_available()
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            if args.warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0,
                    total_iters=args.warmup_epochs
                )
                scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup, scheduler])

            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

            csv_joint = os.path.join(OUTPUT_DIR, f'training_joint_{timestamp}.csv')
            cols = ['epoch', 'ce', 'sc', 'kl', 'bag_recon', 'img_recon', 'train_acc',
                    'val_ce', 'val_acc', 'val_auc', 'val_kl', 'val_bag_recon', 'val_img_recon',
                    'beta', 'lr']
            if args.compute_recon_metrics:
                cols += ['val_ssim', 'val_psnr']
            with open(csv_joint, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(cols)

            best_joint_val_auc = 0.0
            best_joint_val_acc = 0.0
            best_joint_val_loss = float('inf')

            for epoch in range(args.epochs):
                epoch_start = time.time()
                train_dataset.set_epoch(epoch)
                model.train()

                # Compute beta with optional annealing
                if args.beta_anneal:
                    beta = args.beta_start + (args.vae_beta - args.beta_start) * min(epoch / args.beta_warmup_epochs, 1)
                else:
                    beta = args.vae_beta

                run_ce, run_sc, run_kl, run_bag, run_img = 0.0, 0.0, 0.0, 0.0, 0.0
                correct, total_batches = 0, 0

                for images, labels in tqdm(train_loader, desc=f'Joint {epoch}', leave=False):
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        results = model(images, return_attention=True)

                        # MIL losses
                        ce_loss = F.cross_entropy(results['logits'], labels, weight=class_weights)

                        crop_emb = results['crop_embeddings']
                        bag = results['bag']
                        num_crops = crop_emb.shape[1]

                        crop_flat = crop_emb.view(-1, crop_emb.shape[-1]).unsqueeze(1)
                        crop_flat = F.normalize(crop_flat, p=2, dim=-1)
                        inst_labels = labels.repeat_interleave(num_crops)
                        inst_sc = SupConLoss(temperature=max(args.sc_mil_temp, 0.1), contrast_mode='one')(
                            crop_flat, inst_labels
                        )

                        bag_emb = F.normalize(bag, p=2, dim=-1).unsqueeze(1)
                        bag_sc = SupConLoss(temperature=args.sc_mil_temp)(bag_emb, labels)

                        w = args.instance_weight
                        total_sc = w * inst_sc + (1 - w) * bag_sc

                        # VAE losses
                        kl = -0.5 * torch.mean(1 + results['logvar'] - results['mu'].pow(2) - results['logvar'].exp())

                        bag_recon_loss = torch.tensor(0.0, device=device)
                        if model.feature_decoder is not None:
                            bag_recon_loss = F.mse_loss(results['bag_recon'], bag)

                        img_recon_loss = torch.tensor(0.0, device=device)
                        if model.pixel_decoder is not None:
                            center_idx = images.shape[1] // 2
                            img_recon_loss = F.mse_loss(results['img_recon'], images[:, center_idx])

                        vae_loss = beta * kl + bag_recon_loss + img_recon_loss

                        # Combined loss: MIL + weighted VAE
                        loss = (1 - args.sc_mil_weight) * ce_loss + args.sc_mil_weight * total_sc + args.vae_loss_weight * vae_loss

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    run_ce += ce_loss.item()
                    run_sc += total_sc.item()
                    run_kl += kl.item()
                    run_bag += bag_recon_loss.item()
                    run_img += img_recon_loss.item()
                    _, pred = results['logits'].max(1)
                    correct += pred.eq(labels).sum().item()
                    total_batches += labels.size(0)

                scheduler.step()
                train_acc = 100. * correct / max(total_batches, 1)

                # Validation
                model.eval()
                val_ce, v_correct, v_total = 0.0, 0, 0
                v_kl, v_bag, v_img = 0.0, 0.0, 0.0
                all_vp, all_vpr, all_vl = [], [], []
                v_n = 0
                val_ssim_accum, val_psnr_accum = 0.0, 0.0

                with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                    for images, labels in tqdm(val_loader, desc='Validating', leave=False):
                        images, labels = images.to(device), labels.to(device)
                        results = model(images)
                        probs = F.softmax(results['logits'], dim=1)
                        _, pred = results['logits'].max(1)
                        all_vp.extend(pred.cpu().numpy())
                        all_vpr.extend(probs.cpu().numpy())
                        all_vl.extend(labels.cpu().numpy())
                        vloss = F.cross_entropy(results['logits'], labels, weight=class_weights)
                        val_ce += vloss.item()
                        v_correct += pred.eq(labels).sum().item()
                        v_total += labels.size(0)

                        # VAE metrics
                        kl = -0.5 * torch.mean(1 + results['logvar'] - results['mu'].pow(2) - results['logvar'].exp())
                        v_kl += kl.item()

                        if model.feature_decoder is not None:
                            bag_recon = model.feature_decoder(results['z'])
                            v_bag += F.mse_loss(bag_recon, results['bag']).item()

                        if model.pixel_decoder is not None:
                            center_idx = images.shape[1] // 2
                            img_recon = model.pixel_decoder(results['z'])
                            v_img_loss = F.mse_loss(img_recon, images[:, center_idx])
                            v_img += v_img_loss.item()

                            if args.compute_recon_metrics:
                                val_ssim_accum += compute_ssim(img_recon, images[:, center_idx])
                                val_psnr_accum += compute_psnr(img_recon, images[:, center_idx])

                        v_n += 1

                val_acc = 100. * v_correct / max(v_total, 1)
                val_auc = compute_robust_auc(all_vl, all_vpr, num_classes)
                avg_vce = val_ce / max(len(val_loader), 1)
                v_kl /= max(v_n, 1)
                v_bag /= max(v_n, 1)
                v_img /= max(v_n, 1)

                log_str = (
                    f"Joint Epoch {epoch}: CE={run_ce/max(len(train_loader),1):.4f} "
                    f"SC={run_sc/max(len(train_loader),1):.4f} "
                    f"KL={run_kl/max(len(train_loader),1):.4f} "
                    f"TrainAcc={train_acc:.2f}% "
                    f"ValAcc={val_acc:.2f}% ValAUC={val_auc:.4f} "
                    f"β={beta:.4f} "
                    f"Time={time.time()-epoch_start:.1f}s"
                )
                print(log_str)

                row = [epoch,
                       f"{run_ce/max(len(train_loader),1):.4f}",
                       f"{run_sc/max(len(train_loader),1):.4f}",
                       f"{run_kl/max(len(train_loader),1):.4f}",
                       f"{run_bag/max(len(train_loader),1):.4f}",
                       f"{run_img/max(len(train_loader),1):.4f}",
                       f"{train_acc:.2f}",
                       f"{avg_vce:.4f}", f"{val_acc:.2f}", f"{val_auc:.4f}",
                       f"{v_kl:.4f}", f"{v_bag:.4f}", f"{v_img:.4f}",
                       f"{beta:.4f}",
                       f"{optimizer.param_groups[0]['lr']:.6f}"]
                if args.compute_recon_metrics and v_n > 0:
                    row += [f"{val_ssim_accum / v_n:.4f}", f"{val_psnr_accum / v_n:.4f}"]
                with open(csv_joint, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(row)

                tb_writer.add_scalars('Joint_CE', {'train': run_ce/max(len(train_loader),1), 'val': avg_vce}, epoch)
                tb_writer.add_scalars('Joint_Acc', {'train': train_acc, 'val': val_acc}, epoch)
                tb_writer.add_scalar('Joint_Beta', beta, epoch)

                # Save by MIL metrics (matching two-stage convention)
                if not np.isnan(val_auc) and val_auc > best_joint_val_auc:
                    best_joint_val_auc = val_auc
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_mil_vae.pth'))

                if val_acc > best_joint_val_acc:
                    best_joint_val_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_mil_vae_acc.pth'))

                if avg_vce < best_joint_val_loss:
                    best_joint_val_loss = avg_vce
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_mil_vae_loss.pth'))

                if args.mil_checkpoint_every > 0 and (epoch + 1) % args.mil_checkpoint_every == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_auc': val_auc,
                        'val_acc': val_acc,
                        'val_loss': avg_vce,
                    }, os.path.join(OUTPUT_DIR, 'checkpoint_mil_latest.pth'))

            best_val_auc = best_joint_val_auc
            best_vae_loss = best_joint_val_loss
            print(f"Joint training complete! Best val AUC: {best_val_auc:.4f}")

        # =====================================================================
        # TWO-STAGE: Stage 1 - MIL only, Stage 2 - VAE only
        # =====================================================================
        elif args.two_stage:
            # Stage 1: MIL
            print(f"\n{'='*60}")
            print("STAGE 1: MIL Training (SC-MIL: classification + contrastive)")
            print(f"{'='*60}")

            sc_mil_optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                fused=torch.cuda.is_available()
            )
            sc_mil_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                sc_mil_optimizer, T_max=args.epochs
            )
            if args.warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    sc_mil_optimizer, start_factor=0.1, end_factor=1.0,
                    total_iters=args.warmup_epochs
                )
                sc_mil_scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup, sc_mil_scheduler])

            scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
            csv_sc = os.path.join(OUTPUT_DIR, f'training_mil_{timestamp}.csv')
            with open(csv_sc, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['epoch', 'train_ce', 'train_sc', 'train_acc', 'val_ce', 'val_acc', 'val_auc', 'lr'])

            best_val_auc = 0.0
            best_val_acc = 0.0
            best_val_loss = float('inf')

            for epoch in range(args.epochs):
                epoch_start = time.time()
                train_dataset.set_epoch(epoch)
                model.train()
                run_ce, run_sc, correct, total = 0.0, 0.0, 0, 0

                for images, labels in tqdm(train_loader, desc=f'MIL {epoch}', leave=False):
                    images, labels = images.to(device), labels.to(device)
                    sc_mil_optimizer.zero_grad()

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        results = model(images, return_attention=True)

                        ce_loss = F.cross_entropy(results['logits'], labels, weight=class_weights)

                        crop_emb = results['crop_embeddings']
                        bag = results['bag']
                        num_crops = crop_emb.shape[1]

                        crop_flat = crop_emb.view(-1, crop_emb.shape[-1]).unsqueeze(1)
                        crop_flat = F.normalize(crop_flat, p=2, dim=-1)
                        inst_labels = labels.repeat_interleave(num_crops)
                        inst_sc = SupConLoss(temperature=max(args.sc_mil_temp, 0.1), contrast_mode='one')(
                            crop_flat, inst_labels
                        )

                        bag_emb = F.normalize(bag, p=2, dim=-1).unsqueeze(1)
                        bag_sc = SupConLoss(temperature=args.sc_mil_temp)(bag_emb, labels)

                        w = args.instance_weight
                        total_sc = w * inst_sc + (1 - w) * bag_sc
                        loss = (1 - args.sc_mil_weight) * ce_loss + args.sc_mil_weight * total_sc

                    scaler.scale(loss).backward()
                    scaler.step(sc_mil_optimizer)
                    scaler.update()

                    run_ce += ce_loss.item()
                    run_sc += total_sc.item()
                    _, pred = results['logits'].max(1)
                    total += labels.size(0)
                    correct += pred.eq(labels).sum().item()

                sc_mil_scheduler.step()

                train_acc = 100. * correct / total
                avg_ce = run_ce / len(train_loader)
                avg_sc = run_sc / len(train_loader)

                model.eval()
                val_ce, v_correct, v_total = 0.0, 0, 0
                all_vp, all_vpr, all_vl = [], [], []

                with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                    for images, labels in tqdm(val_loader, desc='Validating', leave=False):
                        images, labels = images.to(device), labels.to(device)
                        results = model(images)
                        probs = F.softmax(results['logits'], dim=1)
                        _, pred = results['logits'].max(1)
                        all_vp.extend(pred.cpu().numpy())
                        all_vpr.extend(probs.cpu().numpy())
                        all_vl.extend(labels.cpu().numpy())
                        vloss = F.cross_entropy(results['logits'], labels, weight=class_weights)
                        val_ce += vloss.item()
                        v_correct += pred.eq(labels).sum().item()
                        v_total += labels.size(0)

                val_acc = 100. * v_correct / v_total
                val_auc = compute_robust_auc(all_vl, all_vpr, num_classes)
                avg_vce = val_ce / len(val_loader)

                print(
                    f"MIL Epoch {epoch}: CE={avg_ce:.4f} SC={avg_sc:.4f} "
                    f"TrainAcc={train_acc:.2f}% ValAcc={val_acc:.2f}% ValAUC={val_auc:.4f} "
                    f"Time={time.time()-epoch_start:.1f}s"
                )

                with open(csv_sc, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([epoch, f"{avg_ce:.4f}", f"{avg_sc:.4f}", f"{train_acc:.2f}",
                               f"{avg_vce:.4f}", f"{val_acc:.2f}", f"{val_auc:.4f}",
                               sc_mil_optimizer.param_groups[0]['lr']])

                tb_writer.add_scalars('MIL_Loss', {'train': avg_ce, 'val': avg_vce}, epoch)
                tb_writer.add_scalars('MIL_Acc', {'train': train_acc, 'val': val_acc}, epoch)

                if not np.isnan(val_auc) and val_auc > best_val_auc:
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_mil.pth'))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_mil_acc.pth'))

                if avg_vce < best_val_loss:
                    best_val_loss = avg_vce
                    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_mil_loss.pth'))

                if args.mil_checkpoint_every > 0 and (epoch + 1) % args.mil_checkpoint_every == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': sc_mil_optimizer.state_dict(),
                        'val_auc': val_auc,
                        'val_acc': val_acc,
                        'val_loss': avg_vce,
                    }, os.path.join(OUTPUT_DIR, 'checkpoint_mil_latest.pth'))

            print("Stage 1 complete!")

            mil_checkpoint_path = os.path.join(OUTPUT_DIR, 'best_mil.pth')

            # =================================================================
            # TWO-STAGE: Stage 2 - VAE (freeze backbone)
            # =================================================================
            print(f"\n{'='*60}")
            print(f"STAGE 2: VAE Training (latent_dim={args.latent_dim}, beta={args.vae_beta})")
            print(f"{'='*60}")

            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
            model.encoder.eval()

            vae_params = []
            vae_params.extend(model.vae_mu.parameters())
            vae_params.extend(model.vae_logvar.parameters())
            if model.feature_decoder is not None:
                vae_params.extend(model.feature_decoder.parameters())
            if model.pixel_decoder is not None:
                vae_params.extend(model.pixel_decoder.parameters())

            vae_optimizer = torch.optim.AdamW(vae_params, lr=args.vae_lr, weight_decay=1e-5)
            vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vae_optimizer, T_max=args.vae_epochs)

            csv_vae = os.path.join(OUTPUT_DIR, f'training_vae_{timestamp}.csv')
            with open(csv_vae, 'w', newline='') as f:
                w = csv.writer(f)
                cols = ['epoch', 'vae_loss', 'kl_loss', 'bag_recon', 'val_vae', 'val_kl', 'val_bag_recon']
                if use_pixel_decoder:
                    cols += ['img_recon', 'val_img_recon']
                w.writerow(cols)

            vae_bs = args.vae_batch_size if args.vae_batch_size is not None else args.batch_size * 4
            vae_train_loader = DataLoader(
                train_dataset, batch_size=vae_bs, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=True,
                persistent_workers=NUM_WORKERS > 0,
                worker_init_fn=partial(worker_init_fn, seed=SEED), drop_last=True
            )
            vae_val_loader = DataLoader(
                val_dataset, batch_size=vae_bs, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                persistent_workers=NUM_WORKERS > 0,
                worker_init_fn=partial(worker_init_fn, seed=SEED)
            )
            print(f"VAE batch size: {vae_bs} ({vae_bs}x gradient steps)")

            best_vae_loss = float('inf')

            for epoch in range(args.vae_epochs):
                epoch_start = time.time()
                train_dataset.set_epoch(epoch)
                model.train()
                model.encoder.eval()

                if args.beta_anneal:
                    beta = args.beta_start + (args.vae_beta - args.beta_start) * min(epoch / args.beta_warmup_epochs, 1)
                else:
                    beta = args.vae_beta

                run_vae, run_kl, run_bag, run_img = 0.0, 0.0, 0.0, 0.0
                n_batch = 0

                for images, _ in tqdm(vae_train_loader, desc=f'VAE {epoch}', leave=False):
                    images = images.to(device)
                    vae_optimizer.zero_grad()

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        with torch.no_grad():
                            bag = model.encode_bag(images)
                        mu = model.vae_mu(bag)
                        logvar = model.vae_logvar(bag)
                        z = model.reparameterize(mu, logvar)

                        total_loss = 0.0
                        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                        total_loss = total_loss + beta * kl

                        bag_recon_loss = torch.tensor(0.0, device=device)
                        if model.feature_decoder is not None:
                            bag_recon = model.feature_decoder(z)
                            bag_recon_loss = F.mse_loss(bag_recon, bag)
                            total_loss = total_loss + bag_recon_loss

                        img_recon_loss = torch.tensor(0.0, device=device)
                        if model.pixel_decoder is not None:
                            img_recon = model.pixel_decoder(z)
                            center_idx = images.shape[1] // 2
                            img_recon_loss = F.mse_loss(img_recon, images[:, center_idx])
                            total_loss = total_loss + img_recon_loss

                    total_loss.backward()
                    vae_optimizer.step()

                    run_vae += total_loss.item()
                    run_kl += kl.item()
                    run_bag += bag_recon_loss.item()
                    run_img += img_recon_loss.item()
                    n_batch += 1

                vae_scheduler.step()

                # Validation
                model.eval()
                model.encoder.eval()
                v_vae, v_kl, v_bag, v_img = 0.0, 0.0, 0.0, 0.0
                v_n = 0
                v_ssim, v_psnr = 0.0, 0.0

                with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                    for images, _ in vae_val_loader:
                        images = images.to(device)
                        bag = model.encode_bag(images)
                        mu = model.vae_mu(bag)
                        logvar = model.vae_logvar(bag)
                        z = model.reparameterize(mu, logvar)

                        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                        v_kl += kl.item()

                        if model.feature_decoder is not None:
                            bag_recon = model.feature_decoder(z)
                            v_bag += F.mse_loss(bag_recon, bag).item()

                        if model.pixel_decoder is not None:
                            img_recon = model.pixel_decoder(z)
                            center_idx = images.shape[1] // 2
                            v_img_loss = F.mse_loss(img_recon, images[:, center_idx])
                            v_img += v_img_loss.item()

                            if args.compute_recon_metrics:
                                v_ssim += compute_ssim(img_recon, images[:, center_idx])
                                v_psnr += compute_psnr(img_recon, images[:, center_idx])

                        v_n += 1

                v_vae = (beta * v_kl + v_bag + v_img) / max(v_n, 1)
                v_kl /= max(v_n, 1)
                v_bag /= max(v_n, 1)
                v_img /= max(v_n, 1)

                log_str = (
                    f"VAE Epoch {epoch:3d}: "
                    f"Loss={run_vae/max(n_batch,1):.4f} "
                    f"(KL={run_kl/max(n_batch,1):.4f} "
                    f"Bag={run_bag/max(n_batch,1):.4f} "
                    f"Img={run_img/max(n_batch,1):.4f}) | "
                    f"Val: Loss={v_vae:.4f} KL={v_kl:.4f} Bag={v_bag:.4f} Img={v_img:.4f}"
                )
                if args.compute_recon_metrics:
                    log_str += f" SSIM={v_ssim/max(v_n,1):.4f} PSNR={v_psnr/max(v_n,1):.2f}"
                log_str += f" β={beta:.4f} Time={time.time()-epoch_start:.1f}s"
                print(log_str)

                row = [epoch,
                       f"{run_vae/max(n_batch,1):.4f}", f"{run_kl/max(n_batch,1):.4f}",
                       f"{run_bag/max(n_batch,1):.4f}",
                       f"{v_vae:.4f}", f"{v_kl:.4f}", f"{v_bag:.4f}"]
                if use_pixel_decoder:
                    row += [f"{run_img/max(n_batch,1):.4f}", f"{v_img:.4f}"]
                if args.compute_recon_metrics and v_n > 0:
                    row += [f"{v_ssim / v_n:.4f}", f"{v_psnr / v_n:.4f}"]
                with open(csv_vae, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(row)

                tb_writer.add_scalars('VAE_Loss', {'train': run_vae/max(n_batch,1), 'val': v_vae}, epoch)

                if v_vae < best_vae_loss:
                    best_vae_loss = v_vae
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'vae_mu': model.vae_mu.state_dict(),
                        'vae_logvar': model.vae_logvar.state_dict(),
                        'feature_decoder': model.feature_decoder.state_dict() if model.feature_decoder else None,
                        'pixel_decoder': model.pixel_decoder.state_dict() if model.pixel_decoder else None,
                        'latent_dim': args.latent_dim,
                        'val_loss': v_vae,
                    }, os.path.join(OUTPUT_DIR, 'best_mil_vae.pth'))

                if (epoch + 1) % args.vae_checkpoint_every == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'vae_mu': model.vae_mu.state_dict(),
                        'vae_logvar': model.vae_logvar.state_dict(),
                        'feature_decoder': model.feature_decoder.state_dict() if model.feature_decoder else None,
                        'pixel_decoder': model.pixel_decoder.state_dict() if model.pixel_decoder else None,
                        'latent_dim': args.latent_dim,
                        'val_loss': v_vae,
                    }, os.path.join(OUTPUT_DIR, 'checkpoint_vae_epoch.pth'))

            # Save last epoch as best_mil_vae.pth
            torch.save({
                'model_state_dict': model.state_dict(),
                'vae_mu': model.vae_mu.state_dict(),
                'vae_logvar': model.vae_logvar.state_dict(),
                'feature_decoder': model.feature_decoder.state_dict() if model.feature_decoder else None,
                'pixel_decoder': model.pixel_decoder.state_dict() if model.pixel_decoder else None,
                'latent_dim': args.latent_dim,
                'val_loss': v_vae,
            }, os.path.join(OUTPUT_DIR, 'best_mil_vae.pth'))

            print(f"Stage 2 complete! Best VAE val loss: {best_vae_loss:.4f}")

        # =====================================================================
        # STAGE 2 ONLY: VAE from existing checkpoint
        # =====================================================================
        elif args.stage2_only:
            if args.mil_checkpoint:
                mil_checkpoint_path = args.mil_checkpoint
            else:
                mil_checkpoint_path = os.path.join(OUTPUT_DIR, 'best_mil.pth')

            if not os.path.exists(mil_checkpoint_path):
                print(f"ERROR: No MIL checkpoint at {mil_checkpoint_path}")
                return
            print(f"Loading MIL checkpoint: {mil_checkpoint_path}")
            model.load_state_dict(torch.load(mil_checkpoint_path, map_location=device))

            # VAE stage (same as two-stage stage 2)
            print(f"\n{'='*60}")
            print(f"STAGE 2: VAE Training (latent_dim={args.latent_dim}, beta={args.vae_beta})")
            print(f"{'='*60}")

            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
            model.encoder.eval()

            vae_params = []
            vae_params.extend(model.vae_mu.parameters())
            vae_params.extend(model.vae_logvar.parameters())
            if model.feature_decoder is not None:
                vae_params.extend(model.feature_decoder.parameters())
            if model.pixel_decoder is not None:
                vae_params.extend(model.pixel_decoder.parameters())

            vae_optimizer = torch.optim.AdamW(vae_params, lr=args.vae_lr, weight_decay=1e-5)
            vae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(vae_optimizer, T_max=args.vae_epochs)

            csv_vae = os.path.join(OUTPUT_DIR, f'training_vae_{timestamp}.csv')
            with open(csv_vae, 'w', newline='') as f:
                w = csv.writer(f)
                cols = ['epoch', 'vae_loss', 'kl_loss', 'bag_recon', 'val_vae', 'val_kl', 'val_bag_recon']
                if use_pixel_decoder:
                    cols += ['img_recon', 'val_img_recon']
                w.writerow(cols)

            vae_bs = args.vae_batch_size if args.vae_batch_size is not None else args.batch_size * 4
            vae_train_loader = DataLoader(
                train_dataset, batch_size=vae_bs, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=True,
                persistent_workers=NUM_WORKERS > 0,
                worker_init_fn=partial(worker_init_fn, seed=SEED), drop_last=True
            )
            vae_val_loader = DataLoader(
                val_dataset, batch_size=vae_bs, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                persistent_workers=NUM_WORKERS > 0,
                worker_init_fn=partial(worker_init_fn, seed=SEED)
            )
            print(f"VAE batch size: {vae_bs} ({vae_bs}x gradient steps)")

            best_vae_loss = float('inf')

            for epoch in range(args.vae_epochs):
                epoch_start = time.time()
                train_dataset.set_epoch(epoch)
                model.train()
                model.encoder.eval()

                if args.beta_anneal:
                    beta = args.beta_start + (args.vae_beta - args.beta_start) * min(epoch / args.beta_warmup_epochs, 1)
                else:
                    beta = args.vae_beta

                run_vae, run_kl, run_bag, run_img = 0.0, 0.0, 0.0, 0.0
                n_batch = 0

                for images, _ in tqdm(vae_train_loader, desc=f'VAE {epoch}', leave=False):
                    images = images.to(device)
                    vae_optimizer.zero_grad()

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        with torch.no_grad():
                            bag = model.encode_bag(images)
                        mu = model.vae_mu(bag)
                        logvar = model.vae_logvar(bag)
                        z = model.reparameterize(mu, logvar)

                        total_loss = 0.0
                        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                        total_loss = total_loss + beta * kl

                        bag_recon_loss = torch.tensor(0.0, device=device)
                        if model.feature_decoder is not None:
                            bag_recon = model.feature_decoder(z)
                            bag_recon_loss = F.mse_loss(bag_recon, bag)
                            total_loss = total_loss + bag_recon_loss

                        img_recon_loss = torch.tensor(0.0, device=device)
                        if model.pixel_decoder is not None:
                            img_recon = model.pixel_decoder(z)
                            center_idx = images.shape[1] // 2
                            img_recon_loss = F.mse_loss(img_recon, images[:, center_idx])
                            total_loss = total_loss + img_recon_loss

                    total_loss.backward()
                    vae_optimizer.step()

                    run_vae += total_loss.item()
                    run_kl += kl.item()
                    run_bag += bag_recon_loss.item()
                    run_img += img_recon_loss.item()
                    n_batch += 1

                vae_scheduler.step()

                # Validation
                model.eval()
                model.encoder.eval()
                v_vae, v_kl, v_bag, v_img = 0.0, 0.0, 0.0, 0.0
                v_n = 0
                v_ssim, v_psnr = 0.0, 0.0

                with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                    for images, _ in vae_val_loader:
                        images = images.to(device)
                        bag = model.encode_bag(images)
                        mu = model.vae_mu(bag)
                        logvar = model.vae_logvar(bag)
                        z = model.reparameterize(mu, logvar)

                        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                        v_kl += kl.item()

                        if model.feature_decoder is not None:
                            bag_recon = model.feature_decoder(z)
                            v_bag += F.mse_loss(bag_recon, bag).item()

                        if model.pixel_decoder is not None:
                            img_recon = model.pixel_decoder(z)
                            center_idx = images.shape[1] // 2
                            v_img_loss = F.mse_loss(img_recon, images[:, center_idx])
                            v_img += v_img_loss.item()

                            if args.compute_recon_metrics:
                                v_ssim += compute_ssim(img_recon, images[:, center_idx])
                                v_psnr += compute_psnr(img_recon, images[:, center_idx])

                        v_n += 1

                v_vae = (beta * v_kl + v_bag + v_img) / max(v_n, 1)
                v_kl /= max(v_n, 1)
                v_bag /= max(v_n, 1)
                v_img /= max(v_n, 1)

                log_str = (
                    f"VAE Epoch {epoch:3d}: "
                    f"Loss={run_vae/max(n_batch,1):.4f} "
                    f"(KL={run_kl/max(n_batch,1):.4f} "
                    f"Bag={run_bag/max(n_batch,1):.4f} "
                    f"Img={run_img/max(n_batch,1):.4f}) | "
                    f"Val: Loss={v_vae:.4f} KL={v_kl:.4f} Bag={v_bag:.4f} Img={v_img:.4f}"
                )
                if args.compute_recon_metrics:
                    log_str += f" SSIM={v_ssim/max(v_n,1):.4f} PSNR={v_psnr/max(v_n,1):.2f}"
                log_str += f" β={beta:.4f} Time={time.time()-epoch_start:.1f}s"
                print(log_str)

                row = [epoch,
                       f"{run_vae/max(n_batch,1):.4f}", f"{run_kl/max(n_batch,1):.4f}",
                       f"{run_bag/max(n_batch,1):.4f}",
                       f"{v_vae:.4f}", f"{v_kl:.4f}", f"{v_bag:.4f}"]
                if use_pixel_decoder:
                    row += [f"{run_img/max(n_batch,1):.4f}", f"{v_img:.4f}"]
                if args.compute_recon_metrics and v_n > 0:
                    row += [f"{v_ssim / v_n:.4f}", f"{v_psnr / v_n:.4f}"]
                with open(csv_vae, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(row)

                tb_writer.add_scalars('VAE_Loss', {'train': run_vae/max(n_batch,1), 'val': v_vae}, epoch)

                if v_vae < best_vae_loss:
                    best_vae_loss = v_vae
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'vae_mu': model.vae_mu.state_dict(),
                        'vae_logvar': model.vae_logvar.state_dict(),
                        'feature_decoder': model.feature_decoder.state_dict() if model.feature_decoder else None,
                        'pixel_decoder': model.pixel_decoder.state_dict() if model.pixel_decoder else None,
                        'latent_dim': args.latent_dim,
                        'val_loss': v_vae,
                    }, os.path.join(OUTPUT_DIR, 'best_mil_vae.pth'))

                if (epoch + 1) % args.vae_checkpoint_every == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'vae_mu': model.vae_mu.state_dict(),
                        'vae_logvar': model.vae_logvar.state_dict(),
                        'feature_decoder': model.feature_decoder.state_dict() if model.feature_decoder else None,
                        'pixel_decoder': model.pixel_decoder.state_dict() if model.pixel_decoder else None,
                        'latent_dim': args.latent_dim,
                        'val_loss': v_vae,
                    }, os.path.join(OUTPUT_DIR, 'checkpoint_vae_epoch.pth'))

            torch.save({
                'model_state_dict': model.state_dict(),
                'vae_mu': model.vae_mu.state_dict(),
                'vae_logvar': model.vae_logvar.state_dict(),
                'feature_decoder': model.feature_decoder.state_dict() if model.feature_decoder else None,
                'pixel_decoder': model.pixel_decoder.state_dict() if model.pixel_decoder else None,
                'latent_dim': args.latent_dim,
                'val_loss': v_vae,
            }, os.path.join(OUTPUT_DIR, 'best_mil_vae.pth'))

            print(f"VAE stage complete! Best val loss: {best_vae_loss:.4f}")

        # =====================================================================
        # Finalize
        # =====================================================================
        tb_writer.close()

        training_mode = 'joint' if is_joint else ('two_stage' if args.two_stage else 'stage2_only')
        print(f"Training complete ({training_mode})! Best val AUC: {best_val_auc:.4f}, Best VAE val loss: {best_vae_loss:.4f}")

        with open(os.path.join(OUTPUT_DIR, 'mil_vae_results.json'), 'w') as f:
            json.dump({
                'best_mil_val_auc': float(best_val_auc) if best_val_auc > 0 else None,
                'best_vae_val_loss': float(best_vae_loss),
                'num_classes': num_classes,
                'classes': all_classes,
                'class_to_idx': class_to_idx,
                'training_mode': training_mode,
                'config': vars(args),
            }, f, indent=2)

    if args.run_all_folds:
        for test_plate in all_plates:
            fold_dir = os.path.join(SCRIPT_DIR, f'mil_vae_{args.data_mode}', f'fold_{test_plate}')
            joint_marker = os.path.join(fold_dir, 'best_mil_vae.pth')
            mil_marker = os.path.join(fold_dir, 'best_mil.pth')

            if not args.two_stage and not args.stage2_only:
                if os.path.exists(joint_marker):
                    print(f"Skipping {test_plate}: already trained (joint)")
                    continue
            elif args.two_stage and not args.stage2_only:
                if os.path.exists(mil_marker):
                    print(f"Skipping {test_plate} MIL: already trained")
                    continue
            elif args.stage2_only:
                if os.path.exists(joint_marker):
                    print(f"Skipping {test_plate} VAE: already trained")
                    continue
            train_single_fold(test_plate)
    else:
        train_single_fold(args.test_plate)

    print("Done!")


if __name__ == '__main__':
    main()
