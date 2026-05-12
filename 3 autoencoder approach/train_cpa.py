#!/usr/bin/env python3
"""
CPA-MIL Training — Shared Embedding Space for Drugs & Mutants

Training:
  - MILEncoder produces z_bag [256]
  - Classifier: z_bag → predict drug/gene class
  - Prototype loss: z_bag must be similar to its perturbation's prototype vector
  - PerturbationEmbedding table IS the shared space (row i = prototype for label i)

After training:
  - prototypes[Ciprofloxacin_2x] should be near prototypes[gyrA_1]
  - controls (DMSO, WT, NC) all map to index 0

Usage:
  python3 train_cpa.py --test_plate P6 --epochs 200
  python3 train_cpa.py --run_all_folds --epochs 200
"""

import os, sys, argparse, time, json, glob, random, csv, warnings
from datetime import datetime
from collections import Counter
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*Not enough SMs.*")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from mil_model import MultiCropDataset, extract_well_from_filename
from cpa_model import CPAModel

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--test_plate', type=str, default='P6')
parser.add_argument('--run_all_folds', action='store_true')
parser.add_argument('--neighborhood', type=int, default=3)
parser.add_argument('--grid_size', type=int, default=12)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--num_channels', type=int, default=1)
parser.add_argument('--cls_weight', type=float, default=1.0)
parser.add_argument('--proto_weight', type=float, default=1.0)
parser.add_argument('--proto_temp', type=float, default=10.0)
parser.add_argument('--label_smoothing', type=float, default=0.1)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRUG_BASE = os.path.join(os.path.dirname(SCRIPT_DIR), 'Drugs_Data')
MUTANT_BASE = os.path.join(os.path.dirname(SCRIPT_DIR), 'Mutants_Data')
IC50_PATH = os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')
MUTANT_PATH = os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')

with open(IC50_PATH) as f: ic50_data = json.load(f)
with open(MUTANT_PATH) as f: mutant_data = json.load(f)

all_plates = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']


def build_label_mappings():
    """Build unified label mappings.

    Controls (DMSO, WT, NC) → perturbation 'control'.
    Drugs  → perturbation 'drug_{name}_{conc}' (e.g. drug_Ciprofloxacin_2x)
    Mutants → perturbation 'mutant_{gene_id}' (e.g. mutant_gyrA_1)
    """
    pert_set, class_set = set(), set()
    plate_well_pert = {}
    plate_well_class = {}

    for plate in all_plates:
        plate_well_pert[plate] = {}
        plate_well_class[plate] = {}

        if plate in ic50_data:
            for well, info in ic50_data[plate].items():
                drug = info.get('antibiotic', '').replace(' ', '_')
                ic50 = info.get('ic50_multiple', '')
                if not drug: continue
                if ic50 == 'control':
                    pert_name = 'control'; class_name = 'control'
                else:
                    ic50_str = ic50 if 'x' in ic50 else f'{ic50}x'
                    pert_name = f'drug_{drug}_{ic50_str}'
                    class_name = f'{drug}_{ic50_str}'
                pert_set.add(pert_name); class_set.add(class_name)
                plate_well_pert[plate][f'drug_{well}'] = pert_name
                plate_well_class[plate][f'drug_{well}'] = class_name

        if plate in mutant_data:
            for row, cols in mutant_data[plate].items():
                for col, info in cols.items():
                    gid = info.get('id', '')
                    if not gid: continue
                    well = f"{row}{int(col):02d}"
                    if gid.startswith('WT') or gid.startswith('NC'):
                        pert_name = 'control'
                    else:
                        pert_name = f'mutant_{gid}'
                    pert_set.add(pert_name); class_set.add(gid)
                    plate_well_pert[plate][f'mutant_{well}'] = pert_name
                    plate_well_class[plate][f'mutant_{well}'] = gid

    pert_to_idx = {p: i for i, p in enumerate(sorted(pert_set))}
    class_to_idx = {c: i for i, c in enumerate(sorted(class_set))}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    # class_name → pert_name and class_name → pert_idx lookups
    class_to_pert_name = {}
    for plate in all_plates:
        for wk, cn in plate_well_class.get(plate, {}).items():
            if cn not in class_to_pert_name:
                class_to_pert_name[cn] = plate_well_pert[plate][wk]

    class_to_pert_idx = {cn: pert_to_idx[pn]
                         for cn, pn in class_to_pert_name.items()}

    return (pert_to_idx, class_to_idx, idx_to_class,
            class_to_pert_idx, plate_well_pert, plate_well_class)


def get_image_paths_for_plate(plate_key, plate_well_pert):
    paths = []
    for base_dir in [DRUG_BASE, MUTANT_BASE]:
        pd = os.path.join(base_dir, plate_key)
        if os.path.exists(pd):
            for pat in ['*.tif', '*.tiff', '*.png']:
                paths.extend(glob.glob(os.path.join(pd, '**', pat), recursive=True))
    dp = os.path.join(DRUG_BASE, plate_key)
    mp = os.path.join(MUTANT_BASE, plate_key)
    valid = []
    for p in paths:
        w = extract_well_from_filename(os.path.basename(p))
        if not w: continue
        k = f'drug_{w}' if p.startswith(dp) else f'mutant_{w}'
        if k in plate_well_pert.get(plate_key, {}):
            valid.append(p)
    return valid


def get_paths_and_labels(plates, plate_well_pert, plate_well_class,
                         pert_to_idx, class_to_idx):
    paths, pert_labels, cls_labels = [], [], []
    for plate in plates:
        dp = os.path.join(DRUG_BASE, plate)
        for p in get_image_paths_for_plate(plate, plate_well_pert):
            w = extract_well_from_filename(os.path.basename(p))
            if not w: continue
            k = f'drug_{w}' if p.startswith(dp) else f'mutant_{w}'
            paths.append(p)
            pert_labels.append(pert_to_idx[plate_well_pert[plate][k]])
            cls_labels.append(class_to_idx[plate_well_class[plate][k]])
    return paths, np.array(pert_labels), np.array(cls_labels)


def worker_init_fn(worker_id, seed=42):
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def train_fold(test_plate, pert_to_idx, class_to_idx, idx_to_class,
               class_to_pert_idx, plate_well_pert, plate_well_class):
    output_dir = os.path.join(SCRIPT_DIR, 'cpa', f'fold_{test_plate}')
    os.makedirs(output_dir, exist_ok=True)

    tn = int(test_plate[1])
    val_plate = f'P{(tn - 2) % 6 + 1}'
    train_plates = [p for p in all_plates if p not in (test_plate, val_plate)][:4]

    print(f"\n{'='*60}\nFold: test={test_plate}, val={val_plate}, train={train_plates}")

    train_paths, train_pert, train_cls = get_paths_and_labels(
        train_plates, plate_well_pert, plate_well_class, pert_to_idx, class_to_idx)
    val_paths, val_pert, val_cls = get_paths_and_labels(
        [val_plate], plate_well_pert, plate_well_class, pert_to_idx, class_to_idx)
    test_paths, _, test_cls = get_paths_and_labels(
        [test_plate], plate_well_pert, plate_well_class, pert_to_idx, class_to_idx)

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    num_pert, num_cls = len(pert_to_idx), len(class_to_idx)
    print(f"Perturbations: {num_pert}, Classes: {num_cls}")

    train_ds = MultiCropDataset(train_paths, train_cls.tolist(), None,
        neighborhood=args.neighborhood, grid_size=args.grid_size,
        augment=True, seed=SEED, num_channels=args.num_channels)
    val_ds = MultiCropDataset(val_paths, val_cls.tolist(), None,
        neighborhood=args.neighborhood, grid_size=args.grid_size,
        augment=False, seed=SEED, num_channels=args.num_channels)
    test_ds = MultiCropDataset(test_paths, test_cls.tolist(), None,
        neighborhood=args.neighborhood, grid_size=args.grid_size,
        augment=False, seed=SEED, num_channels=args.num_channels)
    train_ds.set_epoch(0); val_ds.set_epoch(0); test_ds.set_epoch(0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True,
        worker_init_fn=partial(worker_init_fn, seed=SEED))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True,
        worker_init_fn=partial(worker_init_fn, seed=SEED))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=True,
        worker_init_fn=partial(worker_init_fn, seed=SEED))

    model = CPAModel(
        num_perturbations=num_pert, num_classes=num_cls,
        embedding_dim=args.embedding_dim, num_heads=args.num_heads,
        dropout=args.dropout, num_channels=args.num_channels,
        temperature=args.proto_temp,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f'training_{timestamp}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'cls_loss', 'proto_loss', 'total',
                    'val_cls', 'val_acc', 'val_auc', 'lr'])

    best_auc = 0.0
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_ds.set_epoch(epoch)
        model.train()
        cls_s, proto_s, n_b = 0.0, 0.0, 0

        for images, cls_labels in tqdm(train_loader, desc=f'Epoch {epoch}',
                                       leave=False):
            images = images.to(device, non_blocking=True)
            cls_labels = cls_labels.to(device, non_blocking=True)
            pert_labels = torch.tensor(
                [class_to_pert_idx.get(idx_to_class[cl.item()], 0)
                 for cl in cls_labels], device=device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                z_bag, logits, logits_pert = model(images)
                cls_loss = F.cross_entropy(
                    logits, cls_labels, label_smoothing=args.label_smoothing)
                proto_loss = F.cross_entropy(logits_pert, pert_labels)
                loss = args.cls_weight * cls_loss + args.proto_weight * proto_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            cls_s += cls_loss.item(); proto_s += proto_loss.item(); n_b += 1

        scheduler.step()

        # Validate
        model.eval()
        val_c, val_a, val_n = 0.0, 0.0, 0
        all_p, all_l = [], []
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for images, cls_labels in val_loader:
                images = images.to(device, non_blocking=True)
                cls_labels = cls_labels.to(device, non_blocking=True)
                _, logits, _ = model(images)
                val_c += F.cross_entropy(logits, cls_labels).item()
                probs = F.softmax(logits, dim=1)
                val_a += (logits.argmax(1) == cls_labels).sum().item()
                val_n += cls_labels.size(0)
                all_p.extend(probs.cpu().numpy()); all_l.extend(cls_labels.cpu().numpy())

        val_acc = 100.0 * val_a / val_n
        val_auc = float('nan')
        try:
            u = np.unique(all_l)
            if len(u) >= 2:
                val_auc = roc_auc_score(
                    label_binarize(all_l, classes=u),
                    np.array(all_p)[:, u], average='macro')
        except: pass

        avg_c = cls_s / n_b; avg_p = proto_s / n_b
        print(f"Epoch {epoch}: cls={avg_c:.4f} proto={avg_p:.4f} | "
              f"val_cls={val_c/len(val_loader):.4f} "
              f"val_acc={val_acc:.2f}% val_auc={val_auc:.4f} "
              f"time={time.time()-epoch_start:.1f}s")

        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([epoch, avg_c, avg_p, avg_c+avg_p,
                       val_c/len(val_loader), val_acc, val_auc,
                       optimizer.param_groups[0]['lr']])

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            torch.save({'pert_to_idx': pert_to_idx, 'class_to_idx': class_to_idx,
                        'idx_to_class': idx_to_class}, os.path.join(output_dir, 'label_mappings.pth'))

    # Test
    print("\nTesting...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
    model.eval()
    tp, tl = [], []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for images, cls_labels in test_loader:
            images = images.to(device, non_blocking=True)
            cls_labels = cls_labels.to(device, non_blocking=True)
            _, logits, _ = model(images)
            tp.extend(F.softmax(logits, dim=1).cpu().numpy())
            tl.extend(cls_labels.cpu().numpy())
    test_acc = 100.0 * np.mean(np.array(tp).argmax(1) == np.array(tl))

    # Save prototypes and metadata
    prototypes = model.get_prototypes().cpu().numpy()
    np.save(os.path.join(output_dir, 'prototypes.npy'), prototypes)
    pert_names = [None] * num_pert
    for n, i in pert_to_idx.items(): pert_names[i] = n
    with open(os.path.join(output_dir, 'perturbation_names.json'), 'w') as f:
        json.dump(pert_names, f)

    results = {'test_acc': float(test_acc), 'best_val_auc': float(best_auc)}
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f)
    print(f"Test Acc: {test_acc:.2f}%  | Results: {output_dir}")


if __name__ == '__main__':
    (pert_to_idx, class_to_idx, idx_to_class, class_to_pert_idx,
     plate_well_pert, plate_well_class) = build_label_mappings()
    print(f"Perturbations: {len(pert_to_idx)}, Classes: {len(class_to_idx)}")

    if args.run_all_folds:
        for plate in all_plates:
            d = os.path.join(SCRIPT_DIR, 'cpa', f'fold_{plate}')
            if os.path.exists(os.path.join(d, 'best_model.pth')):
                print(f"Skipping {plate}")
                continue
            train_fold(plate, pert_to_idx, class_to_idx, idx_to_class,
                       class_to_pert_idx, plate_well_pert, plate_well_class)
    else:
        train_fold(args.test_plate, pert_to_idx, class_to_idx, idx_to_class,
                   class_to_pert_idx, plate_well_pert, plate_well_class)
