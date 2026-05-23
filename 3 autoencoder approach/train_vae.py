import os
import sys
import argparse
import time
import warnings
import random
import json
import re
import glob
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from vae_model import VAE

warnings.filterwarnings("ignore", message=".*Not enough SMs to use max_autotune_gemm.*")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def load_image(img_path: str, num_channels: int = 1) -> np.ndarray:
    try:
        import tifffile
        img_array = tifffile.imread(img_path)
    except (ImportError, Exception):
        img_array = np.array(Image.open(img_path))
    if len(img_array.shape) == 3:
        img_array = img_array[:, :, 0]
    if img_array.dtype == np.uint16:
        img_array = img_array.astype(np.float32) / 65535.0
    elif img_array.dtype == np.uint8:
        img_array = img_array.astype(np.float32) / 255.0
    elif img_array.dtype in (np.float32, np.float64):
        img_array = img_array.astype(np.float32)
    return img_array


def extract_well_from_filename(filename: str) -> str | None:
    match = re.search(r'Well(\w\d+)_', filename)
    return match.group(1) if match else None


class VAECropDataset(Dataset):
    def __init__(self, image_paths, labels, crop_size=224, crops_per_image=10,
                 augment=True, num_channels=1):
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.augment = augment
        self.num_channels = num_channels

        sample = load_image(image_paths[0], num_channels)
        self.h, self.w = sample.shape

        if augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.3),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_paths) * self.crops_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.crops_per_image
        img = load_image(self.image_paths[img_idx], self.num_channels)
        crop_size = self.crop_size
        if crop_size > self.h or crop_size > self.w:
            crop_img = img
        else:
            top = random.randint(0, self.h - crop_size)
            left = random.randint(0, self.w - crop_size)
            crop_img = img[top:top + crop_size, left:left + crop_size]

        crop_img = (crop_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(crop_img, mode='L')
        crop_np = np.array(pil_img)
        transformed = self.transform(image=crop_np)
        crop_tensor = transformed['image']
        return crop_tensor, self.labels[img_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--beta', type=float, default=1.0, help='Beta-VAE weight for KL')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--crops_per_image', type=int, default=10)
    parser.add_argument('--test_plate', type=str, default='Plate_6')
    parser.add_argument('--data_mode', type=str, default='both', choices=['drug', 'mutant', 'both'])
    parser.add_argument('--drug_no_concentration', action='store_true')
    parser.add_argument('--run_all_folds', action='store_true')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint_every', type=int, default=10)
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

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
                ic50_multiple = info.get('ic50_multiple', '')
                if antibiotic and ic50_multiple:
                    if args.drug_no_concentration:
                        drug_class = antibiotic.replace(' ', '_')
                    else:
                        if ic50_multiple == 'control':
                            drug_class = 'control'
                        else:
                            ic50_str = ic50_multiple if 'x' in ic50_multiple else f"{ic50_multiple}x"
                            drug_class = f"{antibiotic.replace(' ', '_')}_{ic50_str}"
                    plate_maps[plate][f"drug_{well}"] = drug_class

        if args.data_mode in ('mutant', 'both') and plate in mutant_data:
            for row, cols in mutant_data[plate].items():
                for col, info in cols.items():
                    if 'id' in info:
                        well = f"{row}{int(col):02d}"
                        plate_maps[plate][f"mutant_{well}"] = info['id']

    all_plates = ['Plate_1', 'Plate_2', 'Plate_3', 'Plate_4', 'Plate_5', 'Plate_6']

    def get_image_paths(plate: str) -> list[str]:
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
                composite_well = f"{well_prefix}{well}"
                if composite_well and composite_well in plate_maps.get(plate_key, {}):
                    valid.append(path)
        return valid

    def train_single_fold(test_plate):
        OUTPUT_DIR = os.path.join(SCRIPT_DIR, f'vae_{args.data_mode}', f'fold_{test_plate}')
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"VAE Training fold: test_plate={test_plate}, data_mode={args.data_mode}")
        print(f"{'=' * 60}")

        if 'P' in test_plate.upper() and test_plate[-1].isdigit():
            test_norm = f"Plate_{test_plate[-1]}"
        else:
            test_norm = test_plate

        train_val_plates = [p for p in all_plates if p != test_norm]
        test_num = int(test_norm.split('_')[1])
        val_num = (test_num - 2) % 6 + 1
        val_plate = f"Plate_{val_num}"
        val_plates = [val_plate] if val_plate in train_val_plates else [train_val_plates[0]]
        train_plates = [p for p in train_val_plates if p not in val_plates][:4]

        print(f"Train: {train_plates}, Val: {val_plates}, Test: {[test_norm]}")

        plate_key_map = {f'Plate_{i}': f'P{i}' for i in range(1, 7)}

        all_labels_set = set()
        for pm in plate_maps.values():
            for label in pm.values():
                if label:
                    all_labels_set.add(label)
        all_classes = sorted(all_labels_set)
        class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
        num_classes = len(all_classes)
        print(f"Total classes: {num_classes}")

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

        print(f"Train: {len(train_paths)} images, Val: {len(val_paths)}, Test: {len(test_paths)}")

        train_ds = VAECropDataset(train_paths, train_labels, crop_size=args.img_size,
                                  crops_per_image=args.crops_per_image, augment=True)
        val_ds = VAECropDataset(val_paths, val_labels, crop_size=args.img_size,
                                crops_per_image=1, augment=False)
        test_ds = VAECropDataset(test_paths, test_labels, crop_size=args.img_size,
                                 crops_per_image=1, augment=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)

        model = VAE(latent_dim=args.latent_dim, img_size=args.img_size, beta=args.beta).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(OUTPUT_DIR, f'training_vae_{timestamp}.csv')
        with open(csv_path, 'w', newline='') as f:
            import csv
            w = csv.writer(f)
            w.writerow(['epoch', 'train_loss', 'train_recon', 'train_kl', 'val_loss', 'val_recon', 'val_kl'])

        tb_writer = SummaryWriter(log_dir=OUTPUT_DIR)
        best_val_loss = float('inf')

        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            train_loss, train_recon, train_kl = 0.0, 0.0, 0.0
            n_train = 0

            for images, _ in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
                images = images.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    recon, mu, logvar, z = model(images)
                    loss, recon_l, kl_l = model.loss_fn(recon, images, mu, logvar)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                train_recon += recon_l.item()
                train_kl += kl_l.item()
                n_train += 1

            scheduler.step()

            model.eval()
            val_loss, val_recon, val_kl = 0.0, 0.0, 0.0
            n_val = 0
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                for images, _ in val_loader:
                    images = images.to(device, non_blocking=True)
                    recon, mu, logvar, z = model(images)
                    loss, recon_l, kl_l = model.loss_fn(recon, images, mu, logvar)
                    val_loss += loss.item()
                    val_recon += recon_l.item()
                    val_kl += kl_l.item()
                    n_val += 1

            train_loss /= max(n_train, 1)
            train_recon /= max(n_train, 1)
            train_kl /= max(n_train, 1)
            val_loss /= max(n_val, 1)
            val_recon /= max(n_val, 1)
            val_kl /= max(n_val, 1)

            print(
                f"Epoch {epoch:3d}: Train Loss={train_loss:.4f} (Recon={train_recon:.4f}, KL={train_kl:.4f}) | "
                f"Val Loss={val_loss:.4f} (Recon={val_recon:.4f}, KL={val_kl:.4f}) | "
                f"Time={time.time() - epoch_start:.1f}s"
            )

            with open(csv_path, 'a', newline='') as f:
                import csv
                w = csv.writer(f)
                w.writerow([epoch, f"{train_loss:.4f}", f"{train_recon:.4f}", f"{train_kl:.4f}",
                           f"{val_loss:.4f}", f"{val_recon:.4f}", f"{val_kl:.4f}"])

            tb_writer.add_scalars('VAE_Loss', {'train': train_loss, 'val': val_loss}, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'latent_dim': args.latent_dim,
                    'img_size': args.img_size,
                    'beta': args.beta,
                }, os.path.join(OUTPUT_DIR, 'best_vae.pth'))

            if (epoch + 1) % args.checkpoint_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                }, os.path.join(OUTPUT_DIR, f'vae_checkpoint_{epoch}.pth'))

        with open(os.path.join(OUTPUT_DIR, 'vae_training_results.json'), 'w') as f:
            json.dump({
                'best_val_loss': best_val_loss,
                'config': vars(args),
                'num_classes': num_classes,
                'classes': all_classes,
                'class_to_idx': class_to_idx,
            }, f, indent=2)

        tb_writer.close()
        print(f"Best val loss: {best_val_loss:.4f}. Saved to {OUTPUT_DIR}")

        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_vae.pth'), map_location=device)['model_state_dict'])
        model.eval()
        test_loss, test_recon, test_kl = 0.0, 0.0, 0.0
        n_test = 0
        all_z, all_test_labels = [], []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for images, lbls in test_loader:
                images = images.to(device, non_blocking=True)
                recon, mu, logvar, z = model(images)
                loss, recon_l, kl_l = model.loss_fn(recon, images, mu, logvar)
                test_loss += loss.item()
                test_recon += recon_l.item()
                test_kl += kl_l.item()
                n_test += 1
                all_z.append(mu.cpu().numpy())
                all_test_labels.append(lbls.numpy())
        test_loss /= max(n_test, 1)
        print(f"Test Loss: {test_loss:.4f}")

        all_z = np.concatenate(all_z, axis=0)
        all_test_labels = np.concatenate(all_test_labels, axis=0)
        np.savez(os.path.join(OUTPUT_DIR, 'test_latent.npz'),
                 z=all_z, labels=all_test_labels, classes=all_classes)

    if args.run_all_folds:
        for test_plate in all_plates:
            fold_dir = os.path.join(SCRIPT_DIR, f'vae_{args.data_mode}', f'fold_{test_plate}')
            if os.path.exists(os.path.join(fold_dir, 'best_vae.pth')):
                print(f"Skipping {test_plate}: already trained")
                continue
            train_single_fold(test_plate)
    else:
        train_single_fold(args.test_plate)

    print("Done!")


if __name__ == '__main__':
    main()
