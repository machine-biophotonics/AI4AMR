#!/usr/bin/env python3
"""
Self-Supervised Training using SimCLR.
Grayscale-only augmentations.
"""

import os
import sys
import json
import argparse
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message=".*Not enough SMs.*")

os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def get_all_image_paths(data_root, plate):
    drug_dir = os.path.join(data_root, "Drugs_Data", plate)
    mutant_dir = os.path.join(data_root, "Mutants_Data", plate)
    
    drug_paths = []
    mutant_paths = []
    
    if os.path.exists(drug_dir):
        drug_paths = glob.glob(os.path.join(drug_dir, "**", "*.tif"), recursive=True)
        drug_paths += glob.glob(os.path.join(drug_dir, "**", "*.tiff"), recursive=True)
    
    if os.path.exists(mutant_dir):
        mutant_paths = glob.glob(os.path.join(mutant_dir, "**", "*.tif"), recursive=True)
        mutant_paths += glob.glob(os.path.join(mutant_dir, "**", "*.tiff"), recursive=True)
    
    print(f"Plate {plate}: {len(drug_paths)} drugs, {len(mutant_paths)} mutants")
    return drug_paths, mutant_paths


def get_all_plates_image_paths(data_root, plates):
    all_drug = []
    all_mutant = []
    for plate in plates:
        d, m = get_all_image_paths(data_root, plate)
        all_drug.extend(d)
        all_mutant.extend(m)
    print(f"Total: {len(all_drug)} drugs, {len(all_mutant)} mutants")
    return all_drug, all_mutant


class GrayscaleAug:
    def __call__(self, img):
        if random.random() < 0.5:
            from PIL import ImageEnhance
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            from PIL import ImageEnhance
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
        return img


class SelfSupDataset(Dataset):
    def __init__(self, image_paths, crop_size=224, grid_size=12, neighborhood=3, augment=True):
        self.image_paths = image_paths
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.neighborhood = neighborhood
        self.augment = augment
        
        try:
            import tifffile
            arr = tifffile.imread(image_paths[0])
            w, h = arr.shape[1], arr.shape[0]
        except:
            img = Image.open(image_paths[0]).convert('L')
            w, h = img.size
        
        self.image_size = w
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
        half_n = neighborhood // 2
        self.positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if left + crop_size <= w and top + crop_size <= h:
                    if left - half_n * stride >= 0 and left + half_n * stride + crop_size <= w:
                        if top - half_n * stride >= 0 and top + half_n * stride + crop_size <= h:
                            self.positions.append((left, top))
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        if augment:
            self.aug = transforms.Compose([
                transforms.RandomResizedCrop(crop_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(180),
                GrayscaleAug(),
            ])
        else:
            self.aug = None
    
    def __len__(self):
        return len(self.image_paths)
    
    def _load_image(self, idx):
        try:
            import tifffile
            arr = tifffile.imread(self.image_paths[idx])
            if arr.ndim == 3:
                arr = arr[0]
            return arr.astype(np.float32) / 65535.0
        except:
            return np.array(Image.open(self.image_paths[idx]).convert('L')).astype(np.float32) / 255.0
    
    def _extract_crops(self, arr, center_left, center_top):
        crops = []
        half_n = self.neighborhood // 2
        jr = self.stride // 4 if self.augment else 0
        
        for di in range(-half_n, half_n + 1):
            for dj in range(-half_n, half_n + 1):
                jx = random.randint(-jr, jr) if self.augment else 0
                jy = random.randint(-jr, jr) if self.augment else 0
                left = max(0, min(center_left + dj * self.stride + jx, self.image_size - self.crop_size))
                top = max(0, min(center_top + di * self.stride + jy, self.image_size - self.crop_size))
                crop = Image.fromarray((arr[top:top+self.crop_size, left:left+self.crop_size] * 255).astype(np.uint8), mode='L')
                crops.append(crop)
        return crops
    
    def __getitem__(self, idx):
        arr = self._load_image(idx)
        cl, ct = random.choice(self.positions)
        
        crops1 = self._extract_crops(arr, cl, ct)
        crops2 = self._extract_crops(arr, cl, ct)
        
        v1 = torch.stack([self.normalize(self.aug(c) if self.aug else c) for c in crops1])
        v2 = torch.stack([self.normalize(self.aug(c) if self.aug else c) for c in crops2])
        
        return torch.cat([v1, v2], dim=0), 0


class SimCLRHead(nn.Module):
    def __init__(self, in_dim=1280, hidden=512, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class SelfSupMIL(nn.Module):
    def __init__(self, num_crops=9, proj_dim=256):
        super().__init__()
        import torchvision.models as models
        
        base = models.efficientnet_b0(weights='IMAGENET1K_V1')
        base.features[0][0] = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        
        self.backbone = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.feature_dim = 1280
        
        self.attn = nn.Sequential(nn.Linear(self.feature_dim, 256), nn.Tanh(), nn.Linear(256, 1))
        self.proj = SimCLRHead(self.feature_dim, 512, proj_dim)
    
    def forward(self, x, ret_emb=False):
        bs = x.shape[0]
        nc = x.shape[1]
        
        x = x.reshape(bs * nc, *x.shape[2:]).contiguous()
        f = self.backbone(x).reshape(bs, nc, -1)
        
        a = F.softmax(self.attn(f), dim=1)
        pooled = torch.einsum('bn,bnf->bf', a.squeeze(-1), f)
        
        if ret_emb:
            return pooled
        return self.proj(pooled)


def simclr_loss(zi, zj, temp=0.1):
    bs = zi.shape[0]
    zi, zj = F.normalize(zi, dim=1), F.normalize(zj, dim=1)
    z = torch.cat([zi, zj], dim=0)
    s = torch.matmul(z, z.T) / temp
    m = torch.eye(2 * bs, device=zi.device)
    m[bs:, :bs] = 1
    m[:bs, bs:] = 1
    s = s - m * 1e12
    return -torch.logsumexp(s, dim=1).mean()


def train_epoch(model, loader, opt, epoch, args, device, scheduler=None, writer=None):
    model.train()
    total_loss, n = 0, 0
    
    for combined, _ in tqdm(loader, desc=f"Epoch {epoch}"):
        bs = combined.shape[0]
        nc = combined.shape[1] // 2
        
        v1 = combined[:, :nc].to(device)
        v2 = combined[:, nc:].contiguous().to(device)
        
        zi = model(v1)
        zj = model(v2)
        
        loss = simclr_loss(zi, zj, args.temperature)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        # Update scheduler per batch (fixes LR display)
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        n += 1
        
        if writer and n % 50 == 0:
            writer.add_scalar('train/batch', loss.item(), epoch * len(loader) + n)
    
    avg = total_loss / n
    if writer:
        writer.add_scalar('train/epoch', avg, epoch)
        writer.add_scalar('train/lr', opt.param_groups[0]['lr'], epoch)
    return avg


def cosine_warmup(opt, n_warm, n_total):
    def lr_lambda(step):
        if step < n_warm:
            return step / max(1, n_warm)
        prog = (step - n_warm) / max(1, n_total - n_warm)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--plate", type=str, default=None)
    parser.add_argument("--all_plates", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--neighborhood", type=int, default=3)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="self_supervised_trial")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(os.path.join(output_dir, "runs"))
    
    print("=== SimCLR Training ===")
    
    if args.all_plates:
        plates = ["P1", "P2", "P3", "P4", "P5", "P6"]
        drug_paths, mutant_paths = get_all_plates_image_paths(args.data_root, plates)
    elif args.plate:
        drug_paths, mutant_paths = get_all_image_paths(args.data_root, args.plate)
    else:
        drug_paths, mutant_paths = get_all_image_paths(args.data_root, "P1")
    
    if not drug_paths and not mutant_paths:
        print("ERROR: No images found!")
        return
    
    all_paths = drug_paths + mutant_paths
    source_types = ["drug"] * len(drug_paths) + ["mutant"] * len(mutant_paths)
    
    print(f"Total: {len(all_paths)} images")
    
    with open(os.path.join(output_dir, "source_types.json"), "w") as f:
        json.dump(source_types, f)
    
    dataset = SelfSupDataset(all_paths, args.crop_size, args.grid_size, args.neighborhood, True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                      num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    print(f"Batches: {len(loader)}")
    
    nc = args.neighborhood ** 2
    model = SelfSupMIL(nc, args.projection_dim).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_warmup(opt, args.warmup_epochs * len(loader), args.epochs * len(loader))
    
    best_loss = float('inf')
    last_path = os.path.join(output_dir, "last_model.pth")
    
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, opt, epoch, args, device, scheduler, writer)
        
        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f} - LR: {opt.param_groups[0]['lr']:.6f}")
        
        if loss < best_loss:
            best_loss = loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                     "opt_state_dict": opt.state_dict(), "loss": loss, "args": vars(args)},
                    os.path.join(output_dir, "best_model.pth"))
            print(f"  Saved best")
        
        if epoch % args.checkpoint_every == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "loss": loss},
                     os.path.join(output_dir, f"checkpoint_{epoch}.pth"))
            print(f"  Saved checkpoint_{epoch}")
    
    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict(),
              "opt_state_dict": opt.state_dict(), "loss": loss, "args": vars(args)}, last_path)
    
    writer.close()
    print(f"Done! Best: {best_loss:.4f}")
    print(f"Last: {last_path}")


if __name__ == "__main__":
    main()