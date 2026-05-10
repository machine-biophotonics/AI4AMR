#!/usr/bin/env python3
"""
Domain-Adversarial Neural Network (DANN) for drug-mutant domain adaptation.

Architecture:
- Feature Extractor: EfficientNet-B0 (shared backbone)
- Label Classifier: Predicts drug/gene class
- Domain Classifier: Distinguishes drug vs mutant (connected via Gradient Reversal Layer)

Training:
- Source domain: labeled drug images + labeled mutant images
- Goal: Learn domain-invariant features that work on both
"""

import os
import sys
import argparse
import json
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer for domain adaptation.
    
    Forward: Identity function
    Backward: Reverses gradient multiplied by -alpha
    
    This forces the feature extractor to learn domain-invariant features
    by trying to fool the domain classifier.
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return x
    
    def backward(self, grad):
        return -self.alpha * grad


class GradientReversalFunction(torch.autograd.Function):
    """Custom autograd function for gradient reversal."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grl_function(x, alpha):
    """Functional interface for gradient reversal."""
    return GradientReversalFunction.apply(x, alpha)


class DomainAdversarialMIL(nn.Module):
    """Domain-Adversarial Neural Network for MIL."""
    
    def __init__(
        self, 
        num_classes: int,
        num_domains: int = 2,
        feature_dim: int = 1280,
        hidden_dim: int = 256,
        num_crops: int = 9
    ):
        super().__init__()
        
        self.num_crops = num_crops
        self.feature_dim = feature_dim
        
        # Feature Extractor: EfficientNet-B0
        import torchvision.models as models
        base = models.efficientnet_b0(weights='IMAGENET1K_V1')
        base.features[0][0] = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        
        self.feature_extractor = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Attention pooling for MIL
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Label Classifier (for drug/gene classification)
        self.label_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Domain Classifier (with GRL)
        self.grl = GradientReversalLayer(alpha=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains)  # Binary: drug=0, mutant=1
        )
    
    def forward(self, x, return_domain_logits=False):
        """
        Forward pass.
        x: [B, N, C, H, W] where N = num_crops
        """
        bs = x.shape[0]
        nc = x.shape[1]
        
        # Reshape for feature extraction
        x_flat = x.reshape(bs * nc, *x.shape[2:]).contiguous()
        features = self.feature_extractor(x_flat)  # [B*N, 1280]
        features = features.reshape(bs, nc, -1)  # [B, N, 1280]
        
        # Attention pooling
        attn_weights = F.softmax(self.attention(features), dim=1)  # [B, N, 1]
        pooled = torch.einsum('bn,bnf->bf', attn_weights.squeeze(-1), features)  # [B, 1280]
        
        # Label prediction (for classification)
        label_logits = self.label_classifier(pooled)
        
        # Domain prediction (with GRL - gradients reversed during backprop)
        domain_features = grl_function(pooled, self.grl.alpha)
        domain_logits = self.domain_classifier(domain_features)
        
        if return_domain_logits:
            return label_logits, domain_logits, pooled, attn_weights
        
        return label_logits, domain_logits
    
    def set_grl_alpha(self, alpha):
        """Set the GRL alpha for dynamic scheduling."""
        self.grl.alpha = alpha


class DomainAdaptationDataset(Dataset):
    """Dataset that returns both class labels and domain labels."""
    
    def __init__(
        self, 
        image_paths, 
        class_labels, 
        domain_labels,
        crop_size=224, 
        grid_size=12, 
        neighborhood=3,
        augment=True
    ):
        self.image_paths = image_paths
        self.class_labels = class_labels
        self.domain_labels = domain_labels  # 0=drug, 1=mutant
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.neighborhood = neighborhood
        self.augment = augment
        
        # Get image dimensions
        try:
            import tifffile
            arr = tifffile.imread(image_paths[0])
            w, h = arr.shape[1], arr.shape[0]
        except:
            img = Image.open(image_paths[0]).convert('L')
            w, h = img.size
        
        self.image_size = w
        stride = (w - crop_size) // (grid_size - 1) if grid_size > 1 else 0
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
        
        for di in range(-half_n, half_n + 1):
            for dj in range(-half_n, half_n + 1):
                left = center_left + dj * self.stride
                top = center_top + di * self.stride
                crop = arr[top:top+self.crop_size, left:left+self.crop_size]
                crops.append(crop)
        return crops
    
    def __getitem__(self, idx):
        arr = self._load_image(idx)
        cl, ct = random.choice(self.positions)
        
        crops = self._extract_crops(arr, cl, ct)
        crops_tensors = torch.stack([
            self.normalize(self.aug(c) if self.aug else c) for c in crops
        ])
        
        return (
            crops_tensors,  # [9, 1, H, W]
            torch.tensor(self.class_labels[idx], dtype=torch.long),
            torch.tensor(self.domain_labels[idx], dtype=torch.long)
        )


def get_all_image_paths(data_root, plate):
    """Get all image paths from a plate."""
    drug_dir = os.path.join(data_root, "Drugs_Data", plate)
    mutant_dir = os.path.join(data_root, "Mutants_Data", plate)
    
    drug_paths = []
    mutant_paths = []
    
    if os.path.exists(drug_dir):
        drug_paths = sorted(glob.glob(os.path.join(drug_dir, "**", "*.tif"), recursive=True))
        drug_paths += sorted(glob.glob(os.path.join(drug_dir, "**", "*.tiff"), recursive=True))
    
    if os.path.exists(mutant_dir):
        mutant_paths = sorted(glob.glob(os.path.join(mutant_dir, "**", "*.tif"), recursive=True))
        mutant_paths += sorted(glob.glob(os.path.join(mutant_dir, "**", "*.tiff"), recursive=True))
    
    print(f"Plate {plate}: {len(drug_paths)} drugs, {len(mutant_paths)} mutants")
    return drug_paths, mutant_paths


def build_class_mapping(data_root, plate):
    """Build class to index mapping for drugs and mutants."""
    ic50_path = os.path.join(SCRIPT_DIR, '..', 'plate_well_ic50_mapping.json')
    mutant_path = os.path.join(SCRIPT_DIR, '..', 'plate_well_id_path.json')
    
    with open(ic50_path, 'r') as f:
        ic50_data = json.load(f)
    
    with open(mutant_path, 'r') as f:
        mutant_data = json.load(f)
    
    # Get unique drug classes
    drug_classes = set()
    if plate in ic50_data:
        for well, info in ic50_data[plate].items():
            antibiotic = info.get('antibiotic', '')
            ic50_multiple = info.get('ic50_multiple', '')
            if antibiotic and ic50_multiple:
                drug_class = f"{antibiotic}_{ic50_multiple}"
                drug_classes.add(drug_class)
    
    # Get unique mutant classes  
    mutant_classes = set()
    if plate in mutant_data:
        for row, cols in mutant_data[plate].items():
            for col, info in cols.items():
                mutant_id = info.get('id', '')
                if mutant_id:
                    mutant_classes.add(mutant_id)
    
    # Combine and create mapping
    all_classes = sorted(list(drug_classes) + list(mutant_classes))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    
    print(f"Total classes: {len(class_to_idx)} ({len(drug_classes)} drugs, {len(mutant_classes)} mutants)")
    
    return class_to_idx, drug_classes, mutant_classes


def get_class_label(img_path, plate, class_to_idx, drug_classes, mutant_classes):
    """Get class label for an image."""
    filename = os.path.basename(img_path)
    
    # Parse well from filename
    well = None
    for part in filename.split('_'):
        if part.startswith('Well'):
            well = part.replace('Well', '')
            break
    
    if well is None:
        return random.choice(list(class_to_idx.values()))
    
    # Check if drug or mutant based on path
    is_drug = 'Drugs_Data' in img_path
    
    ic50_path = os.path.join(SCRIPT_DIR, '..', 'plate_well_ic50_mapping.json')
    mutant_path = os.path.join(SCRIPT_DIR, '..', 'plate_well_id_path.json')
    
    if is_drug:
        with open(ic50_path, 'r') as f:
            ic50_data = json.load(f)
        
        if plate in ic50_data and well in ic50_data[plate]:
            info = ic50_data[plate][well]
            antibiotic = info.get('antibiotic', '')
            ic50_multiple = info.get('ic50_multiple', '')
            if antibiotic and ic50_multiple:
                drug_class = f"{antibiotic}_{ic50_multiple}"
                return class_to_idx.get(drug_class, 0)
    else:
        with open(mutant_path, 'r') as f:
            mutant_data = json.load(f)
        
        if plate in mutant_data:
            for row, cols in mutant_data[plate].items():
                if well in cols:
                    mutant_id = cols[well].get('id', '')
                    if mutant_id:
                        return class_to_idx.get(mutant_id, 0)
    
    return random.choice(list(class_to_idx.values()))


def dann_loss(label_logits, label_targets, domain_logits, domain_targets, lambda_=1.0):
    """Compute DANN loss: classification loss + lambda * domain loss."""
    # Label loss (standard cross-entropy)
    label_loss = F.cross_entropy(label_logits, label_targets)
    
    # Domain loss (should be minimized - classifier learns to distinguish)
    domain_loss = F.cross_entropy(domain_logits, domain_targets)
    
    # Total loss
    total_loss = label_loss + lambda_ * domain_loss
    
    return total_loss, label_loss, domain_loss


def get_grl_schedule(epoch, total_epochs):
    """Get GRL alpha schedule - gradually increase from 0 to 1.
    
    This follows the paper's suggestion: start with easier task (no domain adaptation),
    then gradually introduce domain adversarial training.
    """
    p = epoch / total_epochs
    # Schedule: 0 -> 1 gradually
    alpha = 2. / (1. + np.exp(-10 * p)) - 1
    return alpha


def train_dann(args):
    """Main training function for DANN."""
    print("=" * 60)
    print("Domain-Adversarial Neural Network (DANN) Training")
    print("=" * 60)
    
    # Get image paths
    drug_paths, mutant_paths = get_all_image_paths(args.data_root, args.plate)
    
    if not drug_paths and not mutant_paths:
        print("ERROR: No images found!")
        return
    
    # Build class mapping
    class_to_idx, drug_classes, mutant_classes = build_class_mapping(args.data_root, args.plate)
    num_classes = len(class_to_idx)
    
    # Prepare dataset
    all_paths = drug_paths + mutant_paths
    all_class_labels = []
    all_domain_labels = []
    
    for img_path in all_paths:
        class_label = get_class_label(img_path, args.plate, class_to_idx, drug_classes, mutant_classes)
        is_drug = 'Drugs_Data' in img_path
        
        all_class_labels.append(class_label)
        all_domain_labels.append(0 if is_drug else 1)
    
    print(f"Total samples: {len(all_paths)}")
    print(f"Domain distribution: Drugs={all_domain_labels.count(0)}, Mutants={all_domain_labels.count(1)}")
    
    # Create dataset and dataloader
    dataset = DomainAdaptationDataset(
        all_paths, all_class_labels, all_domain_labels,
        args.crop_size, args.grid_size, args.neighborhood, True
    )
    
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    print(f"Batches: {len(loader)}")
    
    # Create model
    model = DomainAdversarialMIL(
        num_classes=num_classes,
        num_domains=2,
        hidden_dim=args.hidden_dim,
        num_crops=args.neighborhood ** 2
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    output_dir = os.path.join(SCRIPT_DIR, 'dann_output', f'plate_{args.plate}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save class mapping
    with open(os.path.join(output_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_label_loss = 0
        total_domain_loss = 0
        n = 0
        
        # Update GRL alpha
        alpha = get_grl_schedule(epoch - 1, args.epochs)
        model.set_grl_alpha(alpha)
        
        for crops, class_labels, domain_labels in tqdm(loader, desc=f"Epoch {epoch}"):
            crops = crops.to(device)
            class_labels = class_labels.to(device)
            domain_labels = domain_labels.to(device)
            
            # Forward pass
            label_logits, domain_logits = model(crops)
            
            # Compute loss
            loss, label_loss, domain_loss = dann_loss(
                label_logits, class_labels, 
                domain_logits, domain_labels,
                lambda_=args.lambda_
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_label_loss += label_loss.item()
            total_domain_loss += domain_loss.item()
            n += 1
        
        scheduler.step()
        
        avg_loss = total_loss / n
        avg_label_loss = total_label_loss / n
        avg_domain_loss = total_domain_loss / n
        
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f} (Label: {avg_label_loss:.4f}, Domain: {avg_domain_loss:.4f}) | Alpha: {alpha:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'class_to_idx': class_to_idx,
                'args': vars(args)
            }, os.path.join(output_dir, 'best_model.pth'))
        
        if epoch % args.checkpoint_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss
            }, os.path.join(output_dir, f'checkpoint_{epoch}.pth'))
    
    # Save last model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx,
        'args': vars(args)
    }, os.path.join(output_dir, 'last_model.pth'))
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Models saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='DANN for drug-mutant domain adaptation')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True, help='Path to data root')
    parser.add_argument('--plate', type=str, default='P1', help='Plate to train on')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--lambda_', type=float, default=1.0, help='Domain loss weight')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--checkpoint_every', type=int, default=10, help='Save checkpoint every N epochs')
    
    # Model arguments
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size')
    parser.add_argument('--grid_size', type=int, default=12, help='Grid size')
    parser.add_argument('--neighborhood', type=int, default=3, help='Neighborhood size')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    
    args = parser.parse_args()
    
    train_dann(args)


if __name__ == '__main__':
    main()