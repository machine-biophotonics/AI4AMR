"""
MIL with flexible neighborhood extraction + improvements
Supports: random, cycle, and fixed neighborhood modes
"""

import torch
import torch.nn as nn
import torchvision
import random
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import re
import os

DEBUG = os.environ.get('DEBUG_PSEMIX', '0') == '1'

ALLOWED_NEIGHBORHOODS = [1, 3, 5, 7, 9, 11]


class AttentionPooling(nn.Module):
    """Gated attention MIL pooling (Ilse et al. 2018)"""
    def __init__(self, in_features: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.V = nn.Linear(in_features, in_features // 4)
        self.U = nn.Linear(in_features, in_features // 4)
        self.w = nn.Linear(in_features // 4, num_heads)
    
    def forward(self, x: torch.Tensor, temperature: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        A = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        attn_weights = self.w(A)
        attn_weights = torch.softmax(attn_weights / temperature, dim=1)
        pooled = torch.einsum('bnh,bnf->bhf', attn_weights, x)
        return pooled, attn_weights


class AttentionMILModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_heads: int = 4,
        attention_temp: float = 0.5,
        mammoth = None,
        mildropout = None
    ) -> None:
        super().__init__()
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        feature_dim = 1280
        
        self.use_mammoth = mammoth is not None
        if self.use_mammoth:
            self.mammoth = mammoth
            self.attention_in_dim = 512
        else:
            self.patch_embed = nn.Linear(feature_dim, feature_dim)
            self.attention_in_dim = feature_dim
        
        self.use_mildropout = mildropout is not None
        if self.use_mildropout:
            self.mildropout = mildropout
        
        self.attention_pool = AttentionPooling(self.attention_in_dim, num_heads)
        self.attention_temp = attention_temp
        
        pooled_dim = self.attention_in_dim * num_heads
        self.head_proj = nn.Linear(pooled_dim, pooled_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(pooled_dim, num_classes)
        )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_crops = x.shape[:2]
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)
        if not self.use_mammoth:
            x = self.patch_embed(x)
        return x
    
    def forward_with_features(
        self,
        features: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_crops, feat_dim = features.shape
        x = features
        
        if DEBUG:
            print(f"[DEBUG] features={features.shape}, mammoth={self.use_mammoth}")
        
        if self.use_mammoth:
            x = self.mammoth(x)
            x = x.mean(dim=1)
            x = x.unsqueeze(1).expand(-1, num_crops, -1)
        
        if self.use_mildropout:
            x = self.mildropout(x)
        
        pooled, attn_weights = self.attention_pool(x, temperature=self.attention_temp)
        pooled = pooled.reshape(batch_size, -1)
        pooled = self.head_proj(pooled)
        logits = self.classifier(pooled)
        
        if logits.dim() != 2:
            raise ValueError(f"logits is not 2D! shape={logits.shape}")
        
        if return_attention:
            return logits, attn_weights
        return logits
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_crops = x.shape[:2]
        features = self.extract_features(x)
        
        if self.use_mammoth:
            x = self.mammoth(features)
            x = x.mean(dim=1)
            x = x.unsqueeze(1).expand(-1, num_crops, -1)
        else:
            x = features
        
        if self.use_mildropout:
            x = self.mildropout(x)
        
        pooled, attn_weights = self.attention_pool(x, temperature=self.attention_temp)
        pooled = pooled.reshape(batch_size, -1)
        pooled = self.head_proj(pooled)
        output = self.classifier(pooled)
        
        if return_attention:
            return output, attn_weights
        return output


def compute_valid_centers(grid_size: int, neighborhood: int) -> int:
    """Compute number of valid center positions for a given neighborhood"""
    return (grid_size - neighborhood + 1) ** 2


def compute_total_crops(grid_size: int, neighborhood: int) -> int:
    """Compute total crops per image for a given neighborhood"""
    valid_centers = compute_valid_centers(grid_size, neighborhood)
    return valid_centers * (neighborhood ** 2)


class MultiCropDataset(Dataset):
    """Flexible neighborhood crop extraction for MIL"""
    
    def __init__(
        self,
        image_paths: list,
        labels: list,
        plate_well_map: dict,
        crop_size: int = 224,
        grid_size: int = 12,
        neighborhood: int = 5,
        augment: bool = True,
        seed: int = 42,
        epoch: int = 0,
        # NEW ARGUMENTS
        random_neighborhood: bool = False,
        neighborhood_range: tuple = (1, 11),
        cycle_neighborhood: bool = False,
        cycle_order: list = None,
        max_positions: int = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        self.single_crop = False
        
        # New flexibility options
        self.random_neighborhood = random_neighborhood
        self.neighborhood_range = neighborhood_range
        self.cycle_neighborhood = cycle_neighborhood
        self.max_positions = max_positions
        
        # Default cycle order: 1, 3, 5, 7, 9, 11
        self.cycle_order = cycle_order or [1, 3, 5, 7, 9, 11]
        
        # Determine effective neighborhood for this epoch
        self.neighborhood = self._get_neighborhood_for_epoch(neighborhood)
        
        # Load sample image to get size
        sample_img = Image.open(image_paths[0]).convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
        # Compute valid positions
        self.valid_positions = self._compute_valid_positions(self.neighborhood)
        
        # Apply max_positions limit if set
        if self.max_positions and self.max_positions < len(self.valid_positions):
            random.seed(seed)
            indices = random.sample(range(len(self.valid_positions)), self.max_positions)
            self.valid_positions = [self.valid_positions[i] for i in sorted(indices)]
        
        # Setup augmentations
        if augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1),
                ], p=0.5),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=4, min_height=16, min_width=16, p=0.3),
            ])
        else:
            self.transform = None
    
    def _get_neighborhood_for_epoch(self, default_neighborhood: int) -> int:
        """Determine neighborhood based on mode"""
        if self.random_neighborhood:
            min_n, max_n = self.neighborhood_range
            valid_options = [n for n in ALLOWED_NEIGHBORHOODS if min_n <= n <= max_n]
            return random.choice(valid_options)
        elif self.cycle_neighborhood:
            return self.cycle_order[self.epoch % len(self.cycle_order)]
        else:
            return default_neighborhood
    
    def _compute_valid_positions(self, neighborhood: int) -> list:
        """Compute valid center positions that have room for full neighborhood"""
        radius = neighborhood // 2
        positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                left = j * self.stride
                top = i * self.stride
                if (left - radius * self.stride >= 0 and 
                    top - radius * self.stride >= 0 and
                    left + self.crop_size + radius * self.stride <= self.image_size and
                    top + self.crop_size + radius * self.stride <= self.image_size):
                    positions.append((left, top))
        return positions
    
    def _get_crop_coordinates(self, center_x: int, center_y: int, neighborhood: int) -> list:
        """Get all crop coordinates for a given center and neighborhood size.
        
        IMPORTANT: We use stride as the OFFSET between crops to ensure no black edges.
        Each crop is centered at center + offset where offset goes from -radius*stride to +radius*stride.
        This ensures we extract actual image content at each position, not padding.
        
        Example for neighborhood=3, stride=226:
        - radius = 1
        - offsets: [-226, 0, +226]  
        - crops extracted at: center-226, center, center+226 (no black edges!)
        """
        radius = neighborhood // 2
        
        if neighborhood == 1:
            return [(center_x, center_y)]
        
        coords = []
        # IMPORTANT: step is 1, but multiply by stride to get actual pixel offsets
        # This ensures we get actual image content, never black edges
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                offset_x = dx * self.stride
                offset_y = dy * self.stride
                coords.append((center_x + offset_x, center_y + offset_y))
        return coords
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for dynamic cropping."""
        self.epoch = epoch
        self.neighborhood = self._get_neighborhood_for_epoch(self.neighborhood)
        self.valid_positions = self._compute_valid_positions(self.neighborhood)
    
    def __getitem__(self, idx: int):
        # Get image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Update neighborhood for this sample if random mode
        if self.random_neighborhood:
            neighborhood = self._get_neighborhood_for_epoch(self.neighborhood)
            self.neighborhood = neighborhood
        else:
            neighborhood = self.neighborhood
        
        # Select random center position
        random.seed(self.seed + self.epoch * len(self.image_paths) + idx)
        center_idx = random.randint(0, len(self.valid_positions) - 1)
        center_x, center_y = self.valid_positions[center_idx]
        
        # Get all crops for this center with current neighborhood
        crop_coords = self._get_crop_coordinates(center_x, center_y, neighborhood)
        
        # Extract crops
        crops = []
        for cx, cy in crop_coords:
            crop = img.crop((cx, cy, cx + self.crop_size, cy + self.crop_size))
            if self.transform:
                # Apply augmentation
                result = self.transform(image=np.array(crop))
                crop_np = result['image'] if isinstance(result, dict) else result
                # Convert to float tensor and normalize
                crop = torch.from_numpy(crop_np).permute(2, 0, 1).float() / 255.0
            else:
                crop = torch.from_numpy(np.array(crop)).permute(2, 0, 1).float() / 255.0
            crops.append(crop)
        
        crops = torch.stack(crops)
        
        return {
            'crops': crops,
            'label': self.labels[idx],
            'neighborhood': neighborhood,
            'num_crops': len(crops)
        }


def get_neighborhood_stats(grid_size: int = 12) -> None:
    """Print neighborhood statistics"""
    print("\n=== Neighborhood Statistics (grid_size=12) ===")
    print(f"{'Neighborhood':<12} {'Valid Centers':<15} {'Crops/Pos':<12} {'Total Crops':<12}")
    print("-" * 50)
    for n in ALLOWED_NEIGHBORHOODS:
        valid = compute_valid_centers(grid_size, n)
        total = compute_total_crops(grid_size, n)
        print(f"{n}x{n:<10} {valid:<15} {n*n:<12} {total:<12}")


if __name__ == "__main__":
    get_neighborhood_stats()

def extract_well_from_filename(filename):
    match = re.search(r'Well(\w\d+)_', filename)
    return match.group(1) if match else None


def get_gene_from_path(img_path, plate_maps):
    dirname = os.path.dirname(img_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        return plate_maps[plate][well]
    return 'WT'



class Mildropout(nn.Module):
    """MIL-specific dropout with kernel smoothing - drops top-k instances during training"""
    def __init__(self, topk: int = 3, kernel: int = 7):
        super().__init__()
        self.topk = topk
        self.kernel = kernel
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training or self.topk == 0:
            return features
        batch_size, num_instances, feat_dim = features.shape
        
        # Compute importance scores
        importance = torch.mean(features, dim=-1, keepdim=True)  # (B, N, 1)
        
        # Apply Gaussian kernel smoothing to importance (if kernel > 1)
        if self.kernel > 1:
            importance = self._gaussian_smooth(importance)
        
        # Find top-k indices to drop (lowest importance = most redundant)
        imp_squeezed = importance.squeeze(-1)  # (B, N)
        _, bottomk_idx = torch.topk(imp_squeezed, min(self.topk, num_instances), dim=1, largest=False)
        
        # Create mask - zero out dropped instances
        mask = torch.ones_like(imp_squeezed)
        mask.scatter_(1, bottomk_idx, 0)
        
        # Normalize by kept count
        kept_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        features = features * mask.unsqueeze(-1)
        features = features / kept_count.unsqueeze(-1)
        
        return features
    
    def _gaussian_smooth(self, importance: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian kernel smoothing to importance scores"""
        batch_size, num_instances, _ = importance.shape
        
        # Create Gaussian kernel matrix
        sigma = self.kernel / 6.0
        x = torch.arange(num_instances, device=importance.device, dtype=importance.dtype)
        center = num_instances // 2
        x_centered = x.unsqueeze(1) - x.unsqueeze(0)  # (N, N)
        kernel = torch.exp(-0.5 * (x_centered / sigma) ** 2)  # (N, N)
        kernel = kernel / kernel.sum(dim=1, keepdim=True)  # normalize
        
        # Apply kernel: (B, N, N) @ (B, N, 1) -> (B, N, 1)
        imp_flat = importance.squeeze(-1)  # (B, N)
        smoothed = torch.bmm(kernel.unsqueeze(0).expand(batch_size, -1, -1), imp_flat.unsqueeze(-1)).squeeze(-1)
        
        return smoothed.unsqueeze(-1)


