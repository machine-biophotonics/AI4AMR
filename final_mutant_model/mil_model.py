"""
MIL with cycle-based crop extraction + configurable neighborhood + Contrastive Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pretrained_microscopy_models as pmm
import torch.utils.model_zoo as model_zoo
import random
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import re
import os


class AttentionPooling(nn.Module):
    """Gated attention MIL pooling (Ilse et al. 2018)"""
    def __init__(self, in_features: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = 256
        
        self.V = nn.Linear(in_features, self.hidden_dim)
        self.U = nn.Linear(in_features, self.hidden_dim)
        self.w = nn.Linear(self.hidden_dim, num_heads)
    
    def forward(self, x: torch.Tensor, temperature: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        A = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        attn_weights = self.w(A)
        attn_weights = torch.softmax(attn_weights / temperature, dim=1)
        pooled = torch.einsum('bnh,bnf->bhf', attn_weights, x)
        return pooled, attn_weights


class ContrastiveEncoder(nn.Module):
    """Encoder for contrastive learning"""
    def __init__(self, feature_dim: int = 1280, projection_dim: int = 256) -> None:
        super().__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(x)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.normalize(x, dim=1)


class MILEncoder(nn.Module):
    """MIL encoder with optional contrastive head"""
    def __init__(
        self,
        num_classes: int,
        num_heads: int = 4,
        attention_temp: float = 0.5,
        dropout: float = 0.2,
        use_contrastive: bool = False,
        projection_dim: int = 256,
        num_channels: int = 3, pretrained: str = "imagenet"
    ) -> None:
        super().__init__()
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        # Modify first conv layer for num_channels with proper weight transfer
        if num_channels == 1 and pretrained == 'imagenet':
            # Transfer ImageNet RGB weights to single channel (sum over channels)
            original_conv = base_model.features[0][0]
            original_weights = original_conv.weight.data  # [32, 3, 3, 3]
            new_weights = original_weights.sum(dim=1, keepdim=True)  # [32, 1, 3, 3]
            self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.backbone[0][0].weight.data = new_weights
        elif num_channels == 1 and pretrained == 'micronet':
            # Load NASA MicroNet pretrained weights (ImageNet -> MicroNet)
            print("Loading NASA MicroNet pretrained weights (ImageNet -> MicroNet)...")
            url = pmm.util.get_pretrained_microscopynet_url('efficientnet-b0', 'image-micronet')
            state_dict = model_zoo.load_url(url)
            # Remove module. prefix if present and features. prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('features.'):
                    new_key = k.replace('features.', '')
                    new_state_dict[new_key] = v
            # Load weights into base model
            base_model.load_state_dict(new_state_dict, strict=False)
            # Transfer first conv layer weights
            original_conv = base_model.features[0][0]
            original_weights = original_conv.weight.data
            new_weights = original_weights.sum(dim=1, keepdim=True)
            self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.backbone[0][0].weight.data = new_weights
            print("MicroNet weights loaded and transferred successfully!")
        else:
            self.backbone[0][0] = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.feature_dim = 1280
        self.use_contrastive = use_contrastive
        
        self.attention_pool = AttentionPooling(self.feature_dim, num_heads)
        self.attention_temp = attention_temp
        
        self.head_proj = nn.Linear(self.feature_dim * num_heads, self.feature_dim)
        
        if use_contrastive:
            self.contrastive_head = ContrastiveEncoder(self.feature_dim, projection_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_embedding: bool = False,
        return_crop_embeddings: bool = False,
        return_pooled_embeddings: bool = False,
        return_instance_logits: bool = False
    ) -> tuple | torch.Tensor:
        batch_size, num_crops = x.shape[:2]
        
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        crop_embeddings = x.view(batch_size, num_crops, -1)
        
        pooled, attn_weights = self.attention_pool(crop_embeddings, temperature=self.attention_temp)
        pooled = pooled.reshape(batch_size, -1)
        pooled = self.head_proj(pooled)
        
        if return_embedding and self.use_contrastive:
            embedding = self.contrastive_head.get_embedding(pooled)
            if return_attention:
                return embedding, attn_weights
            return embedding
        
        output = self.classifier(pooled)
        
        results = [output]
        if return_attention:
            results.append(attn_weights)
        if return_crop_embeddings:
            results.append(crop_embeddings)
        if return_pooled_embeddings:
            results.append(pooled)
        if return_instance_logits:
            # Instance-level predictions: apply classifier to each crop
            instance_logits = self.classifier(crop_embeddings)
            results.append(instance_logits)
        
        return results[0] if len(results) == 1 else tuple(results)
    
    def get_contrastive_embedding(self, x):
        batch_size, num_crops = x.shape[:2]
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        
        if num_crops == 1:
            x = x.squeeze(1)
            x = self.head_proj(x)
            return self.contrastive_head.get_embedding(x)
        
        x = x.view(batch_size, num_crops, -1)
        pooled, _ = self.attention_pool(x, temperature=self.attention_temp)
        pooled = pooled.reshape(batch_size, -1)
        pooled = self.head_proj(pooled)
        
        return self.contrastive_head.get_embedding(pooled)
    
    def get_backbone_features(self, x):
        batch_size, num_crops = x.shape[:2]
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        return x
    
    def get_projected_features(self, x):
        if len(x.shape) == 5:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            x = self.backbone(x)
            x = x.view(B, N, -1)
            pooled, _ = self.attention_pool(x, temperature=self.attention_temp)
            pooled = pooled.reshape(B, -1)
        else:
            x = self.backbone(x)
            pooled = x
        
        pooled = self.head_proj(pooled)
        return pooled
    
    def get_mil_embeddings(self, x):
        batch_size, num_crops = x.shape[:2]
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)
        
        pooled, _ = self.attention_pool(x, temperature=self.attention_temp)
        pooled = pooled.reshape(batch_size, -1)
        
        pooled = self.head_proj(pooled)
        
        return pooled
    
    def get_supcon_embeddings(self, x):
        batch_size, num_crops = x.shape[:2]
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        return x.view(batch_size, num_crops, -1)
    
    def get_contrastive_embedding(self, x):
        pooled = self.get_projected_features(x)
        return self.contrastive_head.get_embedding(pooled)
    
    def get_backbone_features(self, x):
        x = self.backbone(x)
        return x


class AttentionMILModel(nn.Module):
    def __init__(self, num_classes, num_heads=4, attention_temp=0.5, dropout=0.2):
        super().__init__()
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        feature_dim = 1280
        
        self.attention_pool = AttentionPooling(feature_dim, num_heads)
        self.attention_temp = attention_temp
        
        self.head_proj = nn.Linear(feature_dim * num_heads, feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        batch_size, num_crops = x.shape[:2]
        
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)
        
        pooled, attn_weights = self.attention_pool(x, temperature=self.attention_temp)
        
        pooled = pooled.reshape(batch_size, -1)
        pooled = self.head_proj(pooled)
        
        output = self.classifier(pooled)
        
        if return_attention:
            return output, attn_weights
        return output


class MultiCropDataset(Dataset):
    """Cycle-based crop extraction with configurable neighborhood for MIL"""
    
    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        plate_well_map: dict | None,
        crop_size: int = 224,
        grid_size: int = 12,
        neighborhood: int = 3,
        augment: bool = True,
        seed: int = 42,
        epoch: int = 0,
        num_channels: int = 1
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.neighborhood = neighborhood
        self.augment = augment
        self.seed = seed
        self.epoch = epoch
        self.single_crop = False
        self.num_channels = num_channels
        
        sample_img = Image.open(image_paths[0])
        # Convert to grayscale ('L') for 1-channel, or RGB for 3-channel
        if num_channels == 1:
            sample_img = sample_img.convert('L')
        else:
            sample_img = sample_img.convert('RGB')
        w, h = sample_img.size
        self.image_size = w
        
        stride = (w - crop_size) // (grid_size - 1)
        self.stride = stride
        
        half_n = neighborhood // 2
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * stride
                top = i * stride
                if left + crop_size <= w and top + crop_size <= h:
                    can_left = left - half_n * stride >= 0
                    can_right = left + half_n * stride + crop_size <= w
                    can_top = top - half_n * stride >= 0
                    can_bottom = top + half_n * stride + crop_size <= h
                    if can_left and can_right and can_top and can_bottom:
                        positions.append((left, top))
        
        self.positions = positions
        self.num_neighbors = neighborhood * neighborhood - 1
        
        # Normalization for 1-channel vs 3-channel
        if num_channels == 1:
            norm_mean = [0.5]
            norm_std = [0.5]
        else:
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
        
        if augment:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.3),
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2(),
            ])
        
        print(f"MIL: {len(positions)} positions, {self.neighborhood}x{self.neighborhood}={self.num_neighbors + 1} crops/image")
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        num_pos = len(self.positions)
        num_images = len(self.image_paths)
        
        if not self.augment:
            center_left = (self.image_size - self.crop_size) // 2
            center_top = (self.image_size - self.crop_size) // 2
            self.epoch_centers = {i: (center_left, center_top) for i in range(num_images)}
            self.single_crop = False
            return
        
        cycle = epoch // num_pos
        pos_in_cycle = epoch % num_pos
        rng = random.Random(self.seed + cycle)
        shuffled = self.positions.copy()
        rng.shuffle(shuffled)
        
        self.epoch_centers = {}
        for idx in range(num_images):
            assigned_idx = (idx + pos_in_cycle) % num_pos
            self.epoch_centers[idx] = shuffled[assigned_idx]
        
        self.single_crop = False
    
    def _load_image(self, img_path: str) -> Image.Image:
        """Load image with proper handling for microscopy images.
        
        Exact approach from trial_daniel and research papers:
        1. Load 16-bit TIFF with tifffile
        2. Clip to 0.1-99.9 percentile to remove outliers
        3. Use skimage.exposure.rescale_intensity to convert to uint8
        4. Output as grayscale (mode='L') for num_channels=1
        """
        # Try tifffile first for 16-bit TIFF, fallback to PIL
        try:
            import tifffile
            img_array = tifffile.imread(img_path)
        except ImportError:
            img_array = np.array(Image.open(img_path))
        except Exception:
            img_array = np.array(Image.open(img_path))
        
        # Handle multi-channel or single-channel
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 0]  # Take first channel if multi-channel
        
        # Normalize based on dtype - exact approach from trial_daniel
        try:
            from skimage import exposure
            clip = True
            if img_array.dtype == np.uint16:
                # Exact trial_daniel approach: clip to 0.1-99.9 percentile
                p_bot, p_top = np.percentile(img_array, 0.1), np.percentile(img_array, 99.9)
                img_array = np.clip(img_array, p_bot, p_top)
                # Use skimage.exposure.rescale_intensity to convert to uint8
                img_array = exposure.rescale_intensity(img_array, out_range='uint8')
            elif img_array.dtype == np.uint8:
                # Already uint8 - just ensure proper range
                img_array = exposure.rescale_intensity(img_array, out_range='uint8')
            elif img_array.dtype == np.float32 or img_array.dtype == np.float64:
                # Float - clip to [0,1] then convert to uint8
                img_array = np.clip(img_array, 0, 1)
                img_array = (img_array * 255).astype(np.uint8)
        except ImportError:
            # Fallback if skimage not available - manual normalization
            if img_array.dtype == np.uint16:
                p1 = np.percentile(img_array, 0.1)
                p99 = np.percentile(img_array, 99.9)
                img_array = np.clip(img_array, p1, p99)
                img_array = ((img_array - p1) / (p99 - p1 + 1e-8) * 255).astype(np.uint8)
            elif img_array.dtype == np.uint8:
                pass  # Already uint8
            elif img_array.dtype in [np.float32, np.float64]:
                img_array = np.clip(img_array, 0, 1)
                img_array = (img_array * 255).astype(np.uint8)
        
        # Convert to PIL Image - use L for grayscale (1 channel), RGB for 3 channels
        if self.num_channels == 1:
            image = Image.fromarray(img_array, mode='L')
        else:
            # Convert grayscale to RGB by stacking
            image = Image.fromarray(img_array, mode='L').convert('RGB')
        
        return image
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        # Load image with proper handling for 16-bit microscopy images
        image = self._load_image(img_path)
        
        center_left, center_top = self.epoch_centers[idx]
        
        if self.single_crop:
            crop = image.crop((center_left, center_top, center_left + self.crop_size, center_top + self.crop_size))
            crop = np.array(crop)
            crop = self.transform(image=crop)['image']
            crops = crop.unsqueeze(0)
        else:
            jitter_range = self.stride // 4
            crops_list = []
            half_n = self.neighborhood // 2
            
            for di in range(-half_n, half_n + 1):
                for dj in range(-half_n, half_n + 1):
                    if self.augment:
                        jitter_x = random.randint(-jitter_range, jitter_range)
                        jitter_y = random.randint(-jitter_range, jitter_range)
                    else:
                        jitter_x = jitter_y = 0
                    
                    left = center_left + dj * self.stride + jitter_x
                    top = center_top + di * self.stride + jitter_y
                    left = max(0, min(left, self.image_size - self.crop_size))
                    top = max(0, min(top, self.image_size - self.crop_size))
                    crop = image.crop((left, top, left + self.crop_size, top + self.crop_size))
                    crop = np.array(crop)
                    crop = self.transform(image=crop)['image']
                    crops_list.append(crop)
            
            num_crops = self.neighborhood * self.neighborhood
            if self.augment:
                perm = list(range(num_crops))
                random.shuffle(perm)
                crops_list = [crops_list[i] for i in perm]
            
            crops = torch.stack(crops_list)
        
        return crops, self.labels[idx]


def extract_well_from_filename(filename: str) -> str | None:
    match = re.search(r'Well(\w\d+)_', filename)
    return match.group(1) if match else None


def get_gene_from_path(img_path: str, plate_maps: dict) -> str:
    """Extract gene/mutant label from image path using plate_maps.
    Handles both mutant mode (data/Plate_X) and drug mode (Drugs_Data/PX).
    """
    path_lower = img_path.lower()
    
    # Determine plate key (P1, P2, etc.)
    for plate_num in range(1, 7):
        if f'/p{plate_num}/' in path_lower or f'\\p{plate_num}\\ ' in path_lower:
            plate_key = f'P{plate_num}'
            break
    else:
        # Try old format: .../Plate_X/...
        dirname = os.path.dirname(img_path)
        plate = os.path.basename(dirname)
        if 'plate' in plate.lower():
            plate_key = f"P{plate.split('_')[-1]}"
        else:
            return 'WT'
    
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if well and plate_key in plate_maps and well in plate_maps[plate_key]:
        return plate_maps[plate_key][well]
    return 'WT'