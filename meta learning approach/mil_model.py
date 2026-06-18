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
from typing import Optional

class AttentionPooling(nn.Module):
    """Gated attention MIL pooling (Ilse et al. 2018, Eq 9 - with gating)"""
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


class SimpleAttentionPooling(nn.Module):
    """Simple attention MIL pooling (Ilse et al. 2018, Eq 8 - no gating)"""
    def __init__(self, in_features: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = 256
        
        self.V = nn.Linear(in_features, self.hidden_dim)
        self.w = nn.Linear(self.hidden_dim, num_heads)
    
    def forward(self, x: torch.Tensor, temperature: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
        A = torch.tanh(self.V(x))
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
        dropout: float = 0.5,
        use_contrastive: bool = False,
        projection_dim: int = 256,
        num_channels: int = 3,
        pretrained: str = "imagenet",
        backbone: str = "efficientnet_b0",
        pooling: str = "attention"
    ) -> None:
        super().__init__()
        self.backbone_type = backbone
        self.pooling = pooling
        
        # Select backbone
        if backbone == "mobilenet_v3_small":
            base_model = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
            # MobileNetV3 Small output features: 576
            self.feature_dim = 576
            self.backbone = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        elif backbone == "mobilenet_v2":
            base_model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
            # MobileNetV2 output features: 1280
            self.feature_dim = 1280
            self.backbone = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        else:  # efficientnet_b0 (default)
            base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
            # EfficientNet-B0 output features: 1280
            self.feature_dim = 1280
            self.backbone = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        # Modify first conv layer for num_channels with proper weight transfer
        if num_channels == 1 and pretrained == 'imagenet':
            if backbone == "mobilenet_v3_small":
                # MobileNetV3 Small first conv: Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
                original_conv = base_model.features[0][0]
                original_weights = original_conv.weight.data  # [16, 3, 3, 3]
                new_weights = original_weights.sum(dim=1, keepdim=True)  # [16, 1, 3, 3]
                self.backbone[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
                self.backbone[0][0].weight.data = new_weights
            elif backbone == "mobilenet_v2":
                # MobileNetV2 first conv: Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
                original_conv = base_model.features[0][0]
                original_weights = original_conv.weight.data  # [32, 3, 3, 3]
                new_weights = original_weights.sum(dim=1, keepdim=True)  # [32, 1, 3, 3]
                self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
                self.backbone[0][0].weight.data = new_weights
            else:  # efficientnet_b0
                # Transfer ImageNet RGB weights to single channel (sum over channels)
                original_conv = base_model.features[0][0]
                original_weights = original_conv.weight.data  # [32, 3, 3, 3]
                new_weights = original_weights.sum(dim=1, keepdim=True)  # [32, 1, 3, 3]
                self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
                self.backbone[0][0].weight.data = new_weights
        elif num_channels == 1 and pretrained == 'micronet':
            # Load NASA MicroNet pretrained weights (ImageNet -> MicroNet)
            # Micronet only available for efficientnet_b0, use ImageNet for other backbones
            if backbone != 'efficientnet_b0':
                print(f"WARNING: Micronet pretrained weights only available for efficientnet_b0, falling back to ImageNet for {backbone}")
                pretrained = 'imagenet'
                original_conv = base_model.features[0][0]
                original_weights = original_conv.weight.data
                new_weights = original_weights.sum(dim=1, keepdim=True)
                self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
                self.backbone[0][0].weight.data = new_weights
            else:
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
        
        if pooling == 'simple_attention':
            self.attention_pool = SimpleAttentionPooling(self.feature_dim, num_heads)
            print(f"MILEncoder: Using SimpleAttentionPooling (no gating)")
        else:
            self.attention_pool = AttentionPooling(self.feature_dim, num_heads)
            print(f"MILEncoder: Using Gated AttentionPooling")
        self.attention_temp = attention_temp
        
        self.head_proj = nn.Linear(self.feature_dim * num_heads, self.feature_dim)
        
        if use_contrastive:
            self.contrastive_head = ContrastiveEncoder(self.feature_dim, projection_dim)
        
        # Simple classifier: single FC layer with dropout=0.5
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
        
# Apply pooling based on self.pooling
        if self.pooling == 'mean':
            pooled = crop_embeddings.mean(dim=1)
            attn_weights = None
            pooled_feat = pooled.clone()
            output = self.classifier(pooled)
            results = [output]
            if return_attention:
                results.append(attn_weights)
            if return_crop_embeddings:
                results.append(crop_embeddings)
            if return_pooled_embeddings:
                results.append(pooled_feat)
            if return_instance_logits:
                batch_size, num_crops = crop_embeddings.shape[:2]
                instance_logits = self.classifier(crop_embeddings)
                instance_logits = instance_logits.view(batch_size, num_crops, -1)
                results.append(instance_logits)
            return results[0] if len(results) == 1 else tuple(results)
        elif self.pooling == 'max':
            pooled, _ = crop_embeddings.max(dim=1)
            attn_weights = None
            pooled_feat = pooled.clone()
            output = self.classifier(pooled)
            results = [output]
            if return_attention:
                results.append(attn_weights)
            if return_crop_embeddings:
                results.append(crop_embeddings)
            if return_pooled_embeddings:
                results.append(pooled_feat)
            if return_instance_logits:
                batch_size, num_crops = crop_embeddings.shape[:2]
                instance_logits = self.classifier(crop_embeddings)
                instance_logits = instance_logits.view(batch_size, num_crops, -1)
                results.append(instance_logits)
            return results[0] if len(results) == 1 else tuple(results)
        else:  # attention (default)
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
            # Instance-level predictions: apply simple classifier to each crop
            batch_size, num_crops = crop_embeddings.shape[:2]
            instance_logits = self.classifier(crop_embeddings)
            instance_logits = instance_logits.view(batch_size, num_crops, -1)
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
    def __init__(self, num_classes, num_heads=4, attention_temp=0.5, dropout=0.5, num_channels=1, pretrained="imagenet", backbone="efficientnet_b0", pooling="attention"):
        super().__init__()
        self.backbone_type = backbone
        self.pooling = pooling
        
        # Select backbone
        if backbone == "mobilenet_v3_small":
            base_model = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1')
            self.feature_dim = 576
            self.backbone = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        elif backbone == "mobilenet_v2":
            base_model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
            self.feature_dim = 1280
            self.backbone = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        else:  # efficientnet_b0
            base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
            self.feature_dim = 1280
            self.backbone = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        
        # Modify first conv layer for num_channels with proper weight transfer
        if num_channels == 1 and pretrained == 'imagenet':
            if backbone == "mobilenet_v3_small":
                original_conv = base_model.features[0][0]
                original_weights = original_conv.weight.data  # [16, 3, 3, 3]
                new_weights = original_weights.sum(dim=1, keepdim=True)  # [16, 1, 3, 3]
                self.backbone[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
                self.backbone[0][0].weight.data = new_weights
            elif backbone == "mobilenet_v2":
                # MobileNetV2 first conv: Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
                original_conv = base_model.features[0][0]
                original_weights = original_conv.weight.data  # [32, 3, 3, 3]
                new_weights = original_weights.sum(dim=1, keepdim=True)  # [32, 1, 3, 3]
                self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
                self.backbone[0][0].weight.data = new_weights
            else:  # efficientnet_b0
                original_conv = base_model.features[0][0]
                original_weights = original_conv.weight.data  # [32, 3, 3, 3]
                new_weights = original_weights.sum(dim=1, keepdim=True)  # [32, 1, 3, 3]
                self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
                self.backbone[0][0].weight.data = new_weights
        elif num_channels == 1 and pretrained == 'micronet':
            # Load NASA MicroNet pretrained weights (ImageNet -> MicroNet)
            # Micronet only available for efficientnet_b0, use ImageNet for other backbones
            if backbone != 'efficientnet_b0':
                print(f"WARNING: Micronet pretrained weights only available for efficientnet_b0, falling back to ImageNet for {backbone}")
                pretrained = 'imagenet'
                original_conv = base_model.features[0][0]
                original_weights = original_conv.weight.data
                new_weights = original_weights.sum(dim=1, keepdim=True)
                self.backbone[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
                self.backbone[0][0].weight.data = new_weights
            else:
                print("Loading NASA MicroNet pretrained weights (ImageNet -> MicroNet) for AttentionMILModel...")
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
            print("MicroNet weights loaded and transferred successfully for AttentionMILModel!")
        else:
            self.backbone[0][0] = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        feature_dim = 1280
        
        if pooling == 'simple_attention':
            self.attention_pool = SimpleAttentionPooling(feature_dim, num_heads)
            print(f"AttentionMILModel: Using SimpleAttentionPooling (no gating)")
        else:
            self.attention_pool = AttentionPooling(feature_dim, num_heads)
            print(f"AttentionMILModel: Using Gated AttentionPooling")
        self.attention_temp = attention_temp
        
        self.head_proj = nn.Linear(feature_dim * num_heads, feature_dim)
        
        # Simple classifier: single FC layer with dropout=0.5
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x, return_attention=False):
        batch_size, num_crops = x.shape[:2]
        
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)
        
        # Apply pooling based on self.pooling
        if self.pooling == 'mean':
            # Mean pooling
            pooled = x.mean(dim=1)
            attn_weights = None
            pooled = self.classifier_dropout(pooled)
            output = self.classifier(pooled)
            if return_attention:
                return output, attn_weights
            return output
        elif self.pooling == 'max':
            # Max pooling
            pooled, _ = x.max(dim=1)
            attn_weights = None
            pooled = self.classifier_dropout(pooled)
            output = self.classifier(pooled)
            if return_attention:
                return output, attn_weights
            return output
        else:  # attention (default)
            pooled, attn_weights = self.attention_pool(x, temperature=self.attention_temp)
            pooled = pooled.reshape(batch_size, -1)
            pooled = self.head_proj(pooled)
            output = self.classifier(pooled)
            if return_attention:
                return output, attn_weights
            return output


class MultiCropDataset(Dataset):
    """Cycle-based crop extraction with configurable neighborhood for MIL
    
    extraction_mode options:
    - 'neighborhood': Extract N×N crops around each valid center position (current behavior)
    - 'raster': Extract all crops in tiling grid as a single bag (like AI4AB/trial_daniel)
    """
    
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
        num_channels: int = 1,
        extraction_mode: str = 'neighborhood',
        raster_crop_size: int = 500,
        raster_resize_size: int = 256,
        raster_num_crops: int = 25,
        raster_grid_size: int = 2500
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
        self.extraction_mode = extraction_mode
        self.raster_crop_size = raster_crop_size
        self.raster_resize_size = raster_resize_size
        self.raster_num_crops = raster_num_crops
        self.raster_grid_size = raster_grid_size
        
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
        
        if extraction_mode == 'raster':
            # New raster mode: centered 1500x1500 grid with 3x3 crops of 500x500 resized to 256
            num_crops_side = int(np.sqrt(raster_num_crops))  # 3 for 9 crops
            positions = []
            
            # Calculate the centered grid region on the image
            grid_left = (w - raster_grid_size) // 2
            grid_top = (h - raster_grid_size) // 2
            
            # Calculate spacing within the grid
            spacing = raster_grid_size / num_crops_side  # 1500/3 = 500
            
            for i in range(num_crops_side):
                for j in range(num_crops_side):
                    # Center of each crop position within the grid
                    center_x = grid_left + (j + 0.5) * spacing
                    center_y = grid_top + (i + 0.5) * spacing
                    
                    # Top-left corner of 500x500 crop
                    left = int(center_x - raster_crop_size / 2)
                    top = int(center_y - raster_crop_size / 2)
                    
                    # Ensure within bounds
                    left = max(0, min(left, w - raster_crop_size))
                    top = max(0, min(top, h - raster_crop_size))
                    
                    positions.append((left, top))
            
            self.all_positions = positions
            self.positions = positions
            self.num_neighbors = len(positions) - 1
            print(f"MIL (raster): {len(positions)} crops ({num_crops_side}x{num_crops_side} grid), grid={raster_grid_size}, crop_size={raster_crop_size}, resize={raster_resize_size}")
        else:
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
            self.all_positions = []
            self.num_neighbors = neighborhood * neighborhood - 1
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
        
        if extraction_mode == 'raster':
            total_crops = len(positions)
            print(f"MIL (raster): {total_crops} crops in {len(positions)} positions")
        else:
            print(f"MIL (neighborhood): {len(positions)} positions, {self.neighborhood}x{self.neighborhood}={self.num_neighbors + 1} crops/image")
    
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
        
        if self.extraction_mode == 'raster':
            center_left = self.positions[0][0] if self.positions else 0
            center_top = self.positions[0][1] if self.positions else 0
            self.epoch_centers = {i: (center_left, center_top) for i in range(num_images)}
        else:
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
        
        EXACT same approach as trial_daniel:
        - For uint16: sample / (2^16 - 1) = sample / 65535
        - For uint8: sample / (2^8 - 1) = sample / 255
        Returns normalized array in [0, 1] range as float32 (same as trial_daniel).
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
        
        # EXACT same normalization as trial_daniel (dataset.py line 207)
        # sample = torch.FloatTensor(sample / (2 ** self.bit_depth - 1))
        if img_array.dtype == np.uint16:
            # uint16: divide by (2^16 - 1) = 65535
            img_array = img_array.astype(np.float32) / 65535.0
        elif img_array.dtype == np.uint8:
            # uint8: divide by (2^8 - 1) = 255
            img_array = img_array.astype(np.float32) / 255.0
        elif img_array.dtype == np.float32 or img_array.dtype == np.float64:
            # Float - already in [0,1] range
            img_array = img_array.astype(np.float32)
        
        # Convert to PIL Image - use L for grayscale (1 channel), RGB for 3 channels
        # The transform pipeline will apply Normalize(mean=0.5, std=0.5) to convert to [-1, 1]
        if self.num_channels == 1:
            image = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
        else:
            image = Image.fromarray((img_array * 255).astype(np.uint8), mode='L').convert('RGB')
        
        return image
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = self._load_image(img_path)
        
        center_left, center_top = self.epoch_centers[idx]
        
        if self.single_crop:
            crop = image.crop((center_left, center_top, center_left + self.crop_size, center_top + self.crop_size))
            crop = np.array(crop)
            crop = self.transform(image=crop)['image']
            crops = crop.unsqueeze(0)
        elif self.extraction_mode == 'raster':
            # New raster mode: use raster_crop_size and raster_resize_size
            jitter_range = self.raster_crop_size // 8 if self.augment else 0
            crops_list = []
            
            for (left, top) in self.all_positions:
                if self.augment:
                    jitter_x = random.randint(-jitter_range, jitter_range)
                    jitter_y = random.randint(-jitter_range, jitter_range)
                else:
                    jitter_x = jitter_y = 0
                
                left = left + jitter_x
                top = top + jitter_y
                # Ensure within bounds using raster_crop_size
                left = max(0, min(left, self.image_size - self.raster_crop_size))
                top = max(0, min(top, self.image_size - self.raster_crop_size))
                
                # Extract 500x500 crop
                crop = image.crop((left, top, left + self.raster_crop_size, top + self.raster_crop_size))
                # Resize to 256x256
                crop = crop.resize((self.raster_resize_size, self.raster_resize_size), Image.BILINEAR)
                crop = np.array(crop)
                crop = self.transform(image=crop)['image']
                crops_list.append(crop)
            
            num_crops = len(self.all_positions)
            if self.augment:
                # Only shuffle if we have more than 9 crops (old raster mode)
                if num_crops > self.raster_num_crops:
                    perm = list(range(num_crops))
                    random.shuffle(perm)
                    crops_list = [crops_list[i] for i in perm]
            
            crops = torch.stack(crops_list)
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
    Uses composite keys (drug_A01 or mutant_A01) for both data mode.
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
    if not well:
        return None
    
    # Determine if this is drug or mutant data based on path
    if '/mutants_data/' in path_lower or '\\mutants_data\\' in path_lower:
        source_prefix = 'mutant_'
    else:
        # Default to drug (for drug mode or fallback)
        source_prefix = 'drug_'
    
    # Use composite key: drug_A01 or mutant_A01
    composite_well = f"{source_prefix}{well}"
    if composite_well and plate_key in plate_maps and composite_well in plate_maps[plate_key]:
        return plate_maps[plate_key][composite_well]
    
    return None
