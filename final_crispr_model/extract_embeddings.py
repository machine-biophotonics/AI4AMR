#!/usr/bin/env python3
"""
Extract image embeddings using the EfficientNet backbone + MIL attention pooling.

For each image:
1. Extract 100 positions × 9 crops (3x3 neighborhood) = 900 crops total
2. Process 9 crops through backbone → attention pooling → 1280-dim position embedding
3. Average 100 position embeddings → 1280-dim image embedding

This gives TRUE image-level visual embeddings, not crop-level.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from tqdm import tqdm
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
    PLATE_WELL_ID = json.load(f)

with open(os.path.join(SCRIPT_DIR, 'classes.txt'), 'r') as f:
    classes = {}
    for line in f:
        idx, name = line.strip().split(',', 1)
        classes[int(idx)] = name

idx_to_label = classes
label_to_idx = {v: k for k, v in classes.items()}


class EmbeddingExtractor(nn.Module):
    """Extract embeddings using EfficientNet backbone + attention pooling."""
    
    def __init__(self, num_classes, num_heads=4, attention_temp=0.5):
        super().__init__()
        base_model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        feature_dim = 1280
        
        self.V = nn.Linear(feature_dim, feature_dim // 4)
        self.U = nn.Linear(feature_dim, feature_dim // 4)
        self.w = nn.Linear(feature_dim // 4, num_heads)
        
        self.num_heads = num_heads
        self.attention_temp = attention_temp
        self.head_proj = nn.Linear(feature_dim * num_heads, feature_dim)
    
    def attention_pool(self, x):
        A = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        attn_weights = torch.softmax(self.w(A) / self.attention_temp, dim=1)
        pooled = torch.einsum('bnh,bnf->bhf', attn_weights, x)
        return pooled, attn_weights
    
    def load_from_mil_model(self, mil_state_dict):
        """Load weights from trained MIL model."""
        state_dict = self.state_dict()
        for key in mil_state_dict:
            if key in state_dict:
                state_dict[key] = mil_state_dict[key]
        self.load_state_dict(state_dict)
    
    def forward(self, x):
        """Extract embedding for batch of crops."""
        batch_size, num_crops = x.shape[:2]
        
        x = x.view(batch_size * num_crops, *x.shape[2:])
        x = self.backbone(x)
        x = x.view(batch_size, num_crops, -1)
        
        pooled, attn_weights = self.attention_pool(x)
        pooled = pooled.reshape(batch_size, -1)
        pooled = self.head_proj(pooled)
        
        return pooled, attn_weights


def extract_mil_crops(img_path, crop_size, grid_size):
    """Extract 100 positions (with 9 crops each) matching training."""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    stride_x = (w - crop_size) // (grid_size - 1) if grid_size > 1 else 0
    stride_y = (h - crop_size) // (grid_size - 1) if grid_size > 1 else 0
    
    valid_positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            left = j * stride_x
            top = i * stride_y
            if left + crop_size <= w and top + crop_size <= h:
                can_left = left - stride_x >= 0
                can_right = left + stride_x + crop_size <= w
                can_top = top - stride_y >= 0
                can_bottom = top + stride_y + crop_size <= h
                if can_left and can_right and can_top and can_bottom:
                    valid_positions.append((left, top))
    
    crops = []
    
    for center_x, center_y in valid_positions:
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                left = center_x + dx * stride_x
                top = center_y + dy * stride_y
                
                crop = img.crop((left, top, left + crop_size, top + crop_size))
                crop_np = np.array(crop).astype(np.float32) / 255.0
                crop_np = np.transpose(crop_np, (2, 0, 1))
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                crop_np = (crop_np - mean) / std
                crop_tensor = torch.from_numpy(crop_np).float()
                
                crops.append(crop_tensor)
    
    return crops


def parse_well_from_filename(img_path):
    """Parse well position from image filename."""
    filename = os.path.basename(img_path)
    match = re.search(r'Well([A-H]\d+)', filename)
    if match:
        well = match.group(1)
        row = well[0]
        col = str(int(well[1:]))  # Convert '01' to '1', '02' to '2', etc.
        return row, col
    return None


def get_ground_truth_label(plate, well):
    """Get ground truth label from plate_well_id_path.json."""
    if plate in PLATE_WELL_ID and well:
        row, col = well[0], well[1:]
        if row in PLATE_WELL_ID[plate]:
            if col in PLATE_WELL_ID[plate][row]:
                return PLATE_WELL_ID[plate][row][col].get('id', None)
    return None


def main():
    parser = argparse.ArgumentParser(description='Extract image embeddings for t-SNE')
    parser.add_argument('--fold', type=str, default='P3', help='Fold to process (P1-P6)')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth', help='Checkpoint file')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--attention_temp', type=float, default=0.5, help='Attention temperature')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    crop_size = 224
    grid_size = 12
    num_classes = len(classes)
    
    fold_dir = os.path.join(SCRIPT_DIR, f'fold_{args.fold}')
    checkpoint_path = os.path.join(fold_dir, args.checkpoint)
    image_dir = os.path.join(BASE_DIR, args.fold)
    
    print(f"\n{'='*60}")
    print(f"Processing fold: {args.fold}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  image_dir: {image_dir}")
    print(f"{'='*60}")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    
    model = EmbeddingExtractor(num_classes=num_classes, num_heads=args.num_heads, attention_temp=args.attention_temp)
    model = model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_from_mil_model(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    
    image_paths = sorted(Path(image_dir).glob('*.tif'))
    print(f"Processing {len(image_paths)} images...")
    
    all_results = []
    
    for img_path in tqdm(image_paths, desc="Extracting embeddings"):
        img_path_str = str(img_path)
        
        all_crops = extract_mil_crops(img_path_str, crop_size, grid_size)
        n_positions = len(all_crops) // 9
        
        embeddings_list = []
        attn_weights_list = []
        probs_list = []
        
        for pos_idx in range(n_positions):
            pos_crops = all_crops[pos_idx * 9:(pos_idx + 1) * 9]
            batch_tensors = torch.stack(pos_crops).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding, attn = model(batch_tensors)
                embeddings_list.append(embedding[0].cpu().numpy())
                attn_weights_list.append(attn[0].cpu().numpy())
        
        mean_embedding = np.mean(embeddings_list, axis=0)
        mean_attn = np.mean(attn_weights_list, axis=0)
        
        well = parse_well_from_filename(img_path_str)
        gt_label = get_ground_truth_label(args.fold, well) if well else None
        gt_idx = label_to_idx.get(gt_label, -1) if gt_label else -1
        
        all_results.append({
            'image_name': os.path.basename(img_path),
            'plate': args.fold,
            'well': f"{well[0]}{well[1]}" if well else None,
            'ground_truth_label': gt_label,
            'ground_truth_idx': gt_idx,
            'embedding': mean_embedding.tolist(),
            'attention_weights': mean_attn.tolist(),
            'num_positions': n_positions,
        })
    
    output_path = args.output or os.path.join(fold_dir, 'image_embeddings.csv')
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)
    print(f"\nSaved embeddings to {output_path}")
    print(f"Embedding dimension: {len(all_results[0]['embedding'])}")


if __name__ == '__main__':
    main()