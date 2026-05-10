#!/usr/bin/env python3
"""
Extract embeddings using self-supervised SimCLR model.
Extracts center crop + neighborhood crops and pools using attention.
"""

import os
import glob
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from pathlib import Path

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    
    def forward(self, x, ret_emb=False, ret_features=False):
        bs = x.shape[0]
        nc = x.shape[1]
        
        x = x.reshape(bs * nc, *x.shape[2:]).contiguous()
        f = self.backbone(x).reshape(bs, nc, -1)
        
        a = F.softmax(self.attn(f), dim=1)
        pooled = torch.einsum('bn,bnf->bf', a.squeeze(-1), f)
        
        if ret_features:
            return f  # Return all crop features
        if ret_emb:
            return pooled
        return self.proj(pooled)


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


def load_image(img_path: str) -> np.ndarray:
    """Load 16-bit grayscale image with proper normalization."""
    try:
        import tifffile
        arr = tifffile.imread(img_path)
        if arr.ndim == 3:
            arr = arr[0]
        arr = arr.astype(np.float32) / 65535.0
    except:
        arr = np.array(Image.open(img_path).convert('L')).astype(np.float32) / 255.0
    return arr


def extract_neighborhood_crops(arr: np.ndarray, center_left: int, center_top: int, 
                               crop_size: int, stride: int, neighborhood: int) -> list:
    """Extract center crop + neighborhood crops."""
    half_n = neighborhood // 2
    crops = []
    
    for di in range(-half_n, half_n + 1):
        for dj in range(-half_n, half_n + 1):
            left = center_left + dj * stride
            top = center_top + di * stride
            
            crop = arr[top:top+crop_size, left:left+crop_size]
            crops.append(crop)
    
    return crops


def normalize_crop(crop: np.ndarray, num_channels: int) -> torch.Tensor:
    """Normalize crop for model input."""
    crop_np = (crop * 255).astype(np.uint8)
    crop_pil = Image.fromarray(crop_np, mode='L')
    
    if num_channels == 1:
        mean = np.array([0.5], dtype=np.float32).reshape(1, 1, 1)
        std = np.array([0.5], dtype=np.float32).reshape(1, 1, 1)
    else:
        crop_pil = crop_pil.convert('RGB')
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    
    crop_np = np.array(crop_pil).astype(np.float32) / 255.0
    if num_channels == 1:
        crop_np = crop_np[np.newaxis, ...]
    else:
        crop_np = np.transpose(crop_np, (2, 0, 1))
    
    crop_np = (crop_np - mean) / std
    return torch.from_numpy(crop_np).float()


def parse_well_from_filename(img_path: str) -> str:
    """Parse well position from image filename."""
    filename = os.path.basename(img_path)
    parts = filename.split('_')
    for part in parts:
        if part.startswith('Well'):
            well_str = part.replace('Well', '')
            if len(well_str) == 3:
                row = well_str[0]
                col = well_str[1:]
                return row + col
            return well_str
    return None


def get_ground_truth(plate: str, well: str, mapping_file: str) -> dict:
    """Get ground truth from mapping file."""
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    plate_key = f"P{plate.split('_')[-1]}" if 'Plate_' in plate else plate
    
    if well and plate_key in mapping and well in mapping[plate_key]:
        return mapping[plate_key][well]
    return {}


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from self-supervised model')
    parser.add_argument('--plate', type=str, default='P1', help='Plate to process (P1-P6)')
    parser.add_argument('--data_root', type=str, default=None, 
                        help='Path to parent folder containing P1-P6')
    parser.add_argument('--checkpoint', type=str, default='last_model.pth',
                        help='Checkpoint file to use')
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size')
    parser.add_argument('--grid_size', type=int, default=12, help='Grid size for positions')
    parser.add_argument('--neighborhood', type=int, default=3, help='Neighborhood size (3=3x3)')
    parser.add_argument('--num_channels', type=int, default=1, help='Input channels (1 or 3)')
    parser.add_argument('--embedding_type', type=str, default='pooled', 
                        choices=['pooled', 'features', 'projection'],
                        help='Type of embedding to extract')
    parser.add_argument('--output', type=str, default='embeddings.csv', help='Output CSV file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--max_images', type=int, default=None, help='Max images to process')
    
    args = parser.parse_args()
    
    if args.data_root:
        DATA_ROOT = args.data_root
    else:
        DATA_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    
    drug_dir = os.path.join(DATA_ROOT, 'Drugs_Data', args.plate)
    mutant_dir = os.path.join(DATA_ROOT, 'Mutants_Data', args.plate)
    
    image_paths = []
    data_type = []
    
    if os.path.exists(drug_dir):
        drug_paths = sorted(Path(drug_dir).rglob('*.tif')) + sorted(Path(drug_dir).rglob('*.tiff'))
        image_paths.extend([str(p) for p in drug_paths])
        data_type.extend(['drug'] * len(drug_paths))
    
    if os.path.exists(mutant_dir):
        mutant_paths = sorted(Path(mutant_dir).rglob('*.tif')) + sorted(Path(mutant_dir).rglob('*.tiff'))
        image_paths.extend([str(p) for p in mutant_paths])
        data_type.extend(['mutant'] * len(mutant_paths))
    
    if args.max_images:
        image_paths = image_paths[:args.max_images]
        data_type = data_type[:args.max_images]
    
    print(f"Found {len(image_paths)} images ({len([d for d in data_type if d == 'drug'])} drugs, {len([d for d in data_type if d == 'mutant'])} mutants)")
    
    if not image_paths:
        print("No images found!")
        return
    
    # Load model
    checkpoint_path = os.path.join(SCRIPT_DIR, 'self_supervised_trial', args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = SelfSupMIL(num_crops=args.neighborhood**2, proj_dim=256)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Determine embedding dimension
    if args.embedding_type == 'pooled':
        embed_dim = model.feature_dim  # 1280
    elif args.embedding_type == 'features':
        embed_dim = model.feature_dim  # 1280
    elif args.embedding_type == 'projection':
        embed_dim = 256
    
    print(f"Embedding type: {args.embedding_type}, dimension: {embed_dim}")
    
    # Load first image to get dimensions
    sample_arr = load_image(image_paths[0])
    h, w = sample_arr.shape
    
    stride = (w - args.crop_size) // (args.grid_size - 1) if args.grid_size > 1 else 0
    half_n = args.neighborhood // 2
    
    # Find center position (middle of grid)
    center_row = args.grid_size // 2
    center_col = args.grid_size // 2
    center_left = center_col * stride
    center_top = center_row * stride
    
    print(f"Using center position: ({center_left}, {center_top}), stride={stride}")
    
    # Extract embeddings
    results = []
    
    for img_path in tqdm(image_paths, desc="Extracting embeddings"):
        try:
            arr = load_image(img_path)
            
            # Extract neighborhood crops
            crops = extract_neighborhood_crops(
                arr, center_left, center_top, 
                args.crop_size, stride, args.neighborhood
            )
            
            # Normalize and stack crops
            crop_tensors = torch.stack([normalize_crop(c, args.num_channels) for c in crops])
            crop_tensors = crop_tensors.unsqueeze(0).to(device)  # [1, 9, C, H, W]
            
            with torch.no_grad():
                if args.embedding_type == 'pooled':
                    embeddings = model(crop_tensors, ret_emb=True)  # [1, 1280]
                elif args.embedding_type == 'features':
                    embeddings = model(crop_tensors, ret_features=True)  # [1, 9, 1280]
                    embeddings = embeddings.mean(dim=1)  # Average over crops
                elif args.embedding_type == 'projection':
                    embeddings = model(crop_tensors, ret_emb=False)  # [1, 256]
            
            embedding = embeddings[0].cpu().numpy()
            
            well = parse_well_from_filename(img_path)
            
            results.append({
                'image_path': img_path,
                'image_name': os.path.basename(img_path),
                'well': well,
                'data_type': data_type[len(results)],
                'embedding': embedding
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Extracted embeddings for {len(results)} images")
    
    # Save embeddings to CSV
    output_path = os.path.join(SCRIPT_DIR, 'self_supervised_trial', args.output)
    
    embeddings_array = np.array([r['embedding'] for r in results])
    embedding_cols = [f'emb_{i}' for i in range(embed_dim)]
    
    df_data = {
        'image_path': [r['image_path'] for r in results],
        'image_name': [r['image_name'] for r in results],
        'well': [r['well'] for r in results],
        'data_type': [r['data_type'] for r in results]
    }
    for i, col in enumerate(embedding_cols):
        df_data[col] = embeddings_array[:, i]
    
    import pandas as pd
    df = pd.DataFrame(df_data)
    df.to_csv(output_path, index=False)
    
    print(f"Saved embeddings to {output_path}")
    print(f"Shape: {embeddings_array.shape} ({len(results)} images x {embed_dim} features)")


if __name__ == '__main__':
    main()