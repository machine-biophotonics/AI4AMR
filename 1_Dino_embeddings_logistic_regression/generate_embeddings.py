#!/usr/bin/env python3
"""
Generate DINOv3 Embeddings for CRISPRi Images

This script:
1. Loads DINOv3 ViT-L pretrained model
2. Extracts 144 crops (12x12 grid) of 256x256 pixels from each image
3. Generates CLS token embeddings for each crop
4. Saves embeddings continuously to disk in organized folder structure

Output structure:
    embeddings/
    ├── P1/
    │   ├── WellA01_Channel.../
    │   │   ├── crop_00_00.npy   (1024-dim embedding)
    │   │   ├── crop_00_01.npy
    │   │   └── ...
    │   └── ...
    ├── P2/ ... P6/ ...
    └── metadata.json
"""

import argparse
import os
import sys
import json
import glob
import re
import random
import gc
import time
from typing import Optional, List, Dict, Tuple
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description='Generate DINOv3 embeddings for CRISPRi images')
parser.add_argument('--model_size', type=str, default='vitl', choices=['vit7b', 'vitl', 'vitb', 'vits'], help='DINOv3 model size')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding extraction')
parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
parser.add_argument('--grid_size', type=int, default=12, help='Grid size for crops (12x12=144)')
parser.add_argument('--crop_size', type=int, default=256, help='Crop size in pixels (DINOv3 expects 256)')
parser.add_argument('--resume', action='store_true', help='Resume interrupted extraction')
args = parser.parse_args()

DINOV3_CHECKPOINT = os.path.join(BASE_DIR, "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "embeddings")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_CONFIGS = {
    'vit7b': {'name': 'dinov3_vit7b16', 'embed_dim': 1536},
    'vitl': {'name': 'dinov3_vitl16', 'embed_dim': 1024},
    'vitb': {'name': 'dinov3_vitb16', 'embed_dim': 768},
    'vits': {'name': 'dinov3_vits16', 'embed_dim': 384},
}

def extract_well_from_filename(filename: str) -> Optional[str]:
    match = re.search(r'Well(\w\d+)_', filename)
    return match.group(1) if match else None

def get_image_name(img_path: str) -> str:
    """Get clean image name for folder naming."""
    filename = os.path.basename(img_path)
    name_without_ext = os.path.splitext(filename)[0]
    return name_without_ext

class CropEmbeddingDataset(Dataset):
    """Dataset that extracts crops and prepares them for embedding extraction."""
    
    def __init__(self, image_paths: List[str], crop_size: int = 256, grid_size: int = 12):
        self.image_paths = image_paths
        self.crop_size = crop_size
        self.grid_size = grid_size
        self.n_crops = grid_size * grid_size
        
        from torchvision.transforms import v2
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((crop_size, crop_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        w, h = Image.open(image_paths[0]).size
        self.step_w = (w - crop_size) / (grid_size - 1) if grid_size > 1 else 0
        self.step_h = (h - crop_size) / (grid_size - 1) if grid_size > 1 else 0
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        img_path = self.image_paths[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        crops = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                left = int(col * self.step_w)
                top = int(row * self.step_h)
                crop = img.crop((left, top, left + self.crop_size, top + self.crop_size))
                crop_tensor = self.transform(crop)
                crops.append(crop_tensor)
        
        crops_tensor = torch.stack(crops)
        
        plate = os.path.basename(os.path.dirname(img_path))
        img_name = get_image_name(img_path)
        
        return crops_tensor, plate, img_name, img_path
    
    def get_crop_position(self, crop_idx: int) -> Tuple[int, int]:
        row = crop_idx // self.grid_size
        col = crop_idx % self.grid_size
        return row, col


def load_dinov3_model(model_size: str = 'vitl'):
    """Load DINOv3 model with local checkpoint."""
    
    config = MODEL_CONFIGS[model_size]
    model_name = config['name']
    embed_dim = config['embed_dim']
    
    dinov3_repo_path = os.path.join(BASE_DIR, "dinov3")
    
    print(f"Loading DINOv3 {model_name}...")
    print(f"  Repo: {dinov3_repo_path}")
    print(f"  Checkpoint: {DINOV3_CHECKPOINT}")
    
    model = torch.hub.load(
        dinov3_repo_path,
        model_name,
        source='local',
        weights=DINOV3_CHECKPOINT,
    )
    
    print(f"Loaded {model_name} successfully! Embedding dim: {embed_dim}")
    return model, embed_dim


def extract_and_save_embeddings(model, dataset: Dataset, batch_size: int, 
                                num_workers: int, resume: bool):
    """Extract embeddings and save continuously to disk."""
    
    model.eval()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    total_images = len(dataset)
    total_batches = len(dataloader)
    
    print(f"Processing {total_images} images in {total_batches} batches...")
    
    metadata = {
        'model': args.model_size,
        'embed_dim': MODEL_CONFIGS[args.model_size]['embed_dim'],
        'crop_size': args.crop_size,
        'grid_size': args.grid_size,
        'n_crops_per_image': args.grid_size * args.grid_size,
        'images': []
    }
    
    start_time = time.time()
    
    for batch_idx, (batch_crops, plates, img_names, img_paths) in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
        batch_size_actual = batch_crops.size(0)
        n_crops = batch_crops.size(1)
        
        crop_size = dataset.crop_size
        batch_crops = batch_crops.view(-1, 3, crop_size, crop_size).to(device)
        
        with torch.no_grad():
            # Use forward_features to get patch tokens
            outputs = model.forward_features(batch_crops)
            
            # DINOv3 returns dict with keys:
            # 'x_norm_clstoken': [batch, 1024]
            # 'x_norm_patchtokens': [batch, 256, 1024]
            cls_token = outputs['x_norm_clstoken']  # [batch, 1024]
            patch_tokens = outputs['x_norm_patchtokens']  # [batch, 256, 1024]
            mean_patches = patch_tokens.mean(dim=1)  # [batch, 1024]
            
            # Concatenate CLS + mean patches = 2048-dim
            combined = torch.cat([cls_token, mean_patches], dim=1)  # [batch, 2048]
        
        combined = combined.cpu().numpy()
        combined = combined.reshape(batch_size_actual, n_crops, -1)
        
        for i in range(batch_size_actual):
            plate = plates[i]
            img_name = img_names[i]
            img_path = img_paths[i]
            
            plate_dir = os.path.join(OUTPUT_DIR, plate)
            img_dir = os.path.join(plate_dir, img_name)
            os.makedirs(img_dir, exist_ok=True)
            
            embeddings = combined[i]
            
            for crop_idx in range(n_crops):
                row, col = dataset.get_crop_position(crop_idx)
                crop_filename = f"crop_{row:02d}_{col:02d}.npy"
                crop_path = os.path.join(img_dir, crop_filename)
                
                np.save(crop_path, embeddings[crop_idx])
            
            metadata['images'].append({
                'plate': plate,
                'image_name': img_name,
                'image_path': img_path,
                'n_crops': n_crops,
                'embeddings_dir': img_dir
            })
        
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) / elapsed
            eta = (total_batches - batch_idx - 1) / rate
            print(f"  Progress: {batch_idx+1}/{total_batches} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            
            gc.collect()
            torch.cuda.empty_cache()
    
    return metadata


def get_all_image_paths() -> Dict[str, List[str]]:
    """Get all image paths organized by plate."""
    plate_paths = {}
    for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        paths = glob.glob(os.path.join(BASE_DIR, plate, '*.tif'))
        plate_paths[plate] = sorted(paths)
        print(f"  {plate}: {len(paths)} images")
    return plate_paths


def main():
    print("\n" + "="*60)
    print("DINOv3 Embedding Generation")
    print("="*60)
    print(f"\nSettings:")
    print(f"  Model: DINOv3 {args.model_size}")
    print(f"  Grid: {args.grid_size}x{args.grid_size} = {args.grid_size**2} crops/image")
    print(f"  Crop size: {args.crop_size}px")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Output: {OUTPUT_DIR}")
    
    plate_paths = get_all_image_paths()
    total_images = sum(len(p) for p in plate_paths.values())
    total_crops = total_images * args.grid_size * args.grid_size
    
    print(f"\nTotal: {total_images} images × {args.grid_size**2} crops = {total_crops} embeddings")
    
    base_embed_dim = MODEL_CONFIGS[args.model_size]['embed_dim']
    embed_dim = base_embed_dim * 2  # CLS + mean pooled patches
    est_size_gb = total_crops * embed_dim * 4 / (1024**3)
    print(f"Estimated size: ~{est_size_gb:.1f} GB (CLS + Mean = {embed_dim}-dim per crop)")
    
    print(f"\nLoading DINOv3 model...")
    model, _ = load_dinov3_model(args.model_size)
    model = model.to(device)
    model.eval()
    
    all_metadata = {
        'model': args.model_size,
        'embed_dim': embed_dim,
        'base_embed_dim': base_embed_dim,
        'crop_size': args.crop_size,
        'grid_size': args.grid_size,
        'n_crops_per_image': args.grid_size * args.grid_size,
        'pooling': 'CLS + mean pooled patches (concatenated)',
        'how_to_unconcat': {
            'cls_only': 'embeddings[:base_embed_dim]',
            'mean_only': 'embeddings[base_embed_dim:]',
            'both': 'embeddings'
        },
        'plates': {}
    }
    
    for plate, paths in plate_paths.items():
        print(f"\n{'='*40}")
        print(f"Processing {plate}: {len(paths)} images")
        print(f"{'='*40}")
        
        plate_output_dir = os.path.join(OUTPUT_DIR, plate)
        
        if args.resume and os.path.exists(plate_output_dir):
            processed = set(os.listdir(plate_output_dir))
            paths = [p for p in paths if get_image_name(p) not in processed]
            print(f"  Already processed: {len(processed)} images")
            print(f"  Remaining: {len(paths)} images to process")
            if len(paths) == 0:
                print(f"  Skipping {plate} (already complete)")
                continue
        
        if len(paths) == 0:
            continue
        
        dataset = CropEmbeddingDataset(paths, crop_size=args.crop_size, grid_size=args.grid_size)
        
        metadata = extract_and_save_embeddings(model, dataset, args.batch_size, args.num_workers, args.resume)
        
        all_metadata['plates'][plate] = {
            'n_images': len(paths),
            'output_dir': plate_output_dir
        }
        
        gc.collect()
        torch.cuda.empty_cache()
    
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Embedding generation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
