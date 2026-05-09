#!/usr/bin/env python3
"""
Generate DINOv3 ViT-Large Embeddings for Mutant and Drug Images (Plate 1)

This script:
1. Loads DINOv3 ViT-Large pretrained model (satellite pretrained)
2. Extracts 500x500 center crop from each image
3. Generates CLS token embeddings (1024-dim)
4. Saves embeddings organized by well

Output structure:
    embeddings/
    ├── metadata.json
    ├── Mutants_P1/
    │   └── WellA01/
    │       └── image_name.npy
    └── Drugs_P1/
        └── WellA01/
            └── image_name.npy
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
from collections import defaultdict

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

OUTPUT_DIR = os.path.join(BASE_DIR, "embeddings")
MUTANTS_DIR = os.path.join(BASE_DIR, "..", "Mutants_Data", "P1")
DRUGS_DIR = os.path.join(BASE_DIR, "..", "Drugs_Data", "P1")
DINOV3_CHECKPOINT = os.path.join(BASE_DIR, "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")

CROP_SIZE = 500
MODEL_INPUT_SIZE = 500  # No resize - use 500x500 directly

def extract_well_from_filename(filename: str) -> Optional[str]:
    """Extract well ID from filename (e.g., WellA01_PointA01_0000 -> WellA01)"""
    match = re.search(r'(Well[A-H]\d+)', filename)
    return match.group(1) if match else None


class CropEmbeddingDataset(Dataset):
    """Dataset that extracts center crop and prepares for embedding extraction."""

    def __init__(self, image_paths: List[str], crop_size: int = 500, model_input_size: int = 256):
        self.image_paths = image_paths
        self.crop_size = crop_size
        self.model_input_size = model_input_size

        from torchvision.transforms import v2
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        img_path = self.image_paths[idx]
        
        # Handle 16-bit images - SAME as final_mutant_model
        try:
            import tifffile
            img_array = tifffile.imread(img_path)
        except:
            img_array = np.array(Image.open(img_path))
        
        # Handle multi-channel
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 0]
        
        # Normalize to 0-1 using theoretical max (SAME as final_mutant_model)
        if img_array.dtype == np.uint16:
            img_array = img_array.astype(np.float32) / 65535.0
        elif img_array.dtype == np.uint8:
            img_array = img_array.astype(np.float32) / 255.0
        else:
            img_array = img_array.astype(np.float32)
        
        # Convert to 0-255 and then to PIL
        img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L').convert('RGB')
        
        w, h = img.size
        
        left = (w - self.crop_size) // 2
        top = (h - self.crop_size) // 2
        crop = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        
        crop_tensor = self.transform(crop)
        
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        well_id = extract_well_from_filename(img_name)
        
        return crop_tensor, well_id, img_name, img_path


def load_dinov3_model():
    """Load DINOv3 ViT-Large model with local checkpoint."""
    
    dinov3_repo_path = os.path.join(BASE_DIR, "dinov3")
    model_name = "dinov3_vitl16"
    embed_dim = 1024  # ViT-Large has 1024-dim embeddings
    
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


def extract_embeddings(model, dataset: Dataset, batch_size: int, num_workers: int):
    """Extract embeddings and return organized by well."""
    
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
    
    well_embeddings = defaultdict(list)
    metadata = []
    
    total_images = len(dataset)
    total_batches = len(dataloader)
    
    print(f"Processing {total_images} images in {total_batches} batches...")
    
    start_time = time.time()
    
    for batch_idx, (batch_crops, well_ids, img_names, img_paths) in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
        batch_crops = batch_crops.to(device)
        
        with torch.no_grad():
            outputs = model.forward_features(batch_crops)
            cls_token = outputs['x_norm_clstoken']
        
        cls_token = cls_token.cpu().numpy()
        
        for i in range(len(well_ids)):
            well_id = well_ids[i]
            img_name = img_names[i]
            embedding = cls_token[i]
            
            if well_id:
                well_embeddings[well_id].append({
                    'image_name': img_name,
                    'embedding': embedding
                })
            
            metadata.append({
                'well_id': well_id,
                'image_name': img_name,
                'image_path': img_paths[i],
                'embedding_shape': embedding.shape
            })
        
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) / elapsed
            eta = (total_batches - batch_idx - 1) / rate
            print(f"  Progress: {batch_idx+1}/{total_batches} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")
            
            gc.collect()
            torch.cuda.empty_cache()
    
    return well_embeddings, metadata


def save_embeddings(well_embeddings: Dict, output_subdir: str, data_type: str):
    """Save embeddings organized by well."""
    
    output_path = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(output_path, exist_ok=True)
    
    saved_files = []
    
    for well_id, embeddings_list in tqdm(well_embeddings.items(), desc=f"Saving {data_type} embeddings"):
        well_dir = os.path.join(output_path, well_id)
        os.makedirs(well_dir, exist_ok=True)
        
        for item in embeddings_list:
            img_name = item['image_name']
            embedding = item['embedding']
            
            filepath = os.path.join(well_dir, f"{img_name}.npy")
            np.save(filepath, embedding)
            saved_files.append({
                'well_id': well_id,
                'image_name': img_name,
                'filepath': filepath
            })
    
    print(f"  Saved {len(saved_files)} embeddings to {output_path}")
    return saved_files


def get_image_paths(data_dir: str, data_type: str) -> List[str]:
    """Get all image paths from data directory."""
    
    if data_type == "mutant":
        tifocus_path = os.path.join(data_dir, "TIFOCUS")
        if os.path.exists(tifocus_path):
            paths = glob.glob(os.path.join(tifocus_path, "*.tif"))
        else:
            paths = glob.glob(os.path.join(data_dir, "*.tif"))
    else:
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if subdirs:
            paths = glob.glob(os.path.join(data_dir, subdirs[0], "*.tiff")) + \
                    glob.glob(os.path.join(data_dir, subdirs[0], "*.tif"))
        else:
            paths = glob.glob(os.path.join(data_dir, "*.tiff")) + \
                    glob.glob(os.path.join(data_dir, "*.tif"))
    
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description='Generate DINOv3 embeddings for mutant and drug images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding extraction')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--resume', action='store_true', help='Resume interrupted extraction')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("DINOv3 Embedding Generation (Plate 1)")
    print("="*60)
    print(f"\nSettings:")
    print(f"  Center crop: {CROP_SIZE}x{CROP_SIZE}")
    print(f"  Model input: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE} (no resize)")
    print(f"  Embedding: CLS token (1024-dim)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers: {args.num_workers}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nLoading DINOv3 model...")
    model, embed_dim = load_dinov3_model()
    model = model.to(device)
    model.eval()
    
    all_metadata = {
        'model': 'dinov3_vitl16',
        'checkpoint': 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        'embed_dim': embed_dim,
        'crop_size': CROP_SIZE,
        'model_input_size': MODEL_INPUT_SIZE,
        'embedding_type': 'CLS token',
        'data_type': 'mutant_and_drug',
        'plate': 'P1',
        'mutant_files': [],
        'drug_files': []
    }
    
    print(f"\n{'='*40}")
    print("Processing Mutant Data (P1)")
    print(f"{'='*40}")
    
    mutant_paths = get_image_paths(MUTANTS_DIR, "mutant")
    print(f"  Found {len(mutant_paths)} mutant images")
    
    if len(mutant_paths) > 0:
        dataset = CropEmbeddingDataset(mutant_paths, crop_size=CROP_SIZE, model_input_size=MODEL_INPUT_SIZE)
        mutant_well_embeddings, mutant_metadata = extract_embeddings(model, dataset, args.batch_size, args.num_workers)
        
        mutant_files = save_embeddings(mutant_well_embeddings, "Mutants_P1", "Mutant")
        all_metadata['mutant_files'] = mutant_metadata
        all_metadata['mutant_wells'] = list(mutant_well_embeddings.keys())
        
        print(f"  Wells found: {len(mutant_well_embeddings)}")
        for well, embeds in sorted(mutant_well_embeddings.items()):
            print(f"    {well}: {len(embeds)} images")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"\n{'='*40}")
    print("Processing Drug Data (P1)")
    print(f"{'='*40}")
    
    drug_paths = get_image_paths(DRUGS_DIR, "drug")
    print(f"  Found {len(drug_paths)} drug images")
    
    if len(drug_paths) > 0:
        dataset = CropEmbeddingDataset(drug_paths, crop_size=CROP_SIZE, model_input_size=MODEL_INPUT_SIZE)
        drug_well_embeddings, drug_metadata = extract_embeddings(model, dataset, args.batch_size, args.num_workers)
        
        drug_files = save_embeddings(drug_well_embeddings, "Drugs_P1", "Drug")
        all_metadata['drug_files'] = drug_metadata
        all_metadata['drug_wells'] = list(drug_well_embeddings.keys())
        
        print(f"  Wells found: {len(drug_well_embeddings)}")
        for well, embeds in sorted(drug_well_embeddings.items()):
            print(f"    {well}: {len(embeds)} images")
        
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