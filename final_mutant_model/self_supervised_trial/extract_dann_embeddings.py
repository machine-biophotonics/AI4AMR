#!/usr/bin/env python3
"""
Extract embeddings from DANN model and visualize with t-SNE.
"""

import os
import argparse
import json
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainAdversarialMIL(nn.Module):
    def __init__(self, num_classes, num_domains=2, feature_dim=1280, hidden_dim=256, num_crops=9):
        super().__init__()
        
        self.num_crops = num_crops
        self.feature_dim = feature_dim
        
        import torchvision.models as models
        base = models.efficientnet_b0(weights='IMAGENET1K_V1')
        base.features[0][0] = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        
        self.feature_extractor = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.label_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.grl = type('GRL', (), {'alpha': 1.0})()
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains)
        )
    
    def forward(self, x, ret_features=False):
        bs = x.shape[0]
        nc = x.shape[1]
        
        x_flat = x.reshape(bs * nc, *x.shape[2:]).contiguous()
        features = self.feature_extractor(x_flat)
        features = features.reshape(bs, nc, -1)
        
        attn_weights = torch.softmax(self.attention(features), dim=1)
        pooled = torch.einsum('bn,bnf->bf', attn_weights.squeeze(-1), features)
        
        if ret_features:
            return pooled
        
        label_logits = self.label_classifier(pooled)
        return label_logits, pooled
    
    def set_grl_alpha(self, alpha):
        self.grl.alpha = alpha


def load_image(img_path, crop_size=224):
    try:
        import tifffile
        arr = tifffile.imread(img_path)
        if arr.ndim == 3:
            arr = arr[0]
        return arr.astype(np.float32) / 65535.0
    except:
        return np.array(Image.open(img_path).convert('L')).astype(np.float32) / 255.0


def extract_crops(arr, center_left, center_top, crop_size, stride, neighborhood):
    half_n = neighborhood // 2
    crops = []
    
    for di in range(-half_n, half_n + 1):
        for dj in range(-half_n, half_n + 1):
            left = center_left + dj * stride
            top = center_top + di * stride
            crop = arr[top:top+crop_size, left:left+crop_size]
            crops.append(crop)
    
    return crops


def normalize_crop(crop):
    crop_np = (crop * 255).astype(np.uint8)
    crop_pil = Image.fromarray(crop_np, mode='L')
    crop_np = np.array(crop_pil).astype(np.float32) / 255.0
    crop_np = (crop_np - 0.5) / 0.5
    return torch.from_numpy(crop_np).float().unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description='Extract DANN embeddings')
    parser.add_argument('--plate', type=str, default='P1')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='best_model.pth')
    parser.add_argument('--output', type=str, default='dann_embeddings.csv')
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--grid_size', type=int, default=12)
    parser.add_argument('--neighborhood', type=int, default=3)
    
    args = parser.parse_args()
    
    output_dir = os.path.join(SCRIPT_DIR, 'dann_output', f'plate_{args.plate}')
    checkpoint_path = os.path.join(output_dir, args.checkpoint)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    class_to_idx = checkpoint['class_to_idx']
    num_classes = len(class_to_idx)
    
    model = DomainAdversarialMIL(
        num_classes=num_classes,
        num_domains=2,
        hidden_dim=256,
        num_crops=args.neighborhood**2
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded!")
    
    # Get image paths
    drug_dir = os.path.join(args.data_root, 'Drugs_Data', args.plate)
    mutant_dir = os.path.join(args.data_root, 'Mutants_Data', args.plate)
    
    drug_paths = sorted(glob.glob(os.path.join(drug_dir, "**", "*.tif"), recursive=True))
    drug_paths += sorted(glob.glob(os.path.join(drug_dir, "**", "*.tiff"), recursive=True))
    
    mutant_paths = sorted(glob.glob(os.path.join(mutant_dir, "**", "*.tif"), recursive=True))
    mutant_paths += sorted(glob.glob(os.path.join(mutant_dir, "**", "*.tiff"), recursive=True))
    
    all_paths = drug_paths + mutant_paths
    data_types = ['drug'] * len(drug_paths) + ['mutant'] * len(mutant_paths)
    
    print(f"Total images: {len(all_paths)} ({len(drug_paths)} drugs, {len(mutant_paths)} mutants)")
    
    # Get center position
    arr = load_image(all_paths[0])
    h, w = arr.shape
    stride = (w - args.crop_size) // (args.grid_size - 1)
    center_row = args.grid_size // 2
    center_col = args.grid_size // 2
    center_left = center_col * stride
    center_top = center_row * stride
    
    # Extract embeddings
    results = []
    
    for img_path in tqdm(all_paths, desc="Extracting embeddings"):
        try:
            arr = load_image(img_path)
            crops = extract_crops(arr, center_left, center_top, args.crop_size, stride, args.neighborhood)
            crop_tensors = torch.stack([normalize_crop(c) for c in crops]).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = model(crop_tensors, ret_features=True)
            
            results.append({
                'image_path': img_path,
                'image_name': os.path.basename(img_path),
                'data_type': data_types[len(results)],
                'embedding': features[0].cpu().numpy()
            })
        except Exception as e:
            print(f"Error: {img_path}: {e}")
            continue
    
    # Save embeddings
    embeddings_array = np.array([r['embedding'] for r in results])
    embed_cols = [f'emb_{i}' for i in range(embeddings_array.shape[1])]
    
    df_data = {
        'image_path': [r['image_path'] for r in results],
        'image_name': [r['image_name'] for r in results],
        'data_type': [r['data_type'] for r in results]
    }
    for i, col in enumerate(embed_cols):
        df_data[col] = embeddings_array[:, i]
    
    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(output_dir, args.output), index=False)
    
    print(f"Saved embeddings: {embeddings_array.shape}")


if __name__ == '__main__':
    main()