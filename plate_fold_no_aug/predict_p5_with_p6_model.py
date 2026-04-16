#!/usr/bin/env python3
"""
Predict on P5 using best model from fold_P6.
"""

import os
import json
import argparse
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json'), 'r') as f:
    PLATE_WELL_ID = json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--grid_size', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    crop_size = args.crop_size
    grid_size = args.grid_size
    
    print(f"Config: crop_size={crop_size}, grid_size={grid_size}")
    
    # Load classes
    classes = {}
    with open(os.path.join(SCRIPT_DIR, 'classes.txt'), 'r') as f:
        for line in f:
            idx, name = line.strip().split(',', 1)
            classes[int(idx)] = name
    
    num_classes = len(classes)
    label_to_idx = {v: k for k, v in classes.items()}
    print(f"Loaded {num_classes} classes")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    class EfficientNetClassifier(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            import torchvision
            self.features = torchvision.models.efficientnet_b0(weights=None).features
            in_features = 1280
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    # Load best model from fold_P6
    model = EfficientNetClassifier(num_classes).to(device)
    checkpoint = torch.load(os.path.join(SCRIPT_DIR, 'fold_P6', 'best_model.pth'), 
                       map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from fold_P6")
    
    def extract_all_crops(img_path, crop_size, grid_size):
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        stride = (w - crop_size) // (grid_size - 1)
        positions = []
        for row in range(grid_size):
            for col in range(grid_size):
                left = col * stride
                top = row * stride
                if left + crop_size <= w and top + crop_size <= h:
                    positions.append((left, top))
        
        crops = []
        crop_idx = 0
        
        for row in range(grid_size):
            for col in range(grid_size):
                if crop_idx >= len(positions):
                    break
                left, top = positions[crop_idx]
                
                crop = img.crop((left, top, left + crop_size, top + crop_size))
                crop_np = np.array(crop).astype(np.float32) / 255.0
                crop_np = np.transpose(crop_np, (2, 0, 1))
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                crop_np = (crop_np - mean) / std
                crop_tensor = torch.from_numpy(crop_np).float()
                
                crops.append((crop_tensor, row, col, crop_idx))
                crop_idx += 1
        
        return crops
    
    def parse_well_from_filename(img_path):
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        for part in parts:
            if part.startswith('Well'):
                well_str = part.replace('Well', '')
                if len(well_str) == 3:
                    row = well_str[0]
                    col = str(int(well_str[1:]))
                    return row + col
                return well_str
        return None
    
    def get_ground_truth_label(plate, well):
        if plate in PLATE_WELL_ID and well:
            row = well[0]
            col = well[1:]
            if row in PLATE_WELL_ID[plate]:
                if col in PLATE_WELL_ID[plate][row]:
                    return PLATE_WELL_ID[plate][row][col].get('id', None)
        return None
    
    # Predict on P5
    test_plate = 'P5'
    test_dir = os.path.join(BASE_DIR, test_plate)
    image_paths = sorted(Path(test_dir).glob('*.tif'))
    
    print(f"Predicting on {test_plate} ({len(image_paths)} images)...")
    
    all_results = []
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc=f'Predicting {test_plate}'):
            img_name = os.path.basename(img_path)
            well = parse_well_from_filename(img_path)
            gt_label = get_ground_truth_label(test_plate, well)
            
            if gt_label is None:
                continue
            
            gt_idx = label_to_idx.get(gt_label, -1)
            
            crops = extract_all_crops(str(img_path), crop_size, grid_size)
            
            for i in range(0, len(crops), args.batch_size):
                batch_crops = crops[i:i+args.batch_size]
                batch_tensors = torch.stack([c[0] for c in batch_crops]).to(device)
                logits = model(batch_tensors)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                for j, (crop_tensor, row, col, crop_idx) in enumerate(batch_crops):
                    pred_class = probs[j].argmax()
                    pred_class_name = classes[pred_class]
                    max_prob = probs[j].max()
                    
                    all_results.append({
                        'image_path': str(img_path),
                        'image_name': img_name,
                        'plate': test_plate,
                        'well': well,
                        'ground_truth_label': gt_label,
                        'ground_truth_idx': gt_idx,
                        'crop_index': crop_idx,
                        'grid_row': row,
                        'grid_col': col,
                        'predicted_class_idx': pred_class,
                        'predicted_class_name': pred_class_name,
                        'confidence': max_prob,
                        'logits': logits[j].cpu().numpy().tolist(),
                        'probs': probs[j].tolist()
                    })
    
    df = pd.DataFrame(all_results)
    output_path = os.path.join(SCRIPT_DIR, 'fold_P6', f'predictions_on_{test_plate}.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} predictions to {output_path}")


if __name__ == '__main__':
    main()