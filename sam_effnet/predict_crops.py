#!/usr/bin/env python3
"""
Predict all crops for each test image with configurable crop_size and grid_size.
Saves results to CSV with: image_path, crop_index, grid_row, grid_col, predicted_class, confidence, all_probabilities
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

def main():
    parser = argparse.ArgumentParser(description='Predict crops for test images')
    parser.add_argument('--crop_size', type=int, default=544, help='Crop size (default: 544)')
    parser.add_argument('--grid_size', type=int, default=5, help='Grid size (default: 5)')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes (default: auto-detect from classes.txt)')
    parser.add_argument('--image_dir', type=str, default=os.path.join(BASE_DIR, 'P6'),
                        help='Directory containing test images')
    parser.add_argument('--output_csv', type=str, default=os.path.join(SCRIPT_DIR, 'crop_predictions.csv'),
                        help='Output CSV file path')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(SCRIPT_DIR, 'best_model.pth'),
                        help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    crop_size = args.crop_size
    grid_size = args.grid_size
    crops_per_image = grid_size * grid_size
    
    print(f"Config: crop_size={crop_size}, grid_size={grid_size}, crops_per_image={crops_per_image}")
    
    # Load class mappings
    classes = {}
    with open(os.path.join(SCRIPT_DIR, 'classes.txt'), 'r') as f:
        for line in f:
            idx, name = line.strip().split(',', 1)
            classes[int(idx)] = name

    idx_to_label = classes
    label_to_idx = {v: k for k, v in classes.items()}
    
    # Allow override of num_classes for excluded classes
    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
        num_classes = len(classes)

    print(f"Loaded {num_classes} classes")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    class EfficientNetClassifier(torch.nn.Module):
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

    model = EfficientNetClassifier(num_classes=num_classes)
    model = model.to(device)

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")

    def extract_all_crops(img_path, crop_size, grid_size):
        """Extract all crops from an image in deterministic order."""
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # Handle grid_size=1 case
        if grid_size == 1:
            stride = 0
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            positions = [(left, top)]
        else:
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

    def predict_image(img_path, batch_size):
        """Predict all crops for a single image."""
        crops = extract_all_crops(img_path, crop_size, grid_size)
        
        results = []
        
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i+batch_size]
            batch_tensors = torch.stack([c[0] for c in batch_crops]).to(device)
            
            with torch.no_grad():
                logits = model(batch_tensors)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                confidences = probs.max(dim=1).values
            
            for j, (crop_tensor, row, col, crop_idx) in enumerate(batch_crops):
                pred_idx = preds[j].item()
                confidence = confidences[j].item()
                
                results.append({
                    'image_path': img_path,
                    'image_name': os.path.basename(img_path),
                    'crop_index': crop_idx,
                    'grid_row': row,
                    'grid_col': col,
                    'predicted_class_idx': pred_idx,
                    'predicted_class_name': idx_to_label[pred_idx],
                    'confidence': confidence,
                    'probs_json': json.dumps(probs[j].cpu().numpy().tolist())
                })
        
        return results

    # Get all tif files
    image_paths = sorted(Path(args.image_dir).glob('*.tif'))
    
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    
    print(f"Processing {len(image_paths)} images from {args.image_dir}")
    
    all_results = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        img_path_str = str(img_path)
        results = predict_image(img_path_str, args.batch_size)
        all_results.extend(results)
    
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(all_results)} crop predictions to {args.output_csv}")
    print(f"  - Images processed: {len(image_paths)}")
    print(f"  - Crops per image: {crops_per_image} ({grid_size}x{grid_size} grid)")
    print(f"  - Total crops: {len(all_results)}")
    
    print(f"\nPrediction summary:")
    print(f"  - Unique predicted classes: {df['predicted_class_name'].nunique()}")
    print(f"  - Average confidence: {df['confidence'].mean():.4f}")
    print(f"  - Min confidence: {df['confidence'].min():.4f}")
    print(f"  - Max confidence: {df['confidence'].max():.4f}")
    
    print(f"\nPredicted class distribution (top 10):")
    class_counts = df['predicted_class_name'].value_counts()
    for cls, count in class_counts.head(10).items():
        print(f"  {cls}: {count} ({100*count/len(df):.1f}%)")

if __name__ == '__main__':
    main()