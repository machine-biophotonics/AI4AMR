#!/usr/bin/env python3
"""
Predict all 144 crops for each test image, tracking which crop came from which image.
Saves results to CSV with: image_path, crop_index, grid_row, grid_col, predicted_class, confidence, all_probabilities
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Load class mappings
classes = {}
with open(os.path.join(SCRIPT_DIR, 'classes.txt'), 'r') as f:
    for line in f:
        idx, name = line.strip().split(',', 1)
        classes[int(idx)] = name

idx_to_label = classes
label_to_idx = {v: k for k, v in classes.items()}
num_classes = len(classes)

print(f"Loaded {num_classes} classes")

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class EfficientNetClassifier(torch.nn.Module):
    """Same architecture as training (train.py uses standard torchvision.models.efficientnet_b0)"""
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

# Initialize model
model = EfficientNetClassifier(num_classes=num_classes)
model = model.to(device)

# Load best checkpoint
checkpoint_path = os.path.join(SCRIPT_DIR, 'best_model.pth')
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully")

# Image preprocessing (same as validation - no augmentation)
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def extract_all_crops(img_path, crop_size=224, grid_size=12):
    """Extract all 144 crops from an image in deterministic order.
    
    Returns:
        crops: list of (crop_tensor, grid_row, grid_col, crop_index)
    """
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    # Compute stride
    stride = (w - crop_size) // (grid_size - 1)
    
    crops = []
    crop_idx = 0
    
    for row in range(grid_size):
        for col in range(grid_size):
            left = col * stride
            top = row * stride
            
            # Ensure crop fits
            if left + crop_size <= w and top + crop_size <= h:
                crop = img.crop((left, top, left + crop_size, top + crop_size))
                
                # Convert to tensor
                crop_np = np.array(crop).astype(np.float32) / 255.0
                # CHW format
                crop_np = np.transpose(crop_np, (2, 0, 1))
                # Normalize
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
                crop_np = (crop_np - mean) / std
                crop_tensor = torch.from_numpy(crop_np).float()
                
                crops.append((crop_tensor, row, col, crop_idx))
                crop_idx += 1
    
    return crops

def predict_image(img_path, batch_size=32):
    """Predict all crops for a single image.
    
    Returns:
        list of dicts with prediction results
    """
    crops = extract_all_crops(img_path)
    
    results = []
    
    # Process in batches
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

def run_predictions(image_dir, output_csv, max_images=None):
    """Run predictions on all images in a directory.
    
    Args:
        image_dir: Directory containing .tif images
        output_csv: Path to save results CSV
        max_images: Maximum number of images to process (None for all)
    """
    # Get all tif files
    image_paths = sorted(Path(image_dir).glob('*.tif'))
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"Processing {len(image_paths)} images from {image_dir}")
    
    all_results = []
    
    for img_path in tqdm(image_paths, desc="Processing images"):
        img_path_str = str(img_path)
        results = predict_image(img_path_str)
        all_results.extend(results)
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(all_results)} crop predictions to {output_csv}")
    print(f"  - Images processed: {len(image_paths)}")
    print(f"  - Crops per image: 144 (12x12 grid)")
    print(f"  - Total crops: {len(all_results)}")
    
    # Print summary
    print(f"\nPrediction summary:")
    print(f"  - Unique predicted classes: {df['predicted_class_name'].nunique()}")
    print(f"  - Average confidence: {df['confidence'].mean():.4f}")
    print(f"  - Min confidence: {df['confidence'].min():.4f}")
    print(f"  - Max confidence: {df['confidence'].max():.4f}")
    
    # Class distribution
    print(f"\nPredicted class distribution:")
    class_counts = df['predicted_class_name'].value_counts()
    for cls, count in class_counts.head(10).items():
        print(f"  {cls}: {count} ({100*count/len(df):.1f}%)")
    
    return df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict all 144 crops for test images')
    parser.add_argument('--image_dir', type=str, default=os.path.join(BASE_DIR, 'P6'),
                        help='Directory containing test images')
    parser.add_argument('--output_csv', type=str, default=os.path.join(SCRIPT_DIR, 'crop_predictions.csv'),
                        help='Output CSV file path')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process (default: all)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    
    args = parser.parse_args()
    
    run_predictions(args.image_dir, args.output_csv, args.max_images)
