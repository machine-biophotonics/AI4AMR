#!/usr/bin/env python3
"""
Generate GradCAM for 9 crops using pytorch-gradcam library.
Applies GradCAM to backbone directly (not full MIL model).
"""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.models as models

from mil_model import AttentionMILModel


def get_base_gene(label):
    if not label or label == 'nan':
        return 'Unknown'
    if '_' in str(label):
        return str(label).rsplit('_', 1)[0]
    return str(label)


def fix_path(path):
    if path.startswith('/mnt/ssd/'):
        path = path.replace('/mnt/ssd/', '')
        path = path.replace('/', '\\')
    elif path.startswith('/'):
        path = 'D:' + path
        path = path.replace('/', '\\')
    return path


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"Script dir: {SCRIPT_DIR}")
    
    # Load predictions
    folds = ['P1', 'P2', 'P3', 'P4', 'P5']
    
    all_pairs = Counter()
    all_examples = {}
    
    for fold in folds:
        csv_path = os.path.join(SCRIPT_DIR, f'fold_{fold}', 'predictions_all_crops_mil_best_model_acc.csv')
        if not os.path.exists(csv_path):
            continue
        
        print(f"Loading {fold}...")
        df = pd.read_csv(csv_path)
        df_valid = df[df['ground_truth_label'].notna()].copy()
        
        for _, row in df_valid.iterrows():
            true_label = row['ground_truth_label']
            pred_label = row['predicted_class_name']
            
            true_gene = get_base_gene(true_label)
            pred_gene = get_base_gene(pred_label)
            
            if true_gene != pred_gene:
                pair_key = (true_label, pred_label)
                all_pairs[pair_key] += 1
                
                if pair_key not in all_examples:
                    all_examples[pair_key] = []
                all_examples[pair_key].append({
                    'image_path': row['image_path'],
                    'confidence': row['confidence']
                })
    
    top_pairs = all_pairs.most_common(6)
    
    print("\nTop cross-gene confused pairs:")
    for (true_cls, pred_cls), count in top_pairs:
        print(f"  {true_cls} -> {pred_cls}: {count}")
    
    # Load MIL model to get predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint_path = os.path.join(SCRIPT_DIR, 'fold_P3', 'best_model_acc.pth')
    
    mil_model = AttentionMILModel(num_classes=96, num_heads=4, attention_temp=0.5)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mil_model.load_state_dict(checkpoint['model_state_dict'])
    mil_model = mil_model.to(device)
    mil_model.eval()
    
    # Create wrapper model that takes single crop and returns logits
    class BackboneWrapper(torch.nn.Module):
        def __init__(self, mil_model):
            super().__init__()
            self.backbone = mil_model.backbone
            self.attention_pool = mil_model.attention_pool
            self.head_proj = mil_model.head_proj
            self.classifier = mil_model.classifier
        
        def forward(self, x):
            features = self.backbone(x)
            pooled, _ = self.attention_pool(features.unsqueeze(1))
            pooled = pooled.reshape(1, -1)
            pooled = self.head_proj(pooled)
            return self.classifier(pooled)
    
    wrapper_model = BackboneWrapper(mil_model)
    wrapper_model = wrapper_model.to(device)
    wrapper_model.eval()
    
    # GradCAM on backbone
    target_layers = [wrapper_model.backbone[-2]]
    cam = GradCAM(model=wrapper_model, target_layers=target_layers)
    
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'analysis', 'gradcam_9crops')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    crop_size = 224
    stride = 68
    
    n_pairs = 6
    fig, axes = plt.subplots(n_pairs, 9, figsize=(27, 3 * n_pairs))
    
    success_count = 0
    
    for pair_idx, ((true_cls, pred_cls), count) in enumerate(top_pairs):
        true_gene = get_base_gene(true_cls)
        pred_gene = get_base_gene(pred_cls)
        
        example = all_examples.get((true_cls, pred_cls), [{}])[0]
        img_path = example.get('image_path')
        
        if not img_path:
            for j in range(9):
                axes[pair_idx, j].axis('off')
            continue
        
        img_path_fixed = fix_path(img_path)
        
        if not os.path.exists(img_path_fixed):
            alt_path = img_path.replace('/mnt/ssd/', 'D:/')
            if os.path.exists(alt_path):
                img_path_fixed = alt_path
        
        if not os.path.exists(img_path_fixed):
            print(f"Image not found: {img_path_fixed}")
            for j in range(9):
                axes[pair_idx, j].text(0.5, 0.5, f'{true_gene}->{pred_gene}', ha='center', va='center', fontsize=8)
                axes[pair_idx, j].axis('off')
            continue
        
        try:
            img = Image.open(img_path_fixed).convert('RGB')
            w, h = img.size
            
            center_x = w // 2 - 4 * stride - crop_size // 2
            center_y = h // 2 - 4 * stride - crop_size // 2
            center_x = max(0, min(center_x, w - crop_size))
            center_y = max(0, min(center_y, h - crop_size))
            
            for j, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1),
                                           (0, -1), (0, 0), (0, 1),
                                           (1, -1), (1, 0), (1, 1)]):
                left = center_x + dx * stride
                top = center_y + dy * stride
                
                crop = img.crop((left, top, left + crop_size, top + crop_size))
                crop_np = np.array(crop, dtype=np.float32) / 255.0
                
                # Preprocess for model
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                crop_norm = (crop_np - mean) / std
                crop_tensor = torch.from_numpy(crop_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                
                # Get prediction
                with torch.no_grad():
                    logits = wrapper_model(crop_tensor)
                    probs = torch.softmax(logits, dim=1)
                
                pred_idx = probs[0].argmax().item()
                pred_prob = probs[0].max().item()
                
                # Generate GradCAM
                targets = [ClassifierOutputTarget(pred_idx)]
                grayscale_cam = cam(input_tensor=crop_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                
                # Create visualization
                visualization = show_cam_on_image(crop_np, grayscale_cam, use_rgb=True, image_weight=0.5)
                
                axes[pair_idx, j].imshow(visualization)
                axes[pair_idx, j].set_title(f'{["TL","TC","TR","ML","C","MR","BL","BC","BR"][j]}\np={pred_prob:.2f}', fontsize=8)
                axes[pair_idx, j].axis('off')
            
            success_count += 1
            axes[pair_idx, 0].set_ylabel(f'{true_gene}->{pred_gene}\n({count}x)', 
                                          fontsize=10, fontweight='bold', rotation=0, ha='right', va='center')
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            for j in range(9):
                axes[pair_idx, j].text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=8)
                axes[pair_idx, j].axis('off')
    
    plt.suptitle(f'GradCAM: 9 Crops (Center + Neighbors)\n{success_count}/{n_pairs} pairs rendered', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.08, 0, 1, 0.97])
    
    output_path = os.path.join(OUTPUT_DIR, 'gradcam_9crops_overlay.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
