"""
Loss Landscape Visualization for CRISPRi Classification.
Creates 2D contour plots showing how loss varies in weight space around the trained model.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cross_entropy
from PIL import Image
from torchvision.transforms import ToTensor, Normalize
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import json
import re
import random
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class MultiPatchDataset(Dataset):
    def __init__(self, image_paths, n_patches=5):
        self.image_paths = image_paths
        self.n_patches = n_patches
        self.patch_size = 224
        self.edge_margin = 200
        self.transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        patches = []
        center_w_start = self.edge_margin
        center_w_end = w - self.edge_margin
        center_h_start = self.edge_margin
        center_h_end = h - self.edge_margin
        
        center_w = center_w_end - center_w_start
        center_h = center_h_end - center_h_start
        
        for _ in range(self.n_patches):
            left = int(np.random.uniform(0, max(1, center_w - self.patch_size)))
            top = int(np.random.uniform(0, max(1, center_h - self.patch_size)))
            left = center_w_start + left
            top = center_h_start + top
            patch = image.crop((left, top, left + self.patch_size, top + self.patch_size))
            patch = ToTensor()(patch)
            patch = self.transform(patch)
            patches.append(patch)
        
        return torch.stack(patches), img_path


def extract_well_from_filename(filename):
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        row = match.group(1)
        col = int(match.group(2))
        return f"{row}{col:02d}"
    return None


def load_plate_data():
    with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
        plate_data = json.load(f)
    
    plate_maps = {}
    for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
        plate_maps[plate] = {}
        for row, wells in plate_data[plate].items():
            for col, info in wells.items():
                well = f"{row}{int(col):02d}"
                plate_maps[plate][well] = info['id']
    return plate_maps


def get_label_from_path(img_path, plate_maps):
    dirname = os.path.basename(os.path.dirname(img_path))
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if dirname in plate_maps and well in plate_maps[dirname]:
        return plate_maps[dirname][well]
    return None


def get_random_direction(state_dict):
    """Get a random direction in weight space (filter normalized for float tensors only)."""
    direction = {}
    for key in state_dict.keys():
        if state_dict[key].dtype in [torch.float32, torch.float64]:
            direction[key] = torch.randn_like(state_dict[key])
            if len(state_dict[key].shape) >= 2:
                direction[key] = direction[key] / (torch.norm(direction[key]) + 1e-10)
        else:
            direction[key] = torch.zeros_like(state_dict[key])
    return direction


def interpolate_weights(original_weights, direction1, direction2, alpha1, alpha2):
    """Interpolate between original weights and directions."""
    new_weights = {}
    for key in original_weights.keys():
        new_weights[key] = original_weights[key] + alpha1 * direction1[key] + alpha2 * direction2[key]
    return new_weights


def compute_loss(model, loader, label_to_idx, device, plate_maps):
    """Compute cross-entropy loss on a sample of data."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, (patches, paths) in enumerate(loader):
            if i >= 5:
                break
            
            batch_size, n_patches, C, H, W = patches.shape
            patches = patches.view(-1, C, H, W).to(device)
            labels = []
            for path in paths:
                label = get_label_from_path(path, plate_maps)
                if label and label in label_to_idx:
                    labels.append(label_to_idx[label])
                else:
                    labels.append(0)
            
            if len(labels) == 0:
                print(f"Warning: No labels found for batch {i}")
                continue
                
            labels = torch.tensor(labels).to(device)
            
            outputs = model(patches)
            outputs = outputs.view(batch_size, n_patches, -1).mean(dim=1)
            loss = cross_entropy(outputs, labels)
            
            if torch.isnan(loss):
                continue
                
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    if total_samples == 0:
        return float('nan')
    return total_loss / total_samples


def compute_loss_surface(model, original_weights, direction1, direction2, alphas, loader, label_to_idx, device, plate_maps):
    """Compute 2D loss surface along two directions."""
    loss_surface = np.zeros((len(alphas), len(alphas)))
    
    for i, alpha1 in enumerate(tqdm(alphas, desc="Computing loss surface")):
        for j, alpha2 in enumerate(alphas):
            new_weights = interpolate_weights(original_weights, direction1, direction2, alpha1, alpha2)
            model.load_state_dict(new_weights)
            loss_surface[i, j] = compute_loss(model, loader, label_to_idx, device, plate_maps)
    
    model.load_state_dict(original_weights)
    return loss_surface


def plot_loss_landscape(loss_surface, alphas, output_path, title="Loss Landscape"):
    """Plot 2D loss landscape as contour and surface plots."""
    X, Y = np.meshgrid(alphas, alphas)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    levels = np.linspace(loss_surface.min(), loss_surface.max(), 30)
    
    ax1 = axes[0]
    contour = ax1.contourf(X, Y, loss_surface, levels=levels, cmap='viridis')
    plt.colorbar(contour, ax=ax1, label='Cross-Entropy Loss')
    ax1.set_xlabel('Direction 1 (alpha)')
    ax1.set_ylabel('Direction 2 (alpha)')
    ax1.set_title('Loss Contour Plot')
    ax1.axhline(0, color='white', linewidth=2, linestyle='--', alpha=0.7)
    ax1.axvline(0, color='white', linewidth=2, linestyle='--', alpha=0.7)
    
    ax2 = axes[1]
    ax2.plot(alphas, loss_surface[len(alphas)//2, :], 'b-', label='Slice along dir 1', linewidth=2)
    ax2.plot(alphas, loss_surface[:, len(alphas)//2], 'r--', label='Slice along dir 2', linewidth=2)
    ax2.set_xlabel('alpha')
    ax2.set_ylabel('Loss')
    ax2.set_title('1D Loss Slices (center row/column)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    fig3d = plt.figure(figsize=(12, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    surf = ax3d.plot_surface(X, Y, loss_surface, cmap='viridis', alpha=0.8)
    ax3d.set_xlabel('Direction 1')
    ax3d.set_ylabel('Direction 2')
    ax3d.set_zlabel('Loss')
    ax3d.set_title('3D Loss Surface')
    plt.colorbar(surf, ax=ax3d, shrink=0.5)
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_3d.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path.replace('.png', '_3d.png')}")


def main():
    print("Loading model...")
    checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model.pth'), map_location=device)
    
    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = checkpoint['idx_to_label']
    num_classes = checkpoint['num_classes']
    
    print(f"Loaded model with {num_classes} classes")
    
    model = models.efficientnet_b0(weights=None)
    in_features = 1280
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    original_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    plate_maps = load_plate_data()
    
    test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))[:100]
    print(f"Using {len(test_paths)} images for loss computation")
    
    test_dataset = MultiPatchDataset(test_paths, n_patches=3)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)
    
    print("Generating random directions...")
    direction1 = get_random_direction(original_weights)
    direction2 = get_random_direction(original_weights)
    
    alphas = np.linspace(-0.5, 0.5, 10)
    
    print(f"\nComputing loss surface ({len(alphas)}x{len(alphas)} grid)...")
    print("This may take a few minutes...")
    
    loss_surface = compute_loss_surface(
        model, original_weights, direction1, direction2, 
        alphas, test_loader, label_to_idx, device, plate_maps
    )
    
    print(f"\nLoss surface stats:")
    print(f"  Min loss: {loss_surface.min():.4f}")
    print(f"  Max loss: {loss_surface.max():.4f}")
    print(f"  Center loss: {loss_surface[len(alphas)//2, len(alphas)//2]:.4f}")
    
    output_path = os.path.join(BASE_DIR, 'loss_landscape.png')
    plot_loss_landscape(loss_surface, alphas, output_path, 
                        title=f"Loss Landscape around Trained Model\n(Center = Trained Model, {len(test_paths)} samples)")
    
    print("\n" + "="*50)
    print("DONE")
    print("="*50)
    print(f"Outputs:")
    print(f"  1. {BASE_DIR}/loss_landscape.png")
    print(f"  2. {BASE_DIR}/loss_landscape_3d.png")


if __name__ == "__main__":
    main()
