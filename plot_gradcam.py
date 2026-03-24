"""
Grad-CAM Visualization for CRISPRi Classification.
Generates heatmaps showing which regions of the image drive the model's predictions.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Normalize
from PIL import Image
import cv2
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


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def compute_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap) + 1e-10
        
        return heatmap.cpu().numpy(), class_idx.cpu().numpy()


class MultiPatchDataset(Dataset):
    def __init__(self, image_paths, n_patches=1):
        self.image_paths = image_paths
        self.n_patches = n_patches
        self.patch_size = 224
        self.edge_margin = 200
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        center_w_start = self.edge_margin
        center_w_end = w - self.edge_margin
        center_h_start = self.edge_margin
        center_h_end = h - self.edge_margin
        
        center_w = center_w_end - center_w_start
        center_h = center_h_end - center_h_start
        
        center_x = center_w_start + center_w // 2
        center_y = center_h_start + center_h // 2
        
        left = center_x - self.patch_size // 2
        top = center_y - self.patch_size // 2
        
        patch = image.crop((left, top, left + self.patch_size, top + self.patch_size))
        patch_tensor = ToTensor()(patch)
        patch_tensor = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(patch_tensor)
        
        patch_np = np.array(patch)
        
        return patch_tensor.unsqueeze(0), patch_np, img_path


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


def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img + 0.4 * heatmap_colored).astype(np.uint8)
    return overlay


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
    model.eval()
    
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    
    plate_maps = load_plate_data()
    
    test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))
    
    classes_to_show = ['WT', 'dnaB_1', 'gyrA_1', 'ftsZ_1', 'parC_1']
    sample_per_class = 3
    
    selected_paths = []
    for class_name in classes_to_show:
        if class_name not in label_to_idx:
            continue
        class_paths = [p for p in test_paths if get_label_from_path(p, plate_maps) == class_name]
        selected_paths.extend(class_paths[:sample_per_class])
    
    print(f"Generating Grad-CAM for {len(selected_paths)} images...")
    
    os.makedirs(os.path.join(BASE_DIR, 'gradcam_samples'), exist_ok=True)
    
    fig, axes = plt.subplots(len(selected_paths), 3, figsize=(12, 4 * len(selected_paths)))
    if len(selected_paths) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (img_path) in enumerate(tqdm(selected_paths, desc="Generating Grad-CAM")):
        dataset = MultiPatchDataset([img_path])
        patch_tensor, patch_img, _ = dataset[0]
        patch_tensor = patch_tensor.to(device)
        
        true_label = get_label_from_path(img_path, plate_maps)
        heatmap, pred_idx = gradcam.compute_cam(patch_tensor, class_idx=None)
        pred_label = idx_to_label[int(pred_idx)]
        
        overlaid = overlay_heatmap(patch_img, heatmap)
        
        axes[idx, 0].imshow(patch_img)
        axes[idx, 0].set_title(f'Crop (224x224)\nTrue: {true_label}', fontsize=10)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(heatmap, cmap='jet')
        axes[idx, 1].set_title('Grad-CAM Heatmap', fontsize=10)
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(overlaid)
        axes[idx, 2].set_title(f'Overlay\nPred: {pred_label}', fontsize=10)
        axes[idx, 2].axis('off')
        
        heatmap_path = os.path.join(BASE_DIR, 'gradcam_samples', f'heatmap_{idx}.png')
        cv2.imwrite(heatmap_path, cv2.cvtColor((overlaid * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'gradcam_overview.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {BASE_DIR}/gradcam_overview.png")
    
    print("\n" + "="*50)
    print("DONE")
    print("="*50)
    print(f"Outputs:")
    print(f"  1. {BASE_DIR}/gradcam_overview.png")
    print(f"  2. {BASE_DIR}/gradcam_samples/")


if __name__ == "__main__":
    main()
