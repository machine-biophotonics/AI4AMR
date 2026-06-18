#!/usr/bin/env python3
"""Extract ALL 144 crops per image from trained flow model's bottleneck.

For each image, extracts the 12x12 grid of 224x224 crops,
runs each through the model at t=0.5 and t=1.0,
saves per-crop bottleneck features (256-dim after GAP).

Output:
  feats_t05.npy   [N_images * 144, 256]  t=0.5
  feats_t10.npy   [N_images * 144, 256]  t=1.0
  labels.npy      [N_images * 144]        true class ID per crop
  crop_indices.npy [N_images * 144]       crop position index (0-143)
  image_indices.npy [N_images * 144]      image index within test set

Usage:
    python3 extract_latents_144.py
    python3 extract_latents_144.py --checkpoint path/to/flow_best.pth
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from mil_model import load_labels, extract_plate_from_path
from flow_model import FreqFlowUNet, FlowUNet

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to flow_best.pth (auto-detect latest)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Sub-batch size for processing 144 crops per image')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--grid_size', type=int, default=12)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Auto-detect checkpoint
if args.checkpoint is None:
    run_dirs = sorted([d for d in os.listdir(SCRIPT_DIR)
                       if d.startswith('flow_run_') and os.path.isdir(os.path.join(SCRIPT_DIR, d))])
    for rd in reversed(run_dirs):
        candidate = os.path.join(SCRIPT_DIR, rd, 'flow_best.pth')
        if os.path.exists(candidate):
            args.checkpoint = candidate
            break
    if args.checkpoint is None:
        print("No flow_best.pth found. Specify --checkpoint.")
        sys.exit(1)

output_dir = args.output_dir or os.path.join(SCRIPT_DIR, 'unsupervised_latents_144')
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("144-Crop Bottleneck Extraction")
print(f"Output: {output_dir}")
print("=" * 60)

# ── Data ──
print("\n[1/4] Loading data ...")
image_list, class_names, label_to_idx = load_labels(PROJECT_ROOT, SCRIPT_DIR)
num_classes = len(class_names)
print(f"  {len(image_list)} images total, {num_classes} classes")

# Read test plate from checkpoint
ckpt_temp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
test_plate = ckpt_temp['args'].get('test_plate', 'P6')
del ckpt_temp

image_list = [x for x in image_list if extract_plate_from_path(x[0]) == test_plate]
n_images = len(image_list)
print(f"  {n_images} images in test plate {test_plate}")
print(f"  {n_images * args.grid_size * args.grid_size} total crops ({args.grid_size}x{args.grid_size})")

# Compute crop positions from first image
sample_img = Image.open(image_list[0][0])
w, h = sample_img.size
stride = (w - args.crop_size) // (args.grid_size - 1)
positions = [(j * stride, i * stride)
             for i in range(args.grid_size) for j in range(args.grid_size)]
n_positions = len(positions)
print(f"  Image size: {w}x{h}, stride={stride}, {n_positions} positions")

# Normalization transform (no augment)
transform = A.Compose([
    A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=1.0),
    ToTensorV2(),
])

# ── Model ──
print("\n[2/4] Loading model ...")
ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
ckpt_args = ckpt['args']

block_channels = tuple(int(x) for x in ckpt_args['block_channels'].split(','))
use_freq = ckpt_args.get('freq_flow', False)
unsupervised = ckpt_args.get('unsupervised', False)
num_class_embeds = 1 if unsupervised else num_classes

if use_freq:
    freq_block_channels = tuple(int(x) for x in ckpt_args.get('freq_block_channels', ckpt_args['block_channels']).split(','))
    model = FreqFlowUNet(
        in_channels=1, sample_size=224,
        block_out_channels=block_channels,
        freq_block_out_channels=freq_block_channels,
        layers_per_block=2, num_class_embeds=num_class_embeds,
        freq_filter_D=ckpt_args.get('freq_filter_D', 8.0),
    ).to(device)
    target_unet = model.spatial_unet
else:
    model = FlowUNet(
        in_channels=1, sample_size=224,
        block_out_channels=block_channels,
        layers_per_block=2, num_class_embeds=num_class_embeds,
    ).to(device)
    target_unet = model.unet

model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  {'FreqFlowUNet' if use_freq else 'FlowUNet'} loaded (epoch {ckpt['epoch']})")

# ── Forward hook ──
mid_features = {}
def make_hook(key):
    def hook(module, input, output):
        mid_features[key] = output[0] if isinstance(output, tuple) else output
    return hook

handle = target_unet.up_blocks[0].register_forward_hook(make_hook('mid'))

# ── Extraction ──
settings = [('t05', 0.5), ('t10', 1.0)]
all_feats = {name: [] for name, _ in settings}
all_labels = []
all_crop_ids = []
all_img_ids = []

print(f"\n[3/4] Extracting {n_positions} crops from {n_images} images ...")
with torch.no_grad():
    for img_idx, (img_path, class_id) in enumerate(tqdm(image_list, desc="Images")):
        img = Image.open(img_path).convert('L')
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Extract all 144 crops
        crops = []
        for (x, y) in positions:
            crop = img_np[y:y + args.crop_size, x:x + args.crop_size]
            crop = transform(image=crop)['image']  # [1, 224, 224]
            crops.append(crop)

        crops = torch.stack(crops, dim=0)  # [144, 1, 224, 224]

        for name, t_val in settings:
            t_batch = torch.full((n_positions,), t_val, device=device)
            if unsupervised:
                labels_for_model = torch.zeros(n_positions, dtype=torch.long, device=device)
            else:
                labels_for_model = torch.full((n_positions,), class_id, dtype=torch.long, device=device)

            # Process in sub-batches if batch_size < 144
            all_pooled = []
            for start in range(0, n_positions, args.batch_size):
                end = min(start + args.batch_size, n_positions)
                batch_crops = crops[start:end].to(device, non_blocking=True)
                batch_t = t_batch[start:end]
                batch_labels = labels_for_model[start:end] if labels_for_model is not None else None

                mid_features.clear()
                with torch.amp.autocast('cuda', enabled=True):
                    if use_freq:
                        _, _ = model(batch_crops, batch_t, class_labels=batch_labels)
                    else:
                        _ = model(batch_crops, batch_t, class_labels=batch_labels)

                feat = mid_features['mid']
                pooled = feat.flatten(2).mean(dim=2).cpu()  # [B, 256]
                all_pooled.append(pooled)

            all_feats[name].append(torch.cat(all_pooled, dim=0))

        all_labels.append(torch.full((n_positions,), class_id, dtype=torch.long))
        all_crop_ids.append(torch.arange(n_positions, dtype=torch.long))
        all_img_ids.append(torch.full((n_positions,), img_idx, dtype=torch.long))

handle.remove()

# ── Save ──
print("\n[4/4] Saving ...")
labels = torch.cat(all_labels, dim=0).numpy()
crop_ids = torch.cat(all_crop_ids, dim=0).numpy()
img_ids = torch.cat(all_img_ids, dim=0).numpy()

np.save(os.path.join(output_dir, 'labels.npy'), labels)
np.save(os.path.join(output_dir, 'crop_indices.npy'), crop_ids)
np.save(os.path.join(output_dir, 'image_indices.npy'), img_ids)
np.save(os.path.join(output_dir, 'class_names.npy'), np.array(class_names, dtype=object))

total_crops = n_images * n_positions
for name, _ in settings:
    feats = torch.cat(all_feats[name], dim=0).numpy()  # [N*144, 256]
    np.save(os.path.join(output_dir, f'feats_{name}.npy'), feats)
    print(f"  feats_{name}.npy: {feats.shape}")

print(f"\nDone. Outputs in: {output_dir}")
print(f"  {total_crops} total crops from {n_images} images")
print(f"  Each image: {n_positions} crops x 2 timesteps = {n_positions * 2} features")
