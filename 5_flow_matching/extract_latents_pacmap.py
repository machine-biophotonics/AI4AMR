#!/usr/bin/env python3
"""Extract bottleneck features from trained flow model.

Extracts at t=0.5 and t=1.0. Saves along with true class labels.

Usage:
    python3 extract_latents_pacmap.py
    python3 extract_latents_pacmap.py --checkpoint path/to/flow_best.pth
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from mil_model import FlowCropDataset, load_labels, extract_plate_from_path
from flow_model import FreqFlowUNet, FlowUNet

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to flow_best.pth (auto-detect latest)')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--output_dir', type=str, default=None)
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

output_dir = args.output_dir or os.path.join(SCRIPT_DIR, 'unsupervised_latents')
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("Bottleneck Feature Extraction")
print(f"Output: {output_dir}")
print("=" * 60)

# ── Data ──
print("\n[1/4] Loading data ...")
image_list, class_names, label_to_idx = load_labels(PROJECT_ROOT, SCRIPT_DIR)
num_classes = len(class_names)
print(f"  {len(image_list)} images total, {num_classes} classes")

# Filter to test plate only (P6 for this unsupervised run)
if args.checkpoint:
    ckpt_temp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    test_plate = ckpt_temp['args'].get('test_plate', 'P6')
else:
    test_plate = 'P6'
image_list = [x for x in image_list if extract_plate_from_path(x[0]) == test_plate]
print(f"  {len(image_list)} images in test plate {test_plate}")

ds = FlowCropDataset(image_list, augment=False)
loader = torch.utils.data.DataLoader(
    ds, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True,
    persistent_workers=True, prefetch_factor=2,
)

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

# ── Forward hook (same location as original extract_latents_pacmap.py) ──
mid_features = {}
def make_hook(key):
    def hook(module, input, output):
        mid_features[key] = output[0] if isinstance(output, tuple) else output
    return hook

handle = target_unet.up_blocks[0].register_forward_hook(make_hook('mid'))

# ── Extraction settings ──
# With --unsupervised, class_labels=0 for all, so conditional=unconditional
settings = [
    ('t05', 0.5),
    ('t10', 1.0),
]

all_feats = {name: [] for name, _ in settings}
all_labels = []

print("\n[3/4] Extracting latents ...")
with torch.no_grad():
    for imgs, class_ids in tqdm(loader, desc="Extract"):
        imgs = imgs.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)

        for name, t_val in settings:
            t_batch = torch.full((imgs.shape[0],), t_val, device=device)
            if unsupervised:
                labels_for_model = torch.zeros(imgs.shape[0], dtype=torch.long, device=device)
            else:
                labels_for_model = class_ids

            mid_features.clear()
            with torch.amp.autocast('cuda', enabled=True):
                if use_freq:
                    _, _ = model(imgs, t_batch, class_labels=labels_for_model)
                else:
                    _ = model(imgs, t_batch, class_labels=labels_for_model)

            feat = mid_features['mid']
            pooled = feat.flatten(2).mean(dim=2).cpu()
            all_feats[name].append(pooled)

        all_labels.append(class_ids.cpu())

handle.remove()

labels = torch.cat(all_labels, dim=0).numpy()
np.save(os.path.join(output_dir, 'labels.npy'), labels)
np.save(os.path.join(output_dir, 'class_names.npy'), np.array(class_names, dtype=object))

for name, _ in settings:
    feats = torch.cat(all_feats[name], dim=0).numpy()
    np.save(os.path.join(output_dir, f'feats_{name}.npy'), feats)
    print(f"  feats_{name}.npy: {feats.shape}")

n_imgs = len(image_list)
print(f"\nDone. Outputs in: {output_dir}")
print(f"  {os.path.join(output_dir, 'feats_t05.npy')}  (t=0.5, {n_imgs} x 256)")
print(f"  {os.path.join(output_dir, 'feats_t10.npy')}  (t=1.0, {n_imgs} x 256)")
print(f"  {os.path.join(output_dir, 'labels.npy')}     ({n_imgs} x 1, true class IDs)")
print(f"  {os.path.join(output_dir, 'class_names.npy')} ({n_imgs} class names)")
