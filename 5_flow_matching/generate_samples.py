#!/usr/bin/env python3
"""Generate 2x concentration antibiotic + guide-1 mutant images using trained flow model.

Generates:
  - 5 antibiotics at 2x MIC: Ciprofloxacin, Rifampicin, Kanamycin, Colistin, Trimethoprim
  - 5 mutants (guide 1): gyrA_1, rpoB_1, rpsL_1, lpxC_1, folA_1
  - 2 images per class = 20 total images
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from mil_model import load_labels
from flow_model import FlowUNet, sample as unet_sample
from dit_model import DiT, build_dit

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Target classes to generate
DRUG_2X = [
    'Ciprofloxacin_2x',
    'Rifampicin_2x',
    'Kanamycin_2x',
    'Colistin_2x',
    'Trimethoprim_2x',
]

MUTANT_G1 = [
    'gyrA_1',
    'rpoB_1',
    'rpsL_1',
    'lpxC_1',
    'folA_1',
]

print("=" * 60)
print("Flow Matching: Generate Antibiotic + Mutant Samples")
print("=" * 60)

print("\n[1/4] Loading class labels ...")
image_list, class_names, label_to_idx = load_labels(PROJECT_ROOT, SCRIPT_DIR)
num_classes = len(class_names)
print(f"  Total classes: {num_classes}")

target_names = DRUG_2X + MUTANT_G1
target_ids = {}
for name in target_names:
    if name in label_to_idx:
        target_ids[name] = label_to_idx[name]
        print(f"  ✓ {name:35s} → class {target_ids[name]}")
    else:
        print(f"  ✗ {name:35s} → NOT FOUND in labels")

if not target_ids:
    print("ERROR: No target classes found!")
    sys.exit(1)

# Find best checkpoint
print("\n[2/4] Loading model ...")
flow_runs = sorted([
    d for d in os.listdir(SCRIPT_DIR)
    if d.startswith('flow_run_')
], reverse=True)

checkpoint_path = None
for run_dir in flow_runs:
    cp = os.path.join(SCRIPT_DIR, run_dir, 'flow_best.pth')
    if os.path.exists(cp):
        checkpoint_path = cp
        break

if checkpoint_path is None:
    print("ERROR: No flow_best.pth found in any flow_run_ directory")
    sys.exit(1)

print(f"  Checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
ckpt_args = ckpt['args']
model_type = ckpt_args.get('model_type', 'unet')

if model_type == 'dit':
    dit_size = ckpt_args.get('dit_size', 'S')
    model = build_dit(
        dit_size, in_channels=1, img_size=224, patch_size=16,
        num_classes=num_classes,
    ).to(device)
    sample_fn = unet_sample
else:
    block_channels = tuple(int(x) for x in ckpt_args['block_channels'].split(','))
    model = FlowUNet(
        in_channels=1, sample_size=224,
        block_out_channels=block_channels,
        layers_per_block=2, num_class_embeds=num_classes,
    ).to(device)
    sample_fn = unet_sample

model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  Model loaded ({model_type}, epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'generated_samples')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n[3/4] Generating images ...")
NUM_IMAGES_PER_CLASS = 2
NUM_STEPS = 100
all_images = {}

for name, cid in target_ids.items():
    print(f"  Generating {name} ...")
    class_tensor = torch.full((NUM_IMAGES_PER_CLASS,), cid, dtype=torch.long, device=device)
    imgs = sample_fn(model, NUM_IMAGES_PER_CLASS, num_steps=NUM_STEPS,
                      class_labels=class_tensor)
    all_images[name] = imgs.cpu()

    # Save individual images
    for i in range(NUM_IMAGES_PER_CLASS):
        save_name = f"{name}_sample_{i+1}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        img_01 = (imgs[i].cpu() * 0.5 + 0.5).clamp(0, 1)
        plt.imsave(save_path, img_01.squeeze(), cmap='gray')
    print(f"    saved {NUM_IMAGES_PER_CLASS} images")

print(f"\n[4/4] Creating composite grid ...")
fig, axes = plt.subplots(len(DRUG_2X) + len(MUTANT_G1), NUM_IMAGES_PER_CLASS + 1,
                         figsize=((NUM_IMAGES_PER_CLASS + 1) * 2, (len(target_names)) * 2))

for row_idx, name in enumerate(target_names):
    axes[row_idx, 0].text(0.5, 0.5, name.replace('_', '\n'),
                          ha='center', va='center', fontsize=7, fontweight='bold',
                          transform=axes[row_idx, 0].transAxes)
    axes[row_idx, 0].axis('off')
    for col_idx in range(NUM_IMAGES_PER_CLASS):
        img = all_images[name][col_idx]
        img_01 = (img * 0.5 + 0.5).clamp(0, 1)
        axes[row_idx, col_idx + 1].imshow(img_01.squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[row_idx, col_idx + 1].axis('off')
        if row_idx == 0:
            axes[row_idx, col_idx + 1].set_title(f'Sample {col_idx+1}', fontsize=8)

plt.suptitle('Flow Matching: Generated Bacterial Microscopy Images', fontsize=12, y=0.98)
plt.tight_layout()
composite_path = os.path.join(OUTPUT_DIR, 'composite_grid.png')
fig.savefig(composite_path, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"  Composite saved: {composite_path}")
print(f"\n{'='*60}")
print(f"Done! {len(all_images)} classes × {NUM_IMAGES_PER_CLASS} images = {len(all_images) * NUM_IMAGES_PER_CLASS} total")
print(f"Output: {OUTPUT_DIR}")
print(f"{'='*60}")
