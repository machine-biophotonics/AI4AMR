#!/usr/bin/env python3
"""Show how a single image looks at different timesteps along the flow path.

For CFM with linear interpolant: x_t = (1-t) * x_0 + t * x_1

Usage:
    python3 visualize_timesteps.py
    python3 visualize_timesteps.py --checkpoint path/to/flow_best.pth
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
from tqdm import tqdm

from mil_model import FlowCropDataset, load_labels
from flow_model import FlowUNet, FreqFlowUNet

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--output', type=str, default='timestep_viz.png')
parser.add_argument('--index', type=int, default=0, help='Index of image to visualize')
parser.add_argument('--timesteps', type=float, nargs='+', default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help='Timesteps to show')
parser.add_argument('--seed', type=int, default=42, help='Random seed for noise x_0')
parser.add_argument('--uncond', action='store_true', default=False,
                    help='Use unconditional (null class) instead of the image label')
args = parser.parse_args()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

print("=" * 60)
print("Timestep Visualization")
print(f"Checkpoint: {args.checkpoint}")
print(f"Image index: {args.index}")
print("=" * 60)

# ── Data ──────────────────────────────────────────────────────
print("\n[1/3] Loading data ...")
image_list, class_names, label_to_idx = load_labels(PROJECT_ROOT, SCRIPT_DIR)
num_classes = len(class_names)
ds = FlowCropDataset(image_list, augment=False)
img, label = ds[args.index]
img = img.unsqueeze(0).to(device)  # (1, 1, H, W)
class_name = class_names[label]
print(f"  Class: {class_name} (id={label})")

# ── Model ─────────────────────────────────────────────────────
print("\n[2/3] Loading model ...")
ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
ckpt_args = ckpt['args']
block_channels = tuple(int(x) for x in ckpt_args['block_channels'].split(','))
use_freq = ckpt_args.get('freq_flow', False)

if use_freq:
    freq_block_channels = tuple(int(x) for x in ckpt_args.get('freq_block_channels', ckpt_args['block_channels']).split(','))
    model = FreqFlowUNet(
        in_channels=1, sample_size=224,
        block_out_channels=block_channels,
        freq_block_out_channels=freq_block_channels,
        layers_per_block=2, num_class_embeds=num_classes,
        freq_filter_D=ckpt_args.get('freq_filter_D', 8.0),
    ).to(device)
else:
    model = FlowUNet(
        in_channels=1, sample_size=224,
        block_out_channels=block_channels,
        layers_per_block=2, num_class_embeds=num_classes,
    ).to(device)

model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  {'FreqFlowUNet' if use_freq else 'FlowUNet'} loaded (epoch {ckpt['epoch']})")

if args.uncond:
    def add_null_embedding(unet, n, device):
        old = unet.class_embedding
        new = torch.nn.Embedding(n + 1, old.embedding_dim, device=device)
        new.weight.data[:n] = old.weight.data.to(device)
        new.weight.data[n] = old.weight.data.mean(dim=0).to(device)
        unet.class_embedding = new

    if use_freq:
        add_null_embedding(model.spatial_unet, num_classes, device)
        add_null_embedding(model.freq_unet, num_classes, device)
    else:
        add_null_embedding(model.unet, num_classes, device)

# ── Generate interpolation path ──────────────────────────────
print("\n[3/3] Generating timestep interpolants ...")

torch.manual_seed(args.seed)
x_1 = img  # clean image
x_0 = torch.randn_like(x_1)  # source noise (fixed seed)

t_steps = sorted(args.timesteps)

# Also get velocity prediction at each timestep
fig, axes = plt.subplots(2, len(t_steps), figsize=(len(t_steps) * 2.5, 5))

with torch.no_grad():
    for i, t_val in enumerate(t_steps):
        t = torch.full((1,), t_val, device=device)
        t_b = t.view(1, 1, 1, 1)

        x_t = (1 - t_b) * x_0 + t_b * x_1  # linear interpolant

        label_id = torch.tensor([num_classes if args.uncond else label], device=device)
        if use_freq:
            _, v_pred = model(x_t, t, class_labels=label_id)
        else:
            v_pred = model(x_t, t, class_labels=label_id)

        # Move to CPU, clamp to [0,1] for display
        img_t = (x_t * 0.5 + 0.5).clamp(0, 1).squeeze().cpu()

        # Row 0: noisy/interpolated image
        ax = axes[0, i] if len(t_steps) > 1 else axes[0]
        ax.imshow(img_t, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f't={t_val:.2f}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # Row 1: velocity field (delta between channels)
        v = v_pred.squeeze().cpu()
        v_mag = v.abs().numpy()
        ax2 = axes[1, i] if len(t_steps) > 1 else axes[1]
        im = ax2.imshow(v_mag, cmap='Reds', vmin=0, vmax=None)
        ax2.set_title(f'|v| max={v_mag.max():.2f}', fontsize=9)
        ax2.set_xticks([])
        ax2.set_yticks([])

plt.suptitle(f'Class: {class_name}  |  Cond: {"null" if args.uncond else "True"}  |  Seed: {args.seed}',
             fontsize=11)
plt.tight_layout()
fig.savefig(args.output, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"\nSaved: {args.output}")
