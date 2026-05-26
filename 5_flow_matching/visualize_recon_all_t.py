#!/usr/bin/env python3
"""Reconstruction visualization across all flow-matching timesteps.

Shows 3 rows for each timestep t in [0.1, 0.2, ..., 1.0]:
  Row 0: x_t = (1-t)*x_0 + t*x_1  — noisy/interpolated image at timestep t
  Row 1: unconditional x₁ reconstruction from x_t (null-class forward pass)
  Row 2: class-conditioned x₁ reconstruction from x_t

Usage:
    python3 visualize_recon_all_t.py
    python3 visualize_recon_all_t.py --checkpoint path/to/flow_best.pth --index 5
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch

from mil_model import FlowCropDataset, load_labels
from flow_model import FlowUNet, FreqFlowUNet, StructFlowUNet, CombinedFlowUNet

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--output', type=str, default='recon_all_t.png')
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ── Auto-detect checkpoint ────────────────────────────────────
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
print("Reconstruction Across All Timesteps")
print(f"Checkpoint: {args.checkpoint}")
print(f"Image index: {args.index}")
print("=" * 60)

# ── Data ──────────────────────────────────────────────────────
print("\n[1/4] Loading data ...")
image_list, class_names, label_to_idx = load_labels(PROJECT_ROOT, SCRIPT_DIR)
num_classes = len(class_names)
ds = FlowCropDataset(image_list, augment=False)
img, label = ds[args.index]
img = img.unsqueeze(0).to(device)
class_name = class_names[label]
print(f"  Class: {class_name} (id={label})")

# ── Model ─────────────────────────────────────────────────────
print("\n[2/4] Loading model ...")
ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
ckpt_args = ckpt['args']
block_channels = tuple(int(x) for x in ckpt_args['block_channels'].split(','))
use_freq = ckpt_args.get('freq_flow', False)
use_struct = ckpt_args.get('struct_flow', False)

if use_struct and use_freq:
    freq_block_channels = tuple(int(x) for x in ckpt_args.get('freq_block_channels', ckpt_args['block_channels']).split(','))
    model = CombinedFlowUNet(
        in_channels=1, sample_size=224,
        block_out_channels=block_channels,
        freq_block_out_channels=freq_block_channels,
        layers_per_block=2, num_class_embeds=num_classes,
        freq_filter_D=ckpt_args.get('freq_filter_D', 8.0),
        use_freq=True, use_struct=True,
        latent_dim=ckpt_args.get('struct_latent_dim', 64),
    ).to(device)
elif use_struct:
    model = StructFlowUNet(
        in_channels=1, sample_size=224,
        block_out_channels=block_channels,
        layers_per_block=2, num_class_embeds=num_classes,
        latent_dim=ckpt_args.get('struct_latent_dim', 64),
    ).to(device)
elif use_freq:
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
print(f"  {type(model).__name__} loaded (epoch {ckpt['epoch']})")

# ── Add null embedding for unconditional guidance ─────────────
def add_null_embedding(module, n, device):
    old = module.class_embedding
    new = torch.nn.Embedding(n + 1, old.embedding_dim, device=device)
    new.weight.data[:n] = old.weight.data.to(device)
    new.weight.data[n] = old.weight.data.mean(dim=0).to(device)
    module.class_embedding = new

if use_struct and use_freq:
    add_null_embedding(model.main_unet, num_classes, device)
    add_null_embedding(model.freq_unet, num_classes, device)
elif use_freq:
    add_null_embedding(model.spatial_unet, num_classes, device)
    add_null_embedding(model.freq_unet, num_classes, device)
else:
    add_null_embedding(model.unet if hasattr(model, 'unet') else model.main_unet, num_classes, device)

# ── Generate ──────────────────────────────────────────────────
print("\n[3/4] Generating reconstructions ...")

torch.manual_seed(args.seed)
x_1 = img
x_0 = torch.randn_like(x_1)

timesteps = [round(i * 0.1, 1) for i in range(1, 11)]  # 0.1, 0.2, ..., 1.0
n = len(timesteps)

fig, axes = plt.subplots(3, n, figsize=(n * 2.2, 6))

with torch.no_grad():
    for i, t_val in enumerate(timesteps):
        t = torch.full((1,), t_val, device=device)
        t_b = t.view(1, 1, 1, 1)

        x_t = (1 - t_b) * x_0 + t_b * x_1

        # Unconditional recon: pass null class
        null_label = torch.tensor([num_classes], device=device)
        out_null = model(x_t, t, class_labels=null_label)
        v_null = out_null[1] if (use_freq or (use_struct and use_freq)) else out_null
        x1_pred_uncond = x_t + (1 - t.view(1, 1, 1, 1)) * v_null

        # Class-conditioned recon: pass image label
        cond_label = torch.tensor([label], device=device)
        out_cond = model(x_t, t, class_labels=cond_label)
        v_cond = out_cond[1] if (use_freq or (use_struct and use_freq)) else out_cond
        x1_pred_cond = x_t + (1 - t.view(1, 1, 1, 1)) * v_cond

        def to_01(tensor):
            return (tensor * 0.5 + 0.5).clamp(0, 1).squeeze().cpu().numpy()

        # Row 0: x_t
        axes[0, i].imshow(to_01(x_t), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f't={t_val:.1f}', fontsize=9)
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

        # Row 1: unconditional reconstruction
        axes[1, i].imshow(to_01(x1_pred_uncond), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'uncond', fontsize=8)
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

        # Row 2: class-conditioned reconstruction
        axes[2, i].imshow(to_01(x1_pred_cond), cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'cond', fontsize=8)
        axes[2, i].set_xticks([]); axes[2, i].set_yticks([])

axes[0, 0].set_ylabel('Noised x_t', fontsize=10)
axes[1, 0].set_ylabel('Uncond recon', fontsize=10)
axes[2, 0].set_ylabel('Cond recon', fontsize=10)
plt.suptitle(f'Class: {class_name}  |  Index: {args.index}',
             fontsize=10, y=0.98)
plt.tight_layout()
fig.savefig(args.output, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"\n[4/4] Saved: {args.output}")
