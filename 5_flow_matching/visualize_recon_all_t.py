#!/usr/bin/env python3
"""Reconstruction visualization across all flow-matching timesteps.

Shows 6 rows for each timestep t in [0.1, 0.2, ..., 1.0]:
  Row 0: x_t = (1-t)*x_0 + t*x_1  — noisy/interpolated image
  Row 1: low-frequency component of x_t
  Row 2: high-frequency component of x_t
  Row 3: class-conditioned x₁ reconstruction from x_t
  Row 4: low-frequency component of reconstructed x₁
  Row 5: high-frequency component of reconstructed x₁

Each signal (x_t, recon) is grouped with its own Fourier decomposition below it.

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
from flow_model import FlowUNet, FreqFlowUNet, StructFlowUNet, CombinedFlowUNet, Fourier_filter

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--output', type=str, default='recon_all_t.png')
parser.add_argument('--index', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--freq_D', type=float, default=8.0,
                    help='Cutoff freq for Fourier decomposition')
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

D = ckpt_args.get('freq_filter_D', args.freq_D)
timesteps = [round(i * 0.1, 1) for i in range(1, 11)]
n = len(timesteps)

fig, axes = plt.subplots(6, n, figsize=(n * 2.2, 12))

with torch.no_grad():
    for i, t_val in enumerate(timesteps):
        t = torch.full((1,), t_val, device=device)
        t_b = t.view(1, 1, 1, 1)

        x_t = (1 - t_b) * x_0 + t_b * x_1

        # Class-conditioned recon
        cond_label = torch.tensor([label], device=device)
        out_cond = model(x_t, t, class_labels=cond_label)
        v_cond = out_cond[1] if (use_freq or (use_struct and use_freq)) else out_cond
        x1_pred_cond = x_t + (1 - t.view(1, 1, 1, 1)) * v_cond

        # Frequency decomposition
        x_t_low, x_t_high = Fourier_filter(x_t, D)
        recon_low, recon_high = Fourier_filter(x1_pred_cond.clamp(-1, 1), D)

        def to_01(tensor):
            return (tensor * 0.5 + 0.5).clamp(0, 1).squeeze().cpu().numpy()

        imgs = [
            (x_t,          f't={t_val:.1f}',  9),
            (x_t_low,       'low',             8),
            (x_t_high,      'high',            8),
            (x1_pred_cond,  'recon',           8),
            (recon_low,     'recon low',       8),
            (recon_high,    'recon high',      8),
        ]
        for r, (tens, title, fs) in enumerate(imgs):
            axes[r, i].imshow(to_01(tens), cmap='gray', vmin=0, vmax=1)
            axes[r, i].set_title(title, fontsize=fs)
            axes[r, i].set_xticks([]); axes[r, i].set_yticks([])

ylabels = ['x_t', 'x_t low-freq', 'x_t high-freq',
           'cond recon', 'recon low-freq', 'recon high-freq']
for r, lbl in enumerate(ylabels):
    axes[r, 0].set_ylabel(lbl, fontsize=9)

plt.suptitle(f'Class: {class_name}  |  Index: {args.index}  |  D={D}',
             fontsize=10, y=0.98)
plt.tight_layout()
fig.savefig(args.output, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"\n[4/4] Saved: {args.output}")
