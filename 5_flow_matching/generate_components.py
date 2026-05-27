#!/usr/bin/env python3
"""Generate sample images from specific GMM components.

Loads a trained checkpoint, samples z from the top-N most populated
GMM components, decodes to image, and refines via ODE.

Usage:
    python3 generate_components.py \
        --checkpoint flow_run_flow_20260526_160835/flow_best.pth \
        --output_dir flow_run_flow_20260526_160835/component_samples \
        --components 18 6 30 3
"""

import os, sys, warnings, argparse
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mil_model import FlowCropDataset, load_labels
from flow_model import CombinedFlowUNet, StructFlowUNet

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='./component_samples')
parser.add_argument('--components', type=int, nargs='+', default=None,
                    help='Component indices to sample from. If None, uses top-4 by weight.')
parser.add_argument('--num_steps', type=int, default=100,
                    help='ODE steps for refinement')
parser.add_argument('--n_per_component', type=int, default=1)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print("\n[1/5] Loading checkpoint ...")
ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
epoch = ckpt['epoch']
ckpt_args = ckpt['args']
print(f"  Epoch {epoch}")

num_classes = ckpt_args.get('num_classes', 185)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

print("\n[2/5] Loading data ...")
image_list, class_names, label_to_idx = load_labels(PROJECT_ROOT, SCRIPT_DIR)
print(f"  {len(image_list)} images, {len(class_names)} classes")

print("\n[3/5] Rebuilding model ...")
block_channels = tuple(int(x) for x in ckpt_args.get('block_channels', '32,64,128,256').split(','))
freq_bc = ckpt_args.get('freq_block_channels', '32,64,128,256')
freq_block_channels = tuple(int(x) for x in freq_bc.split(','))
use_freq = ckpt_args.get('freq_flow', False)
use_struct = ckpt_args.get('struct_flow', False)
use_gmm = ckpt_args.get('gmm', False)
unsupervised_gmm = ckpt_args.get('unsupervised_gmm', False)
gmm_components = ckpt_args.get('gmm_components', 185)
latent_dim = ckpt_args.get('struct_latent_dim', 64)

if use_freq and use_struct:
    model = CombinedFlowUNet(
        in_channels=1, sample_size=224,
        block_out_channels=block_channels,
        freq_block_out_channels=freq_block_channels,
        layers_per_block=2, num_class_embeds=num_classes,
        freq_filter_D=ckpt_args.get('freq_filter_D', 8.0),
        use_freq=True, use_struct=True,
        latent_dim=latent_dim,
        use_gmm=use_gmm, gmm_components=gmm_components,
        unsupervised_gmm=unsupervised_gmm,
    ).to(device)
elif use_struct:
    model = StructFlowUNet(
        in_channels=1, sample_size=224,
        block_out_channels=block_channels,
        layers_per_block=2, num_class_embeds=num_classes,
        latent_dim=latent_dim,
        use_gmm=use_gmm, gmm_components=gmm_components,
        unsupervised_gmm=unsupervised_gmm,
    ).to(device)
else:
    print("ERROR: checkpoint has no structured latent")
    sys.exit(1)

model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

# ─── Get GMM component info ────────────────────────────────────────────────
gmm = model.gmm
means = gmm.means.detach()  # (K, D)
logvars = gmm.logvars.detach()  # (K, D)
logits = gmm.logits.detach()  # (K,)
weights = torch.softmax(logits, dim=-1)  # π_c
stds = (0.5 * logvars).exp()  # σ_c

K = gmm.n_components
print(f"\n  GMM: {K} components, latent_dim={latent_dim}")

# Determine component order (by weight, descending)
comp_order = torch.argsort(weights, descending=True)
comp_ids = args.components if args.components else comp_order[:4].tolist()

print(f"\n  Sampling from components: {comp_ids}")
for c in comp_ids:
    print(f"    C{c}: weight={weights[c]:.3f}, mean_norm={means[c].norm():.3f}, std_mean={stds[c].mean():.3f}")

# ─── Generate images ───────────────────────────────────────────────────────
print(f"\n[4/5] Generating {args.n_per_component} image(s) per component ...")
NUM_STEPS = args.num_steps
dt = 1.0 / NUM_STEPS

all_imgs = []

for c in comp_ids:
    for s in range(args.n_per_component):
        # 1. Sample z from component c
        z = means[c:c+1] + torch.randn(1, latent_dim, device=device) * stds[c:c+1]

        # 2. Decode: x_z = decoder(z)
        x_z = model.decode(z)

        # 3. Add exogenous noise: x_0 = x_z + ε
        x = x_z + torch.randn_like(x_z)

        # 4. Euler ODE integration from t=0 to t=1
        # Class conditioning required by UNet; use class 0 (arbitrary)
        cls_label = torch.zeros(1, dtype=torch.long, device=device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            for i in range(NUM_STEPS):
                t = torch.full((1,), i * dt, device=device)
                out = model(x, t, class_labels=cls_label)
                v = out[1] if use_freq else out
                x = x + v * dt

        img = x.clamp(-1, 1).cpu()
        all_imgs.append((c, img))

# ─── Save: 4-row grid ─────────────────────────────────────────────────────
print(f"\n[5/5] Saving images to {args.output_dir} ...")
n_cols = args.n_per_component
fig, axes = plt.subplots(len(comp_ids), max(1, n_cols),
                         figsize=(max(3, n_cols * 3), len(comp_ids) * 3))
if len(comp_ids) == 1:
    axes = np.array([axes])
if n_cols == 1:
    axes = axes.reshape(-1, 1)

for row, c in enumerate(comp_ids):
    for col in range(n_cols):
        idx = row * n_cols + col
        _, img_tensor = all_imgs[idx]
        ax = axes[row][col]
        ax.imshow(img_tensor[0, 0], cmap='gray', vmin=-1, vmax=1)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(f"C{c}\nπ={weights[c]:.3f}", fontsize=10)

fig.suptitle(f"GMM Component Samples (epoch {epoch})", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(args.output_dir, 'component_samples_grid.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)

# Save individual
for c, img_tensor in all_imgs:
    fp = os.path.join(args.output_dir, f'component_{c:02d}.png')
    fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
    ax2.imshow(img_tensor[0, 0], cmap='gray', vmin=-1, vmax=1)
    ax2.set_title(f"Component C{c} (π={weights[c]:.3f})")
    ax2.axis('off')
    fig2.tight_layout()
    fig2.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig2)

print(f"  Grid:   {args.output_dir}/component_samples_grid.png")
for c, _ in all_imgs:
    print(f"  C{c:02d}:  {args.output_dir}/component_{c:02d}.png")
print("Done.")
