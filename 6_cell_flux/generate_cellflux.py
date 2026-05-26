#!/usr/bin/env python3
"""CellFlux generation — exact CellFlux ODE sampling.

Starts from real control images, runs ODE from t=0 to t=1 to generate
perturbed images conditioned on target perturbation embedding.

Usage:
    python3 generate_cellflux.py --checkpoint path/to/best_model.pth
    python3 generate_cellflux.py --checkpoint ... --cfg_scale 1.5 --num_steps 100
"""
import os, sys, math, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from cellflux_model import UNetModel
from cellflux_dataset import build_datasets, CellFluxTransform, _plate_from_path

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--test_plate', type=str, default='P6')
parser.add_argument('--cfg_scale', type=float, default=0.2,
                    help='CFG scale, w in (1+w)*cond - w*uncond')
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--num_samples', type=int, default=None)
parser.add_argument('--model_channels', type=int, default=128)
parser.add_argument('--num_res_blocks', type=int, default=4)
parser.add_argument('--channel_mult', type=str, default='2,2,2')
parser.add_argument('--attention_resolutions', type=str, default='2')
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--condition_dim', type=int, default=512)
parser.add_argument('--output', type=str, default=None)
args = parser.parse_args()

args.channel_mult = tuple(int(x) for x in args.channel_mult.split(','))
args.attention_resolutions = tuple(int(x) for x in args.attention_resolutions.split(','))

_, _, test_ds, num_pert_classes, class_names, pert2cond = build_datasets(
    PROJECT_ROOT, SCRIPT_DIR,
    test_plate=args.test_plate, val_split=0.0, seed=SEED,
)
cond2class = {v: k for k, v in pert2cond.items()}
print(f"Test: {len(test_ds)} paired samples, {num_pert_classes} perturbation classes")

model = UNetModel(
    in_channels=1, model_channels=args.model_channels, out_channels=1,
    num_res_blocks=args.num_res_blocks, channel_mult=args.channel_mult,
    attention_resolutions=args.attention_resolutions, dropout=args.dropout,
    condition_dim=args.condition_dim,
).to(device)

ckpt = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Loaded checkpoint epoch {ckpt.get('epoch', '?')} (val_loss={ckpt.get('val_loss', '?'):.4f})")

pert_embedding = torch.nn.Embedding(num_pert_classes, args.condition_dim).to(device)
if 'pert_embedding' in ckpt:
    pert_embedding.load_state_dict(ckpt['pert_embedding'])
pert_embedding.eval()

num_samples = min(args.num_samples, len(test_ds)) if args.num_samples else len(test_ds)

print(f"\nGenerating {num_samples} samples, Euler {args.num_steps} steps, CFG={args.cfg_scale}")

model.eval()
pert_embedding.eval()

results = []
with torch.no_grad():
    for i in tqdm(range(num_samples)):
        trt_path, trt_class = test_ds.perturbed_items[i]
        plate = _plate_from_path(trt_path)
        ctrl_path, _ = random.choice(test_ds.ctrl_by_plate[plate])
        cond_idx = test_ds.pert2cond[trt_class]

        ctrl_img = test_ds._load_crop(ctrl_path).unsqueeze(0).to(device)
        real_trt = test_ds._load_crop(trt_path)

        cond_emb = pert_embedding(torch.tensor([cond_idx], device=device))

        x_t = ctrl_img.clone()
        dt = 1.0 / args.num_steps

        for step_i in range(args.num_steps):
            t_val = (step_i + 0.5) * dt
            t_tensor = torch.full((1,), t_val, device=device)
            v_cond = model(x_t, t_tensor, extra={"concat_conditioning": cond_emb})
            v_uncond = model(x_t, t_tensor, extra={})
            v = (1.0 + args.cfg_scale) * v_cond - args.cfg_scale * v_uncond
            x_t = x_t + v * dt

        gen_img = x_t.clamp(-1, 1).cpu()
        results.append((ctrl_img.cpu(), gen_img, real_trt, cond_idx))

# Save visual comparison
os.makedirs(os.path.join(SCRIPT_DIR, 'generated'), exist_ok=True)
n_display = min(num_samples, 16)
fig, axes = plt.subplots(n_display, 3, figsize=(12, 4 * n_display))
for i in range(n_display):
    ctrl, gen, real, cidx = results[i]
    cls_name = class_names[cond2class[cidx.item()]]
    axes[i, 0].imshow(ctrl[0, 0], cmap='gray', vmin=-1, vmax=1)
    axes[i, 0].set_title("Control", fontsize=8)
    axes[i, 0].axis('off')
    axes[i, 1].imshow(gen[0, 0], cmap='gray', vmin=-1, vmax=1)
    axes[i, 1].set_title(f"Gen: {cls_name[:20]}", fontsize=8)
    axes[i, 1].axis('off')
    axes[i, 2].imshow(real[0], cmap='gray', vmin=-1, vmax=1)
    axes[i, 2].set_title(f"Real: {cls_name[:20]}", fontsize=8)
    axes[i, 2].axis('off')
plt.tight_layout()
out = args.output or os.path.join(SCRIPT_DIR, 'generated', f'cellflux_gen_{args.test_plate}.png')
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

mse = float(np.mean([F.mse_loss(g, r).item() for _, g, r, _ in results]))
print(f"Mean MSE: {mse:.6f}")
