#!/usr/bin/env python3
"""Latent trajectory analysis for conditional flow matching.

Extracts UNet bottleneck (mid-block) features at each ODE step for two
classes, projects to 2D via PCA, and visualises how the internal
representations evolve.

Shows both directions (mutant→drug and drug→mutant) with a comparison.

Usage:
    python3 latent_trajectory.py
    python3 latent_trajectory.py --mutant gyrA_1 --drug Ciprofloxacin_2x
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from tqdm import tqdm

from flow_model import FreqFlowUNet

SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ── bottleneck hook ────────────────────────────────────────────────
class BottleneckCapture:
    def __init__(self):
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output.detach()


@torch.no_grad()
def collect_trajectory(
    model, class_id, num_steps=100, save_every=5,
    device='cuda', hook_container=None,
):
    """Run ODE from noise → image, collecting bottleneck latents and
    pixel-space frames at regular intervals."""
    model.eval()
    c_tensor = torch.full((1,), class_id, dtype=torch.long, device=device)

    x = torch.randn(1, 1, 224, 224, device=device)
    dt = 1.0 / num_steps

    frames = [x.cpu().squeeze()]
    latents = []

    for step in range(0, num_steps + 1):
        t = torch.full((1,), step * dt if step < num_steps else 1.0, device=device)
        out = model(x, t, class_labels=c_tensor)
        v = out[1]
        if step < num_steps:
            x = x + v * dt

        # capture bottleneck after forward
        if hook_container is not None and hook_container.features is not None:
            feat = hook_container.features
            pooled = feat.flatten(2).mean(dim=2).squeeze(0)
            latents.append(pooled.cpu())

        if step % save_every == 0:
            frames.append(x.cpu().squeeze())

    return frames, latents


# ── main ──────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mutant', default='gyrA_1')
    parser.add_argument('--drug', default='Ciprofloxacin_2x')
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--output', default='latent_trajectory.png')
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ── 1. load labels ──────────────────────────────────────────────
    from mil_model import load_labels
    _, class_names, label_to_idx = load_labels(PROJECT_ROOT, SCRIPT_DIR)

    for name in [args.mutant, args.drug]:
        if name not in label_to_idx:
            print(f"ERROR: '{name}' not found")
            sys.exit(1)

    mutant_id = label_to_idx[args.mutant]
    drug_id   = label_to_idx[args.drug]
    print(f"Mutant: {args.mutant:30s} → class {mutant_id}")
    print(f"Drug:   {args.drug:30s} → class {drug_id}")

    # ── 2. load model ───────────────────────────────────────────────
    flow_runs = sorted([d for d in os.listdir(SCRIPT_DIR)
                        if d.startswith('flow_run_')], reverse=True)
    ckpt_path = None
    for rd in flow_runs:
        cp = os.path.join(SCRIPT_DIR, rd, 'flow_best.pth')
        if os.path.exists(cp):
            ckpt_path = cp
            break
    if ckpt_path is None:
        print("ERROR: no flow_best.pth found")
        sys.exit(1)
    print(f"\nCheckpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_args = ckpt['args']

    block_channels = tuple(int(x) for x in ckpt_args['block_channels'].split(','))
    freq_block_channels = tuple(int(x) for x in
                                ckpt_args.get('freq_block_channels',
                                              ckpt_args['block_channels']).split(','))
    num_classes = len(class_names)

    model = FreqFlowUNet(
        in_channels=1, sample_size=224,
        block_out_channels=block_channels,
        freq_block_out_channels=freq_block_channels,
        layers_per_block=2,
        num_class_embeds=num_classes,
        freq_filter_D=ckpt_args.get('freq_filter_D', 8.0),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded (epoch {ckpt['epoch']})")

    # ── 3. register bottleneck hook on the spatial UNet ─────────────
    capture = BottleneckCapture()
    handle = model.spatial_unet.mid_block.register_forward_hook(capture.hook_fn)

    # warm-up to populate the hook
    _ = model(
        torch.zeros(1, 1, 224, 224, device=device),
        torch.zeros(1, device=device),
        class_labels=torch.zeros(1, dtype=torch.long, device=device),
    )

    # ── 4. collect trajectories ─────────────────────────────────────
    print(f"\nCollecting trajectories ({args.num_steps} steps, every {args.save_every})...")
    all_frames = {}
    all_latents = {}

    for label, cid, direction in [
        ('mutant→drug', mutant_id, 'forward'),
        ('drug→mutant', drug_id,   'reverse'),
        ('mutant',      mutant_id, 'mutant_only'),
        ('drug',        drug_id,   'drug_only'),
    ]:
        torch.manual_seed(SEED)
        print(f"  {label} (class {cid}) ...")
        # For forward/reverse, we do pure class-conditional trajectory
        frames, latents = collect_trajectory(
            model, cid, args.num_steps, args.save_every,
            device, hook_container=capture,
        )
        all_frames[label] = frames
        all_latents[label] = latents

    handle.remove()
    print(f"  Latent vectors per trajectory: {len(all_latents['mutant'])}")

    n_frames = len(all_frames['mutant'])
    step_labels = [f"t={i*args.save_every/args.num_steps:.2f}"
                   for i in range(n_frames)]

    # ── 5. PCA on ALL bottleneck features ───────────────────────────
    # This gives a shared embedding space so all trajectories are comparable
    all_feats = []
    labels_for_pca = []
    for label in ['mutant', 'drug', 'mutant→drug', 'drug→mutant']:
        for fv in all_latents[label]:
            all_feats.append(fv.numpy())
            labels_for_pca.append(label)

    all_feats = np.array(all_feats)
    pca = PCA(n_components=2)
    pca_feats = pca.fit_transform(all_feats)
    ev_ratio = pca.explained_variance_ratio_
    print(f"\nPCA explained variance: PC1={ev_ratio[0]:.1%}, PC2={ev_ratio[1]:.1%}")

    # split back per trajectory
    n_pts = len(all_latents['mutant'])
    idx = 0
    pca_trajs = {}
    for label in ['mutant', 'drug', 'mutant→drug', 'drug→mutant']:
        pca_trajs[label] = pca_feats[idx:idx + n_pts]
        idx += n_pts

    # ── 6. visualisation ────────────────────────────────────────────
    fig = plt.figure(figsize=(n_frames * 2.0 + 1, 9))
    gs = GridSpec(3, 2, height_ratios=[1.0, 0.8, 0.8], width_ratios=[0.85, 0.15],
                  hspace=0.4, wspace=0.02,
                  left=0.06, right=0.97, bottom=0.07, top=0.93)

    # ─ top: pixel-space trajectories ────────────────────────────────
    gs_top = gs[0, 0].subgridspec(2, n_frames, hspace=0.05, wspace=0.02)

    for col in range(n_frames):
        ax_m = fig.add_subplot(gs_top[0, col])
        img = (all_frames['mutant'][col] * 0.5 + 0.5).clamp(0, 1)
        ax_m.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)
        ax_m.set_xticks([]); ax_m.set_yticks([])
        ax_m.set_title(step_labels[col], fontsize=6)

        ax_d = fig.add_subplot(gs_top[1, col])
        img = (all_frames['drug'][col] * 0.5 + 0.5).clamp(0, 1)
        ax_d.imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)
        ax_d.set_xticks([]); ax_d.set_yticks([])

    # mut/drug labels on the side
    ax_lab = fig.add_subplot(gs[0, 1])
    ax_lab.axis('off')
    ax_lab.text(0.1, 0.72, args.mutant.replace("_", " "),
                fontsize=8, fontweight='bold', rotation=0, va='center')
    ax_lab.text(0.1, 0.28, args.drug.replace("_", " "),
                fontsize=8, fontweight='bold', rotation=0, va='center')

    # ─ middle: PCA latent trajectories ──────────────────────────────
    ax_mid = fig.add_subplot(gs[1, :])
    n_pts = len(pca_trajs['mutant'])
    colors = {
        'mutant':      ('#1f77b4', 'o'),
        'drug':        ('#d62728', 's'),
        'mutant→drug': ('#2ca02c', '^'),
        'drug→mutant': ('#ff7f0e', 'v'),
    }

    for label in ['mutant', 'drug', 'mutant→drug', 'drug→mutant']:
        c, marker = colors[label]
        traj = pca_trajs[label]
        ax_mid.plot(traj[:, 0], traj[:, 1], color=c, linewidth=1.2, alpha=0.6)
        # start
        ax_mid.scatter(traj[0, 0], traj[0, 1],
                       c=c, s=50, marker=marker, edgecolors='k', zorder=5)
        # end
        ax_mid.scatter(traj[-1, 0], traj[-1, 1],
                       c=c, s=100, marker='*', edgecolors='k', zorder=6,
                       label=f'{label} (end)')
        # arrows
        skip = max(1, n_pts // 8)
        for i in range(0, n_pts - 1, skip):
            ax_mid.annotate('', xy=traj[i+1], xytext=traj[i],
                            arrowprops=dict(arrowstyle='->', color=c, lw=0.6, alpha=0.5))

    ax_mid.set_xlabel(f'PC1 ({ev_ratio[0]:.1%})', fontsize=9)
    ax_mid.set_ylabel(f'PC2 ({ev_ratio[1]:.1%})', fontsize=9)
    ax_mid.set_title('UNet Bottleneck Trajectories (PCA)', fontsize=10, fontweight='bold')
    ax_mid.legend(fontsize=6, loc='best', framealpha=0.8, ncol=2)
    ax_mid.grid(True, alpha=0.3)

    # ─ bottom: comparison scatter to highlight class separation ─────
    ax_bot = fig.add_subplot(gs[2, :])
    # L2 distance between mutant and drug trajectories at each step
    traj_m = pca_trajs['mutant']
    traj_d = pca_trajs['drug']
    l2_dist = np.sqrt(np.sum((traj_m - traj_d) ** 2, axis=1))
    steps = np.arange(n_pts) / n_pts

    ax_bot.plot(steps, l2_dist, 'k-', linewidth=1.5, label='mutant vs drug')
    ax_bot.fill_between(steps, l2_dist, alpha=0.15, color='gray')
    ax_bot.set_xlabel('ODE progress t', fontsize=9)
    ax_bot.set_ylabel('L2 distance in PCA space', fontsize=9)
    ax_bot.set_title('Bottleneck Divergence Between Classes', fontsize=10, fontweight='bold')
    ax_bot.legend(fontsize=7)
    ax_bot.grid(True, alpha=0.3)

    out_path = os.path.join(SCRIPT_DIR, args.output)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
