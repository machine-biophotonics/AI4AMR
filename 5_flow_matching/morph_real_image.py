#!/usr/bin/env python3
"""Morph real microscopy images between mutant and drug phenotypes.

Two methods:
  1. Euler ODE inversion + generation
     Invert real→noise with source class (backward ODE),
     generate noise→image with target class (forward ODE).
  2. SDEdit
     Partial noise at t=t_noise, then forward ODE with target class.

Both directions: mutant→drug and drug→mutant.

Usage:
    python3 morph_real_image.py
    python3 morph_real_image.py --mutant gyrA_1 --drug Ciprofloxacin_2x --sdedit_t 0.3
"""
import os, sys, warnings, argparse
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from tqdm import tqdm

from mil_model import load_labels
from flow_model import FreqFlowUNet

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def load_real_image(path, crop_size=224):
    """Load single image, center-crop, normalize to [-1, 1]."""
    img = Image.open(path)
    w, h = img.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    crop = img.crop((left, top, left + crop_size, top + crop_size))
    arr = np.array(crop).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1.0:
        arr = arr / (65535.0 if arr.max() > 255 else 255.0)
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    return tensor.to(device)


def find_first_image(image_list, class_id):
    """Find first image path for given class_id."""
    for path, cid in image_list:
        if cid == class_id:
            return path
    return None


@torch.no_grad()
def ode_inversion(model, x_real, class_label, num_steps=100, freq_flow=True):
    """Backward Euler ODE: real image (t=1) -> noise latent (t=0)."""
    x = x_real.clone()
    dt = 1.0 / num_steps
    frames = [x.cpu().squeeze()]
    for step in range(num_steps, 0, -1):
        t = torch.full((1,), step * dt, device=device)
        out = model(x, t, class_labels=class_label)
        v = out[1] if freq_flow else out
        x = x - v * dt
        frames.append(x.cpu().squeeze())
    return x, frames


@torch.no_grad()
def ode_generation(model, x_noise, class_label, num_steps=100,
                   freq_flow=True, save_every=10):
    """Forward Euler ODE: noise (t=0) -> image (t=1), collect frames."""
    x = x_noise.clone()
    dt = 1.0 / num_steps
    frames = [x.cpu().squeeze()]
    for step in range(num_steps):
        t = torch.full((1,), step * dt, device=device)
        out = model(x, t, class_labels=class_label)
        v = out[1] if freq_flow else out
        x = x + v * dt
        if step % save_every == 0 or step == num_steps - 1:
            frames.append(x.cpu().squeeze())
    return x.clamp(-1, 1), frames


@torch.no_grad()
def sdedit_morph(model, x_real, target_label, t_noise=0.3, num_steps=100,
                 freq_flow=True, save_every=10):
    """SDEdit: mix noise at t_noise, forward ODE with target class."""
    noise = torch.randn_like(x_real)
    x_mixed = (1 - t_noise) * noise + t_noise * x_real
    start_step = int(t_noise * num_steps)
    x = x_mixed.clone()
    dt = 1.0 / num_steps
    frames = [x.cpu().squeeze()]
    for step in range(start_step, num_steps):
        t = torch.full((1,), step * dt, device=device)
        out = model(x, t, class_labels=target_label)
        v = out[1] if freq_flow else out
        x = x + v * dt
        if step % save_every == 0 or step == num_steps - 1:
            frames.append(x.cpu().squeeze())
    return x.clamp(-1, 1), frames


def tensor_to_img(tensor):
    """Convert [-1,1] tensor to [0,1] numpy for display."""
    return (tensor.cpu() * 0.5 + 0.5).clamp(0, 1).squeeze().numpy()


def subsample_indices(n_total, n_wanted):
    """Pick n_wanted evenly spaced indices from [0, n_total)."""
    if n_total <= n_wanted:
        return list(range(n_total))
    return np.linspace(0, n_total - 1, n_wanted, dtype=int).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mutant', default='gyrA_1')
    parser.add_argument('--drug', default='Ciprofloxacin_2x')
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--sdedit_t', type=float, default=0.3,
                        help='Noise level t for SDEdit (0=full noise, 1=no noise)')
    parser.add_argument('--output', default='trajectory_morph.png')
    args = parser.parse_args()

    print("=" * 60)
    print("Real Image Morph: Mutant <-> Drug via Flow Matching")
    print("=" * 60)

    # ── 1. Load labels & find real images ────────────────────────────
    print("\n[1/5] Loading labels & finding real images ...")
    image_list, class_names, label_to_idx = load_labels(PROJECT_ROOT, SCRIPT_DIR)
    num_classes = len(class_names)
    print(f"  Total classes: {num_classes}")

    if args.mutant not in label_to_idx:
        print(f"ERROR: '{args.mutant}' not found in labels")
        sys.exit(1)
    if args.drug not in label_to_idx:
        print(f"ERROR: '{args.drug}' not found in labels")
        sys.exit(1)
    mutant_id = label_to_idx[args.mutant]
    drug_id = label_to_idx[args.drug]

    mut_path = find_first_image(image_list, mutant_id)
    drug_path = find_first_image(image_list, drug_id)
    if mut_path is None or drug_path is None:
        print("ERROR: no real image found for mutant or drug class")
        sys.exit(1)
    print(f"  Mutant image: {mut_path}")
    print(f"  Drug image:   {drug_path}")

    mut_label = torch.full((1,), mutant_id, dtype=torch.long, device=device)
    drug_label = torch.full((1,), drug_id, dtype=torch.long, device=device)

    # ── 2. Load real images ──────────────────────────────────────────
    print("\n[2/5] Loading real images ...")
    x_mut_real = load_real_image(mut_path)
    x_drug_real = load_real_image(drug_path)
    print(f"  Mutant: shape={x_mut_real.shape}, range=[{x_mut_real.min():.2f}, {x_mut_real.max():.2f}]")
    print(f"  Drug:   shape={x_drug_real.shape}, range=[{x_drug_real.min():.2f}, {x_drug_real.max():.2f}]")

    # ── 3. Load model ────────────────────────────────────────────────
    print("\n[3/5] Loading model ...")
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
    print(f"  Checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_args = ckpt['args']
    block_channels = tuple(int(x) for x in ckpt_args['block_channels'].split(','))
    freq_block_channels = tuple(int(x) for x in
                                ckpt_args.get('freq_block_channels',
                                              ckpt_args['block_channels']).split(','))
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
    freq_flow = hasattr(model, 'freq_unet')
    print(f"  Model: FreqFlowUNet (freq={freq_flow}, epoch={ckpt['epoch']})")

    # ── 4. Run morph pipelines ───────────────────────────────────────
    print(f"\n[4/5] Running morph pipelines ({args.num_steps} steps)...")

    # ── Method A: Euler Inversion + Generation ──
    print("\n  -- Euler Inversion + Generation --")

    print("  [M->D] Inverting mutant image (backward ODE) ...")
    z_mut, inv_frames_m = ode_inversion(
        model, x_mut_real, mut_label, args.num_steps, freq_flow)
    print(f"    noise latent: mean={z_mut.mean():.4f}, std={z_mut.std():.4f}")

    print("  [M->D] Generating drug image (forward ODE) ...")
    x_m_to_d, gen_md_frames = ode_generation(
        model, z_mut, drug_label, args.num_steps, freq_flow, args.save_every)
    print(f"    result range: [{x_m_to_d.min():.2f}, {x_m_to_d.max():.2f}]")

    print("  [D->M] Inverting drug image (backward ODE) ...")
    z_drug, inv_frames_d = ode_inversion(
        model, x_drug_real, drug_label, args.num_steps, freq_flow)
    print(f"    noise latent: mean={z_drug.mean():.4f}, std={z_drug.std():.4f}")

    print("  [D->M] Generating mutant image (forward ODE) ...")
    x_d_to_m, gen_dm_frames = ode_generation(
        model, z_drug, mut_label, args.num_steps, freq_flow, args.save_every)
    print(f"    result range: [{x_d_to_m.min():.2f}, {x_d_to_m.max():.2f}]")

    # ── Method B: SDEdit ──
    print(f"\n  -- SDEdit (t_noise={args.sdedit_t}) --")

    print("  [M->D] SDEdit mutant->drug ...")
    x_sde_md, sde_md_frames = sdedit_morph(
        model, x_mut_real, drug_label, args.sdedit_t,
        args.num_steps, freq_flow, args.save_every)

    print("  [D->M] SDEdit drug->mutant ...")
    x_sde_dm, sde_dm_frames = sdedit_morph(
        model, x_drug_real, mut_label, args.sdedit_t,
        args.num_steps, freq_flow, args.save_every)

    # ── 5. Visualization ─────────────────────────────────────────────
    print(f"\n[5/5] Creating visualization ...")

    # Save individual morphs
    morph_dir = os.path.join(SCRIPT_DIR, 'morph_results')
    os.makedirs(morph_dir, exist_ok=True)
    for name, img in [('euler_mutant_to_drug', x_m_to_d),
                      ('euler_drug_to_mutant', x_d_to_m),
                      ('sdedit_mutant_to_drug', x_sde_md),
                      ('sdedit_drug_to_mutant', x_sde_dm)]:
        plt.imsave(os.path.join(morph_dir, f'{name}.png'),
                   tensor_to_img(img), cmap='gray')
        print(f"  Saved: morph_results/{name}.png")

    # Build composite figure
    # Each row: [label] [src] [inv_0..inv_4] [gen_0..gen_4] [result] [ref]
    n_inv_shown = 5
    n_gen_shown = 5
    n_cols = 1 + 1 + n_inv_shown + n_gen_shown + 1 + 1

    inv_idx_m = subsample_indices(len(inv_frames_m), n_inv_shown)
    inv_idx_d = subsample_indices(len(inv_frames_d), n_inv_shown)
    gen_idx_md = subsample_indices(len(gen_md_frames), n_gen_shown)
    gen_idx_dm = subsample_indices(len(gen_dm_frames), n_gen_shown)
    sde_idx_md = subsample_indices(len(sde_md_frames), n_gen_shown)
    sde_idx_dm = subsample_indices(len(sde_dm_frames), n_gen_shown)

    fig = plt.figure(figsize=(n_cols * 2.1 + 1, 11))
    gs = GridSpec(4, n_cols + 1,
                  width_ratios=[0.07] + [1.0] * n_cols,
                  hspace=0.12, wspace=0.04,
                  left=0.02, right=0.98, bottom=0.05, top=0.93)

    titles = ['Src', 'Inv t->0', '', '', '', '',
              'Gen 0->t', '', '', '', '',
              'Result', 'Ref']

    rows_data = [
        # (key, label, src_img, inv_frames, gen_frames, result, ref_img, inv_idx, gen_idx)
        ('euler_md', 'Euler\nM->D',
         x_mut_real, inv_frames_m, gen_md_frames, x_m_to_d, x_drug_real,
         inv_idx_m, gen_idx_md),
        ('euler_dm', 'Euler\nD->M',
         x_drug_real, inv_frames_d, gen_dm_frames, x_d_to_m, x_mut_real,
         inv_idx_d, gen_idx_dm),
        ('sdedit_md', 'SDEdit\nM->D',
         x_mut_real, None, sde_md_frames, x_sde_md, x_drug_real,
         None, sde_idx_md),
        ('sdedit_dm', 'SDEdit\nD->M',
         x_drug_real, None, sde_dm_frames, x_sde_dm, x_mut_real,
         None, sde_idx_dm),
    ]

    for row, (key, label, src, inv_fr, gen_fr, result, ref,
              inv_idx, gen_idx) in enumerate(rows_data):
        # Row label
        ax_lab = fig.add_subplot(gs[row, 0])
        ax_lab.axis('off')
        ax_lab.text(0.5, 0.5, label, ha='center', va='center',
                    fontsize=7, fontweight='bold', transform=ax_lab.transAxes)

        col = 1

        # Source
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(tensor_to_img(src), cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title('Source', fontsize=6)
        col += 1

        if inv_fr is not None:
            # Inversion frames
            for i, idx in enumerate(inv_idx):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(tensor_to_img(inv_fr[idx]), cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                t_val = 1.0 - idx / (len(inv_fr) - 1)
                if row == 0:
                    ax.set_title(f't={t_val:.2f}', fontsize=5, color='#1f77b4')
                col += 1
        else:
            # SDEdit: show mixed frame (gen_fr[0]) repeated
            for i in range(n_inv_shown):
                ax = fig.add_subplot(gs[row, col])
                if i == 0:
                    ax.imshow(tensor_to_img(gen_fr[0]), cmap='gray', vmin=0, vmax=1)
                else:
                    ax.imshow(tensor_to_img(gen_fr[0]), cmap='gray', vmin=0, vmax=1, alpha=0.3)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    if i == 0:
                        ax.set_title(f'mix', fontsize=5, color='purple')
                    else:
                        ax.set_title('', fontsize=5)
                col += 1

        # Generation frames
        for i, idx in enumerate(gen_idx):
            if idx < len(gen_fr):
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(tensor_to_img(gen_fr[idx]), cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if row == 0:
                    ax.set_title(f'gen', fontsize=5, color='#2ca02c')
                col += 1

        # Result
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(tensor_to_img(result), cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title('Result', fontsize=6, fontweight='bold')
        col += 1

        # Target reference
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(tensor_to_img(ref), cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title('Target', fontsize=6)

    fig.suptitle(f'Real Image Morph: {args.mutant} <-> {args.drug}\n'
                 f'Top: Euler inversion + generation | Bottom: SDEdit (t={args.sdedit_t})',
                 fontsize=10, fontweight='bold')
    out_path = os.path.join(SCRIPT_DIR, args.output)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nComposite saved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    # ── 6. Reconstruction quality check ──────────────────────────────
    print("\n[6/6] Reconstruction quality ...")
    # Invert and reconstruct with the SAME class: how well does inversion work?
    print("  Reconstructing mutant (invert + generate with same class) ...")
    z_rec, _ = ode_inversion(model, x_mut_real, mut_label, args.num_steps, freq_flow)
    x_rec_mut, _ = ode_generation(model, z_rec, mut_label, args.num_steps, freq_flow)
    mse_rec = F.mse_loss(x_rec_mut, x_mut_real).item()
    print(f"    MSE: {mse_rec:.6f}")

    print("  Reconstructing drug (invert + generate with same class) ...")
    z_rec_d, _ = ode_inversion(model, x_drug_real, drug_label, args.num_steps, freq_flow)
    x_rec_drug, _ = ode_generation(model, z_rec_d, drug_label, args.num_steps, freq_flow)
    mse_rec_d = F.mse_loss(x_rec_drug, x_drug_real).item()
    print(f"    MSE: {mse_rec_d:.6f}")

    for name, img, ref_img in [('rec_mutant', x_rec_mut, x_mut_real),
                                ('rec_drug', x_rec_drug, x_drug_real)]:
        plt.imsave(os.path.join(morph_dir, f'{name}.png'),
                   tensor_to_img(img), cmap='gray')
        plt.imsave(os.path.join(morph_dir, f'{name}_ref.png'),
                   tensor_to_img(ref_img), cmap='gray')
    print(f"  Reconstruction images saved to morph_results/")

    print(f"\n{'='*60}")
    print(f"Done! All results in {morph_dir}/")
    print(f"Composite: {out_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
