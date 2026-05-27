#!/usr/bin/env python3
"""Analyze structured latents from a trained flow matching checkpoint.

Loads checkpoint, encodes all images through the structured encoder qφ(z|x₁),
and runs t-SNE on μ_z (SCFM paper Fig 4).

Generates:
  - tsne_class_colored.png  — colored by 185-class label (paper-style)
  - tsne_group_colored.png  — colored by drug / mutant / control
  - tsne_component_colored.png — colored by GMM component assignment
  - component_utilization.png
  - component_entropy.png
  - class_component_table.csv

Usage:
    python3 analyze_latents.py \
        --checkpoint /path/to/flow_best.pth \
        --output_dir ./latent_analysis \
        --perplexity 50

References:
    Sumba et al., "Structured Coupling for Flow Matching", arXiv:2605.07676 (2026)
"""

import os, sys, warnings, argparse
warnings.filterwarnings("ignore")
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

from tqdm import tqdm
from mil_model import FlowCropDataset, load_labels
from flow_model import CombinedFlowUNet, StructFlowUNet
from scfm_model import SCFM

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--output_dir', type=str, default='./latent_analysis')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_samples', type=int, default=0,
                    help='Subsample for t-SNE (0 = all). Stratified by class.')
parser.add_argument('--perplexity', type=int, default=50,
                    help='t-SNE perplexity (paper default 50)')
parser.add_argument('--num_workers', type=int, default=8)
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
# Quick group count verification (before encoding)
n_ctrl = sum(1 for n in class_names
             if 'control' in n.lower() or n.startswith('NC_') or n.startswith('WT NC_'))
n_drug = sum(1 for n in class_names
             if n.rsplit('_', 1)[-1].endswith('x') and n.rsplit('_', 1)[-1][:-1].replace('.', '').isdigit())
n_mutant = len(class_names) - n_drug - n_ctrl
print(f"  Classes — Drug: {n_drug}, Mutant: {n_mutant}, Control: {n_ctrl}")
import sys; sys.stdout.flush()

max_samples = args.max_samples if args.max_samples > 0 else len(image_list)

print("\n[3/5] Rebuilding model ...")
block_channels = tuple(int(x) for x in ckpt_args.get('block_channels', '32,64,128,256').split(','))
is_scfm = 'struct_flow' not in ckpt_args and 'freq_flow' not in ckpt_args

if is_scfm:
    use_gmm = ckpt_args.get('use_gmm', False)
    unsupervised_gmm = ckpt_args.get('unsupervised_gmm', False)
    gmm_components = ckpt_args.get('gmm_components', 30)
    latent_dim = ckpt_args.get('latent_dim', 64)
    print(f"  SCFM checkpoint detected (latent_dim={latent_dim}, gmm={gmm_components})")

    model = SCFM(
        in_channels=1, sample_size=224,
        block_out_channels=block_channels,
        layers_per_block=2,
        latent_dim=latent_dim,
        exogenous_dim=ckpt_args.get('exogenous_dim', 64),
        use_gmm=use_gmm,
        gmm_components=gmm_components,
        unsupervised_gmm=unsupervised_gmm,
    ).to(device)
else:
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
        print("ERROR: checkpoint has no structured latent (not struct_flow)")
        sys.exit(1)

model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

# Build dataset — deterministic center crop, no augment (matching paper's eval)
dataset = FlowCropDataset(image_list, augment=False)
dataset.set_epoch(0)
loader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=True,
    persistent_workers=True, prefetch_factor=4,
)

# ─── [4/5] Encode: extract μ_z from qφ(z|x₁) at t=1 ──────────────────────
# This matches SCFM paper Section 3.2: the encoder mean μᶻ_φ(x₁)
# is the first dz coordinates of the shared mean network at t=1.
# Our implementation uses a mid-block hook + projection head (architecturally
# equivalent for high-res 224×224 inputs), which we access via encode_at_t1().
print("\n[4/5] Encoding all images via qφ(z|x₁) at t=1 ...")
all_mu_z = []
all_logvar_z = []
all_class_ids = []

with torch.no_grad():
    for imgs, class_ids in tqdm(loader, desc="Encoding"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            if is_scfm:
                mu_z, logvar_z = model.encode(imgs, return_all=False)
            else:
                class_ids = class_ids.to(device, non_blocking=True)
                mu_z, logvar_z = model.encode_at_t1(imgs, class_labels=class_ids)
        all_mu_z.append(mu_z.cpu())
        all_logvar_z.append(logvar_z.cpu())
        all_class_ids.append(class_ids.cpu())

mu_z_full = torch.cat(all_mu_z, dim=0).float().numpy()
class_ids_full = torch.cat(all_class_ids, dim=0).numpy()
print(f"  Encoded {mu_z_full.shape[0]} images, latent dim {mu_z_full.shape[1]}")

# ─── GMM component assignments ────────────────────────────────────────────
if unsupervised_gmm and hasattr(model, 'gmm'):
    print("  Computing GMM assignments (q(c|x) via VaDE Bayes rule) ...")
    gmm = model.gmm
    mu_z_t = torch.from_numpy(mu_z_full).to(device)
    with torch.no_grad():
        q_c = gmm.responsibilities(mu_z_t)
        hard_assignments = q_c.argmax(dim=-1).cpu().numpy()
        q_c_np = q_c.cpu().numpy()
        component_entropy = (-(q_c * torch.log(q_c + 1e-8)).sum(dim=-1)).cpu().numpy()
    n_components = gmm.n_components
elif hasattr(model, 'gmm') and not unsupervised_gmm:
    print("  Supervised GMM: using class labels as pseudo-components")
    gmm = model.gmm
    mu_z_t = torch.from_numpy(mu_z_full).to(device)
    with torch.no_grad():
        q_c_logits = gmm.categorical_head(mu_z_t) if hasattr(gmm, 'categorical_head') else None
        if q_c_logits is not None:
            q_c = torch.softmax(q_c_logits, dim=-1)
            hard_assignments = q_c.argmax(dim=-1).cpu().numpy()
            q_c_np = q_c.cpu().numpy()
            component_entropy = (-(q_c * torch.log(q_c + 1e-8)).sum(dim=-1)).cpu().numpy()
        else:
            hard_assignments = class_ids_full.copy()
            q_c_np = None
            component_entropy = np.zeros_like(class_ids_full, dtype=np.float32)
    n_components = gmm.n_components
else:
    print("  No GMM: using class labels as pseudo-components")
    hard_assignments = class_ids_full.copy()
    n_components = len(class_names)
    q_c_np = None
    component_entropy = np.zeros_like(class_ids_full, dtype=np.float32)

# ─── Group labels: drug / mutant / control ────────────────────────────────
print("  Assigning drug/mutant/control groups from class names ...")
group_names = np.array(class_names)
group_labels = np.zeros(len(class_ids_full), dtype=np.int32)  # 0=drug, 1=mutant, 2=control
group_color_strs = ['drug', 'mutant', 'control']

for cls_id in range(len(class_names)):
    cls_name = class_names[cls_id]
    mask = class_ids_full == cls_id
    if 'control' in cls_name.lower() or cls_name.startswith('NC_') or cls_name.startswith('WT NC_'):
        group_labels[mask] = 2  # control
    elif cls_name.rsplit('_', 1)[-1].endswith('x') and cls_name.rsplit('_', 1)[-1][:-1].replace('.', '').isdigit():
        group_labels[mask] = 0  # drug (e.g. Ampicillin_0.25x, Gentamicin_1x, Ampicillin_2x)
    else:
        group_labels[mask] = 1  # mutant

n_per_group = np.bincount(group_labels)
print(f"  Drug: {n_per_group[0]}, Mutant: {n_per_group[1]}, Control: {n_per_group[2]}")

# ─── Subsampling for t-SNE ───────────────────────────────────────────────
if mu_z_full.shape[0] > max_samples:
    print(f"\n  Subsampling {max_samples} from {mu_z_full.shape[0]} (stratified) ...")
    unique_classes = np.unique(class_ids_full)
    samples_per_class = max(1, max_samples // len(unique_classes))
    indices = []
    for cls in unique_classes:
        cls_idx = np.where(class_ids_full == cls)[0]
        if len(cls_idx) > samples_per_class:
            cls_idx = np.random.choice(cls_idx, samples_per_class, replace=False)
        indices.extend(cls_idx.tolist())
    if len(indices) > max_samples:
        indices = np.random.choice(indices, max_samples, replace=False).tolist()
    idx = np.array(indices)
    mu_z_vis = mu_z_full[idx]
    class_ids_vis = class_ids_full[idx]
    hard_assignments_vis = hard_assignments[idx]
    component_entropy_vis = component_entropy[idx]
    group_labels_vis = group_labels[idx]
    print(f"  Subsampled to {len(idx)} images")
else:
    idx = np.arange(mu_z_full.shape[0])
    mu_z_vis = mu_z_full
    class_ids_vis = class_ids_full
    hard_assignments_vis = hard_assignments
    component_entropy_vis = component_entropy
    group_labels_vis = group_labels

# ─── [5/5] t-SNE ─────────────────────────────────────────────────────────
# Matches SCFM paper Fig 4: Barnes-Hut t-SNE with perplexity 50,
# PCA initialization for reproducibility, on μ_z from qφ(z|x₁).
print(f"\n[5/5] t-SNE (perplexity={args.perplexity}, {mu_z_vis.shape[0]} points) ...")
tsne = TSNE(n_components=2, perplexity=args.perplexity,
            random_state=SEED, method='barnes_hut', init='pca', verbose=1)
z_tsne = tsne.fit_transform(mu_z_vis)
print(f"  Done: {z_tsne.shape}")

np.savez(os.path.join(args.output_dir, 'latent_analysis.npz'),
         mu_z=mu_z_full, z_tsne=z_tsne, idx=idx,
         class_ids=class_ids_full, hard_assignments=hard_assignments,
         component_entropy=component_entropy, group_labels=group_labels)

# ═══════════════════════════════════════════════════════════════════════════
# Plot 1: Class-colored t-SNE (matches SCFM Fig 4 — colored by class label)
# ═══════════════════════════════════════════════════════════════════════════
print("  Plot 1/4: class-colored t-SNE ...")
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
n_unique = len(np.unique(class_ids_vis))

if n_unique <= 20:
    cmap = plt.cm.tab20
    colors = [cmap(i % 20) for i in range(n_unique)]
    for i, cls in enumerate(np.unique(class_ids_vis)):
        mask = class_ids_vis == cls
        ax.scatter(z_tsne[mask, 0], z_tsne[mask, 1],
                   c=[colors[i]], s=4, alpha=0.6,
                   label=f"{class_names[cls][:20]}")
    ax.legend(markerscale=5, fontsize=5, loc='best', ncol=3)
else:
    cmap = plt.cm.tab20
    colors = [cmap(i % 20) for i in range(n_unique)]
    for i, cls in enumerate(np.unique(class_ids_vis)):
        mask = class_ids_vis == cls
        ax.scatter(z_tsne[mask, 0], z_tsne[mask, 1],
                   c=[colors[i]], s=3, alpha=0.5)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, n_unique), cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, ticks=[0, n_unique])
    cbar.set_label('Class index')

ax.set_title(f't-SNE colored by class ({n_unique} classes, epoch {epoch})')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
fig.tight_layout()
fig.savefig(os.path.join(args.output_dir, 'tsne_class_colored.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# Plot 2: Group-colored t-SNE (drug / mutant / control)
# ═══════════════════════════════════════════════════════════════════════════
print("  Plot 2/4: group-colored t-SNE ...")
group_cmap = ListedColormap(['#E74C3C', '#2ECC71', '#3498DB'])  # red=drug, green=mutant, blue=control

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
for gid, gname in enumerate(['Drug', 'Mutant', 'Control']):
    mask = group_labels_vis == gid
    ax.scatter(z_tsne[mask, 0], z_tsne[mask, 1],
               c=[group_cmap(gid)], s=4, alpha=0.5,
               label=f'{gname} (n={mask.sum()})')

ax.legend(markerscale=5, fontsize=10, loc='best')
ax.set_title(f't-SNE colored by group (drug/mutant/control, epoch {epoch})')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
fig.tight_layout()
fig.savefig(os.path.join(args.output_dir, 'tsne_group_colored.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# Plot 3: Component-colored t-SNE
# ═══════════════════════════════════════════════════════════════════════════
if hard_assignments_vis is not None:
    print("  Plot 3/4: component-colored t-SNE ...")
    n_comp = n_components
    cmap_comp = plt.cm.tab20 if n_comp <= 20 else plt.cm.gist_ncar

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    unique_comps = np.sort(np.unique(hard_assignments_vis))

    for comp in unique_comps:
        mask = hard_assignments_vis == comp
        c = cmap_comp(comp / max(1, n_comp - 1))
        ax.scatter(z_tsne[mask, 0], z_tsne[mask, 1],
                   c=[c], s=4, alpha=0.6, label=f'C{comp}')

    ax.legend(markerscale=5, fontsize=6, loc='best', ncol=5)
    ax.set_title(f't-SNE colored by GMM component ({n_comp} components, epoch {epoch})')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'tsne_component_colored.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# Plot 4: Entropy histogram
# ═══════════════════════════════════════════════════════════════════════════
if component_entropy is not None:
    print("  Plot 4/4: assignment entropy ...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(component_entropy, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
    ax.set_xlabel('Assignment entropy (nats)')
    ax.set_ylabel('Count')
    ax.set_title(f'GMM Assignment Entropy (max={np.log(n_components):.2f} nats)')
    ax.axvline(component_entropy.mean(), color='red', linestyle='--',
               label=f'Mean: {component_entropy.mean():.3f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'component_entropy.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# Component utilization + per-group breakdown
# ═══════════════════════════════════════════════════════════════════════════
if hard_assignments is not None:
    comp_counts = np.bincount(hard_assignments, minlength=n_components)
    active_components = np.where(comp_counts > 0)[0]

    print(f"\n  Component Utilization:")
    print(f"    Active: {len(active_components)}/{n_components} ({100*len(active_components)/n_components:.1f}%)")
    print(f"    Empty: {n_components - len(active_components)} components")
    print(f"    Max: {comp_counts.max()} images (C{comp_counts.argmax()})")
    print(f"    Mean: {comp_counts.mean():.1f} images/component")

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.bar(range(n_components), comp_counts, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Component index')
    ax.set_ylabel('Image count')
    ax.set_title(f'GMM Component Utilization ({len(active_components)}/{n_components} active)')
    ax.axhline(comp_counts.mean(), color='red', linestyle='--',
               label=f'Mean: {comp_counts.mean():.1f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'component_utilization.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Per-component group breakdown
    print(f"\n  Per-component group breakdown (top-20 components):")
    group_names_short = ['Drug', 'Mutant', 'Control']
    comp_group_table = []
    for comp in range(n_components):
        comp_mask = hard_assignments == comp
        n_comp = comp_mask.sum()
        if n_comp == 0:
            continue
        g_counts = np.bincount(group_labels[comp_mask], minlength=3)
        g_pcts = g_counts / n_comp * 100
        dominant_group = g_pcts.argmax()
        comp_group_table.append((comp, n_comp, dominant_group, g_pcts[0], g_pcts[1], g_pcts[2]))

    comp_group_table.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Comp':>4} {'N':>6} {'Dom':>8} {'%Drug':>8} {'%Mutant':>8} {'%Ctrl':>8}")
    print(f"  {'-'*4} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for comp, n, dg, pd, pm, pc in comp_group_table[:20]:
        dg_name = group_names_short[dg]
        print(f"  {comp:>4} {n:>6} {dg_name:>8} {pd:>7.1f}% {pm:>7.1f}% {pc:>7.1f}%")
    print(f"  ... (showing top 20 of {len(comp_group_table)} active components)")

    # Per-class dominant component table
    print(f"\n  Per-class dominant component (top 30 classes):")
    unique_classes = np.unique(class_ids_full)
    table_data = []
    for cls in sorted(unique_classes)[:30]:
        cls_mask = class_ids_full == cls
        cls_comps = hard_assignments[cls_mask]
        if len(cls_comps) > 0:
            dominant = np.bincount(cls_comps, minlength=n_components).argmax()
            dom_count = (cls_comps == dominant).sum()
            purity = dom_count / len(cls_comps)
            table_data.append((class_names[cls][:35], dominant, dom_count, purity, len(cls_comps)))
    print(f"  {'Class':<35} {'Comp':>5} {'DomCnt':>6} {'Purity':>7} {'Total':>5}")
    print(f"  {'-'*35} {'-'*5} {'-'*6} {'-'*7} {'-'*5}")
    for cn, comp, dc, pur, tot in table_data:
        print(f"  {cn:<35} {comp:>5} {dc:>6} {pur:>7.2%} {tot:>5}")

    # Save full tables
    with open(os.path.join(args.output_dir, 'class_component_table.csv'), 'w') as f:
        f.write('class_name,class_id,dominant_component,dom_count,total_count,purity,pct_drug,pct_mutant,pct_control\n')
        for cls in sorted(unique_classes):
            cls_mask = class_ids_full == cls
            cls_comps = hard_assignments[cls_mask]
            if len(cls_comps) > 0:
                dominant = np.bincount(cls_comps, minlength=n_components).argmax()
                dom_count = (cls_comps == dominant).sum()
                purity = dom_count / len(cls_comps)
                g_counts = np.bincount(group_labels[cls_mask], minlength=3)
                g_pcts = g_counts / len(cls_comps) * 100
                f.write(f'{class_names[cls]},{cls},{dominant},{dom_count},{len(cls_comps)},{purity:.4f},{g_pcts[0]:.1f},{g_pcts[1]:.1f},{g_pcts[2]:.1f}\n')

    with open(os.path.join(args.output_dir, 'component_group_breakdown.csv'), 'w') as f:
        f.write('component,n_images,dominant_group,pct_drug,pct_mutant,pct_control\n')
        for comp, n, dg, pd, pm, pc in comp_group_table:
            dg_name = group_names_short[dg]
            f.write(f'{comp},{n},{dg_name},{pd:.1f},{pm:.1f},{pc:.1f}\n')

print(f"\nDone. All outputs in {args.output_dir}")
print(f"  Plot files: tsne_class_colored.png, tsne_group_colored.png, tsne_component_colored.png")
print(f"  Stats: component_utilization.png, component_entropy.png")
print(f"  Tables: class_component_table.csv, component_group_breakdown.csv")
