#!/usr/bin/env python3
"""Extract bottleneck features from trained flow model and visualize with PaCMAP.

Extracts at 4 settings:
    t=0.5 conditional, t=0.5 unconditional,
    t=1.0 conditional, t=1.0 unconditional

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

from mil_model import FlowCropDataset, load_labels
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
parser.add_argument('--timestep_mid', type=float, default=0.5,
                    help='Mid-noise timestep for extraction (default: 0.5)')
parser.add_argument('--timestep_clean', type=float, default=1.0,
                    help='Clean timestep for extraction (default: 1.0)')
parser.add_argument('--tsne_only', action='store_true', default=False,
                    help='Skip extraction, load saved .npy features, only run t-SNE + plot')
parser.add_argument('--tsne_perplexity', type=float, default=50,
                    help='t-SNE perplexity (default: 50)')
parser.add_argument('--tsne_iter', type=int, default=5000,
                    help='t-SNE max iterations (default: 5000)')
parser.add_argument('--wasserstein', action='store_true', default=False,
                    help='Skip extraction/t-SNE, compute Wasserstein distance matrix between drug_2x and mutant_g1')
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

output_dir = args.output_dir or os.path.join(SCRIPT_DIR, 'latent_analysis_pacmap')
os.makedirs(output_dir, exist_ok=True)

if args.wasserstein:
    mode_label = 'Wasserstein Distance Matrix (existing features)'
elif args.tsne_only:
    mode_label = 't-SNE Only (existing features)'
else:
    mode_label = 'Latent Extraction + t-SNE'

print("=" * 60)
print(mode_label)
print(f"Output: {output_dir}")
print("=" * 60)

if not (args.tsne_only or args.wasserstein):
    # ── Data ────────────────────────────────────────────────────
    print("\n[1/4] Loading data ...")
    image_list, class_names, label_to_idx = load_labels(PROJECT_ROOT, SCRIPT_DIR)
    num_classes = len(class_names)
    print(f"  {len(image_list)} images, {num_classes} classes")

    # Classify each class as drug (0), mutant (1), or control (2).
    import json
    ic50_path = os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')
    mutant_path = os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')

    drug_labels_from_source = set()
    if os.path.exists(ic50_path):
        with open(ic50_path) as f:
            ic50_data = json.load(f)
        for plate in ic50_data.values():
            for info in plate.values():
                ab = info.get('antibiotic', '')
                ic = info.get('ic50_multiple', '')
                if ab and ic:
                    lbl = 'control' if ic == 'control' else f"{ab.replace(' ', '_')}_{ic if 'x' in ic else ic + 'x'}"
                    drug_labels_from_source.add(lbl)

    mutant_labels_from_source = set()
    mutant_control_labels = set()
    if os.path.exists(mutant_path):
        with open(mutant_path) as f:
            mutant_data = json.load(f)
        for plate in mutant_data.values():
            for cols in plate.values():
                for cinfo in cols.values():
                    if 'id' in cinfo:
                        lbl = cinfo['id']
                        mutant_labels_from_source.add(lbl)
                        if lbl.startswith('NC_') or lbl.startswith('WT NC_'):
                            mutant_control_labels.add(lbl)

    class_type = {}
    for label_name in label_to_idx:
        if label_name == 'control' or label_name in mutant_control_labels:
            t = 2
        elif label_name in mutant_labels_from_source:
            t = 1
        else:
            t = 0
        class_type[label_name] = t

    class_type_ids = np.array([class_type[cn] for cn in class_names])
    n_drug = (class_type_ids == 0).sum()
    n_mutant = (class_type_ids == 1).sum()
    n_control = (class_type_ids == 2).sum()
    print(f"  Drug={n_drug}, Mutant={n_mutant}, Control={n_control}")

    ds = FlowCropDataset(image_list, augment=False)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    # ── Model ───────────────────────────────────────────────────
    print("\n[2/4] Loading model ...")
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
        target_unet = model.spatial_unet
    else:
        model = FlowUNet(
            in_channels=1, sample_size=224,
            block_out_channels=block_channels,
            layers_per_block=2, num_class_embeds=num_classes,
        ).to(device)
        target_unet = model.unet

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    def add_null_embedding(unet: nn.Module, n: int, device: torch.device):
        old = unet.class_embedding
        new = nn.Embedding(n + 1, old.embedding_dim, device=device)
        new.weight.data[:n] = old.weight.data.to(device)
        new.weight.data[n] = old.weight.data.mean(dim=0).to(device)
        unet.class_embedding = new

    if use_freq:
        add_null_embedding(model.spatial_unet, num_classes, device)
        add_null_embedding(model.freq_unet, num_classes, device)
    else:
        add_null_embedding(model.unet, num_classes, device)

    NULL_LABEL = num_classes
    print(f"  {'FreqFlowUNet' if use_freq else 'FlowUNet'} loaded (epoch {ckpt['epoch']})")

    # ── Forward hook ────────────────────────────────────────────
    mid_features = {}
    def make_hook(key):
        def hook(module, input, output):
            mid_features[key] = output[0] if isinstance(output, tuple) else output
        return hook

    handle = target_unet.up_blocks[0].register_forward_hook(make_hook('mid'))

    # ── Extraction settings ─────────────────────────────────────
    settings = [
        ('t05_cond',  args.timestep_mid,  True),
        ('t05_uncond', args.timestep_mid,  False),
        ('t10_cond',  args.timestep_clean, True),
        ('t10_uncond', args.timestep_clean, False),
    ]

    all_feats = {name: [] for name, _, _ in settings}
    all_labels = []

    print("\n[3/4] Extracting latents (4 modes per batch) ...")

    with torch.no_grad():
        for imgs, class_ids in tqdm(loader, desc="Extract"):
            imgs = imgs.to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)

            for name, t_val, cond in settings:
                t_batch = torch.full((imgs.shape[0],), t_val, device=device)
                labels_for_model = class_ids if cond else torch.full_like(class_ids, NULL_LABEL)

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
    np.save(os.path.join(output_dir, 'class_types.npy'), class_type_ids[labels])

    for name, _, _ in settings:
        feats = torch.cat(all_feats[name], dim=0).numpy()
        np.save(os.path.join(output_dir, f'feats_{name}.npy'), feats)
        print(f"  feats_{name}.npy: {feats.shape}")

if not args.wasserstein:
    # ── t-SNE (load saved features) ─────────────────────────────
    print("\n[4/4] Computing t-SNE ...")

    from sklearn.manifold import TSNE

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    labels = np.load(os.path.join(output_dir, 'labels.npy'))
    class_types_labels = np.load(os.path.join(output_dir, 'class_types.npy'))

    settings = [
        ('t05_cond',  't=0.5, Conditional'),
        ('t05_uncond', 't=0.5, Unconditional'),
        ('t10_cond',  't=1.0, Conditional'),
        ('t10_uncond', 't=1.0, Unconditional'),
    ]

    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.08)

    colors = {0: (1, 0, 0, 0.4), 1: (0, 0.8, 0, 0.4), 2: (0, 0, 1, 0.4)}
    group_names = ('Drug', 'Mutant', 'Control')

    for idx, (name, title) in enumerate(settings):
        feats = np.load(os.path.join(output_dir, f'feats_{name}.npy'))
        print(f"  t-SNE {name}: {feats.shape} (perp={args.tsne_perplexity}, iter={args.tsne_iter}) ...", end=' ', flush=True)

        reducer = TSNE(n_components=2, perplexity=args.tsne_perplexity, max_iter=args.tsne_iter,
                       random_state=SEED, method='barnes_hut', verbose=0)
        embedding = reducer.fit_transform(feats)
        np.save(os.path.join(output_dir, f'tsne_{name}.npy'), embedding)
        print("done")

        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        for t in (0, 1, 2):
            mask = class_types_labels == t
            if mask.sum() == 0:
                continue
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=colors[t], s=2, alpha=0.5, rasterized=True,
                       label=f'{group_names[t]} ({mask.sum()})')

        ax.set_title(f'{title}  (perp={args.tsne_perplexity}, iter={args.tsne_iter})', fontsize=11)
        ax.legend(fontsize=8, markerscale=8, loc='best')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle(f't-SNE: Flow Model Bottleneck Features (perp={args.tsne_perplexity}, iter={args.tsne_iter})', fontsize=12, y=0.98)
    fig.savefig(os.path.join(output_dir, 'tsne_grid.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {os.path.join(output_dir, 'tsne_grid.png')}")

else:
    # ── Wasserstein distance matrix ─────────────────────────────
    print("\n[4/4] Computing Wasserstein distance matrices ...")

    import ot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    class_names = np.load(os.path.join(output_dir, 'class_names.npy'), allow_pickle=True)
    labels = np.load(os.path.join(output_dir, 'labels.npy'))
    class_names_list = list(class_names)

    # Drug classes: 2x concentration + control
    drug_2x_names = sorted([c for c in class_names_list if c.endswith('_2x')])
    if 'control' in class_names_list:
        drug_2x_names.append('control')
    drug_2x_ids = [class_names_list.index(c) for c in drug_2x_names]

    # Mutant guide 1 classes: end with '_1', include controls too
    mutant_g1_names = sorted([c for c in class_names_list if c.endswith('_1')])
    mutant_g1_ids = [class_names_list.index(c) for c in mutant_g1_names]

    print(f"  Drug 2x: {len(drug_2x_names)}, Mutant g1: {len(mutant_g1_names)}")

    settings = [
        ('t05_cond',  't=0.5, Conditional'),
        ('t05_uncond', 't=0.5, Unconditional'),
        ('t10_cond',  't=1.0, Conditional'),
        ('t10_uncond', 't=1.0, Unconditional'),
    ]

    def sinkhorn_wasserstein(X, Y, reg=0.01):
        n, m = len(X), len(Y)
        a = np.ones(n) / n
        b = np.ones(m) / m
        M = ot.dist(X, Y, metric='sqeuclidean')
        M_max = M.max()
        if M_max > 0:
            M /= M_max
        try:
            W = ot.sinkhorn2(a, b, M, reg, numItermax=200)
            return float(W)
        except Exception:
            return np.nan

    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.3)

    for idx, (name, title) in enumerate(settings):
        feats = np.load(os.path.join(output_dir, f'feats_{name}.npy'))
        print(f"  Wasserstein {name} ...", flush=True)

        # Collect samples per class
        drug_samples = {}
        for cid, cname in zip(drug_2x_ids, drug_2x_names):
            mask = labels == cid
            if mask.sum() >= 3:
                drug_samples[cname] = feats[mask]

        mutant_samples = {}
        for cid, cname in zip(mutant_g1_ids, mutant_g1_names):
            mask = labels == cid
            if mask.sum() >= 3:
                mutant_samples[cname] = feats[mask]

        dn = sorted(drug_samples.keys())
        mn = sorted(mutant_samples.keys())

        wd = np.full((len(mn), len(dn)), np.nan)
        for mi, mname in enumerate(mn):
            X = mutant_samples[mname]
            for di, dname in enumerate(dn):
                Y = drug_samples[dname]
                wd[mi, di] = sinkhorn_wasserstein(X, Y)

        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        vmax = np.nanpercentile(wd, 90) if not np.all(np.isnan(wd)) else 1.0
        im = ax.imshow(wd, aspect='auto', cmap='viridis_r', vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Wasserstein dist')

        # Shorten labels
        def shorten_drug(n):
            if n == 'control':
                return 'control'
            return n.replace('_2x', '').replace('_', ' ')
        drug_short = [shorten_drug(n) for n in dn]
        mutant_short = [n.replace('_1', '').replace('_', ' ') for n in mn]

        ax.set_xticks(range(len(dn)))
        ax.set_yticks(range(len(mn)))
        ax.set_xticklabels(drug_short, rotation=90, fontsize=5)
        ax.set_yticklabels(mutant_short, fontsize=6)
        ax.set_xlabel('Drug (2x)', fontsize=9)
        ax.set_ylabel('Mutant (g1)', fontsize=9)
        ax.set_title(title, fontsize=11)

    plt.suptitle('Sinkhorn-Wasserstein: Drug 2x vs Mutant g1', fontsize=13, y=0.98)
    fig.savefig(os.path.join(output_dir, 'wasserstein_grid.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {os.path.join(output_dir, 'wasserstein_grid.png')}")

print(f"\nDone. All outputs in: {output_dir}")
