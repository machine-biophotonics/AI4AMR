#!/usr/bin/env python3
"""
Extract center-3x3 MIL embeddings from control images + t-SNE colored by 7 groups.
Replicates the exact validation logic from train_mil.py's MultiCropDataset
(augment=False: absolute image center, 3x3 neighborhood, no jitter).

Usage:
    python3 extract_tsne_control.py
    python3 extract_tsne_control.py --fold P5 --checkpoint best_model_auc.pth
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']

# ── 7-group mapping (same as generate_control_confusion.py) ──
GROUP_ORDER = ['ACE-1 -ATC', 'ACE-1 +ATC/NC', 'MG1655 -ATC', 'MG1655 +ATC/NC',
               'NC', 'WT NC', 'drug_control']
GROUP_COLORS = {
    'ACE-1 -ATC':     '#1f77b4',
    'ACE-1 +ATC/NC':  '#17becf',
    'MG1655 -ATC':    '#2ca02c',
    'MG1655 +ATC/NC': '#bcbd22',
    'NC':             '#ff7f0e',
    'WT NC':          '#d62728',
    'drug_control':   '#9467bd',
}


def get_group(label: str) -> str:
    if label == 'drug_control':
        return 'drug_control'
    if label.startswith('WT NC'):
        return 'WT NC'
    if label.startswith('NC_'):
        return 'NC'
    if label.startswith('ACE-1_NC') or label == 'ACE-1_plusATC':
        return 'ACE-1 +ATC/NC'
    if label == 'ACE-1_minusATC':
        return 'ACE-1 -ATC'
    if label.startswith('MG1655_NC') or label == 'MG1655_plusATC':
        return 'MG1655 +ATC/NC'
    if label == 'MG1655_minusATC':
        return 'MG1655 -ATC'
    return 'other'


def build_control_classes(CONTROL_DATA, MUTANT_DATA):
    """Build 41-class list from all 3 data sources (control mode)."""
    classes: set = set()
    for plate, rows in CONTROL_DATA.items():
        for row, cols in rows.items():
            for col, info in cols.items():
                if 'id' in info:
                    classes.add(info['id'])
    for plate, rows in MUTANT_DATA.items():
        for row, cols in rows.items():
            for col, info in cols.items():
                mid = info.get('id', '')
                if mid.startswith('NC_') or mid.startswith('WT NC_'):
                    classes.add(mid)
    classes.add('drug_control')
    return sorted(classes)


def build_valid_wells(test_plate_key, CONTROL_DATA, MUTANT_DATA, IC50_DATA):
    """Build set of valid well keys (same logic as predict_all_crops.py)."""
    valid = set()
    if test_plate_key in CONTROL_DATA:
        for row, cols in CONTROL_DATA[test_plate_key].items():
            for col, info in cols.items():
                if 'id' in info:
                    well = f"{row}{int(col):02d}"
                    valid.add(f"controls_data_{well}")
    if test_plate_key in MUTANT_DATA:
        for row, cols in MUTANT_DATA[test_plate_key].items():
            for col, info in cols.items():
                mid = info.get('id', '')
                if mid.startswith('NC_') or mid.startswith('WT NC_'):
                    well = f"{row}{int(col):02d}"
                    valid.add(f"mutants_data_{well}")
    if test_plate_key in IC50_DATA:
        for well, info in IC50_DATA[test_plate_key].items():
            if info.get('ic50_multiple') == 'control':
                valid.add(f"drugs_data_{well}")
    return valid


def parse_well_from_filename(img_path: str):
    filename = os.path.basename(img_path)
    for part in filename.split('_'):
        if part.startswith('Well'):
            return part.replace('Well', '')
    return None


def get_ground_truth_label(plate: str, well, img_path: str,
                           CONTROL_DATA, MUTANT_DATA, IC50_DATA):
    if not well:
        return None
    if 'Plate_' in plate:
        plate_key = 'P' + plate.split('_')[-1]
    else:
        plate_key = plate

    path_lower = img_path.lower()
    if '/controls_data/' in path_lower or '\\controls_data\\' in path_lower:
        row_c = well[0]
        col_c = str(int(well[1:]))
        if plate_key in CONTROL_DATA and row_c in CONTROL_DATA[plate_key] \
                and col_c in CONTROL_DATA[plate_key][row_c]:
            info = CONTROL_DATA[plate_key][row_c][col_c]
            return info.get('id', None)
    if '/mutants_data/' in path_lower or '\\mutants_data\\' in path_lower:
        row = well[0]
        col = well[1:].lstrip('0') or '0'
        if plate_key in MUTANT_DATA and row in MUTANT_DATA[plate_key] \
                and col in MUTANT_DATA[plate_key][row]:
            mid = MUTANT_DATA[plate_key][row][col].get('id', None)
            if mid and (mid.startswith('NC_') or mid.startswith('WT NC_')):
                return mid
    if '/drugs_data/' in path_lower or '\\drugs_data\\' in path_lower:
        if plate_key in IC50_DATA and well in IC50_DATA[plate_key]:
            info = IC50_DATA[plate_key][well]
            if info.get('ic50_multiple') == 'control':
                return 'drug_control'
    return None


def load_image(img_path: str, num_channels: int = 1):
    """Same _load_image as predict_all_crops.py — returns PIL Image."""
    try:
        import tifffile
        img_array = tifffile.imread(img_path)
    except Exception:
        img_array = np.array(Image.open(img_path))

    if len(img_array.shape) == 3:
        img_array = img_array[:, :, 0]

    if img_array.dtype == np.uint16:
        img_array = img_array.astype(np.float32) / 65535.0
    elif img_array.dtype == np.uint8:
        img_array = img_array.astype(np.float32) / 255.0
    elif img_array.dtype in (np.float32, np.float64):
        img_array = img_array.astype(np.float32)

    if num_channels == 1:
        return Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
    else:
        return Image.fromarray((img_array * 255).astype(np.uint8), mode='L').convert('RGB')


def extract_center_crops(image: Image.Image, crop_size: int, grid_size: int,
                         neighborhood: int, num_channels: int = 1):
    """
    Extract only the center-position 3x3 neighborhood (matching validation in train_mil.py).
    Returns tensor of shape (9, C, 224, 224) where C=1 for grayscale, C=3 for RGB.
    Normalization: mean=0.5,std=0.5 for 1-channel; ImageNet stats for 3-channel.
    """
    w, h = image.size
    stride = (w - crop_size) // (grid_size - 1) if grid_size > 1 else 0
    half_n = neighborhood // 2
    center = (w - crop_size) // 2

    # Normalization params matching train_mil.py validation transform
    if num_channels == 1:
        norm_mean = [0.5]
        norm_std = [0.5]
    else:
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

    tfm = A.Compose([
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2(),
    ])

    crops_list = []
    for di in range(-half_n, half_n + 1):
        for dj in range(-half_n, half_n + 1):
            left = center + dj * stride
            top = center + di * stride
            left = max(0, min(left, w - crop_size))
            top = max(0, min(top, h - crop_size))
            crop_pil = image.crop((left, top, left + crop_size, top + crop_size))
            crop_np = np.array(crop_pil)
            # grayscale: shape (H,W), expand to (H,W,1) for albumentations
            if crop_np.ndim == 2:
                crop_np = crop_np[..., np.newaxis]
            transformed = tfm(image=crop_np)
            crops_list.append(transformed['image'])

    return torch.stack(crops_list)  # (9, C, 224, 224)


def main():
    parser = argparse.ArgumentParser(description='Extract center MIL embeddings + t-SNE')
    parser.add_argument('--fold', type=str, default='P6',
                        help='Test plate (P1-P6)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_epoch.pth',
                        help='Checkpoint filename')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Path to parent folder with P1-P6 (default: parent of script dir)')
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--grid_size', type=int, default=12)
    parser.add_argument('--crop_neighborhood', type=int, default=3, choices=[3, 5, 7, 9, 11])
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for MIL inference')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--pretrained', type=str, default='micronet',
                        choices=['imagenet', 'micronet'])
    parser.add_argument('--max_images', type=int, default=None,
                        help='Limit number of images (for debugging)')
    parser.add_argument('--tsne_perplexity', type=float, default=50.0)
    parser.add_argument('--tsne_random_state', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: control/fold_Plate_X)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load JSONs ──
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    with open(os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')) as f:
        IC50_DATA = json.load(f)
    with open(os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')) as f:
        MUTANT_DATA = json.load(f)
    with open(os.path.join(SCRIPT_DIR, 'plate_well_control_id_path.json')) as f:
        CONTROL_DATA = json.load(f)

    # ── Classes ──
    all_classes = build_control_classes(CONTROL_DATA, MUTANT_DATA)
    label_to_idx = {l: i for i, l in enumerate(all_classes)}
    idx_to_label = {i: l for l, i in label_to_idx.items()}
    print(f"Loaded {len(all_classes)} classes")

    # ── Determine fold key ──
    test_plate = args.fold
    if 'Plate_' in test_plate:
        plate_num = test_plate.split('_')[-1]
        image_plate_key = f'P{plate_num}'
    else:
        image_plate_key = test_plate

    # ── Load model ──
    from mil_model import MILEncoder

    fold_key_p = f'fold_{test_plate}'
    fold_key_plate = f'fold_Plate_{test_plate.replace("P", "")}'
    fold_dir_p = os.path.join(SCRIPT_DIR, 'control', fold_key_p)
    fold_dir_plate = os.path.join(SCRIPT_DIR, 'control', fold_key_plate)

    ckpt_p = os.path.join(fold_dir_p, args.checkpoint)
    ckpt_plate = os.path.join(fold_dir_plate, args.checkpoint)

    if os.path.exists(ckpt_p):
        checkpoint_path = ckpt_p
    elif os.path.exists(ckpt_plate):
        checkpoint_path = ckpt_plate
    else:
        print(f"ERROR: Checkpoint not found at {ckpt_p} or {ckpt_plate}")
        sys.exit(1)

    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = MILEncoder(
        num_classes=len(all_classes),
        num_heads=args.num_heads,
        num_channels=args.num_channels,
        pretrained=args.pretrained,
    ).to(device)
    model.eval()

    state_dict = checkpoint['model_state_dict']
    # Remove contrastive head keys (training had use_contrastive=True, model doesn't)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('contrastive_head.')}
    model.load_state_dict(state_dict, strict=True)
    print("  Model loaded")

    # ── Build valid wells for this plate ──
    valid_wells = build_valid_wells(image_plate_key, CONTROL_DATA, MUTANT_DATA, IC50_DATA)
    print(f"  Valid wells for {image_plate_key}: {len(valid_wells)}")

    # ── Search images across all 3 dirs ──
    control_search_dirs = [
        (os.path.join(PARENT_DIR, 'Controls_Data'), 'controls_data'),
        (os.path.join(PARENT_DIR, 'Mutants_Data'), 'mutants_data'),
        (os.path.join(PARENT_DIR, 'Drugs_Data'), 'drugs_data'),
    ]

    image_paths = []
    for search_dir, source_tag in control_search_dirs:
        plate_search = os.path.join(search_dir, image_plate_key)
        if not os.path.exists(plate_search):
            continue
        for fpath in sorted(Path(plate_search).rglob('*.tif')) + sorted(Path(plate_search).rglob('*.tiff')):
            well = parse_well_from_filename(str(fpath))
            if well and f"{source_tag}_{well}" in valid_wells:
                image_paths.append(fpath)

    if args.max_images:
        image_paths = image_paths[:args.max_images]
    print(f"  Found {len(image_paths)} images")

    # ── Extract embeddings ──
    embeddings_list = []
    labels_list = []
    groups_list = []
    plates_list = []
    img_paths_list = []

    for img_path in tqdm(image_paths, desc="Extracting embeddings"):
        img_path_str = str(img_path)
        well = parse_well_from_filename(img_path_str)
        gt = get_ground_truth_label(test_plate, well, img_path_str,
                                     CONTROL_DATA, MUTANT_DATA, IC50_DATA)
        if gt is None:
            continue

        image = load_image(img_path_str, num_channels=args.num_channels)
        crops = extract_center_crops(
            image,
            crop_size=args.crop_size,
            grid_size=args.grid_size,
            neighborhood=args.crop_neighborhood,
            num_channels=args.num_channels,
        ).to(device)  # (9, C, 224, 224)

        # MIL forward: reshape to (1, 9, C, 224, 224) → get_projected_features → (1, 1280)
        with torch.no_grad():
            bag = crops.unsqueeze(0)  # (1, 9, C, 224, 224)
            embedding = model.get_projected_features(bag)  # (1, 1280)
            embedding = embedding.squeeze(0).cpu().numpy()  # (1280,)

        embeddings_list.append(embedding)
        labels_list.append(gt)
        groups_list.append(get_group(gt))
        plates_list.append(image_plate_key)
        img_paths_list.append(img_path_str)

    if not embeddings_list:
        print("ERROR: No valid embeddings extracted")
        sys.exit(1)

    embeddings = np.array(embeddings_list, dtype=np.float32)
    labels = np.array(labels_list)
    groups = np.array(groups_list)
    plates = np.array(plates_list)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Labels: {len(set(labels))} unique")
    print(f"Groups: {len(set(groups))} unique: {sorted(set(groups))}")

    # ── Determine output directory ──
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir_candidate = os.path.join(SCRIPT_DIR, 'control', fold_key_p)
        if os.path.exists(out_dir_candidate):
            out_dir = out_dir_candidate
        else:
            out_dir = os.path.join(SCRIPT_DIR, 'control', fold_key_plate)
    os.makedirs(out_dir, exist_ok=True)

    # ── Save embeddings ──
    npz_path = os.path.join(out_dir, 'control_embeddings.npz')
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        labels=labels,
        groups=groups,
        plates=plates,
        image_paths=np.array(img_paths_list),
    )
    print(f"Saved: {npz_path}")

    # ── t-SNE ──
    print(f"\nRunning t-SNE (perplexity={args.tsne_perplexity}, random_state={args.tsne_random_state})...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        random_state=args.tsne_random_state,
        n_iter=1000,
        init='pca',
    )
    tsne_result = tsne.fit_transform(embeddings)

    # ── Plot t-SNE colored by 7 groups ──
    fig, ax = plt.subplots(figsize=(12, 10))

    group_names_in_data = [g for g in GROUP_ORDER if g in set(groups)]
    for group_name in group_names_in_data:
        mask = groups == group_name
        ax.scatter(
            tsne_result[mask, 0],
            tsne_result[mask, 1],
            c=GROUP_COLORS[group_name],
            label=group_name,
            alpha=0.7,
            s=20,
            edgecolors='none',
        )

    ax.set_title(f't-SNE of Center MIL Embeddings — {image_plate_key}\n'
                 f'{len(embeddings)} images, 1280-dim → 2D (perp={args.tsne_perplexity})',
                 fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    tsne_path = os.path.join(out_dir, f'tsne_7groups.png')
    fig.savefig(tsne_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {tsne_path}")

    # ── Optional: per-class t-SNE ──
    fig2, ax2 = plt.subplots(figsize=(14, 12))
    unique_labels = sorted(set(labels))
    label_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax2.scatter(
            tsne_result[mask, 0],
            tsne_result[mask, 1],
            c=[label_colors[i]],
            label=label,
            alpha=0.6,
            s=15,
            edgecolors='none',
        )

    ax2.set_title(f't-SNE of Center MIL Embeddings — {image_plate_key} (41 classes)', fontsize=14)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(fontsize=5, markerscale=2, loc='upper left', ncol=2)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    tsne41_path = os.path.join(out_dir, f'tsne_41classes.png')
    fig2.savefig(tsne41_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"Saved: {tsne41_path}")

    print(f"\nDone! All outputs in: {out_dir}")


if __name__ == '__main__':
    main()
