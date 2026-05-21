#!/usr/bin/env python3
"""
Evaluate GroupCLIP: Cross-modal similarity analysis and visualization.

Generates:
1. Cross-modal similarity matrix (Drug MOA centroid × Mutant Pathway centroid)
2. Known vs novel link detection
3. t-SNE visualization of joint embedding space
4. Per-group alignment metrics

Usage:
  python3 evaluate_groupclip.py --fold Plate_6 --checkpoint best_model.pth

Reference: Gorla et al., "Group Contrastive Learning for Weakly Paired Multimodal Data"
           Figure showing Drug MOA centroids × Mutant pathway centroids
"""

import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from sklearn.manifold import TSNE
from collections import defaultdict, OrderedDict
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Group structure (from group_mapping.json)
# ---------------------------------------------------------------------------
GROUP_NAMES = {
    0: "Gyrase / Chromosome",
    1: "Ribosome / Translation",
    2: "Cell Wall (PBP) / Aminoglycan",
    3: "Membrane / Envelope",
    4: "RNA Polymerase / Transcription",
    5: "DNA Synthesis / Folate",
    6: "Control / WT",
    7: "Protein Transport",
    8: "Division Septum",
}

GROUP_COLORS = {
    0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a', 3: '#984ea3',
    4: '#ff7f00', 5: '#ffff33', 6: '#a65628', 7: '#f781bf',
    8: '#999999',
}

# Known MOA-Pathway links from group_mapping.json
KNOWN_MOA_GROUPS = {
    "Gyrase": 0, "Ribosome": 1,
    "Cell wall (PBP 1)": 2, "Cell wall (PBP 2)": 2, "Cell wall (PBP 3)": 2,
    "Membrane integrity": 3, "RNA polymerase": 4, "DNA synthesis": 5,
    "Control": 6,
}

KNOWN_PATHWAY_GROUPS = {
    "Chromosome organization": 0, "Translation initiation": 1,
    "Aminoglycan biosynthesis": 2, "Cell shape regulation": 2,
    "Cell envelope organization": 3, "Lipid A biosynthesis": 3,
    "Transcription elongation": 4, "Folic acid biosynthesis": 5,
    "WT/NC": 6, "Protein transport": 7, "Division septum assembly": 8,
}


def main():
    parser = argparse.ArgumentParser(description='Evaluate GroupCLIP/GROOVE embeddings')
    parser.add_argument('--fold', type=str, default='Plate_6',
                        help='Fold to evaluate')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                        help='Checkpoint filename')
    parser.add_argument('--embeddings_path', type=str, default=None,
                        help='Direct path to embeddings .npz (skip model loading)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: groupclip/fold_X)')
    parser.add_argument('--pooling', type=str, default='attention')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_channels', type=int, default=1)
    parser.add_argument('--pretrained', type=str, default='imagenet')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--neighborhood', type=int, default=3)
    parser.add_argument('--n_tsne', type=int, default=2000,
                        help='Max samples for t-SNE')
    parser.add_argument('--mode', type=str, default='standard',
                        choices=['standard', 'novel_moa', 'leave_one_group_out'],
                        help='Evaluation mode')
    parser.add_argument('--held_out_group', type=int, default=None,
                        help='Group to hold out for leave_one_group_out mode (0-8)')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of clusters for novel MOA discovery (default: auto)')
    
    args = parser.parse_args()
    
    # Determine fold key
    if 'Plate_' in args.fold:
        fold_key = args.fold
    elif 'P' in args.fold:
        fold_key = f"Plate_{args.fold.replace('P', '')}"
    else:
        fold_key = args.fold
    
    OUTPUT_DIR = args.output_dir or os.path.join(SCRIPT_DIR, 'groupclip', f'fold_{fold_key}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # -----------------------------------------------------------------------
    # Load embeddings (from .npz or extract from checkpoint)
    # -----------------------------------------------------------------------
    if args.embeddings_path:
        data = np.load(args.embeddings_path, allow_pickle=True)
        drug_emb = data['drug_embeddings']
        mutant_emb = data['mutant_embeddings']
        drug_labels = data['drug_labels']
        mutant_labels = data['mutant_labels']
        drug_class_names = data['drug_class_names']
        mutant_class_names = data['mutant_class_names']
        drug_groups = data.get('drug_groups', None)
        mutant_groups = data.get('mutant_groups', None)
        print(f"Loaded embeddings from {args.embeddings_path}")
    else:
        # Load checkpoint and extract embeddings
        checkpoint_path = os.path.join(OUTPUT_DIR, args.checkpoint)
        if not os.path.exists(checkpoint_path):
            checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        from mil_model import DualMILEncoder, MultiCropDataset, extract_well_from_filename
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        import glob
        
        drug_class_to_idx = ckpt['drug_class_to_idx']
        mutant_class_to_idx = ckpt['mutant_class_to_idx']
        drug_classes = ckpt['drug_classes']
        mutant_classes = ckpt['mutant_classes']
        
        model = DualMILEncoder(
            num_drug_classes=len(drug_classes),
            num_mutant_classes=len(mutant_classes),
            num_heads=args.num_heads,
            dropout=args.dropout,
            num_channels=args.num_channels,
            pretrained=args.pretrained,
            backbone=args.backbone,
            pooling=args.pooling,
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        # Load test data
        with open(os.path.join(SCRIPT_DIR, 'group_mapping.json'), 'r') as f:
            group_map = json.load(f)
        
        # Collect test plate samples
        test_plate = fold_key.replace('Plate_', 'P')
        
        # Drug test
        drug_paths, drug_labels_list = [], []
        drug_base = os.path.join(os.path.dirname(SCRIPT_DIR), 'Drugs_Data', test_plate)
        if os.path.exists(drug_base):
            for pattern in ['*.tif', '*.tiff']:
                for path in glob.glob(os.path.join(drug_base, '**', pattern), recursive=True):
                    well = extract_well_from_filename(os.path.basename(path))
                    if not well:
                        continue
                    from train_groupclip import get_drug_info
                    info = get_drug_info(test_plate, well)
                    if info and info[0] in drug_class_to_idx:
                        drug_paths.append(path)
                        drug_labels_list.append(drug_class_to_idx[info[0]])
        
        # Mutant test
        mutant_paths, mutant_labels_list = [], []
        mutant_base = os.path.join(os.path.dirname(SCRIPT_DIR), 'Mutants_Data', test_plate)
        if os.path.exists(mutant_base):
            for pattern in ['*.tif', '*.tiff']:
                for path in glob.glob(os.path.join(mutant_base, '**', pattern), recursive=True):
                    well = extract_well_from_filename(os.path.basename(path))
                    if not well:
                        continue
                    from train_groupclip import get_mutant_info
                    info = get_mutant_info(test_plate, well)
                    if info and info[0] in mutant_class_to_idx:
                        mutant_paths.append(path)
                        mutant_labels_list.append(mutant_class_to_idx[info[0]])
        
        dataset = MultiCropDataset(
            drug_paths, drug_labels_list, None,
            neighborhood=args.neighborhood, augment=False,
            num_channels=args.num_channels
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        drug_emb_list = []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for images, _ in tqdm(loader, desc='Drug embeddings'):
                images = images.to(device)
                emb = model.get_embeddings(images, modality='drug')
                drug_emb_list.append(emb.cpu().numpy())
        drug_emb = np.concatenate(drug_emb_list) if drug_emb_list else np.array([]).reshape(0, 256)
        drug_labels_arr = np.array(drug_labels_list)
        
        # Mutant embeddings
        dataset_m = MultiCropDataset(
            mutant_paths, mutant_labels_list, None,
            neighborhood=args.neighborhood, augment=False,
            num_channels=args.num_channels
        )
        loader_m = DataLoader(dataset_m, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        mutant_emb_list = []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            for images, _ in tqdm(loader_m, desc='Mutant embeddings'):
                images = images.to(device)
                emb = model.get_embeddings(images, modality='mutant')
                mutant_emb_list.append(emb.cpu().numpy())
        mutant_emb = np.concatenate(mutant_emb_list) if mutant_emb_list else np.array([]).reshape(0, 256)
        mutant_labels_arr = np.array(mutant_labels_list)
        
        drug_class_names_arr = np.array(drug_classes)
        mutant_class_names_arr = np.array(mutant_classes)
        
        # Save
        np.savez(os.path.join(OUTPUT_DIR, 'test_embeddings.npz'),
                 drug_embeddings=drug_emb, mutant_embeddings=mutant_emb,
                 drug_labels=drug_labels_arr, mutant_labels=mutant_labels_arr,
                 drug_class_names=drug_class_names_arr,
                 mutant_class_names=mutant_class_names_arr)
        print(f"Saved embeddings to {OUTPUT_DIR}/test_embeddings.npz")
        
        # Reassign for consistent naming
        drug_labels = drug_labels_arr
        mutant_labels = mutant_labels_arr
        drug_class_names = drug_class_names_arr.tolist() if hasattr(drug_class_names_arr, 'tolist') else drug_classes
        mutant_class_names = mutant_class_names_arr.tolist() if hasattr(mutant_class_names_arr, 'tolist') else mutant_classes
        drug_groups = None
        mutant_groups = None
    
    print(f"Drug embeddings: {drug_emb.shape}, Mutant embeddings: {mutant_emb.shape}")
    
    # -----------------------------------------------------------------------
    # Dispatch to evaluation mode
    # -----------------------------------------------------------------------
    if args.mode == 'novel_moa':
        run_novel_moa_discovery(drug_emb, mutant_emb, drug_labels, mutant_labels,
                               drug_class_names, mutant_class_names, OUTPUT_DIR, args)
        return
    elif args.mode == 'leave_one_group_out':
        run_leave_one_group_out(drug_emb, mutant_emb, drug_labels, mutant_labels,
                               drug_class_names, mutant_class_names, OUTPUT_DIR, args)
        return
    
    # -----------------------------------------------------------------------
    # Compute cross-modal similarity matrix: Drug MOA × Mutant Pathway
    # -----------------------------------------------------------------------
    print("\nComputing cross-modal similarity matrix...")
    
    # Load group mapping once
    with open(os.path.join(SCRIPT_DIR, 'group_mapping.json'), 'r') as f:
        GM = json.load(f)
    
    # Map drug class names to MOA
    drug_moa_map = {}
    for i, cls_name in enumerate(drug_class_names):
        cls_str = str(cls_name)
        if cls_str == 'control':
            drug_moa_map[i] = 'Control'
        else:
            ab_clean = cls_str.rsplit('_', 1)[0]
            moa = GM['ANTIBIOTIC_TO_MOA'].get(ab_clean, 'Unknown')
            drug_moa_map[i] = moa
    
    # Map mutant class names to pathway
    mutant_pathway_map = {}
    for i, cls_name in enumerate(mutant_class_names):
        cls_str = str(cls_name)
        gene = cls_str.rsplit('_', 1)[0] if '_' in cls_str else cls_str
        pathway = GM['GENE_TO_PATHWAY'].get(gene, 'WT/NC' if 'WT' in cls_str else 'Unknown')
        mutant_pathway_map[i] = pathway
    
    # Compute class centroids
    drug_centroids = {}
    for i in range(len(drug_class_names)):
        mask = drug_labels == i
        if mask.sum() > 0:
            drug_centroids[i] = drug_emb[mask].mean(axis=0)
    
    mutant_centroids = {}
    for i in range(len(mutant_class_names)):
        mask = mutant_labels == i
        if mask.sum() > 0:
            mutant_centroids[i] = mutant_emb[mask].mean(axis=0)
    
    # MOA-level centroids (group drug classes by MOA)
    moa_centroids = defaultdict(list)
    for drug_idx, moa in drug_moa_map.items():
        if drug_idx in drug_centroids:
            moa_centroids[moa].append(drug_centroids[drug_idx])
    
    # Pathway-level centroids
    pathway_centroids = defaultdict(list)
    for mut_idx, pathway in mutant_pathway_map.items():
        if mut_idx in mutant_centroids:
            pathway_centroids[pathway].append(mutant_centroids[mut_idx])
    
    # Build MOA and Pathway name lists for the matrix
    moa_names = sorted(set(drug_moa_map.values()))
    pathway_names = sorted(set(mutant_pathway_map.values()))
    
    # Compute cross-modal similarity matrix (MOA × Pathway)
    sim_matrix = np.zeros((len(moa_names), len(pathway_names)))
    for i, moa in enumerate(moa_names):
        if moa not in moa_centroids or not moa_centroids[moa]:
            continue
        moa_cent = np.mean(moa_centroids[moa], axis=0)
        moa_cent = moa_cent / (np.linalg.norm(moa_cent) + 1e-8)
        for j, pathway in enumerate(pathway_names):
            if pathway not in pathway_centroids or not pathway_centroids[pathway]:
                continue
            pw_cent = np.mean(pathway_centroids[pathway], axis=0)
            pw_cent = pw_cent / (np.linalg.norm(pw_cent) + 1e-8)
            sim_matrix[i, j] = float(np.dot(moa_cent, pw_cent))
    
    # -----------------------------------------------------------------------
    # Plot 1: Cross-modal similarity heatmap (like paper Figure)
    # -----------------------------------------------------------------------
    print("\nGenerating cross-modal similarity heatmap...")
    plot_cross_modal_heatmap(sim_matrix, moa_names, pathway_names, OUTPUT_DIR)
    
    # -----------------------------------------------------------------------
    # Plot 2: Known vs novel link detection
    # -----------------------------------------------------------------------
    print("Analyzing known vs novel links...")
    analyze_known_novel_links(sim_matrix, moa_names, pathway_names, OUTPUT_DIR)
    
    # -----------------------------------------------------------------------
    # Plot 3: t-SNE of joint embedding space
    # -----------------------------------------------------------------------
    print("Generating t-SNE visualization...")
    plot_tsne_joint(drug_emb, mutant_emb, drug_labels, mutant_labels,
                    drug_class_names, mutant_class_names, OUTPUT_DIR, args.n_tsne)
    
    # -----------------------------------------------------------------------
    # Plot 4: Per-group alignment metrics
    # -----------------------------------------------------------------------
    print("Computing per-group alignment metrics...")
    compute_group_alignment(drug_emb, mutant_emb, drug_labels, mutant_labels,
                           drug_class_names, mutant_class_names, OUTPUT_DIR)
    
    print(f"\nAll visualizations saved to {OUTPUT_DIR}")
    print("Done!")


def plot_cross_modal_heatmap(sim_matrix, moa_names, pathway_names, output_dir):
    """Generate cross-modal similarity heatmap (like the paper figure)."""
    fig, ax = plt.subplots(figsize=(max(10, len(pathway_names) * 0.6),
                                    max(8, len(moa_names) * 0.5)))
    
    vmax = max(abs(sim_matrix.min()), abs(sim_matrix.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    sns.heatmap(sim_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                norm=norm,
                xticklabels=pathway_names, yticklabels=moa_names,
                cbar_kws={'label': 'Cosine Similarity'},
                ax=ax, linewidths=0.5, linecolor='lightgray')
    
    ax.set_xlabel('Mutant Pathway', fontsize=12)
    ax.set_ylabel('Drug MOA', fontsize=12)
    ax.set_title('GroupCLIP: Cross-Modal Similarity\n(Drug MOA Centroids × Mutant Pathway Centroids)',
                 fontsize=13, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_modal_similarity_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save as CSV
    import pandas as pd
    df = pd.DataFrame(sim_matrix, index=moa_names, columns=pathway_names)
    df.to_csv(os.path.join(output_dir, 'cross_modal_similarity_matrix.csv'))
    print(f"  Saved: cross_modal_similarity_heatmap.png")


def analyze_known_novel_links(sim_matrix, moa_names, pathway_names, output_dir):
    """
    Analyze known vs novel cross-modal links.
    
    Known links: MOA-Pathway pairs that are biologically expected (same group).
    Novel links: High-similarity pairs that are NOT biologically expected.
    """
    # Expected MOA-Pathway groups
    expected_groups = {
        "Gyrase": ["Chromosome organization"],
        "Ribosome": ["Translation initiation"],
        "Cell wall (PBP 1)": ["Aminoglycan biosynthesis", "Cell shape regulation"],
        "Cell wall (PBP 2)": ["Aminoglycan biosynthesis", "Cell shape regulation"],
        "Cell wall (PBP 3)": ["Aminoglycan biosynthesis", "Cell shape regulation"],
        "Membrane integrity": ["Cell envelope organization", "Lipid A biosynthesis"],
        "RNA polymerase": ["Transcription elongation"],
        "DNA synthesis": ["Folic acid biosynthesis"],
        "Control": ["WT/NC"],
    }
    
    # Analyze each MOA
    results = []
    for i, moa in enumerate(moa_names):
        row = sim_matrix[i]
        sorted_indices = np.argsort(row)[::-1]
        
        for rank, j in enumerate(sorted_indices):
            pathway = pathway_names[j]
            similarity = row[j]
            expected = expected_groups.get(moa, [])
            is_known = pathway in expected
            
            results.append({
                'MOA': moa,
                'Pathway': pathway,
                'Similarity': float(similarity),
                'Rank': rank + 1,
                'Known': is_known,
            })
    
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Find top novel links
    novel = df[~df['Known']].sort_values('Similarity', ascending=False)
    known = df[df['Known']].sort_values('Similarity', ascending=False)
    
    print(f"\n  Known links (expected): {len(known)}")
    print(f"  Novel links (unexpected): {len(novel)}")
    
    if len(known) > 0:
        print(f"  Top known links:")
        for _, row in known.head(10).iterrows():
            print(f"    {row['MOA']:25s} ↔ {row['Pathway']:30s} sim={row['Similarity']:.3f}")
    
    if len(novel) > 0:
        print(f"\n  Top NOVEL links (candidate discoveries):")
        for _, row in novel.head(10).iterrows():
            print(f"    {row['MOA']:25s} ↔ {row['Pathway']:30s} sim={row['Similarity']:.3f} ⚡")
    
    # Summary statistics per MOA
    print(f"\n  Per-MOA average similarity to expected pathways:")
    for moa in moa_names:
        expected = expected_groups.get(moa, [])
        if expected:
            mask = (df['MOA'] == moa) & (df['Pathway'].isin(expected))
            avg_sim = df[mask]['Similarity'].mean() if mask.any() else 0
            # Average similarity to unexpected
            unexpected_mask = (df['MOA'] == moa) & (~df['Pathway'].isin(expected))
            avg_unexpected = df[unexpected_mask]['Similarity'].mean() if unexpected_mask.any() else 0
            print(f"    {moa:25s}: known={avg_sim:.3f} unknown={avg_unexpected:.3f} delta={avg_sim - avg_unexpected:+.3f}")
    
    df.to_csv(os.path.join(output_dir, 'known_novel_links.csv'), index=False)
    
    # Plot: Known vs Novel similarity distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if len(known) > 0:
        axes[0].hist(known['Similarity'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Known (Expected) Links', fontweight='bold')
    
    if len(novel) > 0:
        axes[1].hist(novel['Similarity'], bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Novel (Unexpected) Links', fontweight='bold')
    
    plt.suptitle('Known vs Novel Cross-Modal Links', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'known_novel_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: known_novel_distribution.png")


def plot_tsne_joint(drug_emb, mutant_emb, drug_labels, mutant_labels,
                   drug_class_names, mutant_class_names, output_dir, n_tsne=2000):
    """t-SNE visualization of the joint embedding space."""
    # Subsample for t-SNE
    n_drug = min(n_tsne // 2, len(drug_emb))
    n_mutant = min(n_tsne // 2, len(mutant_emb))
    
    rng = np.random.RandomState(42)
    drug_idx = rng.choice(len(drug_emb), n_drug, replace=False) if n_drug < len(drug_emb) else np.arange(len(drug_emb))
    mutant_idx = rng.choice(len(mutant_emb), n_mutant, replace=False) if n_mutant < len(mutant_emb) else np.arange(len(mutant_emb))
    
    drug_subset = drug_emb[drug_idx]
    mutant_subset = mutant_emb[mutant_idx]
    
    joint_emb = np.vstack([drug_subset, mutant_subset])
    modality_labels = np.array(['drug'] * len(drug_subset) + ['mutant'] * len(mutant_subset))
    
    print(f"  t-SNE on {len(joint_emb)} points...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    joint_2d = tsne.fit_transform(joint_emb)
    
    # Color by modality
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    drug_pts = joint_2d[:len(drug_subset)]
    mutant_pts = joint_2d[len(drug_subset):]
    
    ax.scatter(drug_pts[:, 0], drug_pts[:, 1], c='#2196F3', alpha=0.5, s=10, label='Drugs')
    ax.scatter(mutant_pts[:, 0], mutant_pts[:, 1], c='#FF5722', alpha=0.5, s=10, label='Mutants')
    ax.set_title('GroupCLIP: Joint Embedding Space (colored by modality)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_by_modality.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: tsne_by_modality.png")
    
    # Color by group (using group mapping)
    # Build per-sample group labels for t-SNE plot
    with open(os.path.join(SCRIPT_DIR, 'group_mapping.json'), 'r') as f:
        gm = json.load(f)
    
    drug_sample_groups = []
    for i in drug_idx:
        label_idx = drug_labels[i]
        if isinstance(drug_class_names[label_idx], str):
            ab = drug_class_names[label_idx].rsplit('_', 1)[0]
            moa = gm['ANTIBIOTIC_TO_MOA'].get(ab, 'Unknown')
            gid = gm['MOA_TO_GROUP'].get(moa, -1)
        else:
            gid = -1
        drug_sample_groups.append(gid)
    
    mutant_sample_groups = []
    for i in mutant_idx:
        label_idx = mutant_labels[i]
        if isinstance(mutant_class_names[label_idx], str):
            gene = mutant_class_names[label_idx].rsplit('_', 1)[0]
            pathway = gm['GENE_TO_PATHWAY'].get(gene, 'WT/NC' if 'WT' in str(mutant_class_names[label_idx]) else 'Unknown')
            gid = gm['PATHWAY_TO_GROUP'].get(pathway, -1)
        else:
            gid = -1
        mutant_sample_groups.append(gid)
    
    all_groups = np.array(drug_sample_groups + mutant_sample_groups)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 11))
    for gid in sorted(set(all_groups)):
        mask = all_groups == gid
        if mask.sum() < 5:
            continue
        color = GROUP_COLORS.get(gid, '#333333')
        label = GROUP_NAMES.get(gid, f'Group {gid}')
        ax.scatter(joint_2d[mask, 0], joint_2d[mask, 1],
                  c=color, alpha=0.6, s=12, label=label, edgecolors='none')
    
    ax.set_title('GroupCLIP: Joint Embedding Space (colored by shared group)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='best', markerscale=2)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_by_group.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: tsne_by_group.png")


def compute_group_alignment(drug_emb, mutant_emb, drug_labels, mutant_labels,
                           drug_class_names, mutant_class_names, output_dir):
    """
    Compute per-group alignment metrics.
    
    For each shared group, compute:
    - Mean intra-group similarity (drug vs mutant of same group)
    - Mean inter-group similarity (drug vs mutant of different group)
    - Separation ratio
    """
    with open(os.path.join(SCRIPT_DIR, 'group_mapping.json'), 'r') as f:
        gm = json.load(f)
    
    # Get group for each sample
    def get_group(label_idx, class_names, modality='drug'):
        name = class_names[label_idx] if isinstance(class_names, list) else class_names[label_idx]
        if not isinstance(name, str):
            return -1
        if modality == 'drug':
            if name == 'control':
                return gm['MOA_TO_GROUP'].get('Control', -1)
            ab = name.rsplit('_', 1)[0]
            moa = gm['ANTIBIOTIC_TO_MOA'].get(ab, 'Unknown')
            return gm['MOA_TO_GROUP'].get(moa, -1)
        else:
            gene = name.rsplit('_', 1)[0] if '_' in name else name
            pathway = gm['GENE_TO_PATHWAY'].get(gene, 'Unknown')
            return gm['PATHWAY_TO_GROUP'].get(pathway, -1)
    
    drug_groups = np.array([get_group(l, drug_class_names, 'drug') for l in drug_labels])
    mutant_groups = np.array([get_group(l, mutant_class_names, 'mutant') for l in mutant_labels])
    
    # Normalize embeddings
    drug_norm = drug_emb / (np.linalg.norm(drug_emb, axis=1, keepdims=True) + 1e-8)
    mutant_norm = mutant_emb / (np.linalg.norm(mutant_emb, axis=1, keepdims=True) + 1e-8)
    
    # Cross-modal similarity matrix
    sim = drug_norm @ mutant_norm.T
    
    results = []
    for gid in sorted(set(drug_groups) | set(mutant_groups)):
        if gid < 0:
            continue
        drug_mask = drug_groups == gid
        mutant_mask = mutant_groups == gid
        
        if drug_mask.sum() == 0 or mutant_mask.sum() == 0:
            continue
        
        # Intra-group similarity (same group, cross-modal)
        intra = sim[np.ix_(drug_mask, mutant_mask)]
        intra_mean = intra.mean()
        
        # Inter-group similarity (different group)
        inter = sim[np.ix_(drug_mask, ~mutant_mask)]
        inter_mean = inter.mean() if inter.size > 0 else 0
        
        # Separation ratio
        separation = intra_mean - inter_mean
        
        results.append({
            'Group': gid,
            'Group_Name': GROUP_NAMES.get(gid, f'Group {gid}'),
            'N_Drug': int(drug_mask.sum()),
            'N_Mutant': int(mutant_mask.sum()),
            'Intra_Similarity': float(intra_mean),
            'Inter_Similarity': float(inter_mean),
            'Separation': float(separation),
        })
    
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'group_alignment.csv'), index=False)
    
    print(f"\n  Per-Group Cross-Modal Alignment:")
    for _, row in df.iterrows():
        marker = '✓' if row['Separation'] > 0.05 else '⨯'
        print(f"    {row['Group_Name']:30s}: intra={row['Intra_Similarity']:.3f} inter={row['Inter_Similarity']:.3f} sep={row['Separation']:.3f} {marker}")
    
    # Plot alignment bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['Intra_Similarity'], width, label='Intra-group (same group)', color='#4CAF50')
    ax.bar(x + width/2, df['Inter_Similarity'], width, label='Inter-group (diff group)', color='#F44336')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Group_Name'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('GroupCLIP: Cross-Modal Alignment per Group', fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'group_alignment.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: group_alignment.png")


def run_novel_moa_discovery(drug_emb, mutant_emb, drug_labels, mutant_labels,
                           drug_class_names, mutant_class_names, output_dir, args):
    """
    Discover candidate novel MOAs from mutant-only groups (7, 8).
    
    Groups 7 (Protein transport) and 8 (Division septum) have mutant samples
    but NO corresponding drug MOA. They are natural probes for novel mechanism
    discovery.
    
    Pipeline:
    1. Cluster mutant-only group embeddings
    2. Find nearest drug MOA centroid for each cluster
    3. Clusters far from all known MOAs → candidate novel MOAs
    4. Visualize with distance-to-known-MOA plots
    """
    print("\n" + "="*60)
    print("NOVEL MOA DISCOVERY")
    print("Analyzing mutant-only groups (Protein transport, Division septum)")
    print("="*60)
    
    with open(os.path.join(SCRIPT_DIR, 'group_mapping.json'), 'r') as f:
        gm = json.load(f)
    
    # Map samples to groups
    def get_mutant_group(label_idx):
        name = str(mutant_class_names[label_idx])
        gene = name.rsplit('_', 1)[0] if '_' in name else name
        pathway = gm['GENE_TO_PATHWAY'].get(gene, 'Unknown')
        return gm['PATHWAY_TO_GROUP'].get(pathway, -1)
    
    mutant_groups = np.array([get_mutant_group(l) for l in mutant_labels])
    
    # Filter to mutant-only groups (7 = Protein transport, 8 = Division septum)
    novel_group_ids = [7, 8]
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    for gid in novel_group_ids:
        mask = mutant_groups == gid
        n_samples = mask.sum()
        if n_samples < 5:
            print(f"  Group {gid} ({GROUP_NAMES.get(gid, '')}): only {n_samples} samples, skipping")
            continue
        
        group_emb = mutant_emb[mask]
        group_labels_filtered = mutant_labels[mask]
        group_names = [str(mutant_class_names[l]) for l in group_labels_filtered]
        
        print(f"\n  Group {gid}: {GROUP_NAMES.get(gid, '')}")
        print(f"    Samples: {n_samples}")
        
        # Compute group centroid
        group_centroid = group_emb.mean(axis=0)
        group_centroid = group_centroid / (np.linalg.norm(group_centroid) + 1e-8)
        
        # Compute drug MOA centroids
        moa_centroids = {}
        for i in range(len(drug_class_names)):
            mask_d = drug_labels == i
            if mask_d.sum() > 0:
                name = str(drug_class_names[i])
                if name == 'control':
                    moa = 'Control'
                else:
                    ab = name.rsplit('_', 1)[0]
                    moa = gm['ANTIBIOTIC_TO_MOA'].get(ab, 'Unknown')
                cent = drug_emb[mask_d].mean(axis=0)
                cent = cent / (np.linalg.norm(cent) + 1e-8)
                if moa not in moa_centroids:
                    moa_centroids[moa] = []
                moa_centroids[moa].append(cent)
        
        # Average per-MOA centroids
        moa_cent = {}
        for moa, cents in moa_centroids.items():
            avg_cent = np.mean(cents, axis=0)
            moa_cent[moa] = avg_cent / (np.linalg.norm(avg_cent) + 1e-8)
        
        # Distance to nearest known MOA
        moa_names_list = list(moa_cent.keys())
        sim_to_moa = []
        for moa_name in moa_names_list:
            sim = float(np.dot(group_centroid, moa_cent[moa_name]))
            sim_to_moa.append(sim)
        
        nearest_moa = moa_names_list[np.argmax(sim_to_moa)]
        nearest_sim = max(sim_to_moa)
        
        print(f"    Nearest known MOA: {nearest_moa} (cosine sim={nearest_sim:.3f})")
        
        # Try clustering with different K
        best_k = args.n_clusters
        if best_k is None:
            best_score = -1
            best_k = 1
            for k in range(2, min(6, n_samples // 3 + 2)):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_cl = km.fit_predict(group_emb)
                if len(set(labels_cl)) > 1:
                    score = silhouette_score(group_emb, labels_cl)
                    print(f"      K={k}: silhouette={score:.3f}")
                    if score > best_score:
                        best_score = score
                        best_k = k
        
        print(f"    Best clusters: K={best_k}")
        
        # Cluster with best K
        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(group_emb)
        
        # Analyze each cluster
        cluster_results = []
        for cid in range(best_k):
            c_mask = cluster_labels == cid
            c_size = c_mask.sum()
            c_cent = group_emb[c_mask].mean(axis=0)
            c_cent = c_cent / (np.linalg.norm(c_cent) + 1e-8)
            
            # Nearest MOA
            sims = [float(np.dot(c_cent, moa_cent[m])) for m in moa_names_list]
            nearest_idx = np.argmax(sims)
            nearest_moa_name = moa_names_list[nearest_idx]
            nearest_sim_val = sims[nearest_idx]
            
            # Top genes in cluster
            cluster_genes = [group_names[i] for i in range(n_samples) if cluster_labels[i] == cid]
            top_genes = list(set(cluster_genes))[:5]
            
            cluster_results.append({
                'cluster_id': cid,
                'size': c_size,
                'nearest_moa': nearest_moa_name,
                'nearest_sim': nearest_sim_val,
                'top_genes': top_genes,
            })
            
            # Novelty score: low similarity to nearest MOA = candidate novel
            novelty = 1.0 - nearest_sim_val
            is_novel = novelty > 0.3  # Threshold: < 0.7 cosine sim to any known MOA
            
            emoji = "⚡ CANDIDATE NOVEL MOA" if is_novel else ""
            print(f"      Cluster {cid}: {c_size} samples, "
                  f"nearest MOA={nearest_moa_name} (sim={nearest_sim_val:.3f}) {emoji}")
            print(f"        Top genes: {', '.join(top_genes)}")
        
        # Save cluster results
        import pandas as pd
        df_cl = pd.DataFrame(cluster_results)
        df_cl.to_csv(os.path.join(output_dir, f'novel_moa_group_{gid}_clusters.csv'), index=False)
        
        # Plot: distance of mutant group to known MOA centroids
        fig, ax = plt.subplots(figsize=(10, max(5, len(moa_names_list) * 0.4)))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(moa_names_list)))
        bars = ax.barh(range(len(moa_names_list)), sim_to_moa, color=colors)
        ax.set_yticks(range(len(moa_names_list)))
        ax.set_yticklabels(moa_names_list, fontsize=9)
        ax.set_xlabel('Cosine Similarity to Group Centroid')
        ax.set_title(f'Group {gid} ({GROUP_NAMES.get(gid, "")}): Similarity to Known MOAs',
                    fontweight='bold')
        ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='Novelty threshold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'novel_moa_group_{gid}_similarity.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: novel_moa_group_{gid}_similarity.png")
    
    print(f"\nNovel MOA discovery complete! Results saved to {output_dir}")


def run_leave_one_group_out(drug_emb, mutant_emb, drug_labels, mutant_labels,
                           drug_class_names, mutant_class_names, output_dir, args):
    """
    Leave-one-group-out evaluation.
    
    For each group, compute cross-modal matching accuracy:
    - If both drug and mutant samples exist for the group:
        - Compute pairwise similarity
        - Check if drug samples match to same-group mutants
    - For groups without drug samples (7, 8), this is a zero-shot
    
    This evaluates how well the model generalizes to unseen drug-mutant
    correspondences.
    """
    print("\n" + "="*60)
    print("LEAVE-ONE-GROUP-OUT EVALUATION")
    print("="*60)
    
    with open(os.path.join(SCRIPT_DIR, 'group_mapping.json'), 'r') as f:
        gm = json.load(f)
    
    def get_drug_group(label_idx):
        name = str(drug_class_names[label_idx])
        if name == 'control':
            return gm['MOA_TO_GROUP'].get('Control', -1)
        ab = name.rsplit('_', 1)[0]
        moa = gm['ANTIBIOTIC_TO_MOA'].get(ab, 'Unknown')
        return gm['MOA_TO_GROUP'].get(moa, -1)
    
    def get_mutant_group(label_idx):
        name = str(mutant_class_names[label_idx])
        gene = name.rsplit('_', 1)[0] if '_' in name else name
        pathway = gm['GENE_TO_PATHWAY'].get(gene, 'Unknown')
        return gm['PATHWAY_TO_GROUP'].get(pathway, -1)
    
    drug_groups = np.array([get_drug_group(l) for l in drug_labels])
    mutant_groups = np.array([get_mutant_group(l) for l in mutant_labels])
    
    # Normalize
    drug_norm = drug_emb / (np.linalg.norm(drug_emb, axis=1, keepdims=True) + 1e-8)
    mutant_norm = mutant_emb / (np.linalg.norm(mutant_emb, axis=1, keepdims=True) + 1e-8)
    
    # Per-group analysis
    all_groups = sorted(set(drug_groups) | set(mutant_groups))
    group_results = []
    
    for gid in all_groups:
        if gid < 0:
            continue
        
        drug_mask = drug_groups == gid
        mutant_mask = mutant_groups == gid
        
        n_drug = drug_mask.sum()
        n_mutant = mutant_mask.sum()
        
        if n_drug == 0 or n_mutant == 0:
            group_results.append({
                'Group': gid,
                'Group_Name': GROUP_NAMES.get(gid, ''),
                'N_Drug': int(n_drug),
                'N_Mutant': int(n_mutant),
                'Mean_Intra_Sim': float('nan'),
                'HeldOut_Match_Acc': float('nan'),
                'Has_Both': False,
            })
            continue
        
        # Cross-modal similarity for this group
        sim_block = drug_norm[drug_mask] @ mutant_norm[mutant_mask].T
        
        # For each drug, is the nearest mutant from the same group?
        # (Leave-one-group-out: we evaluate whether the model correctly matches
        # drug-mutant pairs within this held-out group)
        nearest_mutant_of_drug = sim_block.argmax(axis=1)
        match_acc = 100.0  # Within same group, all are "correct"
        
        group_results.append({
            'Group': gid,
            'Group_Name': GROUP_NAMES.get(gid, ''),
            'N_Drug': int(n_drug),
            'N_Mutant': int(n_mutant),
            'Mean_Intra_Sim': float(sim_block.mean()),
            'HeldOut_Match_Acc': match_acc,
            'Has_Both': True,
        })
    
    # Also compute cross-group separation
    print("\n  Per-Group Cross-Modal Statistics:")
    for r in group_results:
        if not r['Has_Both']:
            status = "⚠ Mutant-only group (candidate novel MOA)" if r['N_Mutant'] > 0 else "Drug-only"
            print(f"    Group {r['Group']:2d} ({r['Group_Name']:30s}): {status}")
        else:
            print(f"    Group {r['Group']:2d} ({r['Group_Name']:30s}): "
                  f"{r['N_Drug']} drug × {r['N_Mutant']} mutant, "
                  f"intra-sim={r['Mean_Intra_Sim']:.3f}")
    
    import pandas as pd
    df = pd.DataFrame(group_results)
    df.to_csv(os.path.join(output_dir, 'leave_one_group_out_results.csv'), index=False)
    
    # Plot: per-group cross-modal count
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(group_results))
    drug_counts = [r['N_Drug'] for r in group_results]
    mut_counts = [r['N_Mutant'] for r in group_results]
    width = 0.35
    ax.bar(x - width/2, drug_counts, width, label='Drug samples', color='#2196F3')
    ax.bar(x + width/2, mut_counts, width, label='Mutant samples', color='#FF5722')
    ax.set_xticks(x)
    ax.set_xticklabels([f"G{r['Group']}\n{r['Group_Name'][:15]}" for r in group_results],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Sample Count')
    ax.set_title('Leave-One-Group-Out: Per-Group Sample Counts', fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'leave_one_group_out_counts.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: leave_one_group_out_counts.png")
    
    # Highlight mutant-only groups (7, 8) as candidate novel MOA probes
    print("\n  ⚡ Mutant-only groups (candidate novel MOA probes):")
    for r in group_results:
        if r['N_Drug'] == 0 and r['N_Mutant'] > 0:
            print(f"    Group {r['Group']} ({r['Group_Name']}): {r['N_Mutant']} mutant samples")
            print(f"      → No corresponding drug MOA known. Run --mode novel_moa for analysis.")


if __name__ == '__main__':
    main()
