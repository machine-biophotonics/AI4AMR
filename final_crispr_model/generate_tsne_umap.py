#!/usr/bin/env python3
"""
Generate t-SNE and UMAP plots with gene colors and guide shapes.
- Color: Gene-based coloring (biological pathway colors)
- Shape: Guide number (1=circle, 2=square, 3=triangle, 4=pentagon, 5=star, 6=cross)
- All folds option: combine all plates with different color schemes
- Pathway coloring: color by biological pathway using --color_by pathway
"""

import argparse
import json
import os
import json
import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import Optional, Dict, Tuple

import plotly.express as px
import plotly.io as pio
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not available, will skip UMAP plots")


GENE_COLORS = {
    # Folic acid biosynthetic process
    'folP': '#D95F02',
    'folA': '#F28E2B',

    # Intracellular protein transport
    'secY': '#E377C2',
    'secA': '#F2A2D6',

    # Regulation of DNA-templated transcription elongation
    'rpoB': '#9467BD',
    'rpoA': '#B79BD9',

    # Cell envelope organization
    'lptC': '#00A65A',
    'lptA': '#2FBF71',
    'msbA': '#66D18E',

    # Division septum assembly
    'ftsZ': '#7CAE00',

    # Regulation of translational initiation
    'rplC': '#4C78A8',
    'rplA': '#6A91C7',
    'rpsA': '#8AAAE5',
    'rpsL': '#AEC7FF',

    # Aminoglycan biosynthetic process
    'murC': '#17A2B8',
    'murA': '#45B8C6',
    'mrcB': '#FF7043',

    # Regulation of cell shape
    'mrdA': '#B58900',
    'mrcA': '#D4A017',
    'ftsI': '#E3BF3A',

    # Lipid A biosynthetic process
    'lpxC': '#20B2AA',
    'lpxA': '#4FC3BD',

    # Chromosome organization
    'gyrB': '#F17CB0',
    'gyrA': '#F29CB3',
    'dnaB': '#F4B6C2',
    'parE': '#F7C7CE',
    'parC': '#FAD9DF',
    'dnaE': '#F9A3A3',

    # Controls
    'WT NC_1': '#000000', 'WT NC_2': '#000000', 'WT NC_3': '#000000',
    'WT NC_4': '#000000', 'WT NC_5': '#000000', 'WT NC_6': '#000000',
    'NC_1': '#808080', 'NC_2': '#808080', 'NC_3': '#808080',
    'NC_4': '#808080', 'NC_5': '#808080', 'NC_6': '#808080',
    'WT': '#424242', 'wt': '#424242'
}

# Create lowercase lookup for case-insensitive matching
GENE_COLORS_LOWER = {k.lower(): v for k, v in GENE_COLORS.items()}
GENE_COLORS_LOWER['nc'] = '#808080'  # Gray for NC, WT NC, NC_1-6, WT NC_1-6
GENE_COLORS_LOWER['wt_nc'] = '#808080'  # Legacy - also gray

GUIDE_SHAPES = {
    1: 'circle',
    2: 'square',
    3: 'triangle-up',
    4: 'pentagon',
    5: 'star',
    6: 'x'
}

PLATE_COLORS = {
    'P1': '#1f77b4',  # blue
    'P2': '#ff7f0e',  # orange
    'P3': '#2ca02c',  # green
    'P4': '#d62728',  # red
    'P5': '#9467bd',  # purple
    'P6': '#8c564b',  # brown
}

# Trial pathway mapping - user's biological process table
TRIAL_PATHWAY = {
    # Controls - map to 'WT' pathway (will use gray/black colors)
    'NC_1': 'WT', 'NC_2': 'WT', 'NC_3': 'WT',
    'NC_4': 'WT', 'NC_5': 'WT', 'NC_6': 'WT',
    'WT NC_1': 'WT', 'WT NC_2': 'WT', 'WT NC_3': 'WT',
    'WT NC_4': 'WT', 'WT NC_5': 'WT', 'WT NC_6': 'WT',
    
    # Folic acid biosynthetic process
    'folP': 'folic acid biosynthetic process',
    'folA': 'folic acid biosynthetic process',
    
    # Intracellular protein transport
    'secY': 'intracellular protein transport',
    'secA': 'intracellular protein transport',
    
    # Regulation of DNA-templated transcription elongation
    'rpoB': 'regulation of DNA-templated transcription elongation',
    'rpoA': 'regulation of DNA-templated transcription elongation',
    
    # Cell envelope organization
    'lptC': 'cell envelope organization',
    'lptA': 'cell envelope organization',
    'msbA': 'cell envelope organization',
    
    # Division septum assembly
    'ftsZ': 'division septum assembly',
    
    # Regulation of translational initiation
    'rplC': 'regulation of translational initiation',
    'rplA': 'regulation of translational initiation',
    'rpsA': 'regulation of translational initiation',
    'rpsL': 'regulation of translational initiation',
    
    # Aminoglycan biosynthetic process
    'murC': 'aminoglycan biosynthetic process',
    'murA': 'aminoglycan biosynthetic process',
    'mrcB': 'aminoglycan biosynthetic process',
    
    # Regulation of cell shape
    'mrdA': 'regulation of cell shape',
    'mrcA': 'regulation of cell shape',
    'ftsI': 'regulation of cell shape',
    
    # Lipid A biosynthetic process
    'lpxC': 'lipid A biosynthetic process',
    'lpxA': 'lipid A biosynthetic process',
    
    # Chromosome organization
    'gyrB': 'chromosome organization',
    'gyrA': 'chromosome organization',
    'dnaB': 'chromosome organization',
    'parE': 'chromosome organization',
    'parC': 'chromosome organization',
    'dnaE': 'chromosome organization',
}

# Pathway colors - unique colors for each pathway
PATHWAY_COLORS = {
    'folic acid biosynthetic process': '#D95F02',     # orange
    'intracellular protein transport': '#E377C2',      # pink
    'regulation of DNA-templated transcription elongation': '#9467BD',  # purple
    'cell envelope organization': '#00A65A',           # green
    'division septum assembly': '#7CAE00',             # lime
    'regulation of translational initiation': '#4C78A8',  # blue
    'aminoglycan biosynthetic process': '#17A2B8',     # teal
    'regulation of cell shape': '#B58900',             # gold
    'lipid A biosynthetic process': '#20B2AA',        # cyan
    'chromosome organization': '#F17CB0',             # rose
    'WT': '#424242',                                  # black for WT/NC
}


def get_base_gene(label: str) -> str:
    """Extract base gene name from label like 'dnaA_1' or 'NC_1' or 'WT NC_1'."""
    if not label or label == 'nan':
        return 'WT'
    label = str(label)
    
    # Handle NC variants - extract 'NC' as base
    if label.startswith('NC_'):
        return 'NC'
    if label.startswith('WT NC_'):
        return 'WT NC'
    
    # Regular gene: extract base
    if '_' in label:
        return label.rsplit('_', 1)[0]
    return label


def get_pathway(label: str) -> str:
    """Get pathway for a gene label."""
    base = get_base_gene(label)
    if base.upper() in ['WT', 'NC', 'WT NC']:
        return 'WT'
    return TRIAL_PATHWAY.get(base, 'WT')


def get_gene_and_guide(label: str) -> Tuple[str, int]:
    """Extract gene name and guide number from label like 'dnaA_1'."""
    if not label or label == 'nan':
        return 'wt', 0
    label = str(label)
    
    # Handle NC variants - WT NC maps to 'wt', NC maps to 'nc'
    if label.startswith('WT NC_'):
        parts = label.split('_')
        guide = int(parts[-1])
        return 'wt', guide
    if label.startswith('NC_'):
        parts = label.split('_')
        guide = int(parts[-1])
        return 'nc', guide
    
    # Regular gene: extract base and guide
    if '_' in label:
        parts = label.rsplit('_', 1)
        gene = parts[0].lower()
        try:
            guide = int(parts[1])
        except (ValueError, IndexError):
            guide = 0
    else:
        gene = label.lower()
        guide = 0
    return gene, guide


def load_predictions(fold_dir: str, prefer_mil: bool = True, prediction_csv: str = None, mixed_checkpoints: bool = False) -> pd.DataFrame:
    """Load prediction CSV for a fold."""
    if prediction_csv:
        csv_path = os.path.join(fold_dir, prediction_csv)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
    
    csv_files = [
        'predictions_all_crops_mil_best_model.csv',
        'predictions_all_crops_mil_best_model_acc.csv',
        'predictions_all_crops_mil_100pos.csv',
        'predictions_all_crops_best_model.csv',
        'predictions_all_crops.csv',
    ]
    
    csv_path = None
    for f in csv_files:
        path = os.path.join(fold_dir, f)
        if os.path.exists(path):
            csv_path = path
            break
    
    if csv_path is None:
        raise FileNotFoundError(f"No prediction CSV found in {fold_dir}")
    
    df = pd.read_csv(csv_path)
    return df


def load_embeddings(fold_dir: str) -> pd.DataFrame:
    """Load embeddings CSV for a fold."""
    csv_path = os.path.join(fold_dir, 'image_embeddings.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Embeddings CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df['embedding'] = df['embedding'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    # Re-derive ground_truth_label from image_name (fix for NaN values)
    # Find plate_well_id_path.json by searching up from fold_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plate_data_path = os.path.join(script_dir, 'plate_well_id_path.json')
    with open(plate_data_path, 'r') as f:
        plate_data = json.load(f)
    
    def get_label_from_image(row):
        plate = row['plate']
        match = re.search(r'Well([A-H]\d+)', row['image_name'])
        if match:
            well = match.group(1)
            row_letter = well[0]
            col = str(int(well[1:]))  # '01' -> '1', '10' -> '10'
            if plate in plate_data and row_letter in plate_data[plate]:
                if col in plate_data[plate][row_letter]:
                    return plate_data[plate][row_letter][col].get('id', None)
        return None
    
    # Compute new labels for all rows
    new_labels = df.apply(get_label_from_image, axis=1)
    
    # If original labels are NaN, replace with computed ones
    df['ground_truth_label'] = df['ground_truth_label'].where(df['ground_truth_label'].notna(), new_labels)
    
    return df


def aggregate_from_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """Create image_df from embeddings."""
    image_results = []
    
    for _, row in df.iterrows():
        image_results.append({
            'image_name': row['image_name'],
            'true_label': row['ground_truth_label'],
            'pred_label': row['ground_truth_label'],
            'probs': row['embedding'],
        })
    
    return pd.DataFrame(image_results)


def aggregate_to_image(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate crop predictions to image level."""
    image_results = []
    
    for img_name, group in df.groupby('image_name'):
        true_label = group['ground_truth_label'].iloc[0]
        pred_counts = group['predicted_class_name'].value_counts()
        majority_pred = pred_counts.index[0]
        
        probs_list = []
        for p in group['probs']:
            if isinstance(p, str):
                try:
                    probs_list.append(json.loads(p))
                except json.JSONDecodeError:
                    probs_list.append([0.0] * 96)
            else:
                probs_list.append([0.0] * 96)
        
        mean_probs = np.mean(probs_list, axis=0)
        
        image_results.append({
            'image_name': img_name,
            'true_label': true_label,
            'pred_label': majority_pred,
            'probs': mean_probs
        })
    
    return pd.DataFrame(image_results)


def main():
    parser = argparse.ArgumentParser(description='Generate t-SNE/UMAP plots')
    parser.add_argument('--fold', type=str, default='P6', help='Fold to plot (P1-P6)')
    parser.add_argument('--run_all_folds', action='store_true', help='Generate plots for all folds')
    parser.add_argument('--method', type=str, default='both', choices=['tsne', 'umap', 'both'])
    parser.add_argument('--perplexity', type=str, default='50', help='Comma-separated perplexity values (e.g., "30,50,100") - see t-SNE best practices')
    parser.add_argument('--iterations', type=int, default=5000, help='t-SNE iterations (recommended: 1000-5000 for convergence)')
    parser.add_argument('--learning_rate', type=int, default=200, help='t-SNE learning rate (default: 200)')
    parser.add_argument('--dim3', action='store_true', help='Generate 3D t-SNE visualization')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--prediction_csv', type=str, default=None, help='Specific prediction CSV file to use')
    parser.add_argument('--mixed_checkpoints', action='store_true', help='Use best_model_acc for P1-P5, best_model for P6')
    parser.add_argument('--color_by', type=str, default='gene', choices=['gene', 'pathway'], help='Color by gene or pathway')
    parser.add_argument('--use_embeddings', action='store_true', help='Use embeddings instead of probabilities')
    args = parser.parse_args()
    
    # Parse perplexity values
    perplexities = [int(p.strip()) for p in args.perplexity.split(',')]
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    def process_single_fold(fold: str, output_dir: str) -> None:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{fold}')
        if not os.path.exists(fold_dir):
            print(f"Fold directory not found: {fold_dir}")
            return
        
        print(f"\n{'='*60}")
        print(f"Processing fold {fold}")
        print(f"{'='*60}")
        
        if args.use_embeddings:
            print("Using image embeddings (1280-dim) for t-SNE...")
            emb_df = load_embeddings(fold_dir)
            image_df = aggregate_from_embeddings(emb_df)
            feature_dim = len(image_df['probs'].iloc[0])
            print(f"Loaded {len(image_df)} image embeddings (dim={feature_dim})")
        else:
            df = load_predictions(fold_dir, prediction_csv=args.prediction_csv, mixed_checkpoints=args.mixed_checkpoints)
            df_valid = df[df['ground_truth_label'].notna()].copy()
            image_df = aggregate_to_image(df_valid)
            print(f"Aggregated to {len(image_df)} images")
        
        gene_guide = image_df['true_label'].apply(lambda x: get_gene_and_guide(x))
        image_df['gene'] = gene_guide.apply(lambda x: x[0])
        image_df['guide'] = gene_guide.apply(lambda x: x[1])
        
        # Use lowercase lookup for case-insensitive matching - default unknown to WT
        valid_genes = set(GENE_COLORS_LOWER.keys())
        image_df.loc[~image_df['gene'].isin(valid_genes), 'gene'] = 'wt'
        
        # Map gene to pathway for coloring
        image_df['pathway'] = image_df['true_label'].apply(lambda x: get_pathway(x))
        
        image_df['guide'] = image_df['guide'].fillna(0).astype(int)
        
        print(f"Genes: {image_df['gene'].nunique()}, Pathways: {image_df['pathway'].nunique()}, Guides: {sorted(image_df['guide'].unique())}")
        
        X = np.array(image_df['probs'].tolist())
        
        image_df['plate'] = fold
        
        results = {'image_df': image_df, 'X': X}
        
        if args.method in ['tsne', 'both']:
            print(f"Running t-SNE (perplexity={perplexities}, iterations={args.iterations})...")
            
            # Use first perplexity for main plot (or run multiple)
            # t-SNE is stochastic - use fixed random state for reproducibility
            perp = min(perplexities[0], len(X) - 1)
            
            tsne = TSNE(
                n_components=2,
                random_state=42,
                perplexity=perp,
                max_iter=args.iterations,
                learning_rate=args.learning_rate,
                n_iter_without_progress=300
            )
            X_tsne = tsne.fit_transform(X)
            image_df['tsne_x'] = X_tsne[:, 0]
            image_df['tsne_y'] = X_tsne[:, 1]
            
            # Determine color column and discrete map
            color_col = 'pathway' if args.color_by == 'pathway' else 'gene'
            discrete_map = PATHWAY_COLORS if args.color_by == 'pathway' else GENE_COLORS_LOWER
            
            # Add pathway to hover data when coloring by gene
            hover_cols = ['image_name', 'true_label', 'pred_label']
            if args.color_by == 'gene':
                hover_cols.append('pathway')
            
            fig = px.scatter(
                image_df,
                x='tsne_x',
                y='tsne_y',
                color=color_col,
                symbol='guide',
                color_discrete_map=discrete_map,
                symbol_map=GUIDE_SHAPES,
                hover_data=hover_cols,
                title=f't-SNE Embeddings (1280-dim) - Fold {fold} ({len(image_df)} images) [{args.color_by}]',
                labels={'tsne_x': 't-SNE 1', 'tsne_y': 't-SNE 2'}
            )
            fig.update_traces(marker=dict(size=5, opacity=0.8))
            fig.update_layout(width=1200, height=900)
            
            # Output filename includes color_by and embedding flag
            color_suffix = 'pathway' if args.color_by == 'pathway' else 'gene_guide'
            out_path = os.path.join(output_dir, f'tsne_embedding_{color_suffix}_{fold}.html')
            pio.write_html(fig, out_path)
            print(f"Saved t-SNE: {out_path}")
        
        if args.method in ['umap', 'both'] and HAS_UMAP:
            print(f"Running UMAP...")
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            X_umap = reducer.fit_transform(X)
            image_df['umap_x'] = X_umap[:, 0]
            image_df['umap_y'] = X_umap[:, 1]
            
            # Determine color column and discrete map
            color_col = 'pathway' if args.color_by == 'pathway' else 'gene'
            discrete_map = PATHWAY_COLORS if args.color_by == 'pathway' else GENE_COLORS_LOWER
            
            # Add pathway to hover data when coloring by gene
            hover_cols = ['image_name', 'true_label', 'pred_label']
            if args.color_by == 'gene':
                hover_cols.append('pathway')
            
            fig = px.scatter(
                image_df,
                x='umap_x',
                y='umap_y',
                color=color_col,
                symbol='guide',
                color_discrete_map=discrete_map,
                symbol_map=GUIDE_SHAPES,
                hover_data=hover_cols,
                title=f'UMAP - Fold {fold} ({len(image_df)} images) [{args.color_by}]',
                labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2'}
            )
            fig.update_traces(marker=dict(size=5, opacity=0.8))
            fig.update_layout(width=1200, height=900)
            
            # Output filename includes color_by
            color_suffix = 'pathway' if args.color_by == 'pathway' else 'gene_guide'
            out_path = os.path.join(output_dir, f'umap_{color_suffix}_{fold}.html')
            pio.write_html(fig, out_path)
            print(f"Saved UMAP: {out_path}")
    
    def process_all_folds(output_dir: str) -> None:
        all_dfs = []
        all_X = []
        
        for fold in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
            fold_dir = os.path.join(SCRIPT_DIR, f'fold_{fold}')
            if not os.path.exists(fold_dir):
                continue
            
            try:
                df = load_predictions(fold_dir, prediction_csv=args.prediction_csv, mixed_checkpoints=args.mixed_checkpoints)
                df_valid = df[df['ground_truth_label'].notna()].copy()
                image_df = aggregate_to_image(df_valid)
                
                gene_guide = image_df['true_label'].apply(lambda x: get_gene_and_guide(x))
                image_df['gene'] = gene_guide.apply(lambda x: x[0])
                image_df['guide'] = gene_guide.apply(lambda x: x[1])
                
                # Use lowercase lookup for case-insensitive matching - default unknown to WT
                valid_genes = set(GENE_COLORS_LOWER.keys())
                image_df.loc[~image_df['gene'].isin(valid_genes), 'gene'] = 'wt'
                
                # Map gene to pathway
                image_df['pathway'] = image_df['true_label'].apply(lambda x: get_pathway(x))
                
                image_df['guide'] = image_df['guide'].fillna(0).astype(int)
                image_df['plate'] = fold
                
                all_dfs.append(image_df)
                all_X.append(np.array(image_df['probs'].tolist()))
                print(f"Loaded fold {fold}: {len(image_df)} images")
            except Exception as e:
                print(f"Skipping fold {fold}: {e}")
        
        if not all_dfs:
            print("No folds loaded!")
            return
        
        X_all = np.vstack(all_X)
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        print(f"\nTotal: {len(combined_df)} images from {len(all_dfs)} plates")
        
        if args.method in ['tsne', 'both']:
            print(f"Running t-SNE (perplexity={perplexities}, iterations={args.iterations}, lr={args.learning_rate})...")
            
            # Run t-SNE with multiple perplexities as recommended in t-SNE best practices
            # Different perplexities reveal different structures (local vs global)
            for perp_val in perplexities:
                perp = min(perp_val, len(X_all) - 1)
                print(f"  Processing perplexity={perp}...")
                
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=perp,
                    max_iter=args.iterations,
                    learning_rate=args.learning_rate,
                    n_iter_without_progress=300
                )
                X_tsne = tsne.fit_transform(X_all)
                
                # Add suffix to distinguish different perplexities
                suffix = f"_perp{perp}" if len(perplexities) > 1 else ""
                
                combined_df[f'tsne_x{suffix}'] = X_tsne[:, 0]
                combined_df[f'tsne_y{suffix}'] = X_tsne[:, 1]
                
                # Determine color column and discrete map
                color_col = 'pathway' if args.color_by == 'pathway' else 'gene'
                discrete_map = PATHWAY_COLORS if args.color_by == 'pathway' else GENE_COLORS_LOWER
                
                # Add pathway to hover data when coloring by gene
                hover_cols = ['image_name', 'plate', 'true_label']
                if args.color_by == 'gene':
                    hover_cols.append('pathway')
                
                fig1 = px.scatter(
                    combined_df,
                    x=f'tsne_x{suffix}',
                    y=f'tsne_y{suffix}',
                    color=color_col,
                    symbol='guide',
                    color_discrete_map=discrete_map,
                    symbol_map=GUIDE_SHAPES,
                    hover_data=hover_cols,
                    title=f't-SNE (perplexity={perp}) - {color_col.capitalize()} Color + Guide Shape ({len(combined_df)} images)',
                )
                fig1.update_traces(marker=dict(size=5, opacity=0.8))
                fig1.update_layout(width=1400, height=1000)
                
                color_suffix = 'pathway' if args.color_by == 'pathway' else 'gene_guide'
                out_path1 = os.path.join(output_dir, f'tsne_all_folds_{color_suffix}{suffix}.html')
                pio.write_html(fig1, out_path1)
                print(f"Saved: {out_path1}")
                
                combined_df['is_wt'] = combined_df['gene'] == 'wt'
                
                fig2 = px.scatter(
                    combined_df,
                    x=f'tsne_x{suffix}',
                    y=f'tsne_y{suffix}',
                    color='plate',
                    symbol='is_wt',
                    color_discrete_map=PLATE_COLORS,
                    symbol_map={True: 'circle', False: 'square'},
                    hover_data=['image_name', 'gene', 'guide'],
                    title=f't-SNE (perplexity={perp}) - Plate Color + WT Shape ({len(combined_df)} images)',
                )
                fig2.update_traces(marker=dict(size=8, opacity=0.7))
                fig2.update_layout(width=1400, height=1000)
                
                out_path2 = os.path.join(output_dir, f'tsne_all_folds_plate_shape{suffix}.html')
                pio.write_html(fig2, out_path2)
                print(f"Saved: {out_path2}")
                
                # Generate 3D t-SNE if requested
                if args.dim3:
                    print(f"  Running 3D t-SNE (perplexity={perp})...")
                    tsne_3d = TSNE(
                        n_components=3,
                        random_state=42,
                        perplexity=perp,
                        max_iter=args.iterations,
                        learning_rate=args.learning_rate,
                        n_iter_without_progress=300
                    )
                    X_tsne_3d = tsne_3d.fit_transform(X_all)
                    
                    combined_df['tsne_3d_x'] = X_tsne_3d[:, 0]
                    combined_df['tsne_3d_y'] = X_tsne_3d[:, 1]
                    combined_df['tsne_3d_z'] = X_tsne_3d[:, 2]
                    
                    fig3d_genes = px.scatter_3d(
                        combined_df,
                        x='tsne_3d_x',
                        y='tsne_3d_y',
                        z='tsne_3d_z',
                        color='gene',
                        symbol='guide',
                        color_discrete_map=GENE_COLORS_LOWER,
                        symbol_map=GUIDE_SHAPES,
                        hover_data=['image_name', 'plate', 'true_label'],
                        title=f'3D t-SNE (perplexity={perp}) - Gene Color + Guide Shape ({len(combined_df)} images)',
                    )
                    fig3d_genes.update_traces(marker=dict(size=5, opacity=0.7))
                    fig3d_genes.update_layout(width=1400, height=1000)
                    
                    out_path3d = os.path.join(output_dir, f'tsne3d_all_folds_gene_guide_perp{perp}.html')
                    pio.write_html(fig3d_genes, out_path3d)
                    print(f"Saved 3D: {out_path3d}")
        
        if args.method in ['umap', 'both'] and HAS_UMAP:
            print("Running UMAP on all folds...")
            
            # UMAP also needs careful parameter selection
            # n_neighbors controls local vs global (low = local, high = global)
            # min_dist controls cluster tightness (0 = tight, 1 = spread)
            n_neighbors_list = [15, 30, 50]  # Test multiple values
            
            for n_neigh in n_neighbors_list:
                print(f"  UMAP n_neighbors={n_neigh}...")
                reducer = umap.UMAP(
                    n_components=2, 
                    random_state=42, 
                    n_neighbors=n_neigh, 
                    min_dist=0.1,
                    metric='cosine'  # Cosine works better for probability distributions
                )
                X_umap = reducer.fit_transform(X_all)
                
                suffix = f"_nn{n_neigh}" if len(n_neighbors_list) > 1 else ""
                
                combined_df[f'umap_x{suffix}'] = X_umap[:, 0]
                combined_df[f'umap_y{suffix}'] = X_umap[:, 1]
                
                # Determine color column and discrete map
                color_col = 'pathway' if args.color_by == 'pathway' else 'gene'
                discrete_map = PATHWAY_COLORS if args.color_by == 'pathway' else GENE_COLORS_LOWER
                
                # Add pathway to hover data when coloring by gene
                hover_cols = ['image_name', 'plate', 'true_label']
                if args.color_by == 'gene':
                    hover_cols.append('pathway')
                
                fig1 = px.scatter(
                    combined_df,
                    x=f'umap_x{suffix}',
                    y=f'umap_y{suffix}',
                    color=color_col,
                    symbol='guide',
                    color_discrete_map=discrete_map,
                    symbol_map=GUIDE_SHAPES,
                    hover_data=hover_cols,
                    title=f'UMAP (n_neighbors={n_neigh}) - {color_col.capitalize()} Color + Guide Shape',
                )
                fig1.update_traces(marker=dict(size=5, opacity=0.8))
                fig1.update_layout(width=1400, height=1000)
                
                color_suffix = 'pathway' if args.color_by == 'pathway' else 'gene_guide'
                out_path1 = os.path.join(output_dir, f'umap_all_folds_{color_suffix}{suffix}.html')
                pio.write_html(fig1, out_path1)
                print(f"Saved: {out_path1}")
    
    if args.run_all_folds:
        output_dir = args.output_dir or os.path.join(SCRIPT_DIR, 'train_test_results')
        os.makedirs(output_dir, exist_ok=True)
        process_all_folds(output_dir)
    else:
        output_dir = args.output_dir or os.path.join(SCRIPT_DIR, f'fold_{args.fold}')
        os.makedirs(output_dir, exist_ok=True)
        process_single_fold(args.fold, output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()