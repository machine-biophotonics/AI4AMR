#!/usr/bin/env python3
"""
Generate t-SNE plots for final_mutant_model predictions.
Aggregates crop-level predictions to image-level via majority voting.
Uses gene-based coloring from user-provided color scheme.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.manifold import TSNE


GENE_COLORS = {
    # Peptidoglycan synthesis
    'mrcB': '#E53935',
    'mrcA': '#EF5350',
    'mrdA': '#F0625D',
    'ftsI': '#E57373',
    'murA': '#FF8A80',
    'murC': '#FFAB91',
    'lpxA': '#FFCCBC',
    'lpxC': '#FF7043',

    # Ribosome proteins
    'rpsL': '#FDD835',
    'rpsA': '#FBC02D',
    'rplA': '#F9A825',
    'rplC': '#F57F17',

    # LPS transport
    'msbA': '#00ACC1',
    'lptA': '#26C6DA',
    'lptC': '#4DD0E1',

    # DNA topology
    'gyrA': '#3949AB',
    'gyrB': '#5C6BC0',
    'parC': '#7986CB',
    'parE': '#9FA8DA',

    # Protein translocation
    'secA': '#00897B',
    'secY': '#26A69A',

    # DNA replication
    'dnaB': '#7E57C2',
    'dnaE': '#9575CD',

    # RNA polymerase
    'rpoA': '#43A047',
    'rpoB': '#66BB6A',

    # Cell division
    'ftsZ': '#D81B60',

    # Folate biosynthesis
    'folA': '#7CB342',
    'folP': '#9CCC65',

    # Control
    'WT': '#424242', 'wt': '#424242'
}

GENE_COLORS_LOWER = {k.lower(): v for k, v in GENE_COLORS.items()}
GENE_COLORS_LOWER['nc'] = '#424242'
GENE_COLORS_LOWER['wt nc'] = '#424242'

PLATE_COLORS = {
    'P1': '#1f77b4',
    'P2': '#ff7f0e',
    'P3': '#2ca02c',
    'P4': '#d62728',
    'P5': '#9467bd',
    'P6': '#8c564b',
}


def get_base_gene(label) -> str:
    """Extract base gene name from label."""
    if pd.isna(label):
        return 'wt'
    label = str(label).strip()
    label = label.replace(' ', '').replace('-', '').replace('_', '')
    if 'wt' in label.lower() or 'nc' in label.lower():
        return 'wt'
    import re
    gene = re.sub(r'(\D+)\d*$', r'\1', label)
    return gene.lower()


def load_predictions(fold_dir: str, csv_file: str = None) -> pd.DataFrame:
    """Load prediction CSV for a fold."""
    if csv_file:
        csv_path = os.path.join(fold_dir, csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
    else:
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
    return df, os.path.basename(csv_path)


def aggregate_crop_to_image(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate crop-level predictions to image-level using majority voting.
    
    For each image:
    1. Count predicted_class_name votes across all crops
    2. Select majority prediction (mode)
    3. Compute mean probability vector across all crops
    
    Returns image-level DataFrame with:
    - image_name
    - ground_truth_label (from any crop, they're all same)
    - predicted_class_name (majority vote)
    - pred_confidence (max probability from mean probs)
    - probs (mean probability vector)
    - num_crops (number of crops used)
    """
    image_results: list[dict] = []
    
    for img_name, group in df.groupby('image_name'):
        # Ground truth is same for all crops in same image
        true_label = group['ground_truth_label'].iloc[0]
        
        # MAJORITY VOTING: Get most common prediction
        pred_counts = group['predicted_class_name'].value_counts()
        majority_pred = pred_counts.index[0]
        pred_votes = pred_counts.to_dict()
        
        # Parse and average probabilities
        probs_list: list = []
        for p in group['probs']:
            if isinstance(p, str):
                try:
                    probs_list.append(json.loads(p))
                except json.JSONDecodeError:
                    probs_list.append([0.0] * 96)
            else:
                probs_list.append([0.0] * 96)
        
        mean_probs = np.mean(probs_list, axis=0)
        max_prob = np.max(mean_probs)
        
        image_results.append({
            'image_name': img_name,
            'ground_truth_label': true_label,
            'predicted_class_name': majority_pred,
            'pred_confidence': max_prob,
            'probs': mean_probs,
            'num_crops': len(group),
            'pred_votes': pred_votes
        })
    
    return pd.DataFrame(image_results)


def generate_tsne(df: pd.DataFrame, fold: str, output_dir: str) -> None:
    """Generate t-SNE plot for image-level predictions."""
    
    # Extract gene labels
    df['gene'] = df['ground_truth_label'].apply(get_base_gene).astype(str).str.lower()
    
    valid_genes = set(GENE_COLORS_LOWER.keys())
    df.loc[~df['gene'].isin(valid_genes), 'gene'] = 'wt'
    
    # Extract prediction gene
    df['pred_gene'] = df['predicted_class_name'].apply(get_base_gene).astype(str).str.lower()
    df.loc[~df['pred_gene'].isin(valid_genes), 'pred_gene'] = 'wt'
    
    # Check if correct prediction
    df['correct'] = (df['gene'] == df['pred_gene']).astype(int)
    
    print(f"  Images: {len(df)}, Genes: {df['gene'].nunique()}, Accuracy: {df['correct'].mean():.3f}")
    
    # Get probability matrix
    X = np.array(df['probs'].tolist())
    
    # Run t-SNE
    perplexity = min(30, len(X) - 1)
    print(f"  Running t-SNE (perplexity={perplexity})...")
    
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        max_iter=1000
    )
    
    X_tsne = tsne.fit_transform(X)
    
    df['tsne_x'] = X_tsne[:, 0]
    df['tsne_y'] = X_tsne[:, 1]
    
    # Create plot
    fig = px.scatter(
        df,
        x='tsne_x',
        y='tsne_y',
        color='gene',
        color_discrete_map=GENE_COLORS_LOWER,
        hover_data=['image_name', 'ground_truth_label', 'predicted_class_name', 'pred_confidence'],
        title=f't-SNE - Final Mutant Model - Fold {fold} ({len(df)} images, majority voting)',
        labels={'tsne_x': 't-SNE 1', 'tsne_y': 't-SNE 2'}
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        width=1200,
        height=900,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        )
    )
    
    output_path = os.path.join(output_dir, f'tsne_image_majority_fold_{fold}.html')
    pio.write_html(fig, output_path)
    print(f"  Saved: {output_path}")
    
    # Save CSV
    csv_path = os.path.join(output_dir, f'tsne_image_majority_fold_{fold}.csv')
    df[['image_name', 'ground_truth_label', 'predicted_class_name', 'pred_confidence', 
        'gene', 'pred_gene', 'correct', 'tsne_x', 'tsne_y']].to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate t-SNE plots for final_mutant_model')
    parser.add_argument('--fold', type=str, default='P1', help='Fold to process (P1, P2, P6)')
    parser.add_argument('--csv', type=str, default=None, help='Specific CSV file to use')
    parser.add_argument('--folds', type=str, default=None, help='Comma-separated folds (e.g., P1,P2,P6)')
    parser.add_argument('--all_folds', action='store_true', help='Process all available folds')
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--max_iter', type=int, default=1000)
    args = parser.parse_args()
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Determine folds to process
    fold_list = []
    if args.all_folds:
        fold_list = ['P1', 'P2', 'P6']  # Available
    elif args.folds:
        fold_list = args.folds.split(',')
    elif args.fold:
        fold_list = [args.fold]
    
    output_dir = os.path.join(SCRIPT_DIR, 'train_test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Final Mutant Model - t-SNE with Majority Voting")
    print("="*60)
    print(f"Folds: {fold_list}")
    print(f"Output: {output_dir}")
    print()
    
    for fold in fold_list:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{fold}')
        
        if not os.path.exists(fold_dir):
            print(f"Skipping {fold}: folder not found")
            continue
        
        # Determine CSV files to process
        csv_files = []
        if args.csv:
            csv_files = [args.csv]
        else:
            csv_files = [
                'predictions_all_crops_mil_best_model.csv',
                'predictions_all_crops_mil_best_model_acc.csv',
            ]
        
        for csv_file in csv_files:
            csv_path = os.path.join(fold_dir, csv_file)
            if not os.path.exists(csv_path):
                print(f"  Skipping {csv_file}: not found")
                continue
            
            print(f"\nProcessing fold {fold}, file {csv_file}...")
            
            try:
                # Load predictions
                df, csv_name = load_predictions(fold_dir, csv_file)
                print(f"  Loaded {len(df)} crop predictions")
                
                # Aggregate to image level
                image_df = aggregate_crop_to_image(df)
                print(f"  Aggregated to {len(image_df)} images (majority voting)")
                
                # Generate t-SNE with custom name
                csv_base = csv_name.replace('.csv', '')
                output_name = f"tsne_image_majority_{csv_base}_{fold}"
                
                # Extract gene labels for the image_df
                image_df['gene'] = image_df['ground_truth_label'].apply(get_base_gene).astype(str).str.lower()
                valid_genes = set(GENE_COLORS_LOWER.keys())
                image_df.loc[~image_df['gene'].isin(valid_genes), 'gene'] = 'wt'
                image_df['pred_gene'] = image_df['predicted_class_name'].apply(get_base_gene).astype(str).str.lower()
                image_df.loc[~image_df['pred_gene'].isin(valid_genes), 'pred_gene'] = 'wt'
                image_df['correct'] = (image_df['gene'] == image_df['pred_gene']).astype(int)
                
                print(f"  Images: {len(image_df)}, Genes: {image_df['gene'].nunique()}, Accuracy: {image_df['correct'].mean():.3f}")
                
                X = np.array(image_df['probs'].tolist())
                perplexity = min(args.perplexity, len(X) - 1)
                print(f"  Running t-SNE (perplexity={perplexity})...")
                
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=args.max_iter)
                X_tsne = tsne.fit_transform(X)
                image_df = image_df.copy()
                image_df['tsne_x'] = X_tsne[:, 0]
                image_df['tsne_y'] = X_tsne[:, 1]
                
                fig = px.scatter(image_df, x='tsne_x', y='tsne_y', color='gene', color_discrete_map=GENE_COLORS_LOWER,
                    hover_data=['image_name', 'ground_truth_label', 'predicted_class_name', 'pred_confidence'],
                    title=f't-SNE - Final Mutant - {csv_base} - {fold} ({len(image_df)} images)')
                fig.update_traces(marker=dict(size=8, opacity=0.7))
                fig.update_layout(width=1200, height=900, legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=10)))
                
                out_html = os.path.join(output_dir, f'{output_name}.html')
                pio.write_html(fig, out_html)
                print(f"  Saved: {out_html}")
                
                out_csv = os.path.join(output_dir, f'{output_name}.csv')
                image_df[['image_name', 'ground_truth_label', 'predicted_class_name', 'pred_confidence', 'gene', 'pred_gene', 'correct', 'tsne_x', 'tsne_y']].to_csv(out_csv, index=False)
                print(f"  Saved: {out_csv}")
                
            except Exception as e:
                import traceback
                print(f"  Error: {e}")
                traceback.print_exc()
                continue
    
    print("\nDone!")


if __name__ == '__main__':
    main()