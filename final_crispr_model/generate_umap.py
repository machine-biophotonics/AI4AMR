#!/usr/bin/env python3
"""
Generate interactive UMAP plot for predictions.
Color by gene name using fixed biological color map.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import umap

from utils import (
    GENE_COLORS,
    find_prediction_csv,
    logger
)


def get_base_gene(label) -> str:
    """Extract gene name from label."""
    from utils import get_base_gene as _get_base_gene
    return _get_base_gene(label)


def aggregate_to_image(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate crop predictions to image level."""
    image_results: list[dict] = []
    
    for img_name, group in df.groupby('image_name'):
        true_label = group['ground_truth_label'].iloc[0]
        
        pred_counts = group['predicted_class_name'].value_counts()
        majority_pred = pred_counts.index[0]
        
        probs_list: list = []
        for p in group['probs']:
            if isinstance(p, str):
                try:
                    probs_list.append(json.loads(p))
                except json.JSONDecodeError:
                    probs_list.append(p)
            else:
                probs_list.append(p)
        
        mean_probs = np.mean(probs_list, axis=0)
        
        image_results.append({
            'image_name': img_name,
            'true_label': true_label,
            'pred_label': majority_pred,
            'probs': mean_probs
        })
    
    return pd.DataFrame(image_results)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate UMAP plot for fold predictions')
    parser.add_argument('--fold', type=str, default='P3')
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--output', type=str, default='umap_gene_interactive.html')
    parser.add_argument('--n_neighbors', type=int, default=15)
    parser.add_argument('--min_dist', type=float, default=0.1)
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    if args.csv:
        csv_path = args.csv
    else:
        fold_dir = os.path.join(SCRIPT_DIR, f'fold_{args.fold}')
        csv_path = find_prediction_csv(fold_dir, prefer_mil=True)
        if csv_path is None:
            csv_path = os.path.join(fold_dir, 'predictions_all_crops_mil_best_model_acc.csv')

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    df_valid = df[df['ground_truth_label'].notna()].copy()
    
    print(f"Loaded {len(df_valid)} crop predictions")

    image_df = aggregate_to_image(df_valid)

    image_df['gene'] = image_df['true_label'].apply(get_base_gene).astype(str).str.lower()

    valid_genes = set(GENE_COLORS.keys())
    image_df.loc[~image_df['gene'].isin(valid_genes), 'gene'] = 'wt'

    print(f"Aggregated to {len(image_df)} images")
    print(f"Unique genes: {image_df['gene'].nunique()}")

    X = np.array(image_df['probs'].tolist())

    print(f"Running UMAP on {X.shape} features...")

    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric='euclidean'
    )

    X_umap = reducer.fit_transform(X)

    image_df['umap_x'] = X_umap[:, 0]
    image_df['umap_y'] = X_umap[:, 1]

    fig = px.scatter(
        image_df,
        x='umap_x',
        y='umap_y',
        color='gene',
        color_discrete_map=GENE_COLORS,
        hover_data=['image_name', 'true_label', 'pred_label'],
        title=f'UMAP Visualization - Fold {args.fold} ({len(image_df)} images)',
        labels={'umap_x': 'UMAP 1', 'umap_y': 'UMAP 2', 'gene': 'Gene'}
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

    output_dir = os.path.join(SCRIPT_DIR, 'train_test_results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output)

    pio.write_html(fig, output_path)
    print(f"Saved interactive UMAP to {output_path}")

    try:
        fig.write_image(output_path.replace('.html', '.png'), width=1200, height=900)
        print(f"Saved static PNG")
    except Exception as e:
        print(f"Could not save static version: {e}")


if __name__ == '__main__':
    main()
