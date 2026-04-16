#!/usr/bin/env python3
"""
Generate interactive UMAP plot for the best fold.
Color by gene name using custom palette.
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
import umap
import plotly.express as px
import plotly.io as pio


# =========================
# CUSTOM GENE COLOR MAP
# =========================
GENE_COLORS = {
    # Cell wall synthesis (warm reds → oranges)
    'mrcA': '#E57373',
    'mrcB': '#EF5350',
    'mrdA': '#F06292',
    'ftsI': '#EC407A',
    'mreB': '#FF8A65',
    'murA': '#FFB74D',
    'murC': '#FFA726',

    # LPS synthesis (teal → cyan → turquoise)
    'lpxA': '#4DB6AC',
    'lpxC': '#26A69A',
    'lptA': '#4DD0E1',
    'lptC': '#26C6DA',
    'msbA': '#80DEEA',

    # DNA metabolism (indigo → violet → lavender)
    'gyrA': '#5C6BC0',
    'gyrB': '#3F51B5',
    'parC': '#7986CB',
    'parE': '#9FA8DA',
    'dnaE': '#9575CD',
    'dnaB': '#B39DDB',

    # Transcription & translation (greens → golds)
    'rpoA': '#81C784',
    'rpoB': '#66BB6A',
    'rpsA': '#FFF176',
    'rpsL': '#FFEE58',
    'rplA': '#FFD54F',
    'rplC': '#FFCA28',

    # Metabolism & protein export (lime → mint)
    'folA': '#AED581',
    'folP': '#9CCC65',
    'secY': '#80CBC4',
    'secA': '#4DB6AC',

    # Cell division (magenta → rose)
    'ftsZ': '#F06292',
    'minC': '#F48FB1',

    # Control
    'WT': '#424242'
}


def get_base_gene(label):
    if not label or label == 'nan':
        return 'Unknown'
    if '_' in str(label):
        return str(label).rsplit('_', 1)[0]
    return str(label)


def main():
    parser = argparse.ArgumentParser(description='Generate UMAP plot for fold predictions')
    parser.add_argument('--fold', type=str, default='P3', help='Fold to use')
    parser.add_argument('--csv', type=str, default=None, help='Prediction CSV file')
    parser.add_argument('--output', type=str, default='umap_gene_interactive.html', help='Output HTML file')
    parser.add_argument('--n_neighbors', type=int, default=15, help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1, help='UMAP min_dist parameter')
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # -------------------------
    # Load CSV
    # -------------------------
    if args.csv:
        csv_path = args.csv
    else:
        csv_options = [
            f'fold_{args.fold}/predictions_all_crops_mil_best_model_acc.csv',
            f'fold_{args.fold}/predictions_all_crops.csv',
            f'fold_{args.fold}/image_predictions_all_crops.csv',
        ]
        csv_path = None
        for opt in csv_options:
            full_path = os.path.join(SCRIPT_DIR, opt)
            if os.path.exists(full_path):
                csv_path = full_path
                break

        if csv_path is None:
            csv_path = os.path.join(SCRIPT_DIR, csv_options[0])

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # -------------------------
    # Filter valid predictions
    # -------------------------
    df_valid = df[df['ground_truth_label'].notna()].copy()

    # -------------------------
    # Aggregate to image level
    # -------------------------
    image_results = []

    for img_name, group in df_valid.groupby('image_name'):
        true_label = group['ground_truth_label'].iloc[0]

        # majority vote
        pred_counts = group['predicted_class_name'].value_counts()
        majority_pred = pred_counts.index[0]

        # average probabilities
        probs_list = []
        for p in group['probs']:
            if isinstance(p, str):
                probs_list.append(json.loads(p))
            else:
                probs_list.append(p)

        mean_probs = np.mean(probs_list, axis=0)

        image_results.append({
            'image_name': img_name,
            'true_label': true_label,
            'pred_label': majority_pred,
            'probs': mean_probs
        })

    image_df = pd.DataFrame(image_results)

    # -------------------------
    # Gene extraction
    # -------------------------
    image_df['gene'] = image_df['true_label'].apply(get_base_gene)

    # handle missing / unknown genes
    image_df['gene'] = image_df['gene'].fillna('WT')
    image_df.loc[~image_df['gene'].isin(GENE_COLORS.keys()), 'gene'] = 'WT'

    print(f"Aggregated to {len(image_df)} images")
    print(f"Unique genes: {image_df['gene'].nunique()}")

    # -------------------------
    # UMAP
    # -------------------------
    X = np.array(image_df['probs'].tolist())

    print(f"Running UMAP on {X.shape} features (n_neighbors={args.n_neighbors}, min_dist={args.min_dist})...")

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

    # -------------------------
    # Plot
    # -------------------------
    fig = px.scatter(
        image_df,
        x='umap_x',
        y='umap_y',
        color='gene',
        color_discrete_map=GENE_COLORS,
        hover_data=['image_name', 'true_label', 'pred_label'],
        title=f'UMAP Visualization - Fold {args.fold} ({len(image_df)} images, {image_df["gene"].nunique()} genes)',
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

    # -------------------------
    # Save output
    # -------------------------
    output_path = os.path.join(SCRIPT_DIR, 'train_test_results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pio.write_html(fig, output_path)
    print(f"Saved interactive UMAP to {output_path}")

    # Optional PNG export
    static_output = output_path.replace('.html', '.png')
    try:
        fig.write_image(static_output, width=1200, height=900)
        print(f"Saved static UMAP to {static_output}")
    except Exception as e:
        print(f"Could not save static version: {e}")


if __name__ == '__main__':
    main()