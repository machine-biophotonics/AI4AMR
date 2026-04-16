#!/usr/bin/env python3
"""
Generate interactive t-SNE plot for the best fold.
Color by gene name using fixed biological color map.
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.io as pio


# =========================
# FIXED BIOLOGICAL COLOR MAP
# =========================
GENE_COLORS = {
    'mrcA': '#E57373',
    'mrcB': '#EF5350',
    'mrdA': '#F06292',
    'ftsI': '#EC407A',
    'mreB': '#FF8A65',
    'murA': '#FFB74D',
    'murC': '#FFA726',

    'lpxA': '#4DB6AC',
    'lpxC': '#26A69A',
    'lptA': '#4DD0E1',
    'lptC': '#26C6DA',
    'msbA': '#80DEEA',

    'gyrA': '#5C6BC0',
    'gyrB': '#3F51B5',
    'parC': '#7986CB',
    'parE': '#9FA8DA',
    'dnaE': '#9575CD',
    'dnaB': '#B39DDB',

    'rpoA': '#81C784',
    'rpoB': '#66BB6A',
    'rpsA': '#FFF176',
    'rpsL': '#FFEE58',
    'rplA': '#FFD54F',
    'rplC': '#FFCA28',

    'folA': '#AED581',
    'folP': '#9CCC65',
    'secY': '#80CBC4',
    'secA': '#4DB6AC',

    'ftsZ': '#F06292',
    'minC': '#F48FB1',

    'WT': '#424242'
}


def get_base_gene(label):
    if not label or label == 'nan':
        return 'WT'
    if '_' in str(label):
        return str(label).rsplit('_', 1)[0]
    return str(label)


def main():
    parser = argparse.ArgumentParser(description='Generate t-SNE plot for fold predictions')
    parser.add_argument('--fold', type=str, default='P3')
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--output', type=str, default='tsne_gene_interactive.html')
    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # -------------------------
    # Load CSV
    # -------------------------
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(
            SCRIPT_DIR,
            f'fold_{args.fold}',
            'predictions_all_crops.csv'
        )

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    df_valid = df[df['ground_truth_label'].notna()].copy()

    # -------------------------
    # Aggregate to image level
    # -------------------------
    image_results = []

    for img_name, group in df_valid.groupby('image_name'):
        true_label = group['ground_truth_label'].iloc[0]

        pred_counts = group['predicted_class_name'].value_counts()
        majority_pred = pred_counts.index[0]

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
    # FIX: DO NOT LOWERCASE
    # -------------------------
    image_df['gene'] = image_df['true_label'].apply(get_base_gene)

    # handle missing
    image_df['gene'] = image_df['gene'].fillna('WT')

    # force unknown → WT
    image_df.loc[~image_df['gene'].isin(GENE_COLORS.keys()), 'gene'] = 'WT'

    print(f"Aggregated to {len(image_df)} images")
    print(f"Unique genes: {image_df['gene'].nunique()}")

    # -------------------------
    # t-SNE
    # -------------------------
    X = np.array(image_df['probs'].tolist())

    print(f"Running t-SNE on {X.shape} features...")

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(30, len(X) - 1),
        max_iter=1000
    )

    X_tsne = tsne.fit_transform(X)

    image_df['tsne_x'] = X_tsne[:, 0]
    image_df['tsne_y'] = X_tsne[:, 1]

    # -------------------------
    # COLOR MAP (DIRECT FIX)
    # -------------------------
    color_map = GENE_COLORS  # IMPORTANT FIX

    # -------------------------
    # PLOT
    # -------------------------
    fig = px.scatter(
        image_df,
        x='tsne_x',
        y='tsne_y',
        color='gene',
        color_discrete_map=color_map,
        hover_data=['image_name', 'true_label', 'pred_label'],
        title=f't-SNE Visualization - Fold {args.fold}',
        labels={'tsne_x': 't-SNE 1', 'tsne_y': 't-SNE 2', 'gene': 'Gene'}
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
    # SAVE
    # -------------------------
    output_path = os.path.join(SCRIPT_DIR, 'train_test_results', args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pio.write_html(fig, output_path)
    print(f"Saved interactive t-SNE to {output_path}")

    try:
        fig.write_image(output_path.replace('.html', '.png'), width=1200, height=900)
    except Exception as e:
        print(f"Could not save PNG: {e}")


if __name__ == '__main__':
    main()