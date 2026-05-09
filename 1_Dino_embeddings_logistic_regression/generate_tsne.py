#!/usr/bin/env python3
"""
Generate t-SNE visualization for DINO embeddings.
Colors:
- Mutants/Genes: Green (dark shades for different genes)
- Drugs: Blue (different shades for different drugs)
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'embeddings')


def load_embeddings():
    """Load all embeddings and their labels."""
    embeddings = []
    labels = []
    domain = []  # 'mutant' or 'drug'
    
    # Load gene mapping for mutants
    gene_mapping_path = os.path.join(BASE_DIR, 'plate_well_id_path.json')
    with open(gene_mapping_path, 'r') as f:
        gene_mapping = json.load(f)['P1']
    
    # Load mutants
    mutants_dir = os.path.join(EMBEDDINGS_DIR, 'Mutants_P1')
    if os.path.exists(mutants_dir):
        for well_folder in sorted(glob.glob(os.path.join(mutants_dir, 'Well*'))):
            well_name = os.path.basename(well_folder)  # e.g., WellA01
            # Extract row (A) and col (1, not 01)
            row = well_name[4]  # A from WellA01
            col = str(int(well_name[5:]))  # 1 from WellA01 (remove leading zero)
            
            # Get gene ID from mapping
            if row in gene_mapping and col in gene_mapping[row]:
                gene_id = gene_mapping[row][col]['id']
            else:
                gene_id = 'Unknown'
            
            for emb_file in sorted(glob.glob(os.path.join(well_folder, '*.npy'))):
                emb = np.load(emb_file)
                if len(emb.shape) == 1:
                    embeddings.append(emb)
                    labels.append(gene_id)
                    domain.append('mutant')
    
    # Load IC50 mapping for drugs
    ic50_path = os.path.join(BASE_DIR, 'plate_well_ic50_mapping.json')
    with open(ic50_path, 'r') as f:
        ic50_data = json.load(f)
    
    # Load drugs
    drugs_dir = os.path.join(EMBEDDINGS_DIR, 'Drugs_P1')
    if os.path.exists(drugs_dir):
        for well_folder in sorted(glob.glob(os.path.join(drugs_dir, 'Well*'))):
            well_name = os.path.basename(well_folder)  # e.g., WellA01 -> A01
            wellname = well_name[4:]  # A01 (remove 'Well')
            
            for emb_file in sorted(glob.glob(os.path.join(well_folder, '*.npy'))):
                emb = np.load(emb_file)
                if len(emb.shape) == 1:
                    embeddings.append(emb)
                    # Get drug name from IC50 mapping
                    if 'P1' in ic50_data and wellname in ic50_data['P1']:
                        antibiotic = ic50_data['P1'][wellname].get('antibiotic', 'Unknown')
                        ic50 = ic50_data['P1'][wellname].get('ic50_multiple', '1x')
                        drug_name = f"{antibiotic}_{ic50}"
                        labels.append(drug_name)
                    else:
                        labels.append('Unknown')
                    domain.append('drug')
    
    return np.array(embeddings), labels, domain


def create_color_map(domain, labels):
    """Create color map - green for mutants (with special markers for WT NC/NC), blue for drugs."""
    colors = {}
    markers = {}
    
    # Assign colors to mutants (genes) - various green shades
    gene_labels = sorted(set([l for l, d in zip(labels, domain) if d == 'mutant']))
    green_shades = plt.cm.Greens(np.linspace(0.3, 0.9, len(gene_labels)))
    for i, gene in enumerate(gene_labels):
        colors[gene] = f'#{int(green_shades[i][0]*255):02x}{int(green_shades[i][1]*255):02x}{int(green_shades[i][2]*255):02x}'
        markers[gene] = 'o'  # default marker
    
    # Override for WT NC (red triangles) and NC (black squares)
    for gene in gene_labels:
        base = gene.rsplit('_', 1)[0]
        if base == 'WT NC':
            colors[gene] = 'red'
            markers[gene] = '^'  # triangle
        elif base == 'NC':
            colors[gene] = 'black'
            markers[gene] = 's'  # square
    
    # Assign colors to drugs - various blue shades
    drug_labels = sorted(set([l for l, d in zip(labels, domain) if d == 'drug']))
    blue_shades = plt.cm.Blues(np.linspace(0.3, 0.9, len(drug_labels)))
    for i, drug in enumerate(drug_labels):
        colors[drug] = f'#{int(blue_shades[i][0]*255):02x}{int(blue_shades[i][1]*255):02x}{int(blue_shades[i][2]*255):02x}'
        markers[drug] = 'o'
    
    return colors, markers


def run_tsne(embeddings, perplexity=30):
    """Run t-SNE on embeddings."""
    print(f"Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=2000,
        learning_rate='auto',
        init='pca',
        random_state=42,
        method='barnes_hut'
    )
    return tsne.fit_transform(embeddings)


def main():
    print("Loading embeddings...")
    embeddings, labels, domain = load_embeddings()
    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Mutants: {sum(1 for d in domain if d == 'mutant')}")
    print(f"Drugs: {sum(1 for d in domain if d == 'drug')}")
    
    # Run t-SNE
    tsne_coords = run_tsne(embeddings, perplexity=30)
    
    # Create color map
    color_map, marker_map = create_color_map(domain, labels)
    
    # Create DataFrame
    df = pd.DataFrame({
        'tSNE_1': tsne_coords[:, 0],
        'tSNE_2': tsne_coords[:, 1],
        'Label': labels,
        'Domain': domain,
        'Color': [color_map[l] for l in labels]
    })
    
    # Save to CSV
    csv_path = os.path.join(BASE_DIR, 'tsne_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved t-SNE results to {csv_path}")
    
    # Create interactive plot with plotly
    print("Creating interactive t-SNE plot...")
    
    # Separate for different markers
    df_mutant = df[df['Domain'] == 'mutant']
    df_drug = df[df['Domain'] == 'drug']
    
    fig = go.Figure()
    
    # Add mutant points - with special markers for WT NC and NC
    # Separate by marker type
    df_wt_nc = df_mutant[df_mutant['Label'].apply(lambda x: x.rsplit('_',1)[0] == 'WT NC')]
    df_nc = df_mutant[df_mutant['Label'].apply(lambda x: x.rsplit('_',1)[0] == 'NC')]
    df_other_mutant = df_mutant[~df_mutant['Label'].apply(lambda x: x.rsplit('_',1)[0] in ['WT NC', 'NC'])]
    
    # Other mutants (green)
    fig.add_trace(go.Scatter(
        x=df_other_mutant['tSNE_1'],
        y=df_other_mutant['tSNE_2'],
        mode='markers',
        marker=dict(
            size=8,
            color='green',
            opacity=0.7
        ),
        text=df_other_mutant['Label'],
        name='Mutant (Gene)',
        hoverinfo='text'
    ))
    
    # WT NC - red triangles
    fig.add_trace(go.Scatter(
        x=df_wt_nc['tSNE_1'],
        y=df_wt_nc['tSNE_2'],
        mode='markers',
        marker=dict(
            size=12,
            color='red',
            symbol='triangle-up',
            line=dict(width=2, color='darkred')
        ),
        text=df_wt_nc['Label'],
        name='WT NC',
        hoverinfo='text'
    ))
    
    # NC - black squares
    fig.add_trace(go.Scatter(
        x=df_nc['tSNE_1'],
        y=df_nc['tSNE_2'],
        mode='markers',
        marker=dict(
            size=10,
            color='black',
            symbol='square'
        ),
        text=df_nc['Label'],
        name='NC',
        hoverinfo='text'
    ))
    
    # Add drug points (blue)
    fig.add_trace(go.Scatter(
        x=df_drug['tSNE_1'],
        y=df_drug['tSNE_2'],
        mode='markers',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.7
        ),
        text=df_drug['Label'],
        name='Drug',
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title='t-SNE of DINOv3 Embeddings<br><sub>Green circles: Mutants/Genes | Red triangles: WT NC | Black squares: NC | Blue: Drugs</sub>',
        xaxis_title='t-SNE 1',
        yaxis_title='t-SNE 2',
        width=1200,
        height=900,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    html_path = os.path.join(BASE_DIR, 'tsne_dino_embeddings.html')
    fig.write_html(html_path)
    print(f"Saved interactive plot to {html_path}")
    
    # Create static matplotlib version
    print("Creating static plot...")
    fig_static, ax = plt.subplots(figsize=(14, 10))
    
    # Separate mutants by type
    df_wt_nc_static = df_mutant[df_mutant['Label'].apply(lambda x: x.rsplit('_',1)[0] == 'WT NC')]
    df_nc_static = df_mutant[df_mutant['Label'].apply(lambda x: x.rsplit('_',1)[0] == 'NC')]
    df_other_mutant_static = df_mutant[~df_mutant['Label'].apply(lambda x: x.rsplit('_',1)[0] in ['WT NC', 'NC'])]
    
    # Plot other mutants in green
    ax.scatter(
        df_other_mutant_static['tSNE_1'], df_other_mutant_static['tSNE_2'],
        c='green', s=30, alpha=0.6, label=f'Mutant (Gene) n={len(df_other_mutant_static)}'
    )
    
    # Plot WT NC - red triangles
    ax.scatter(
        df_wt_nc_static['tSNE_1'], df_wt_nc_static['tSNE_2'],
        c='red', marker='^', s=150, edgecolors='darkred', linewidths=2,
        label=f'WT NC n={len(df_wt_nc_static)}'
    )
    
    # Plot NC - black squares
    ax.scatter(
        df_nc_static['tSNE_1'], df_nc_static['tSNE_2'],
        c='black', marker='s', s=80,
        label=f'NC n={len(df_nc_static)}'
    )
    
    # Plot drugs in blue
    ax.scatter(
        df_drug['tSNE_1'], df_drug['tSNE_2'],
        c='blue', s=30, alpha=0.6, label=f'Drug n={len(df_drug)}'
    )
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE of DINOv3 Embeddings\nGreen circles: Mutants | Red triangles: WT NC | Black squares: NC | Blue: Drugs', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    png_path = os.path.join(BASE_DIR, 'tsne_dino_embeddings.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved static plot to {png_path}")
    
    print("\n=== DONE ===")
    print(f"Mutants: {len(df_mutant)} points (green)")
    print(f"Drugs: {len(df_drug)} points (blue)")


if __name__ == '__main__':
    main()