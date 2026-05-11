#!/usr/bin/env python3
"""
Generate t-SNE visualization for DINO embeddings (ALL Plates P1-P6).
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


PLATES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']


def load_embeddings():
    """Load all embeddings from all plates and their labels."""
    embeddings = []
    labels = []
    domain = []  # 'mutant' or 'drug'
    plate_info = []  # which plate the embedding came from
    
    # Load gene mapping for mutants (all plates)
    gene_mapping_path = os.path.join(BASE_DIR, 'plate_well_id_path.json')
    with open(gene_mapping_path, 'r') as f:
        all_gene_mapping = json.load(f)
    
    # Load IC50 mapping for drugs (all plates)
    ic50_path = os.path.join(BASE_DIR, 'plate_well_ic50_mapping.json')
    with open(ic50_path, 'r') as f:
        all_ic50_data = json.load(f)
    
    for plate in PLATES:
        gene_mapping = all_gene_mapping.get(plate, {})
        ic50_data = all_ic50_data.get(plate, {})
        
        # Load mutants from this plate
        mutants_dir = os.path.join(EMBEDDINGS_DIR, f'Mutants_{plate}')
        if os.path.exists(mutants_dir):
            for well_folder in sorted(glob.glob(os.path.join(mutants_dir, 'Well*'))):
                well_name = os.path.basename(well_folder)  # e.g., WellA01
                row = well_name[4]  # A from WellA01
                col = str(int(well_name[5:]))  # 1 from WellA01
                
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
                        plate_info.append(plate)
        
        # Load drugs from this plate
        drugs_dir = os.path.join(EMBEDDINGS_DIR, f'Drugs_{plate}')
        if os.path.exists(drugs_dir):
            for well_folder in sorted(glob.glob(os.path.join(drugs_dir, 'Well*'))):
                well_name = os.path.basename(well_folder)  # e.g., WellA01 -> A01
                wellname = well_name[4:]  # A01
                
                for emb_file in sorted(glob.glob(os.path.join(well_folder, '*.npy'))):
                    emb = np.load(emb_file)
                    if len(emb.shape) == 1:
                        embeddings.append(emb)
                        if wellname in ic50_data:
                            antibiotic = ic50_data[wellname].get('antibiotic', 'Unknown')
                            ic50 = ic50_data[wellname].get('ic50_multiple', '1x')
                            drug_name = f"{antibiotic}_{ic50}"
                            labels.append(drug_name)
                        else:
                            labels.append('Unknown')
                        domain.append('drug')
                        plate_info.append(plate)
    
    return np.array(embeddings), labels, domain, plate_info


def create_color_map(domain, labels):
    """Create color map with plate-based intensity shading.
    Within each domain, lighter shade = P1, darker shade = P6.
    """
    colors = {}
    markers = {}
    
    gene_labels = sorted(set([l for l, d in zip(labels, domain) if d == 'mutant']))
    green_shades = plt.cm.Greens(np.linspace(0.3, 0.9, len(gene_labels)))
    for i, gene in enumerate(gene_labels):
        colors[gene] = green_shades[i]
        markers[gene] = 'o'
    
    for gene in gene_labels:
        base = gene.rsplit('_', 1)[0]
        if base == 'WT NC':
            colors[gene] = plt.cm.Reds(np.linspace(0.4, 0.9, len(gene_labels)))[gene_labels.index(gene)]
            markers[gene] = '^'
        elif base == 'NC':
            colors[gene] = plt.cm.Greys(np.linspace(0.4, 0.9, len(gene_labels)))[gene_labels.index(gene)]
            markers[gene] = 's'
    
    drug_labels = sorted(set([l for l, d in zip(labels, domain) if d == 'drug']))
    blue_shades = plt.cm.Blues(np.linspace(0.3, 0.9, len(drug_labels)))
    for i, drug in enumerate(drug_labels):
        colors[drug] = blue_shades[i]
        markers[drug] = 'o'
    
    return colors, markers


def get_plate_color(label, plate, label_base_colors):
    """Get color with plate-based intensity adjustment.
    P1 = lightest (0.5), P6 = darkest (1.0)
    """
    plate_idx = {'P1': 0, 'P2': 1, 'P3': 2, 'P4': 3, 'P5': 4, 'P6': 5}[plate]
    intensity = 0.4 + (plate_idx / 5) * 0.6  # P1=0.4, P6=1.0
    
    base_color = label_base_colors.get(label, np.array([0.5, 0.5, 0.5]))
    if isinstance(base_color, str):
        if base_color == 'red':
            base_color = np.array([1.0, 0.0, 0.0])
        elif base_color == 'black':
            base_color = np.array([0.0, 0.0, 0.0])
        else:
            base_color = np.array([0.0, 0.5, 0.0])
    
    adjusted = np.clip(base_color * intensity, 0, 1)
    return f'rgb({int(adjusted[0]*255)},{int(adjusted[1]*255)},{int(adjusted[2]*255)})'


def run_tsne(embeddings, perplexity=30):
    """Run t-SNE on embeddings."""
    print(f"Running t-SNE with perplexity={perplexity} on {len(embeddings)} embeddings...")
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
    import argparse
    parser = argparse.ArgumentParser(description='Generate t-SNE from DINO embeddings (all plates P1-P6)')
    parser.add_argument('--plates', type=str, default='P1,P2,P3,P4,P5,P6',
                        help='Comma-separated plates to include (default: all 6)')
    parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity')
    parser.add_argument('--max_iter', type=int, default=2000, help='t-SNE max iterations')
    args = parser.parse_args()

    global PLATES
    PLATES = [p.strip() for p in args.plates.split(',')]
    
    print("Loading embeddings from all plates...")
    embeddings, labels, domain, plate_info = load_embeddings()
    
    mutant_count = sum(1 for d in domain if d == 'mutant')
    drug_count = sum(1 for d in domain if d == 'drug')
    print(f"Loaded {len(embeddings)} embeddings total")
    print(f"  Mutants: {mutant_count}")
    print(f"  Drugs: {drug_count}")
    print(f"  Plates: {PLATES}")
    
    # Run t-SNE
    tsne_coords = run_tsne(embeddings, perplexity=args.perplexity)
    
    # Create color map (returns numpy arrays for base colors)
    label_base_colors, marker_map = create_color_map(domain, labels)
    
    # Create DataFrame
    df = pd.DataFrame({
        'tSNE_1': tsne_coords[:, 0],
        'tSNE_2': tsne_coords[:, 1],
        'Label': labels,
        'Domain': domain,
        'Plate': plate_info,
        'Color': [get_plate_color(l, p, label_base_colors) for l, p in zip(labels, plate_info)]
    })
    
    # Save to CSV
    csv_path = os.path.join(BASE_DIR, 'tsne_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved t-SNE results to {csv_path}")
    
    # Create interactive plot with plotly
    print("Creating interactive t-SNE plot...")
    
    fig = go.Figure()
    
    # Define base colors for each domain
    base_colors = {
        'mutant': {'base': 'rgb(0,128,0)', 'wt_nc': 'rgb(255,0,0)', 'nc': 'rgb(0,0,0)'},
        'drug': {'base': 'rgb(0,0,255)'}
    }
    
    # Plot by domain and plate (6 plates)
    # Each plate gets its own trace with intensity based on plate number
    plate_opacity = {f'P{i}': 0.3 + (i-1)/5 * 0.5 for i in range(1, 7)}  # P1=0.3, P6=0.8
    
    for plate in PLATES:
        # Mutants in this plate
        df_plate_mutant = df[(df['Domain'] == 'mutant') & (df['Plate'] == plate)]
        if len(df_plate_mutant) > 0:
            # Separate by label type
            df_wt_nc = df_plate_mutant[df_plate_mutant['Label'].apply(lambda x: x.rsplit('_',1)[0] == 'WT NC')]
            df_nc = df_plate_mutant[df_plate_mutant['Label'].apply(lambda x: x.rsplit('_',1)[0] == 'NC')]
            df_other = df_plate_mutant[~df_plate_mutant['Label'].apply(lambda x: x.rsplit('_',1)[0] in ['WT NC', 'NC'])]
            
            opacity = plate_opacity[plate]
            
            if len(df_wt_nc) > 0:
                fig.add_trace(go.Scatter(
                    x=df_wt_nc['tSNE_1'], y=df_wt_nc['tSNE_2'],
                    mode='markers',
                    marker=dict(size=10, color='red', opacity=opacity, symbol='triangle-up'),
                    text=[f"Label: {l}<br>Plate: {p}<br>Domain: {d}" for l, p, d in zip(df_wt_nc['Label'], df_wt_nc['Plate'], df_wt_nc['Domain'])],
                    name=f'WT NC - {plate}',
                    hoverinfo='text', showlegend=True
                ))
            
            if len(df_nc) > 0:
                fig.add_trace(go.Scatter(
                    x=df_nc['tSNE_1'], y=df_nc['tSNE_2'],
                    mode='markers',
                    marker=dict(size=8, color='black', opacity=opacity, symbol='square'),
                    text=[f"Label: {l}<br>Plate: {p}<br>Domain: {d}" for l, p, d in zip(df_nc['Label'], df_nc['Plate'], df_nc['Domain'])],
                    name=f'NC - {plate}',
                    hoverinfo='text', showlegend=True
                ))
            
            if len(df_other) > 0:
                fig.add_trace(go.Scatter(
                    x=df_other['tSNE_1'], y=df_other['tSNE_2'],
                    mode='markers',
                    marker=dict(size=6, color='green', opacity=opacity),
                    text=[f"Label: {l}<br>Plate: {p}<br>Domain: {d}" for l, p, d in zip(df_other['Label'], df_other['Plate'], df_other['Domain'])],
                    name=f'Mutant - {plate}',
                    hoverinfo='text', showlegend=True
                ))
        
        # Drugs in this plate
        df_plate_drug = df[(df['Domain'] == 'drug') & (df['Plate'] == plate)]
        if len(df_plate_drug) > 0:
            opacity = plate_opacity[plate]
            fig.add_trace(go.Scatter(
                x=df_plate_drug['tSNE_1'], y=df_plate_drug['tSNE_2'],
                mode='markers',
                marker=dict(size=6, color='blue', opacity=opacity),
                text=[f"Label: {l}<br>Plate: {p}<br>Domain: {d}" for l, p, d in zip(df_plate_drug['Label'], df_plate_drug['Plate'], df_plate_drug['Domain'])],
                name=f'Drug - {plate}',
                hoverinfo='text', showlegend=True
            ))
    
    fig.update_layout(
        title='t-SNE of DINOv3 Embeddings (All Plates P1-P6)<br><sub>Lighter shade = P1, Darker shade = P6</sub>',
        xaxis_title='t-SNE 1',
        yaxis_title='t-SNE 2',
        width=1400,
        height=1000,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=8)
        )
    )
    
    html_path = os.path.join(BASE_DIR, 'tsne_dino_embeddings.html')
    fig.write_html(html_path)
    print(f"Saved interactive plot to {html_path}")
    
    # Create static matplotlib version
    print("Creating static plot...")
    fig_static, ax = plt.subplots(figsize=(16, 12))
    
    plate_opacity_static = {f'P{i}': 0.3 + (i-1)/5 * 0.5 for i in range(1, 7)}
    
    for plate in PLATES:
        df_plate = df[df['Plate'] == plate]
        opacity = plate_opacity_static[plate]
        
        df_mutant_plate = df_plate[df_plate['Domain'] == 'mutant']
        df_drug_plate = df_plate[df_plate['Domain'] == 'drug']
        
        df_wt_nc = df_mutant_plate[df_mutant_plate['Label'].apply(lambda x: x.rsplit('_',1)[0] == 'WT NC')]
        df_nc = df_mutant_plate[df_mutant_plate['Label'].apply(lambda x: x.rsplit('_',1)[0] == 'NC')]
        df_other = df_mutant_plate[~df_mutant_plate['Label'].apply(lambda x: x.rsplit('_',1)[0] in ['WT NC', 'NC'])]
        
        if len(df_wt_nc) > 0:
            ax.scatter(df_wt_nc['tSNE_1'], df_wt_nc['tSNE_2'],
                      c='red', marker='^', s=80, alpha=opacity, label=f'WT NC - {plate}' if plate == 'P1' else None)
        
        if len(df_nc) > 0:
            ax.scatter(df_nc['tSNE_1'], df_nc['tSNE_2'],
                      c='black', marker='s', s=40, alpha=opacity, label=f'NC - {plate}' if plate == 'P1' else None)
        
        if len(df_other) > 0:
            ax.scatter(df_other['tSNE_1'], df_other['tSNE_2'],
                      c='green', s=20, alpha=opacity, label=f'Mutant - {plate}' if plate == 'P1' else None)
        
        if len(df_drug_plate) > 0:
            ax.scatter(df_drug_plate['tSNE_1'], df_drug_plate['tSNE_2'],
                      c='blue', s=20, alpha=opacity, label=f'Drug - {plate}' if plate == 'P1' else None)
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('t-SNE of DINOv3 Embeddings (All Plates P1-P6)\nLighter shade = P1, Darker shade = P6', fontsize=14)
    
    # Create custom legend for plate intensities
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Mutant'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=8, label='WT NC'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=8, label='NC'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Drug'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    png_path = os.path.join(BASE_DIR, 'tsne_dino_embeddings.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved static plot to {png_path}")
    
    print("\n=== DONE ===")
    print(f"Mutants: {sum(1 for d in domain if d == 'mutant')} points")
    print(f"Drugs: {sum(1 for d in domain if d == 'drug')} points")


if __name__ == '__main__':
    main()