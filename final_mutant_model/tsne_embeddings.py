#!/usr/bin/env python3
"""
t-SNE visualization for embeddings.
Supports:
- drug mode: antibiotic colored, concentration shaded
- mutant mode: gene/pathway colored, guide RNA shaped
- combined mode: drug (red) + mutant (blue) together, control (green)
"""

import os
import sys
import json
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


SCRIPT_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Gene to Pathway mapping
GENE_PATHWAY = {
    'folP': 'folic acid biosynthetic process', 'folA': 'folic acid biosynthetic process',
    'secY': 'intracellular protein transport', 'secA': 'intracellular protein transport',
    'rpoB': 'regulation of DNA-templated transcription elongation', 'rpoA': 'regulation of DNA-templated transcription elongation',
    'lptC': 'cell envelope organization', 'lptA': 'cell envelope organization', 'msbA': 'cell envelope organization',
    'ftsZ': 'division septum assembly',
    'rplC': 'regulation of translational initiation', 'rplA': 'regulation of translational initiation',
    'rpsA': 'regulation of translational initiation', 'rpsL': 'regulation of translational initiation',
    'murC': 'aminoglycan biosynthetic process', 'murA': 'aminoglycan biosynthetic process',
    'mrdA': 'regulation of cell shape', 'mrcA': 'regulation of cell shape', 'ftsI': 'regulation of cell shape',
    'lpxC': 'lipid A biosynthetic process', 'lpxA': 'lipid A biosynthetic process',
    'gyrB': 'chromosome organization', 'gyrA': 'chromosome organization', 'dnaB': 'chromosome organization',
    'parE': 'chromosome organization', 'parC': 'chromosome organization', 'dnaE': 'chromosome organization',
}

GENE_COLORS = {
    'folP': '#D95F02', 'folA': '#F28E2B', 'secY': '#E377C2', 'secA': '#F2A2D6',
    'rpoB': '#9467BD', 'rpoA': '#B79BD9', 'lptC': '#00A65A', 'lptA': '#2FBF71', 'msbA': '#66D18E',
    'ftsZ': '#7CAE00', 'rplC': '#4C78A8', 'rplA': '#6A91C7', 'rpsA': '#8AAAE5', 'rpsL': '#AEC7FF',
    'murC': '#17A2B8', 'murA': '#45B8C6', 'mrdA': '#B58900', 'mrcA': '#D4A017', 'ftsI': '#E3BF3A',
    'lpxC': '#20B2AA', 'lpxA': '#4FC3BD', 'gyrB': '#F17CB0', 'gyrA': '#F29CB3', 'dnaB': '#F4B6C2',
    'parE': '#F7C7CE', 'parC': '#FAD9DF', 'dnaE': '#F9A3A3',
}

ANTIBIOTIC_COLORS = {
    'Chloramphenicol': '#1F77B4', 'Ciprofloxacin': '#FF7F0E', 'Meropenem': '#2CA02C', 'Penicillin': '#D62728',
    'Cefepim': '#9467BD', 'Ceftriaxone': '#8C564B', 'Aztreonam': '#E377C2', 'Kanamycin': '#7F7F7F',
    'Doxicyclin': '#BCBD22', 'Rifampicin': '#17BECF', 'Colistin': '#FF9896', 'Mecillinam': '#AEC7E8',
    'Avibactam': '#FFBB78', 'Sulbactam': '#98DF8A', 'Cefsulodin': '#FF9896', 'DMSO': '#8C8C8C',
}

COMBINED_DRUG_COLOR = '#E41A1C'  # Red
COMBINED_MUTANT_COLOR = '#377EB8'  # Blue
COMBINED_CONTROL_COLOR = '#4DAF4A'  # Green

CONCENTRATION_ORDER = ['0.25x', '0.5x', '1x', '2x', '4x']
SHADE_FACTORS = {'0.25x': 0.3, '0.5x': 0.5, '1x': 0.7, '2x': 0.9, '4x': 1.0, 'control': 1.0}

GUIDE_SHAPES = {'1': 'circle', '2': 'square', '3': 'diamond', '4': 'cross', '5': 'x', '6': 'triangle-up', '7': 'triangle-down', '8': 'hexagon', '9': 'star'}


def get_gene_from_id(label: str) -> str:
    if not label: return 'unknown'
    if '_' in label: return label.rsplit('_', 1)[0]
    return label


def get_guide_number(label: str) -> str:
    if not label: return 'unknown'
    if '_' in label:
        parts = label.rsplit('_', 1)
        if len(parts) > 1: return parts[1]
    return 'unknown'


def parse_drug_label(label: str):
    if not label or label == 'unknown': return 'unknown', 'unknown'
    if label.lower() == 'control': return 'control', 'control'
    if '_' in label:
        parts = label.rsplit('_', 1)
        if len(parts) == 2: return parts[0], parts[1]
    return label, 'unknown'


def get_pathway(gene: str) -> str:
    return GENE_PATHWAY.get(gene, 'unknown')


def is_control_or_wt(label: str) -> bool:
    if not label: return False
    lower = label.lower()
    return 'control' in lower or 'wt' in lower or 'wild' in lower or 'nc' in lower


def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization')
    parser.add_argument('--fold', type=str, default='P6')
    parser.add_argument('--embeddings', type=str, default=None)
    parser.add_argument('--data_mode', type=str, default='drug',
                        choices=['drug', 'mutant', 'mutant_on_drug', 'drug_on_mutant', 'combined', 'both'])
    parser.add_argument('--embedding_type', type=str, default='mil')
    parser.add_argument('--neighborhood', type=int, default=3,
                        help='Neighborhood size in filename (default: 3)')
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--n_iter', type=int, default=5000)
    parser.add_argument('--color_by', type=str, default='gene',
                        choices=['gene', 'pathway', 'drug'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--combined_fold', type=str, default='P1',
                        help='Fold for combined mode (default: P1)')
    
    args = parser.parse_args()
    
    test_plate = args.fold if args.fold else args.combined_fold
    fold_key = test_plate if 'Plate_' in test_plate else f'Plate_{test_plate.replace("P", "")}'
    
    combined_fold_key = args.combined_fold if 'Plate_' in args.combined_fold else f'Plate_{args.combined_fold.replace("P", "")}'
    
    if args.output_dir:
        output_dir = args.output_dir
    elif args.data_mode == 'both':
        output_dir = os.path.join(SCRIPT_DIR, 'both', f'fold_{fold_key}')
    else:
        output_dir = os.path.join(SCRIPT_DIR, 'combined', f'fold_{combined_fold_key}')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output dir: {output_dir}")
    
    def is_drug_label(label) -> bool:
        if not label: return False
        if label.lower() == 'control': return True
        if '_' in label:
            suffix = label.rsplit('_', 1)[1]
            if suffix.endswith('x') or suffix.endswith('X'):
                return True
        return False
    
    if args.data_mode == 'both':
        emb_path = os.path.join(SCRIPT_DIR, 'both', f'fold_{fold_key}', f'embeddings_{fold_key}_{args.embedding_type}_n{args.neighborhood}.npz')
        
        print(f"Loading: {emb_path}")
        if not os.path.exists(emb_path):
            print(f"ERROR: Not found: {emb_path}")
            return
        
        data = np.load(emb_path)
        embeddings = data['embeddings']
        paths = data['paths']
        
        # Load label mappings to fix labels (stored labels are wrong — all mutant due to well overlap)
        ic50_path = os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')
        mut_path = os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')
        IC50 = json.load(open(ic50_path)) if os.path.exists(ic50_path) else {}
        MUT = json.load(open(mut_path)) if os.path.exists(mut_path) else {}
        
        def _fix_label(img_path):
            path_lower = img_path.lower()
            if '/drugs_data/' in path_lower:
                src = 'drug'
            elif '/mutants_data/' in path_lower:
                src = 'mutant'
            else:
                return 'unknown'
            match = re.search(r'Well(\w\d+)_', os.path.basename(img_path))
            well = match.group(1) if match else None
            if not well:
                return 'unknown'
            pk = None
            for pn in range(1, 7):
                if f'/p{pn}/' in path_lower:
                    pk = f'P{pn}'
                    break
            if not pk:
                return 'unknown'
            if src == 'drug':
                if pk in IC50 and well in IC50[pk]:
                    info = IC50[pk][well]
                    ab = info.get('antibiotic', '')
                    ic = info.get('ic50_multiple', '')
                    if ab and ic:
                        if ic == 'control':
                            return 'control'
                        return f"{ab.replace(' ', '_')}_{ic if 'x' in str(ic) else f'{ic}x'}"
            else:
                row, col_raw = well[0], well[1:].lstrip('0') or '0'
                try:
                    if pk in MUT and row in MUT[pk] and col_raw in MUT[pk][row]:
                        return MUT[pk][row][col_raw].get('id', None)
                except:
                    pass
            return 'unknown'
        
        correct_labels = [_fix_label(p) for p in paths]
        
        print(f"Loaded {len(embeddings)} embeddings from both model")
        
        def _ctrl_type(label):
            if not label: return ''
            l = label.lower()
            if 'wt' in l or 'wild' in l: return 'WT'
            if l == 'nc' or l.startswith('nc_'): return 'NC'
            if 'control' in l: return 'control'
            return ''
        
        def _source(label):
            if not label: return 'mutant'
            if label == 'unknown': return 'unknown'
            if _ctrl_type(label): return 'control'
            if '_' in label and label.rsplit('_', 1)[1].endswith('x'):
                return 'drug'
            return 'mutant'
        
        df = pd.DataFrame({'label': correct_labels, 'path': paths})
        df['source'] = df['label'].apply(_source)
        df['ctrl_type'] = df['label'].apply(_ctrl_type)
        
        n_drug = (df['source'] == 'drug').sum()
        n_mut = (df['source'] == 'mutant').sum()
        n_ctrl = (df['source'] == 'control').sum()
        print(f"  Drug: {n_drug}, Mutant: {n_mut}, Control: {n_ctrl}")
        
        print(f"Running t-SNE (perplexity={args.perplexity}, n_iter={args.n_iter})...")
        tsne = TSNE(n_components=2, perplexity=args.perplexity, max_iter=args.n_iter, random_state=42, init='pca')
        embeddings_2d = tsne.fit_transform(embeddings)
        df['x'] = embeddings_2d[:, 0]
        df['y'] = embeddings_2d[:, 1]
        
        csv_path = os.path.join(output_dir, 'tsne_combined.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        CTRL_MARKERS = {'WT': '*', 'NC': '^', 'control': 's'}
        
        drug = df[df['source'] == 'drug']
        if len(drug):
            ax.scatter(drug['x'], drug['y'], c=COMBINED_DRUG_COLOR, label='Drug', s=20, alpha=0.6)
        
        mutant = df[df['source'] == 'mutant']
        if len(mutant):
            ax.scatter(mutant['x'], mutant['y'], c=COMBINED_MUTANT_COLOR, label='Mutant', s=20, alpha=0.6)
        
        for ctype, marker in CTRL_MARKERS.items():
            sub = df[df['ctrl_type'] == ctype]
            if len(sub):
                ax.scatter(sub['x'], sub['y'], marker=marker, s=60,
                          facecolors='none', edgecolors=COMBINED_CONTROL_COLOR, linewidth=2, label=ctype, alpha=0.8)
        
        ax.legend(loc='upper left', fontsize=12)
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title(f't-SNE: Drug (red) + Mutant (blue) + Control\nFold {fold_key}', fontsize=14)
        plt.tight_layout()
        png_path = os.path.join(output_dir, 'tsne_combined_drug_mutant.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved PNG: {png_path}")
        
        fig_html = go.Figure()
        if len(drug):
            fig_html.add_trace(go.Scatter(
                x=drug['x'], y=drug['y'], mode='markers',
                marker=dict(color=COMBINED_DRUG_COLOR, size=6),
                name='Drug', text=[f"Label: {l}" for l in drug['label']], hoverinfo='text'))
        if len(mutant):
            fig_html.add_trace(go.Scatter(
                x=mutant['x'], y=mutant['y'], mode='markers',
                marker=dict(color=COMBINED_MUTANT_COLOR, size=6),
                name='Mutant', text=[f"Label: {l}" for l in mutant['label']], hoverinfo='text'))
        PLOTLY_SYMBOLS = {'WT': 'star', 'NC': 'triangle-up', 'control': 'square'}
        for ctype, sym in PLOTLY_SYMBOLS.items():
            sub = df[df['ctrl_type'] == ctype]
            if len(sub):
                fig_html.add_trace(go.Scatter(
                    x=sub['x'], y=sub['y'], mode='markers',
                    marker=dict(color=COMBINED_CONTROL_COLOR, size=10, symbol=sym, line=dict(width=2)),
                    name=ctype, text=[f"Label: {l}" for l in sub['label']], hoverinfo='text'))
        fig_html.update_layout(
            title=f't-SNE: Drug (red) + Mutant (blue) + Control',
            xaxis_title='t-SNE 1', yaxis_title='t-SNE 2',
            width=1400, height=1200,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=12)),
            template='plotly_white')
        html_path = os.path.join(output_dir, 'tsne_combined_drug_mutant.html')
        fig_html.write_html(html_path, include_plotlyjs='cdn')
        print(f"Saved HTML: {html_path}")
        return
    
    if args.data_mode == 'combined':
        # Load both drug and mutant embeddings
        drug_path = os.path.join(SCRIPT_DIR, 'drug', f'fold_{combined_fold_key}', f'embeddings_{combined_fold_key}_{args.embedding_type}_n3.npz')
        mutant_path = os.path.join(SCRIPT_DIR, 'mutant', f'fold_{combined_fold_key}', f'embeddings_{combined_fold_key}_{args.embedding_type}_n3.npz')
        
        print(f"Loading drug embeddings: {drug_path}")
        print(f"Loading mutant embeddings: {mutant_path}")
        
        drug_data = np.load(drug_path)
        mutant_data = np.load(mutant_path)
        
        drug_emb = drug_data['embeddings']
        drug_labels = drug_data['labels']
        mutant_emb = mutant_data['embeddings']
        mutant_labels = mutant_data['labels']
        
        print(f"Drug: {len(drug_emb)}, Mutant: {len(mutant_emb)}")
        
        # Combine embeddings
        all_embeddings = np.vstack([drug_emb, mutant_emb])
        all_labels = np.concatenate([drug_labels, mutant_labels])
        
        print(f"Combined: {len(all_embeddings)} embeddings")
        
        # Run t-SNE
        print(f"Running t-SNE (perplexity={args.perplexity}, n_iter={args.n_iter})...")
        tsne = TSNE(n_components=2, perplexity=args.perplexity, max_iter=args.n_iter, random_state=42, init='pca')
        embeddings_2d = tsne.fit_transform(all_embeddings)
        print("t-SNE complete")
        
        # Build dataframe
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'label': all_labels,
        })
        
        # Add source column and colors
        n_drug = len(drug_emb)
        df['source'] = ['drug'] * n_drug + ['mutant'] * len(mutant_emb)
        
        # Determine color based on source and control
        colors = []
        for i, row in df.iterrows():
            label = row['label']
            source = row['source']
            if is_control_or_wt(label):
                colors.append(COMBINED_CONTROL_COLOR)
            elif source == 'drug':
                colors.append(COMBINED_DRUG_COLOR)
            else:
                colors.append(COMBINED_MUTANT_COLOR)
        
        df['hex'] = colors
        
        # Save CSV
        csv_path = os.path.join(output_dir, 'tsne_combined.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Plot drug (red)
        drug_mask = df['source'] == 'drug'
        drug_not_ctrl = drug_mask & ~df['label'].apply(is_control_or_wt)
        if drug_not_ctrl.sum() > 0:
            ax.scatter(df.loc[drug_not_ctrl, 'x'], df.loc[drug_not_ctrl, 'y'], 
                    c=COMBINED_DRUG_COLOR, label='Drug', s=20, alpha=0.6)
        
        # Plot mutant (blue)
        mutant_mask = df['source'] == 'mutant'
        mutant_not_ctrl = mutant_mask & ~df['label'].apply(is_control_or_wt)
        if mutant_not_ctrl.sum() > 0:
            ax.scatter(df.loc[mutant_not_ctrl, 'x'], df.loc[mutant_not_ctrl, 'y'], 
                    c=COMBINED_MUTANT_COLOR, label='Mutant', s=20, alpha=0.6)
        
        # Plot control/WT/NC (green)
        ctrl_mask = df['label'].apply(is_control_or_wt)
        if ctrl_mask.sum() > 0:
            ax.scatter(df.loc[ctrl_mask, 'x'], df.loc[ctrl_mask, 'y'], 
                    c=COMBINED_CONTROL_COLOR, marker='s', s=40, facecolors='none', 
                    linewidth=2, label='Control/WT/NC')
        
        ax.legend(loc='upper left', fontsize=12)
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title(f't-SNE: Drug (red) + Mutant (blue) + Control (green)\nFold {combined_fold_key}', fontsize=14)
        
        plt.tight_layout()
        
        # Save PNG
        png_path = os.path.join(output_dir, 'tsne_combined_drug_mutant.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved PNG: {png_path}")
        
        # Create interactive HTML with plotly
        fig_html = go.Figure()
        
        # Drug traces
        drug_df = df[(df['source'] == 'drug') & ~df['label'].apply(is_control_or_wt)]
        fig_html.add_trace(go.Scatter(
            x=drug_df['x'], y=drug_df['y'],
            mode='markers',
            marker=dict(color=COMBINED_DRUG_COLOR, size=6),
            name='Drug',
            text=[f"Label: {l}" for l in drug_df['label']],
            hoverinfo='text'
        ))
        
        # Mutant traces
        mutant_df = df[(df['source'] == 'mutant') & ~df['label'].apply(is_control_or_wt)]
        fig_html.add_trace(go.Scatter(
            x=mutant_df['x'], y=mutant_df['y'],
            mode='markers',
            marker=dict(color=COMBINED_MUTANT_COLOR, size=6),
            name='Mutant',
            text=[f"Label: {l}" for l in mutant_df['label']],
            hoverinfo='text'
        ))
        
        # Control
        ctrl_df = df[df['label'].apply(is_control_or_wt)]
        fig_html.add_trace(go.Scatter(
            x=ctrl_df['x'], y=ctrl_df['y'],
            mode='markers',
            marker=dict(color=COMBINED_CONTROL_COLOR, size=10, symbol='circle-open', line=dict(width=2)),
            name='Control/WT/NC',
            text=[f"Label: {l}" for l in ctrl_df['label']],
            hoverinfo='text'
        ))
        
        fig_html.update_layout(
            title=f't-SNE: Drug (red) + Mutant (blue) + Control (green)',
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            width=1400,
            height=1200,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=12)),
            template='plotly_white'
        )
        
        html_path = os.path.join(output_dir, 'tsne_combined_drug_mutant.html')
        fig_html.write_html(html_path, include_plotlyjs='cdn')
        print(f"Saved HTML: {html_path}")
        
        return
    
    # Original single-mode code continues below for drug/mutant modes
    if args.embeddings:
        emb_path = args.embeddings
    else:
        emb_path = os.path.join(SCRIPT_DIR, args.data_mode, f'fold_{fold_key}', f'embeddings_{fold_key}_{args.embedding_type}_n3.npz')
    
    print(f"Loading: {emb_path}")
    
    if not os.path.exists(emb_path):
        print(f"ERROR: Not found: {emb_path}")
        return
    
    data = np.load(emb_path)
    embeddings = data['embeddings']
    labels = data['labels']
    
    print(f"Loaded {len(embeddings)} embeddings")
    
    print(f"Running t-SNE (perplexity={args.perplexity}, n_iter={args.n_iter})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, max_iter=args.n_iter, random_state=42, init='pca')
    embeddings_2d = tsne.fit_transform(embeddings)
    print("t-SNE complete")
    
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels,
    })
    
    if args.data_mode == 'drug':
        df['antibiotic'], df['concentration'] = zip(*df['label'].apply(parse_drug_label))
        
        def get_drug_color(row):
            if is_control_or_wt(row['label']): return COMBINED_CONTROL_COLOR
            ab = row['antibiotic']
            conc = row['concentration']
            if ab == 'unknown': return '#888888'
            base = ANTIBIOTIC_COLORS.get(ab, '#888888')
            shade = SHADE_FACTORS.get(conc, 0.7)
            hc = base.lstrip('#')
            r = int(int(hc[0:2], 16) * shade)
            g = int(int(hc[2:4], 16) * shade)
            b = int(int(hc[4:6], 16) * shade)
            return f'#{r:02x}{g:02x}{b:02x}'
        
        df['hex'] = df.apply(get_drug_color, axis=1)
        title_prefix = 'drug'
        
    else:
        df['gene'] = df['label'].apply(get_gene_from_id)
        df['guide'] = df['label'].apply(get_guide_number)
        df['pathway'] = df['gene'].apply(get_pathway)
        
        def get_mutant_color(row):
            if is_control_or_wt(row['label']): return COMBINED_CONTROL_COLOR
            if args.color_by == 'gene': return GENE_COLORS.get(row['gene'], '#888888')
            return '#888888'
        
        df['hex'] = df.apply(get_mutant_color, axis=1)
        title_prefix = args.data_mode
    
    csv_path = os.path.join(output_dir, f'tsne_data_{args.color_by}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")
    
    # Create matplotlib figure
    fig_mpl, ax = plt.subplots(figsize=(14, 12))
    
    for hex_val, group_df in df.groupby('hex'):
        label_name = 'Control/WT/NC' if hex_val == COMBINED_CONTROL_COLOR else title_prefix
        ax.scatter(group_df['x'], group_df['y'], c=hex_val, label=label_name, s=15, alpha=0.7)
    
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(f't-SNE of {title_prefix} embeddings', fontsize=14)
    
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, f'tsne_{args.color_by}_{args.embedding_type}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PNG: {png_path}")
    
    # Interactive HTML
    fig_html = go.Figure()
    
    for hex_val in df['hex'].unique():
        subset = df[df['hex'] == hex_val]
        name = 'Control/WT/NC' if hex_val == COMBINED_CONTROL_COLOR else title_prefix
        fig_html.add_trace(go.Scatter(
            x=subset['x'], y=subset['y'],
            mode='markers',
            marker=dict(color=hex_val, size=6),
            name=name,
            text=[f"Label: {l}" for l in subset['label']],
            hoverinfo='text'
        ))
    
    fig_html.update_layout(
        title=f't-SNE of {title_prefix} embeddings',
        xaxis_title='t-SNE 1',
        yaxis_title='t-SNE 2',
        width=1400,
        height=1200,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=8)),
        template='plotly_white'
    )
    
    html_path = os.path.join(output_dir, f'tsne_{args.color_by}_{args.embedding_type}.html')
    fig_html.write_html(html_path, include_plotlyjs='cdn')
    print(f"Saved HTML: {html_path}")


if __name__ == '__main__':
    main()