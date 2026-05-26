#!/usr/bin/env python3
"""
PaCMAP visualization for embeddings.
Same structure as tsne_embeddings.py but uses PaCMAP instead of t-SNE.
Supports: drug, mutant, combined, both modes.
"""

import os, sys, json, re, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import pacmap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

COMBINED_DRUG_COLOR = '#E41A1C'
COMBINED_MUTANT_COLOR = '#377EB8'
COMBINED_CONTROL_COLOR = '#000000'

CONCENTRATION_ORDER = ['0.25x', '0.5x', '1x', '2x', '4x']
SHADE_FACTORS = {'0.25x': 0.3, '0.5x': 0.5, '1x': 0.7, '2x': 0.9, '4x': 1.0, 'control': 1.0}


def get_gene_from_id(label):
    if not label: return 'unknown'
    if '_' in label: return label.rsplit('_', 1)[0]
    return label


def get_guide_number(label):
    if not label: return 'unknown'
    if '_' in label:
        parts = label.rsplit('_', 1)
        if len(parts) > 1: return parts[1]
    return 'unknown'


def parse_drug_label(label):
    if not label or label == 'unknown': return 'unknown', 'unknown'
    if label.lower() == 'control': return 'control', 'control'
    if '_' in label:
        parts = label.rsplit('_', 1)
        if len(parts) == 2: return parts[0], parts[1]
    return label, 'unknown'


def get_pathway(gene):
    return GENE_PATHWAY.get(gene, 'unknown')


def is_control_or_wt(label):
    if not label: return False
    lower = label.lower()
    return 'control' in lower or 'wt' in lower or 'wild' in lower or 'nc' in lower


def parse_path_info(paths):
    wells, images = [], []
    for p in paths:
        match = re.search(r'Well(\w\d+)', p)
        wells.append(match.group(1) if match else 'unknown')
        images.append(os.path.basename(p))
    return wells, images


def make_hover(label, well, img):
    return f"Well: {well}<br>Image: {img}<br>Label: {label}"


def main():
    parser = argparse.ArgumentParser(description='PaCMAP visualization')
    parser.add_argument('--fold', type=str, default='P6')
    parser.add_argument('--embeddings', type=str, default=None)
    parser.add_argument('--data_mode', type=str, default='drug',
                        choices=['drug', 'mutant', 'combined', 'both'])
    parser.add_argument('--embedding_type', type=str, default='mil')
    parser.add_argument('--neighborhood', type=int, default=3)
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='PaCMAP n_neighbors (default: 15)')
    parser.add_argument('--MN_ratio', type=float, default=0.5,
                        help='PaCMAP mid-near ratio (default: 0.5)')
    parser.add_argument('--FP_ratio', type=float, default=2.0,
                        help='PaCMAP far-pair ratio (default: 2.0)')
    parser.add_argument('--color_by', type=str, default='gene',
                        choices=['gene', 'pathway', 'drug'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--combined_fold', type=str, default='P1')
    args = parser.parse_args()

    test_plate = args.fold if args.fold else args.combined_fold
    fold_key = test_plate if 'Plate_' in test_plate else f'Plate_{test_plate.replace("P", "")}'
    combined_fold_key = args.combined_fold if 'Plate_' in args.combined_fold else f'Plate_{args.combined_fold.replace("P", "")}'

    if args.output_dir:
        output_dir = args.output_dir
    elif args.data_mode == 'both':
        output_dir = os.path.join(SCRIPT_DIR, 'both', f'fold_{fold_key}')
    else:
        output_dir = os.path.join(SCRIPT_DIR, args.data_mode, f'fold_{fold_key}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    if args.data_mode == 'both':
        emb_path = os.path.join(SCRIPT_DIR, 'both', f'fold_{fold_key}',
                                f'embeddings_{fold_key}_{args.embedding_type}_n{args.neighborhood}.npz')
        print(f"Loading: {emb_path}")
        if not os.path.exists(emb_path):
            print(f"ERROR: Not found: {emb_path}")
            return

        data = np.load(emb_path, allow_pickle=True)
        embeddings = data['embeddings']
        paths = data['paths']

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
            if not well: return 'unknown'
            pk = None
            for pn in range(1, 7):
                if f'/p{pn}/' in path_lower:
                    pk = f'P{pn}'
                    break
            if not pk: return 'unknown'
            if src == 'drug':
                if pk in IC50 and well in IC50[pk]:
                    info = IC50[pk][well]
                    ab = info.get('antibiotic', '')
                    ic = info.get('ic50_multiple', '')
                    if ab and ic:
                        if ic == 'control': return 'control'
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

        wells, images = parse_path_info(paths)
        df = pd.DataFrame({'label': correct_labels, 'path': paths, 'well': wells, 'image': images})
        df['source'] = df['label'].apply(_source)
        df['ctrl_type'] = df['label'].apply(_ctrl_type)
        n_drug = (df['source'] == 'drug').sum()
        n_mut = (df['source'] == 'mutant').sum()
        n_ctrl = (df['source'] == 'control').sum()
        print(f"  Drug: {n_drug}, Mutant: {n_mut}, Control: {n_ctrl}")

        print(f"Running PaCMAP (n_neighbors={args.n_neighbors}, MN_ratio={args.MN_ratio}, FP_ratio={args.FP_ratio})...")
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=args.n_neighbors,
                                MN_ratio=args.MN_ratio, FP_ratio=args.FP_ratio,
                                random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        df['x'] = embeddings_2d[:, 0]
        df['y'] = embeddings_2d[:, 1]
        df['source'] = df['label'].apply(_source)
        df['ctrl_type'] = df['label'].apply(_ctrl_type)

        csv_path = os.path.join(output_dir, 'pacmap_combined.csv')
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
                          facecolors='none', edgecolors=COMBINED_CONTROL_COLOR,
                          linewidth=2, label=ctype, alpha=0.8)

        ax.legend(loc='upper left', fontsize=12)
        ax.set_xlabel('PaCMAP 1', fontsize=12)
        ax.set_ylabel('PaCMAP 2', fontsize=12)
        ax.set_title(f'PaCMAP: Drug (red) + Mutant (blue) + Control\nFold {fold_key}', fontsize=14)
        plt.tight_layout()
        png_path = os.path.join(output_dir, 'pacmap_combined_drug_mutant.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved PNG: {png_path}")

        fig_html = go.Figure()
        if len(drug):
            fig_html.add_trace(go.Scatter(
                x=drug['x'], y=drug['y'], mode='markers',
                marker=dict(color=COMBINED_DRUG_COLOR, size=6),
                name='Drug',
                text=[make_hover(l, w, i) for l, w, i in zip(drug['label'], drug['well'], drug['image'])],
                hoverinfo='text'))
        if len(mutant):
            fig_html.add_trace(go.Scatter(
                x=mutant['x'], y=mutant['y'], mode='markers',
                marker=dict(color=COMBINED_MUTANT_COLOR, size=6),
                name='Mutant',
                text=[make_hover(l, w, i) for l, w, i in zip(mutant['label'], mutant['well'], mutant['image'])],
                hoverinfo='text'))
        PLOTLY_SYMBOLS = {'WT': 'star', 'NC': 'triangle-up', 'control': 'square'}
        for ctype, sym in PLOTLY_SYMBOLS.items():
            sub = df[df['ctrl_type'] == ctype]
            if len(sub):
                fig_html.add_trace(go.Scatter(
                    x=sub['x'], y=sub['y'], mode='markers',
                    marker=dict(color=COMBINED_CONTROL_COLOR, size=10, symbol=sym, line=dict(width=2)),
                    name=ctype,
                    text=[make_hover(l, w, i) for l, w, i in zip(sub['label'], sub['well'], sub['image'])],
                    hoverinfo='text'))
        fig_html.update_layout(
            title='PaCMAP: Drug (red) + Mutant (blue) + Control',
            xaxis_title='PaCMAP 1', yaxis_title='PaCMAP 2',
            width=1400, height=1200,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=12)),
            template='plotly_white')
        html_path = os.path.join(output_dir, 'pacmap_combined_drug_mutant.html')
        fig_html.write_html(html_path, include_plotlyjs='cdn')
        print(f"Saved HTML: {html_path}")
        return

    # Combined mode (drug + mutant from separate nps)
    if args.data_mode == 'combined':
        drug_path = os.path.join(SCRIPT_DIR, 'drug', f'fold_{combined_fold_key}',
                                 f'embeddings_{combined_fold_key}_{args.embedding_type}_n{args.neighborhood}.npz')
        mutant_path = os.path.join(SCRIPT_DIR, 'mutant', f'fold_{combined_fold_key}',
                                   f'embeddings_{combined_fold_key}_{args.embedding_type}_n{args.neighborhood}.npz')
        print(f"Loading drug: {drug_path}")
        print(f"Loading mutant: {mutant_path}")
        drug_data = np.load(drug_path, allow_pickle=True)
        mutant_data = np.load(mutant_path, allow_pickle=True)
        drug_emb = drug_data['embeddings']
        drug_labels = drug_data['labels']
        drug_paths = drug_data['paths']
        mutant_emb = mutant_data['embeddings']
        mutant_labels = mutant_data['labels']
        mutant_paths = mutant_data['paths']
        print(f"Drug: {len(drug_emb)}, Mutant: {len(mutant_emb)}")
        all_embeddings = np.vstack([drug_emb, mutant_emb])
        all_labels = np.concatenate([drug_labels, mutant_labels])
        all_paths = np.concatenate([drug_paths, mutant_paths])
        drug_wells, drug_imgs = parse_path_info(drug_paths)
        mut_wells, mut_imgs = parse_path_info(mutant_paths)
        all_wells = drug_wells + mut_wells
        all_images = drug_imgs + mut_imgs
        print(f"Combined: {len(all_embeddings)} embeddings")

        print("Running PaCMAP...")
        reducer = pacmap.PaCMAP(n_components=2, n_neighbors=args.n_neighbors,
                                MN_ratio=args.MN_ratio, FP_ratio=args.FP_ratio,
                                random_state=42)
        embeddings_2d = reducer.fit_transform(all_embeddings)

        n_drug = len(drug_emb)
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0], 'y': embeddings_2d[:, 1],
            'label': all_labels, 'well': all_wells, 'image': all_images,
        })
        df['source'] = ['drug'] * n_drug + ['mutant'] * len(mutant_emb)

        colors = []
        for _, row in df.iterrows():
            if is_control_or_wt(row['label']):
                colors.append(COMBINED_CONTROL_COLOR)
            elif row['source'] == 'drug':
                colors.append(COMBINED_DRUG_COLOR)
            else:
                colors.append(COMBINED_MUTANT_COLOR)
        df['hex'] = colors

        csv_path = os.path.join(output_dir, 'pacmap_combined.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        fig, ax = plt.subplots(figsize=(14, 12))
        drug_mask = df['source'] == 'drug'
        drug_not_ctrl = drug_mask & ~df['label'].apply(is_control_or_wt)
        if drug_not_ctrl.sum() > 0:
            ax.scatter(df.loc[drug_not_ctrl, 'x'], df.loc[drug_not_ctrl, 'y'],
                       c=COMBINED_DRUG_COLOR, label='Drug', s=20, alpha=0.6)
        mutant_mask = df['source'] == 'mutant'
        mutant_not_ctrl = mutant_mask & ~df['label'].apply(is_control_or_wt)
        if mutant_not_ctrl.sum() > 0:
            ax.scatter(df.loc[mutant_not_ctrl, 'x'], df.loc[mutant_not_ctrl, 'y'],
                       c=COMBINED_MUTANT_COLOR, label='Mutant', s=20, alpha=0.6)
        ctrl_mask = df['label'].apply(is_control_or_wt)
        if ctrl_mask.sum() > 0:
            ax.scatter(df.loc[ctrl_mask, 'x'], df.loc[ctrl_mask, 'y'],
                       c=COMBINED_CONTROL_COLOR, marker='s', s=40,
                       facecolors='none', linewidth=2, label='Control/WT/NC')
        ax.legend(loc='upper left', fontsize=12)
        ax.set_xlabel('PaCMAP 1', fontsize=12)
        ax.set_ylabel('PaCMAP 2', fontsize=12)
        ax.set_title(f'PaCMAP: Drug (red) + Mutant (blue) + Control\nFold {combined_fold_key}', fontsize=14)
        plt.tight_layout()
        png_path = os.path.join(output_dir, 'pacmap_combined_drug_mutant.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved PNG: {png_path}")

        fig_html = go.Figure()
        drug_df = df[(df['source'] == 'drug') & ~df['label'].apply(is_control_or_wt)]
        fig_html.add_trace(go.Scatter(
            x=drug_df['x'], y=drug_df['y'], mode='markers',
            marker=dict(color=COMBINED_DRUG_COLOR, size=6), name='Drug',
            text=[make_hover(l, w, i) for l, w, i in zip(drug_df['label'], drug_df['well'], drug_df['image'])],
            hoverinfo='text'))
        mutant_df = df[(df['source'] == 'mutant') & ~df['label'].apply(is_control_or_wt)]
        fig_html.add_trace(go.Scatter(
            x=mutant_df['x'], y=mutant_df['y'], mode='markers',
            marker=dict(color=COMBINED_MUTANT_COLOR, size=6), name='Mutant',
            text=[make_hover(l, w, i) for l, w, i in zip(mutant_df['label'], mutant_df['well'], mutant_df['image'])],
            hoverinfo='text'))
        ctrl_df = df[df['label'].apply(is_control_or_wt)]
        fig_html.add_trace(go.Scatter(
            x=ctrl_df['x'], y=ctrl_df['y'], mode='markers',
            marker=dict(color=COMBINED_CONTROL_COLOR, size=10, symbol='circle-open', line=dict(width=2)),
            name='Control/WT/NC',
            text=[make_hover(l, w, i) for l, w, i in zip(ctrl_df['label'], ctrl_df['well'], ctrl_df['image'])],
            hoverinfo='text'))
        fig_html.update_layout(
            title='PaCMAP: Drug (red) + Mutant (blue) + Control',
            xaxis_title='PaCMAP 1', yaxis_title='PaCMAP 2',
            width=1400, height=1200,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=12)),
            template='plotly_white')
        html_path = os.path.join(output_dir, 'pacmap_combined_drug_mutant.html')
        fig_html.write_html(html_path, include_plotlyjs='cdn')
        print(f"Saved HTML: {html_path}")
        return

    # Single mode (drug or mutant)
    if args.embeddings:
        emb_path = args.embeddings
    else:
        emb_path = os.path.join(SCRIPT_DIR, args.data_mode, f'fold_{fold_key}',
                                f'embeddings_{fold_key}_{args.embedding_type}_n{args.neighborhood}.npz')
    print(f"Loading: {emb_path}")
    if not os.path.exists(emb_path):
        print(f"ERROR: Not found: {emb_path}")
        return
    data = np.load(emb_path, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels']
    paths = data['paths']
    wells, images = parse_path_info(paths)
    print(f"Loaded {len(embeddings)} embeddings")

    print(f"Running PaCMAP (n_neighbors={args.n_neighbors})...")
    reducer = pacmap.PaCMAP(n_components=2, n_neighbors=args.n_neighbors,
                            MN_ratio=args.MN_ratio, FP_ratio=args.FP_ratio,
                            random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    print("PaCMAP complete")

    df = pd.DataFrame({'x': embeddings_2d[:, 0], 'y': embeddings_2d[:, 1],
                       'label': labels, 'well': wells, 'image': images})

    if args.data_mode == 'drug':
        df['antibiotic'], df['concentration'] = zip(*df['label'].apply(parse_drug_label))
        def get_drug_color(row):
            if is_control_or_wt(row['label']): return COMBINED_CONTROL_COLOR
            ab = row['antibiotic']; conc = row['concentration']
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

    csv_path = os.path.join(output_dir, f'pacmap_data_{title_prefix}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    fig, ax = plt.subplots(figsize=(14, 12))
    if args.data_mode == 'mutant':
        nc_df = df[df['gene'] == 'NC']
        if len(nc_df):
            ax.scatter(nc_df['x'], nc_df['y'], c='#888888', label='NC', s=15, alpha=0.7)
        wt_df = df[df['gene'] == 'WT NC']
        if len(wt_df):
            ax.scatter(wt_df['x'], wt_df['y'], c='#000000', label='WT NC', s=15, alpha=0.7)
        for gene, group_df in df[df['hex'] != COMBINED_CONTROL_COLOR].groupby('gene'):
            color = GENE_COLORS.get(gene, '#888888')
            ax.scatter(group_df['x'], group_df['y'], c=color, label=gene, s=15, alpha=0.7)
    else:
        for hex_val, group_df in df.groupby('hex'):
            label_name = 'Control/WT/NC' if hex_val == COMBINED_CONTROL_COLOR else title_prefix
            ax.scatter(group_df['x'], group_df['y'], c=hex_val, label=label_name, s=15, alpha=0.7)
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.set_xlabel('PaCMAP 1', fontsize=12)
    ax.set_ylabel('PaCMAP 2', fontsize=12)
    ax.set_title('PaCMAP of CRISPRi knockdown embeddings', fontsize=14)
    plt.tight_layout()
    png_path = os.path.join(output_dir, f'pacmap_{title_prefix}_mil.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PNG: {png_path}")

    fig_html = go.Figure()
    if args.data_mode == 'mutant':
        nc_df = df[df['gene'] == 'NC']
        if len(nc_df):
            fig_html.add_trace(go.Scatter(
                x=nc_df['x'], y=nc_df['y'], mode='markers',
                marker=dict(color='#888888', size=6), name='NC',
                text=[make_hover(l, w, i) for l, w, i in zip(nc_df['label'], nc_df['well'], nc_df['image'])],
                hoverinfo='text'))
        wt_df = df[df['gene'] == 'WT NC']
        if len(wt_df):
            fig_html.add_trace(go.Scatter(
                x=wt_df['x'], y=wt_df['y'], mode='markers',
                marker=dict(color='#000000', size=6), name='WT NC',
                text=[make_hover(l, w, i) for l, w, i in zip(wt_df['label'], wt_df['well'], wt_df['image'])],
                hoverinfo='text'))
        for gene, subset in df[df['hex'] != COMBINED_CONTROL_COLOR].groupby('gene'):
            color = GENE_COLORS.get(gene, '#888888')
            fig_html.add_trace(go.Scatter(
                x=subset['x'], y=subset['y'], mode='markers',
                marker=dict(color=color, size=6), name=gene,
                text=[make_hover(l, w, i) for l, w, i in zip(subset['label'], subset['well'], subset['image'])],
                hoverinfo='text'))
    else:
        for hex_val in df['hex'].unique():
            subset = df[df['hex'] == hex_val]
            name = 'Control/WT/NC' if hex_val == COMBINED_CONTROL_COLOR else title_prefix
            fig_html.add_trace(go.Scatter(
                x=subset['x'], y=subset['y'], mode='markers',
                marker=dict(color=hex_val, size=6), name=name,
                text=[make_hover(l, w, i) for l, w, i in zip(subset['label'], subset['well'], subset['image'])],
                hoverinfo='text'))
    fig_html.update_layout(
        title='PaCMAP of CRISPRi knockdown embeddings',
        xaxis_title='PaCMAP 1', yaxis_title='PaCMAP 2',
        width=1400, height=1200,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=10)),
        template='plotly_white')
    html_path = os.path.join(output_dir, f'pacmap_{title_prefix}_mil.html')
    fig_html.write_html(html_path, include_plotlyjs='cdn')
    print(f"Saved HTML: {html_path}")


if __name__ == '__main__':
    main()

