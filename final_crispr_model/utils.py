"""Utility functions for CRISPRi reference plate analysis."""

import os
import pandas as pd
import json

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
    'WT': '#424242'
}

GENE_COLORS_LOWER = {k.lower(): v for k, v in GENE_COLORS.items()}
GENE_COLORS_LOWER['nc'] = '#424242'
GENE_COLORS_LOWER['wt nc'] = '#424242'


def get_base_gene(label):
    """Extract base gene name from label."""
    if pd.isna(label):
        return 'wt'
    label = str(label).strip()
    # Remove _1, _2, _3 suffix and handle NC, WT NC
    label = label.replace(' ', '').replace('-', '').replace('_', '')
    # Handle special cases
    if 'wt' in label.lower() or 'nc' in label.lower():
        return 'wt'
    # Remove trailing digits (e.g., gyrB_1 -> gyrB)
    import re
    gene = re.sub(r'(\D+)\d*$', r'\1', label)
    return gene.lower()


def find_prediction_csv(fold_dir, prefer_mil=False):
    """Find the prediction CSV file in fold directory."""
    csv_path = None
    
    csv_files = [
        'predictions_all_crops.csv',
        'predictions.csv',
    ]
    
    if prefer_mil:
        csv_files = [
            'predictions_all_crops_mil.csv',
            'predictions_all_crops_mil_best_model_acc.csv',
            'predictions_all_crops_mil_best_model_auc.csv',
        ] + csv_files
    
    for csv_file in csv_files:
        candidate = os.path.join(fold_dir, csv_file)
        if os.path.exists(candidate):
            csv_path = candidate
            break
    
    return csv_path


def logger(msg):
    """Simple logger."""
    print(msg)