"""Generate Figure 2-style analysis plots for CRISPRi test results."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
import os

DATA_DIR = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/trial_daniel/data'
RESULTS_DIR = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/trial_daniel/models/train_1-4_val_5_test_6/Plate_6_260501_2136'
OUTPUT_DIR = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/trial_daniel/analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_DICT = {
    0: 'Avibactam_0.125xIC50', 1: 'Avibactam_0.25xIC50', 2: 'Avibactam_0.5xIC50', 3: 'Avibactam_1xIC50',
    4: 'Aztreonam_0.125xIC50', 5: 'Aztreonam_0.25xIC50', 6: 'Aztreonam_0.5xIC50', 7: 'Aztreonam_1xIC50',
    8: 'Cefepime_0.125xIC50', 9: 'Cefepime_0.25xIC50', 10: 'Cefepime_0.5xIC50', 11: 'Cefepime_1xIC50',
    12: 'Cefsulodin_0.125xIC50', 13: 'Cefsulodin_0.25xIC50', 14: 'Cefsulodin_0.5xIC50', 15: 'Cefsulodin_1xIC50',
    16: 'Ceftriaxone_0.125xIC50', 17: 'Ceftriaxone_0.25xIC50', 18: 'Ceftriaxone_0.5xIC50', 19: 'Ceftriaxone_1xIC50',
    20: 'Chloramphenicol_0.125xIC50', 21: 'Chloramphenicol_0.25xIC50', 22: 'Chloramphenicol_0.5xIC50', 23: 'Chloramphenicol_1xIC50',
    24: 'Ciprofloxacin_0.125xIC50', 25: 'Ciprofloxacin_0.25xIC50', 26: 'Ciprofloxacin_0.5xIC50', 27: 'Ciprofloxacin_1xIC50',
    28: 'Clarithromycin_0.125xIC50', 29: 'Clarithromycin_0.25xIC50', 30: 'Clarithromycin_0.5xIC50', 31: 'Clarithromycin_1xIC50',
    32: 'Clavulanate_0.125xIC50', 33: 'Clavulanate_0.25xIC50', 34: 'Clavulanate_0.5xIC50', 35: 'Clavulanate_1xIC50',
    36: 'Colistin_0.125xIC50', 37: 'Colistin_0.25xIC50', 38: 'Colistin_0.5xIC50', 39: 'Colistin_1xIC50',
    40: 'DMSO', 41: 'Doxycycline_0.125xIC50', 42: 'Doxycycline_0.25xIC50', 43: 'Doxycycline_0.5xIC50', 44: 'Doxycycline_1xIC50',
    45: 'Kanamycin_0.125xIC50', 46: 'Kanamycin_0.25xIC50', 47: 'Kanamycin_0.5xIC50', 48: 'Kanamycin_1xIC50',
    49: 'Levofloxacin_0.125xIC50', 50: 'Levofloxacin_0.25xIC50', 51: 'Levofloxacin_0.5xIC50', 52: 'Levofloxacin_1xIC50',
    53: 'Mecillinam_0.125xIC50', 54: 'Mecillinam_0.25xIC50', 55: 'Mecillinam_0.5xIC50', 56: 'Mecillinam_1xIC50',
    57: 'Meropenem_0.125xIC50', 58: 'Meropenem_0.25xIC50', 59: 'Meropenem_0.5xIC50', 60: 'Meropenem_1xIC50',
    61: 'Norfloxacin_0.125xIC50', 62: 'Norfloxacin_0.25xIC50', 63: 'Norfloxacin_0.5xIC50', 64: 'Norfloxacin_1xIC50',
    65: 'PenicillinG_0.125xIC50', 66: 'PenicillinG_0.25xIC50', 67: 'PenicillinG_0.5xIC50', 68: 'PenicillinG_1xIC50',
    69: 'PolymyxinB_0.125xIC50', 70: 'PolymyxinB_0.25xIC50', 71: 'PolymyxinB_0.5xIC50', 72: 'PolymyxinB_1xIC50',
    73: 'Relebactam_0.125xIC50', 74: 'Relebactam_0.25xIC50', 75: 'Relebactam_0.5xIC50', 76: 'Relebactam_1xIC50',
    77: 'Rifampicin_0.125xIC50', 78: 'Rifampicin_0.25xIC50', 79: 'Rifampicin_0.5xIC50', 80: 'Rifampicin_1xIC50',
    81: 'Sulbactam_0.125xIC50', 82: 'Sulbactam_0.25xIC50', 83: 'Sulbactam_0.5xIC50', 84: 'Sulbactam_1xIC50',
    85: 'Trimethoprim_0.125xIC50', 86: 'Trimethoprim_0.25xIC50', 87: 'Trimethoprim_0.5xIC50', 88: 'Trimethoprim_1xIC50'
}

MOA_DICT = {
    'Avibactam': 'Cell wall (PBP 2)', 'Aztreonam': 'Cell wall (PBP 3)', 'Cefepime': 'Cell wall (PBP 3)',
    'Cefsulodin': 'Cell wall (PBP 1)', 'Ceftriaxone': 'Cell wall (PBP 3)', 'Chloramphenicol': 'Ribosome',
    'Ciprofloxacin': 'Gyrase', 'Clarithromycin': 'Ribosome', 'Clavulanate': 'Cell wall (PBP 2)',
    'Colistin': 'Membrane integrity', 'DMSO': 'Control', 'Doxycycline': 'Ribosome', 'Kanamycin': 'Ribosome',
    'Levofloxacin': 'Gyrase', 'Mecillinam': 'Cell wall (PBP 2)', 'Meropenem': 'Cell wall (PBP 2)',
    'Norfloxacin': 'Gyrase', 'PenicillinG': 'Cell wall (PBP 1)', 'PolymyxinB': 'Membrane integrity',
    'Relebactam': 'Cell wall (PBP 2)', 'Rifampicin': 'RNA polymerase', 'Sulbactam': 'Cell wall (PBP 1)',
    'Trimethoprim': 'DNA synthesis'
}

def get_compound_name(class_label):
    return class_label.split('_')[0] if '_' in class_label else class_label

def get_moa(class_label):
    compound = get_compound_name(class_label)
    return MOA_DICT.get(compound, 'Unknown')

def load_data():
    labels = np.loadtxt(f'{RESULTS_DIR}/labels.txt')
    preds = np.loadtxt(f'{RESULTS_DIR}/preds.txt')
    feat_vecs = np.loadtxt(f'{RESULTS_DIR}/feat_vecs.txt')
    return labels, preds, feat_vecs

def plot_confusion_matrix(y_true, y_pred, classes, title, save_name, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
    
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(title=title, ylabel='True label', xlabel='Predicted label')
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{save_name}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_name}')

def plot_moa_confusion_matrix(y_true, y_pred, class_dict, title, save_name, normalize=True):
    unique_moa = sorted(set(get_moa(class_dict[c]) for c in class_dict.keys()))
    moa_to_idx = {m: i for i, m in enumerate(unique_moa)}
    
    n = len(y_true)
    y_true_moa = np.array([moa_to_idx.get(get_moa(class_dict.get(int(l), '')), -1) for l in y_true])
    y_pred_moa = np.array([moa_to_idx.get(get_moa(class_dict.get(int(p), '')), -1) for p in y_pred])
    
    valid_mask = (y_true_moa >= 0) & (y_pred_moa >= 0)
    y_true_moa = y_true_moa[valid_mask]
    y_pred_moa = y_pred_moa[valid_mask]
    
    cm = confusion_matrix(y_true_moa, y_pred_moa, labels=range(len(unique_moa)))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(title=title, ylabel='True MoA', xlabel='Predicted MoA')
    
    ax.set_xticks(np.arange(len(unique_moa)))
    ax.set_yticks(np.arange(len(unique_moa)))
    ax.set_xticklabels(unique_moa, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(unique_moa, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{save_name}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_name}')

def plot_tsne(feat_vecs, labels, class_dict, title, save_name, use_moa=False):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    feat_2d = tsne.fit_transform(feat_vecs)
    
    if use_moa:
        unique_labels = sorted(set(get_moa(class_dict.get(int(l), '')) for l in labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}
        point_colors = [label_to_color.get(get_moa(class_dict.get(int(l), '')), 'gray') for l in labels]
    else:
        unique_labels = sorted(set(int(l) for l in labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}
        point_colors = [label_to_color.get(int(l), 'gray') for l in labels]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    scatter = ax.scatter(feat_2d[:, 0], feat_2d[:, 1], c=point_colors, s=10, alpha=0.6)
    ax.set(title=title, xlabel='t-SNE 1', ylabel='t-SNE 2')
    
    if use_moa:
        legend_elements = [plt.scatter([], [], c=[c], label=l, s=30) for l, c in label_to_color.items()]
    else:
        legend_elements = [plt.scatter([], [], c=[c], label=CLASS_DICT.get(l, str(l)), s=30) for l, c in label_to_color.items()]
    
    ax.legend(handles=legend_elements, loc='best', fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{save_name}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_name}')

def plot_umap(feat_vecs, labels, class_dict, title, save_name, use_moa=False):
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    feat_2d = reducer.fit_transform(feat_vecs)
    
    if use_moa:
        unique_labels = sorted(set(get_moa(class_dict.get(int(l), '')) for l in labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}
        point_colors = [label_to_color.get(get_moa(class_dict.get(int(l), '')), 'gray') for l in labels]
    else:
        unique_labels = sorted(set(int(l) for l in labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}
        point_colors = [label_to_color.get(int(l), 'gray') for l in labels]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(feat_2d[:, 0], feat_2d[:, 1], c=point_colors, s=10, alpha=0.6)
    ax.set(title=title, xlabel='UMAP 1', ylabel='UMAP 2')
    
    if use_moa:
        legend_elements = [plt.scatter([], [], c=[c], label=l, s=30) for l, c in label_to_color.items()]
    else:
        legend_elements = [plt.scatter([], [], c=[c], label=CLASS_DICT.get(l, str(l)), s=30) for l, c in label_to_color.items()]
    
    ax.legend(handles=legend_elements, loc='best', fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{save_name}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_name}')

def plot_cosine_similarity(feat_vecs, labels, class_dict, title, save_name):
    unique_classes = sorted(set(int(l) for l in labels))
    class_indices = {c: np.where(labels == c)[0] for c in unique_classes}
    
    class_means = []
    class_names = []
    for c in unique_classes:
        idx = class_indices[c]
        if len(idx) > 0:
            class_means.append(feat_vecs[idx].mean(axis=0))
            class_names.append(CLASS_DICT.get(c, str(c)))
    
    class_means = np.array(class_means)
    norms = np.linalg.norm(class_means, axis=1, keepdims=True)
    norms[norms == 0] = 1
    class_means_norm = class_means / norms
    cosine_sim = class_means_norm @ class_means_norm.T
    
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cosine_sim, cmap='RdYlBu_r', vmin=-1, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set(title=title)
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{save_name}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_name}')

if __name__ == '__main__':
    print('Loading data...')
    labels, preds, feat_vecs = load_data()
    print(f'Loaded {len(labels)} samples, {feat_vecs.shape[1]} features')
    
    print('\n--- Generating Confusion Matrices ---')
    classes = sorted(set(int(l) for l in labels))
    class_names = [CLASS_DICT.get(c, str(c)) for c in classes]
    
    plot_confusion_matrix(labels, preds, classes, 
                         'Confusion Matrix (Compound Level)', 
                         'confusion_matrix_compound.svg')
    
    plot_moa_confusion_matrix(labels, preds, CLASS_DICT,
                             'Confusion Matrix (MoA Level)',
                             'confusion_matrix_moa.svg')
    
    print('\n--- Generating UMAP/t-SNE Plots ---')
    plot_tsne(feat_vecs, labels, CLASS_DICT,
             't-SNE by Compound', 'tsne_by_compound.svg', use_moa=False)
    
    plot_tsne(feat_vecs, labels, CLASS_DICT,
             't-SNE by MoA', 'tsne_by_moa.svg', use_moa=True)
    
    plot_umap(feat_vecs, labels, CLASS_DICT,
             'UMAP by Compound', 'umap_by_compound.svg', use_moa=False)
    
    plot_umap(feat_vecs, labels, CLASS_DICT,
             'UMAP by MoA', 'umap_by_moa.svg', use_moa=True)
    
    print('\n--- Generating Cosine Similarity Matrix ---')
    plot_cosine_similarity(feat_vecs, labels, CLASS_DICT,
                         'Cosine Similarity of Class Centroids',
                         'cosine_similarity.svg')
    
    print('\n--- All plots saved to:', OUTPUT_DIR)