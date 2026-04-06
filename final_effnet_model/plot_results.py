import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import json
import seaborn as sns
import os
import ast
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import json
import seaborn as sns
import os
import ast

plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/final_effnet_model'
METRICS_FILE = os.path.join(OUTPUT_DIR, 'training_metrics_20260403_171749.csv')
RESULTS_FILE = os.path.join(OUTPUT_DIR, 'training_results_20260404_111438.json')
CLASSES_FILE = os.path.join(OUTPUT_DIR, 'classes.txt')
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, 'crop_predictions_test.csv')
LABEL_JSON = os.path.join(OUTPUT_DIR, 'plate_well_id_path.json')

def load_label_mapping():
    with open(LABEL_JSON, 'r') as f:
        plate_map = json.load(f)
    
    label_map = {}
    for plate, wells in plate_map.items():
        for row, cols in wells.items():
            for col, data in cols.items():
                well = f"Well{row}{col}"
                label_map[well] = data['id']
    return label_map

def load_data():
    label_map = load_label_mapping()
    
    metrics = pd.read_csv(METRICS_FILE)
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    with open(CLASSES_FILE, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    predictions = pd.read_csv(PREDICTIONS_FILE)
    
    def get_true_label(name):
        match = re.match(r'Well([A-H])(\d+)', name)
        if match:
            row, col = match.groups()
            well = f'Well{row}{col.zfill(2)}'
            if well in label_map:
                return class_to_idx.get(label_map[well], -1)
        return -1
    
    predictions['true_label_idx'] = predictions['image_name'].apply(get_true_label)
    
    return metrics, results, classes, predictions

def plot_training_curves(metrics):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = metrics['epoch']
    
    axes[0, 0].plot(epochs, metrics['train_loss'], label='Train Loss', color='#2ecc71')
    axes[0, 0].plot(epochs, metrics['val_loss'], label='Val Loss', color='#e74c3c')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, metrics['train_acc'], label='Train Acc', color='#2ecc71')
    axes[0, 1].plot(epochs, metrics['val_acc'], label='Val Acc', color='#e74c3c')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training & Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(epochs, metrics['val_balanced_acc'], label='Balanced Acc', color='#3498db')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Balanced Accuracy (%)')
    axes[1, 0].set_title('Validation Balanced Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(epochs, metrics['lr'], color='#9b59b6')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: training_curves.png")

def plot_confusion_matrix(predictions, classes):
    valid = predictions[predictions['true_label_idx'] >= 0]
    y_true = valid['true_label_idx'].values
    y_pred = valid['predicted_class_idx'].values
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        if cm[i].sum() > 0:
            cm_normalized[i] = cm[i].astype(float) / cm[i].sum()
    
    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(cm_normalized, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Normalized Confusion Matrix')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: confusion_matrix.png")

def plot_class_performance(predictions, classes, results):
    valid = predictions[predictions['true_label_idx'] >= 0]
    print(f"DEBUG: {len(valid)} valid predictions for class performance")
    if len(valid) == 0:
        print("Skipping class_performance - no valid predictions")
        return
    y_true = valid['true_label_idx'].values
    y_prob = np.array([ast.literal_eval(p) for p in valid['probs_json'].values])
    
    num_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    
    per_class_auc = []
    per_class_ap = []
    
    for i in range(num_classes):
        if y_true_bin[:, i].sum() > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            per_class_auc.append(auc(fpr, tpr))
            prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            per_class_ap.append(average_precision_score(y_true_bin[:, i], y_prob[:, i]))
        else:
            per_class_auc.append(0)
            per_class_ap.append(0)
    
    sorted_indices = np.argsort(per_class_auc)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].barh(range(20), np.array(per_class_auc)[sorted_indices[-20:]], color='#3498db')
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels([classes[i] for i in sorted_indices[-20:]], fontsize=8)
    axes[0].set_xlabel('ROC AUC')
    axes[0].set_title('Top 20 Classes by ROC AUC')
    axes[0].invert_yaxis()
    
    axes[1].barh(range(20), np.array(per_class_ap)[sorted_indices[-20:]], color='#e74c3c')
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels([classes[i] for i in sorted_indices[-20:]], fontsize=8)
    axes[1].set_xlabel('Average Precision')
    axes[1].set_title('Top 20 Classes by Average Precision')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: class_performance.png")
    
    with open(os.path.join(OUTPUT_DIR, 'per_class_metrics.csv'), 'w') as f:
        f.write('class,auc,ap\n')
        for i in range(num_classes):
            f.write(f'{classes[i]},{per_class_auc[i]:.4f},{per_class_ap[i]:.4f}\n')
    print("Saved: per_class_metrics.csv")

def plot_metrics_summary(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ['Test Acc', 'Mean ROC AUC', 'Mean AP']
    values = [results['results']['test_acc'], results['results']['mean_roc_auc'], results['results']['mean_ap']]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = ax.bar(metrics, values, color=colors)
    ax.set_ylabel('Value')
    ax.set_title('Test Metrics Summary')
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: metrics_summary.png")

def plot_per_gene_accuracy(predictions, classes):
    valid = predictions[predictions['true_label_idx'] >= 0]
    y_true = valid['true_label_idx'].values
    y_pred = valid['predicted_class_idx'].values
    
    per_gene_acc = {}
    for gene in set(y_true):
        mask = y_true == gene
        per_gene_acc[classes[gene]] = (y_pred[mask] == gene).mean() * 100
    
    sorted_genes = sorted(per_gene_acc.items(), key=lambda x: x[1])
    
    fig, ax = plt.subplots(figsize=(20, 10))
    genes = [x[0] for x in sorted_genes]
    accs = [x[1] for x in sorted_genes]
    
    colors = ['#e74c3c' if a < 20 else '#f39c12' if a < 50 else '#2ecc71' for a in accs]
    ax.bar(range(len(genes)), accs, color=colors)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=90, fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Gene Test Accuracy')
    ax.axhline(y=float(np.mean(accs)), color='black', linestyle='--', label=f'Mean: {np.mean(accs):.1f}%')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_gene_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: per_gene_accuracy.png")

if __name__ == '__main__':
    print("Loading data...")
    metrics, results, classes, predictions = load_data()
    
    print("Generating plots...")
    plot_training_curves(metrics)
    plot_confusion_matrix(predictions, classes)
    plot_class_performance(predictions, classes, results)
    plot_metrics_summary(results)
    plot_per_gene_accuracy(predictions, classes)
    
    print("Done!")
