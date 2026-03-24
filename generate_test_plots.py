import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torchvision.models as models
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import os
import json
import re
import glob
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model.pth'), map_location=device)

label_to_idx = checkpoint['label_to_idx']
idx_to_label = checkpoint['idx_to_label']
num_classes = checkpoint['num_classes']
all_labels = checkpoint['all_labels']
best_val_acc = checkpoint.get('best_val_acc', 0)
trained_epoch = checkpoint.get('epoch', 'N/A')

print(f"Loaded model from epoch {trained_epoch}")
print(f"Best val acc: {best_val_acc:.2f}%")
print(f"Number of classes: {num_classes}")

model = models.efficientnet_b0(weights=None)
in_features = 1280
model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, num_classes)
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
    plate_data = json.load(f)

plate_maps = {}
for plate in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    plate_maps[plate] = {}
    for row, wells in plate_data[plate].items():
        for col, info in wells.items():
            well = f"{row}{int(col):02d}"
            plate_maps[plate][well] = info['id']

def extract_well_from_filename(filename):
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        row = match.group(1)
        col = int(match.group(2))
        return f"{row}{col:02d}"
    return None

def get_label_from_path(img_path):
    dirname = os.path.dirname(img_path)
    plate = os.path.basename(dirname)
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)
    if plate in plate_maps and well in plate_maps[plate]:
        return plate_maps[plate][well]
    return None

class CRISPRDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        patch_size = 224
        edge_margin = 200
        n_patches = 5
        
        patches = []
        for _ in range(n_patches):
            center_w = (w - 2 * edge_margin - patch_size) // 2 + edge_margin
            center_h = (h - 2 * edge_margin - patch_size) // 2 + edge_margin
            patch = image.crop((center_w, center_h, center_w + patch_size, center_h + patch_size))
            if self.transform:
                patch = self.transform(patch)
            patches.append(patch)
        
        patches = torch.stack(patches)
        label = self.labels[idx]
        return patches, label, img_path

transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))
print(f"Found {len(test_paths)} test images")

test_labels = []
valid_test_paths = []
for path in test_paths:
    label = get_label_from_path(path)
    if label and label in label_to_idx:
        test_labels.append(label_to_idx[label])
        valid_test_paths.append(path)

test_labels = np.array(test_labels)
print(f"Valid test samples: {len(valid_test_paths)}")

test_dataset = CRISPRDataset(valid_test_paths, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

print("Running inference on test set...")
all_preds = []
all_labels_arr = []
all_probs = []

with torch.no_grad():
    for patches, labels, _ in tqdm(test_loader, desc="Testing"):
        batch_size, n_patches, C, H, W = patches.shape
        patches = patches.view(-1, C, H, W).to(device, non_blocking=True)
        
        outputs = model(patches)
        outputs = outputs.view(batch_size, n_patches, -1).mean(dim=1)
        
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels_arr.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels_arr = np.array(all_labels_arr)
all_probs = np.array(all_probs)

test_acc = (all_preds == all_labels_arr).mean() * 100
print(f"\nTest Accuracy: {test_acc:.2f}%")

test_labels_bin = label_binarize(all_labels_arr, classes=list(range(num_classes)))
classes_with_samples = [i for i in range(test_labels_bin.shape[1]) if test_labels_bin[:, i].sum() > 0]

fpr = {}
tpr = {}
roc_auc = {}
precision_vals = {}
recall_vals = {}
ap = {}

print("\nComputing metrics...")
for i in tqdm(classes_with_samples, desc="Metrics"):
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], all_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    precision_vals[i], recall_vals[i], _ = precision_recall_curve(test_labels_bin[:, i], all_probs[:, i])
    ap[i] = average_precision_score(test_labels_bin[:, i], all_probs[:, i])

mean_roc_auc = np.mean([roc_auc[i] for i in classes_with_samples])
mean_ap = np.mean([ap[i] for i in classes_with_samples])

print(f"\nAverage ROC AUC: {mean_roc_auc:.4f}")
print(f"Average Precision: {mean_ap:.4f}")

unique_labels_in_test = sorted(set(all_labels_arr))
display_labels = [idx_to_label[i] for i in unique_labels_in_test]

cm = confusion_matrix(all_labels_arr, all_preds, labels=unique_labels_in_test)
fig, ax = plt.subplots(figsize=(20, 18))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(ax=ax, include_values=False, cmap='Blues', xticks_rotation=90)
ax.set_title(f'Confusion Matrix (Test Accuracy: {test_acc:.2f}%)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

sorted_by_auc = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)
best_classes = []
seen_names = set()
for i, auc_val in sorted_by_auc:
    name = idx_to_label[i]
    if name not in seen_names:
        best_classes.append(i)
        seen_names.add(name)
        if len(best_classes) >= 8:
            break

fig_roc, axes_roc = plt.subplots(2, 4, figsize=(16, 8))
axes_roc = axes_roc.flatten()

for idx, i in enumerate(best_classes):
    axes_roc[idx].plot(fpr[i], tpr[i], label=f'AUC = {roc_auc[i]:.2f}')
    axes_roc[idx].plot([0, 1], [0, 1], 'k--')
    axes_roc[idx].set_xlabel('False Positive Rate')
    axes_roc[idx].set_ylabel('True Positive Rate')
    axes_roc[idx].set_title(f'{idx_to_label[i]}')
    axes_roc[idx].legend(loc='lower right')
    axes_roc[idx].grid(True, alpha=0.3)

for j in range(len(best_classes), len(axes_roc)):
    axes_roc[j].axis('off')

plt.suptitle(f'ROC Curves (Best 8 Classes by AUC)\nMean AUC: {mean_roc_auc:.4f}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roc_curves.png")

sorted_by_ap = sorted(ap.items(), key=lambda x: x[1], reverse=True)
best_classes_ap = []
seen_names_ap = set()
for i, ap_val in sorted_by_ap:
    name = idx_to_label[i]
    if name not in seen_names_ap:
        best_classes_ap.append(i)
        seen_names_ap.add(name)
        if len(best_classes_ap) >= 8:
            break

fig_pr, axes_pr = plt.subplots(2, 4, figsize=(16, 8))
axes_pr = axes_pr.flatten()

for idx, i in enumerate(best_classes_ap):
    axes_pr[idx].plot(recall_vals[i], precision_vals[i], label=f'AP = {ap[i]:.2f}')
    axes_pr[idx].set_xlabel('Recall')
    axes_pr[idx].set_ylabel('Precision')
    axes_pr[idx].set_title(f'{idx_to_label[i]}')
    axes_pr[idx].legend(loc='lower left')
    axes_pr[idx].grid(True, alpha=0.3)

for j in range(len(best_classes_ap), len(axes_pr)):
    axes_pr[j].axis('off')

plt.suptitle(f'Precision-Recall Curves (Best 8 Classes by AP)\nMean AP: {mean_ap:.4f}', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'precision_recall_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: precision_recall_curves.png")

sorted_by_auc = sorted(roc_auc.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 5 Classes by AUC:")
for i, val in sorted_by_auc[:5]:
    print(f"  {idx_to_label[i]}: {val:.4f}")
print(f"Bottom 5 Classes by AUC:")
for i, val in sorted_by_auc[-5:]:
    print(f"  {idx_to_label[i]}: {val:.4f}")

sorted_by_ap = sorted(ap.items(), key=lambda x: x[1], reverse=True)
print(f"\nTop 5 Classes by AP:")
for i, val in sorted_by_ap[:5]:
    print(f"  {idx_to_label[i]}: {val:.4f}")
print(f"Bottom 5 Classes by AP:")
for i, val in sorted_by_ap[-5:]:
    print(f"  {idx_to_label[i]}: {val:.4f}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_json = {
    "timestamp": timestamp,
    "source": "best_model.pth (epoch {})".format(trained_epoch),
    "results": {
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "mean_roc_auc": float(mean_roc_auc),
        "mean_ap": float(mean_ap),
    },
    "class_metrics": {
        idx_to_label[i]: {
            "idx": int(i),
            "auc": float(roc_auc.get(i, 0)),
            "ap": float(ap.get(i, 0)),
            "sample_count": int(test_labels_bin[:, i].sum())
        } for i in classes_with_samples
    },
    "top_5_auc": [{"class": idx_to_label[i], "auc": float(val)} for i, val in sorted_by_auc[:5]],
    "bottom_5_auc": [{"class": idx_to_label[i], "auc": float(val)} for i, val in sorted_by_auc[-5:]],
    "top_5_ap": [{"class": idx_to_label[i], "ap": float(val)} for i, val in sorted_by_ap[:5]],
    "bottom_5_ap": [{"class": idx_to_label[i], "ap": float(val)} for i, val in sorted_by_ap[-5:]]
}

with open(os.path.join(BASE_DIR, f'test_results_{timestamp}.json'), 'w') as f:
    json.dump(results_json, f, indent=2)
print(f"Saved: test_results_{timestamp}.json")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Best Val Accuracy: {best_val_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Mean ROC AUC: {mean_roc_auc:.4f}")
print(f"Mean Average Precision: {mean_ap:.4f}")
print("="*50)
print("\nAll plots generated successfully!")
