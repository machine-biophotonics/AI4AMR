#!/usr/bin/env python3
"""Extract VAE latents from BOTH Drugs_Data AND Mutants_Data with separate labels."""
import os, sys, json, glob, re, random, warnings
warnings.filterwarnings("ignore")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mil_model import MultiCropDataset, extract_well_from_filename
from vae_model import MILVAE

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

IC50_PATH = os.path.join(PROJECT_ROOT, 'final_mutant_model', 'plate_well_ic50_mapping.json')
MUTANT_PATH = os.path.join(PROJECT_ROOT, 'final_mutant_model', 'plate_well_id_path.json')
CHECKPOINT = os.path.join(SCRIPT_DIR, 'mil_vae_both', 'fold_Plate_1', 'best_mil_vae.pth')
OUTPUT = os.path.join(SCRIPT_DIR, 'mil_vae_both', 'fold_Plate_1', 'contour_data.json')

TEST_PLATE = 'Plate_1'
TEST_PLATE_KEY = 'P1'

# Load mappings
with open(IC50_PATH) as f:
    ic50_data = json.load(f)
with open(MUTANT_PATH) as f:
    mutant_data = json.load(f)

# Build drug lookup: plate -> well -> drug_class
drug_map = {}
for plate_key in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    drug_map[plate_key] = {}
    if plate_key in ic50_data:
        for well, info in ic50_data[plate_key].items():
            drug = info.get('antibiotic', '')
            conc = info.get('ic50_multiple', '')
            if drug and conc:
                if conc == 'control':
                    drug_class = 'control'
                else:
                    conc_str = conc if 'x' in conc else f"{conc}x"
                    drug_class = f"{drug.replace(' ', '_')}_{conc_str}"
                drug_map[plate_key][well] = drug_class

# Build mutant lookup: plate -> well -> mutant_id
mutant_map = {}
for plate_key in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']:
    mutant_map[plate_key] = {}
    if plate_key in mutant_data:
        for row, cols in mutant_data[plate_key].items():
            for col, info in cols.items():
                if 'id' in info:
                    well = f"{row}{int(col):02d}"
                    mutant_map[plate_key][well] = info['id']

# ---------------------------------------------------------------------------
# Collect all images from BOTH directories
# ---------------------------------------------------------------------------
paths_drug = []
drug_dir = os.path.join(PROJECT_ROOT, 'Drugs_Data', TEST_PLATE_KEY)
if os.path.exists(drug_dir):
    for root, dirs, files in os.walk(drug_dir):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff', '.png')):
                paths_drug.append(os.path.join(root, f))
print(f"Drugs_Data images: {len(paths_drug)}")

paths_mutant = []
mutant_dir = os.path.join(PROJECT_ROOT, 'Mutants_Data', TEST_PLATE_KEY)
if os.path.exists(mutant_dir):
    for root, dirs, files in os.walk(mutant_dir):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff', '.png')):
                paths_mutant.append(os.path.join(root, f))
print(f"Mutants_Data images: {len(paths_mutant)}")

# ---------------------------------------------------------------------------
# Assign labels: drug images get drug label, mutant images get mutant label
# Also store the OTHER label as metadata for matching
# ---------------------------------------------------------------------------
all_points_meta = []  # (path, primary_label, source_type, well, drug_label, mutant_label)

for p in paths_drug:
    well = extract_well_from_filename(os.path.basename(p))
    if well:
        drug = drug_map.get(TEST_PLATE_KEY, {}).get(well, 'unknown')
        mutant = mutant_map.get(TEST_PLATE_KEY, {}).get(well, 'unknown')
        if drug != 'unknown':
            all_points_meta.append((p, drug, 'drug', well, drug, mutant))

for p in paths_mutant:
    well = extract_well_from_filename(os.path.basename(p))
    if well:
        drug = drug_map.get(TEST_PLATE_KEY, {}).get(well, 'unknown')
        mutant = mutant_map.get(TEST_PLATE_KEY, {}).get(well, 'unknown')
        if mutant != 'unknown':
            all_points_meta.append((p, mutant, 'mutant', well, drug, mutant))

print(f"Total labeled points: {len(all_points_meta)}")

# Collect all unique classes
all_drug_classes = sorted(set(drug_map.get(TEST_PLATE_KEY, {}).values()))
all_mutant_classes = sorted(set(mutant_map.get(TEST_PLATE_KEY, {}).values()))
all_classes = sorted(set(all_drug_classes + all_mutant_classes))
class_to_idx = {c: i for i, c in enumerate(all_classes)}
print(f"Drug classes: {len(all_drug_classes)}, Mutant: {len(all_mutant_classes)}, Total: {len(all_classes)}")

# Build paths/labels for MultiCropDataset
valid_paths = [m[0] for m in all_points_meta]
valid_labels = [class_to_idx[m[1]] for m in all_points_meta]

# ---------------------------------------------------------------------------
# MultiCropDataset + latent extraction
# ---------------------------------------------------------------------------
train_dataset = MultiCropDataset(
    valid_paths, valid_labels, None,
    neighborhood=3, grid_size=12,
    augment=False, seed=SEED, num_channels=1,
    extraction_mode='neighborhood'
)
train_dataset.set_epoch(0)

class LatentDataset(Dataset):
    def __init__(self, base, meta):
        self.base = base
        self.meta = meta
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, _ = self.base[idx]
        _, _, source, well, drug, mutant = self.meta[idx]
        meta_str = f"{source}|{well}|{drug}|{mutant}"
        return img, meta_str

loader = DataLoader(
    LatentDataset(train_dataset, all_points_meta),
    batch_size=32, shuffle=False, num_workers=0
)

# Load model
ckpt = torch.load(CHECKPOINT, map_location=device)
model = MILVAE(
    num_classes=len(all_classes),
    latent_dim=ckpt.get('latent_dim', 32),
    beta=0.1,
    num_heads=4, dropout=0.5, use_contrastive=True,
    num_channels=1, feature_decoder=True, pixel_decoder=True,
).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print("Model loaded.")

# Extract latents
all_z = []
all_meta = []
with torch.no_grad():
    for images, meta_batch in tqdm(loader, desc='Extracting latents'):
        images = images.to(device)
        bag = model.encode_bag(images)
        mu = model.vae_mu(bag)
        all_z.append(mu.cpu().numpy())
        all_meta.extend(meta_batch)

z = np.concatenate(all_z, axis=0)
print(f"Latents: {z.shape}")

# t-SNE to 2D
print("Running t-SNE...")
scaler = StandardScaler()
z_scaled = scaler.fit_transform(z)
perplexity = min(30, len(z) - 1)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED, max_iter=1000)
z_2d = tsne.fit_transform(z_scaled)
print(f"t-SNE done: {z_2d.shape}")

# Build JSON
points = []
for i, meta_str in enumerate(all_meta):
    parts = meta_str.split('|')
    source = parts[0]
    well = parts[1]
    drug = parts[2]
    mutant = parts[3]
    points.append({
        'x': round(float(z_2d[i, 0]), 4),
        'y': round(float(z_2d[i, 1]), 4),
        'z': [round(float(v), 6) for v in z[i]],
        'source': source,
        'well': well,
        'drug': drug,
        'mutant': mutant,
    })

data = {
    'points': points,
    'drug_classes': all_drug_classes,
    'mutant_classes': all_mutant_classes,
}

with open(OUTPUT, 'w') as f:
    json.dump(data, f)
print(f"Saved: {OUTPUT} ({len(points)} points)")
