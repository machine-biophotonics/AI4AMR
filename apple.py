"""
Interactive UMAP and t-SNE HTML plots using Plotly
FIXED: Loads backbone correctly from multi-head checkpoint
"""

import plotly.express as px
import numpy as np
import torch
from torch import nn
import torchvision.models as models
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import glob
import json
import re
import random
from tqdm import tqdm
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =========================
# YOUR COLOR MAP
# =========================
GENE_COLORS = {
    'mrcA': '#E57373','mrcB': '#EF5350','mrdA': '#F06292','ftsI': '#EC407A',
    'mreB': '#FF8A65','murA': '#FFB74D','murC': '#FFA726',
    'lpxA': '#4DB6AC','lpxC': '#26A69A','lptA': '#4DD0E1','lptC': '#26C6DA','msbA': '#80DEEA',
    'gyrA': '#5C6BC0','gyrB': '#3F51B5','parC': '#7986CB','parE': '#9FA8DA',
    'dnaE': '#9575CD','dnaB': '#B39DDB',
    'rpoA': '#81C784','rpoB': '#66BB6A','rpsA': '#FFF176','rpsL': '#FFEE58',
    'rplA': '#FFD54F','rplC': '#FFCA28',
    'folA': '#AED581','folP': '#9CCC65','secY': '#80CBC4','secA': '#4DB6AC',
    'ftsZ': '#F06292','minC': '#F48FB1',
    'WT': '#424242'
}

# =========================
# HELPERS
# =========================

def parse_gene_subgroup(label):
    if label and '_' in label and label != 'WT':
        parts = label.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ['1', '2', '3']:
            return parts[0], parts[1]
    return label, None

def get_base_name(label):
    if not label:
        return None
    if label.startswith('WT') or label.startswith('NC'):
        return label.split('_')[0]
    parts = label.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return label

# =========================
# DATASET
# =========================

class FeatureDataset(Dataset):
    def __init__(self, image_paths, transform=None, n_patches=10):
        self.image_paths = image_paths
        self.transform = transform
        self.n_patches = n_patches
        self.patch_size = 224
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        w, h = image.size

        patches = []
        for _ in range(self.n_patches):
            left = random.randint(200, w - 200 - self.patch_size)
            top = random.randint(200, h - 200 - self.patch_size)
            patch = image.crop((left, top, left + self.patch_size, top + self.patch_size))
            patches.append(patch)

        images = torch.stack([self.transform(p) for p in patches])
        return images, img_path

# =========================
# UTILITIES
# =========================

def extract_well_from_filename(filename):
    match = re.search(r'Well(\w)(\d+)_', filename)
    if match:
        return f"{match.group(1)}{int(match.group(2)):02d}"
    return None

def load_plate_data():
    with open(os.path.join(BASE_DIR, 'plate_well_id_path.json'), 'r') as f:
        return json.load(f)

def get_label_from_path(img_path, plate_maps):
    dirname = os.path.basename(os.path.dirname(img_path))
    filename = os.path.basename(img_path)
    well = extract_well_from_filename(filename)

    if dirname in plate_maps and well in plate_maps[dirname]:
        return plate_maps[dirname][well]['id']
    return None

# =========================
# PLOTTING
# =========================

def create_plot(df, x, y, title, filename):

    missing = set(df['gene_family']) - set(GENE_COLORS.keys())
    if missing:
        print(f"⚠ Missing colors: {missing}")

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color='gene_family',
        symbol='subgroup',
        color_discrete_map=GENE_COLORS,
        opacity=0.7,
        title=title,
        hover_data={'label': True}
    )

    fig.update_traces(marker=dict(size=8))
    fig.write_html(os.path.join(BASE_DIR, filename))
    print(f"Saved: {filename}")

# =========================
# MAIN
# =========================

def main():

    print("Loading checkpoint...")
    checkpoint = torch.load(os.path.join(BASE_DIR, 'best_model.pth'), map_location=device)

    # 🔥 FIX: load ONLY backbone
    state_dict = checkpoint['model_state_dict']

    backbone_dict = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }

    model = models.efficientnet_b0(weights=None)
    model.load_state_dict(backbone_dict, strict=False)
    model = model.to(device).eval()

    feature_extractor = nn.Sequential(*list(model.children())[:-1])

    # =========================
    # DATA
    # =========================

    plate_maps = load_plate_data()
    paths = glob.glob(os.path.join(BASE_DIR, 'P6', '*.tif'))

    transform = Compose([
        ToTensor(),
        Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = FeatureDataset(paths, transform)
    loader = DataLoader(dataset, batch_size=16)

    features, labels = [], []

    print("Extracting features...")
    with torch.no_grad():
        for images, paths in tqdm(loader):
            b, n, C, H, W = images.shape
            images = images.view(-1, C, H, W).to(device)

            feat = feature_extractor(images)
            feat = feat.view(b, n, -1).mean(1)

            features.append(feat.cpu().numpy())

            for p in paths:
                labels.append(get_label_from_path(p, plate_maps))

    features = np.vstack(features)

    df = pd.DataFrame({
        'label': labels,
        'gene_family': [get_base_name(l) for l in labels],
        'subgroup': [parse_gene_subgroup(l)[1] or '1' for l in labels]
    })

    # =========================
    # UMAP
    # =========================
    if HAS_UMAP:
        print("Running UMAP...")
        emb = umap.UMAP(random_state=42).fit_transform(features)
        df['UMAP1'], df['UMAP2'] = emb[:,0], emb[:,1]
        create_plot(df, 'UMAP1','UMAP2',"UMAP","umap.html")

    # =========================
    # TSNE
    # =========================
    print("Running t-SNE...")
    emb = TSNE(n_components=2, random_state=42).fit_transform(features)
    df['TSNE1'], df['TSNE2'] = emb[:,0], emb[:,1]
    create_plot(df, 'TSNE1','TSNE2',"t-SNE","tsne.html")

    print("\nDONE!")

if __name__ == "__main__":
    main()