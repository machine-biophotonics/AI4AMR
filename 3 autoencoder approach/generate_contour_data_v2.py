#!/usr/bin/env python3
"""Generate contour data with CORRECT label separation (no cross-labeling).
Drug points → only drug label
Mutant points → only mutant label

Outputs: contour_data_v2.json + standalone latent_contour_v2.html
"""
import os, sys, json, warnings, math
warnings.filterwarnings("ignore")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import umap
from tqdm import tqdm

from mil_model import MultiCropDataset, extract_well_from_filename
from vae_model import MILVAE

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IC50_PATH = os.path.join(SCRIPT_DIR, 'plate_well_ic50_mapping.json')
if not os.path.exists(IC50_PATH):
    IC50_PATH = os.path.join(PROJECT_ROOT, 'final_mutant_model', 'plate_well_ic50_mapping.json')

MUTANT_PATH = os.path.join(SCRIPT_DIR, 'plate_well_id_path.json')
if not os.path.exists(MUTANT_PATH):
    MUTANT_PATH = os.path.join(PROJECT_ROOT, 'final_mutant_model', 'plate_well_id_path.json')

CHECKPOINT = os.path.join(SCRIPT_DIR, 'mil_vae_both', 'fold_P1', 'best_mil_vae.pth')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'mil_vae_both', 'fold_P1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_PLATE_KEY = 'P1'
TEST_PLATE = 'Plate_1'

# ---------------------------------------------------------------------------
# Load mappings
# ---------------------------------------------------------------------------
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
# Collect all images from BOTH directories with CORRECT labels
# ---------------------------------------------------------------------------
all_points_meta = []  # (path, source, well, drug_label, mutant_label)

drug_dir = os.path.join(PROJECT_ROOT, 'Drugs_Data', TEST_PLATE_KEY)
if os.path.exists(drug_dir):
    for root, dirs, files in os.walk(drug_dir):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff', '.png')):
                well = extract_well_from_filename(f)
                if well:
                    drug = drug_map.get(TEST_PLATE_KEY, {}).get(well, '')
                    if drug:
                        all_points_meta.append((os.path.join(root, f), 'drug', well, drug, ''))
print(f"Drug images: {len([m for m in all_points_meta if m[1]=='drug'])}")

mutant_dir = os.path.join(PROJECT_ROOT, 'Mutants_Data', TEST_PLATE_KEY)
if os.path.exists(mutant_dir):
    for root, dirs, files in os.walk(mutant_dir):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff', '.png')):
                well = extract_well_from_filename(f)
                if well:
                    mutant = mutant_map.get(TEST_PLATE_KEY, {}).get(well, '')
                    if mutant:
                        all_points_meta.append((os.path.join(root, f), 'mutant', well, '', mutant))
print(f"Mutant images: {len([m for m in all_points_meta if m[1]=='mutant'])}")
print(f"Total: {len(all_points_meta)}")

all_drug_classes = sorted(set(d for _, _, _, d, _ in all_points_meta if d))
all_mutant_classes = sorted(set(m for _, _, _, _, m in all_points_meta if m))
print(f"Unique drug classes: {len(all_drug_classes)}, mutant: {len(all_mutant_classes)}")

# ---------------------------------------------------------------------------
# Build unified dataset with dummy labels (model only needs encoder)
# ---------------------------------------------------------------------------
valid_paths = [m[0] for m in all_points_meta]
dummy_labels = [0] * len(valid_paths)

train_dataset = MultiCropDataset(
    valid_paths, dummy_labels, None,
    neighborhood=3, grid_size=12,
    augment=False, seed=SEED, num_channels=1,
    extraction_mode='neighborhood'
)
train_dataset.set_epoch(0)

# Force all images to use exact image center for the 3x3 neighborhood
img_w = train_dataset.image_size
crop_size = train_dataset.crop_size
center = (img_w - crop_size) // 2
train_dataset.epoch_centers = [(center, center)] * len(train_dataset)
print(f"Using image center: ({center}, {center}) for all {len(train_dataset)} images")


class LatentDataset(Dataset):
    def __init__(self, base, meta):
        self.base = base
        self.meta = meta
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, _ = self.base[idx]
        _, source, well, drug, mutant = self.meta[idx]
        return img, f"{source}|{well}|{drug}|{mutant}"


loader = DataLoader(
    LatentDataset(train_dataset, all_points_meta),
    batch_size=32, shuffle=False, num_workers=0
)

# ---------------------------------------------------------------------------
# Load model (same architecture as training)
# ---------------------------------------------------------------------------
print(f"Loading checkpoint: {CHECKPOINT}")
state_dict = torch.load(CHECKPOINT, map_location=device)

# Infer params from state_dict
latent_dim = state_dict['vae_mu.weight'].shape[0]
cls_weight_key = [k for k in state_dict if 'classifier' in k and k.endswith('.weight')][0]
num_classes = state_dict[cls_weight_key].shape[0]
has_contrastive = any('contrastive' in k for k in state_dict)
has_feature_decoder = any('feature_decoder' in k for k in state_dict)
has_pixel_decoder = any('pixel_decoder' in k for k in state_dict)

model = MILVAE(
    num_classes=num_classes,
    latent_dim=latent_dim,
    beta=0.1,
    num_heads=4, dropout=0.5, use_contrastive=has_contrastive,
    num_channels=1,
    pretrained='imagenet', backbone='efficientnet_b0',
    pooling='attention', img_size=224,
    feature_decoder=has_feature_decoder,
    pixel_decoder=has_pixel_decoder,
).to(device)
model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded. Latent dim: {latent_dim}, Classes: {num_classes}")

# ---------------------------------------------------------------------------
# Extract latents (deterministic: mu, no sampling)
# ---------------------------------------------------------------------------
all_z = []
all_meta = []
with torch.no_grad():
    for images, meta_batch in tqdm(loader, desc='Extracting latents'):
        images = images.to(device)
        z = model.encode_deterministic(images)
        all_z.append(z.cpu().numpy())
        all_meta.extend(meta_batch)

z = np.concatenate(all_z, axis=0)
print(f"Latents: {z.shape}")

# ---------------------------------------------------------------------------
# UMAP to 2D
# ---------------------------------------------------------------------------
print("Running UMAP...")
scaler = StandardScaler()
z_scaled = scaler.fit_transform(z)
n_neighbors = min(30, len(z) // 5)
reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.3,
                     random_state=SEED, n_jobs=-1)
z_2d = reducer.fit_transform(z_scaled)
print(f"UMAP done: {z_2d.shape} (n_neighbors={n_neighbors}, min_dist=0.3)")

# ---------------------------------------------------------------------------
# Parse metadata
# ---------------------------------------------------------------------------
points_data = []
drug_vecs = {}   # drug_class -> list of latent vectors
mutant_vecs = {} # mutant_class -> list of latent vectors

for i, meta_str in enumerate(all_meta):
    parts = meta_str.split('|')
    source = parts[0]
    well = parts[1]
    drug = parts[2]
    mutant = parts[3]
    latent = z[i]

    points_data.append({
        'x': round(float(z_2d[i, 0]), 4),
        'y': round(float(z_2d[i, 1]), 4),
        's': source, 'w': well, 'd': drug, 'm': mutant,
    })

    if drug:
        drug_vecs.setdefault(drug, []).append(latent)
    if mutant:
        mutant_vecs.setdefault(mutant, []).append(latent)

# ---------------------------------------------------------------------------
# Compute centroids
# ---------------------------------------------------------------------------
def compute_centroid(vecs):
    return np.mean(vecs, axis=0)

drug_centroids = {k: compute_centroid(v) for k, v in drug_vecs.items()}
mutant_centroids = {k: compute_centroid(v) for k, v in mutant_vecs.items()}
print(f"Drug centroids: {len(drug_centroids)}, Mutant centroids: {len(mutant_centroids)}")

# ---------------------------------------------------------------------------
# EXPECTED_MATCHES (known drug-target relationships)
# Compact version of derive_group_mapping.py
# ---------------------------------------------------------------------------
EXPECTED_MATCHES = {
    'Cefsulodin': ['mrcA', 'mrcB'],
    'Penicillin': ['mrcA', 'mrcB', 'ftsI'],
    'Sulbactam': ['mrcA', 'mrcB', 'ftsI'],
    'Mecillinam': ['mrdA'],
    'Meropenem': ['mrdA', 'ftsI', 'mrcA', 'mrcB'],
    'Aztreonam': ['ftsI'],
    'Cefepim': ['ftsI', 'mrcA', 'mrcB', 'mrdA'],
    'Ceftriaxone': ['ftsI', 'mrcA', 'mrcB'],
    'Chloramphenicol': ['rplA', 'rplC'],
    'Clarithromycin': ['rplA', 'rplC'],
    'Doxicyclin': ['rpsA', 'rpsL'],
    'Kanamycin': ['rpsA', 'rpsL'],
    'Ciprofloxacin': ['gyrA', 'gyrB', 'parC', 'parE'],
    'Levofloxacin': ['gyrA', 'gyrB', 'parC', 'parE'],
    'Norfloxacin': ['gyrA', 'gyrB', 'parC', 'parE'],
    'Rifampicin': ['rpoA', 'rpoB'],
    'Trimethoprim': ['folA', 'folP'],
    'Colistin': ['lpxA', 'lpxC', 'lptA', 'lptC'],
    'Polymyxin_B': ['lpxA', 'lpxC', 'lptA', 'lptC'],
}

# Build known lookup: drug_base_name -> set of expected mutant_base_names
# Match across concentration variants (e.g., Ciprofloxacin_1x -> {'gyrA_1','gyrA_2','gyrA_3',...})
known_map = {}
for drug_base, gene_list in EXPECTED_MATCHES.items():
    # Find all drug concentrations
    drug_variants = [d for d in all_drug_classes if d.startswith(drug_base + '_')]
    if not drug_variants:
        known_map[drug_base] = []
        continue
    # Find all mutant variants
    mutant_variants = set()
    for gene in gene_list:
        for m in all_mutant_classes:
            if m.startswith(gene + '_'):
                mutant_variants.add(m)
    for dv in drug_variants:
        known_map[dv] = sorted(mutant_variants)

# ---------------------------------------------------------------------------
# Compute pairwise distances: drug centroid -> mutant centroid (cosine distance)
# ---------------------------------------------------------------------------
def cosine_dist(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(1.0 - np.dot(a_norm, b_norm))

# Drug -> Mutant distances
drug_dists = {}
for d_name, d_cent in drug_centroids.items():
    if d_name == 'control':
        continue
    dists = []
    for m_name, m_cent in mutant_centroids.items():
        if m_name.startswith('NC_') or m_name.startswith('WT NC_'):
            continue
        dist = cosine_dist(d_cent, m_cent)
        known = 0
        if d_name in known_map and m_name in known_map[d_name]:
            known = 1
        dists.append({'n': m_name, 'd': round(dist, 6), 'k': known})
    dists.sort(key=lambda x: x['d'])
    drug_dists[d_name] = dists

# Mutant -> Drug distances
mut_dists = {}
for m_name, m_cent in mutant_centroids.items():
    if m_name.startswith('NC_') or m_name.startswith('WT NC_'):
        continue
    dists = []
    for d_name, d_cent in drug_centroids.items():
        if d_name == 'control':
            continue
        dist = cosine_dist(m_cent, d_cent)
        known = 0
        if d_name in known_map and m_name in known_map[d_name]:
            known = 1
        dists.append({'n': d_name, 'd': round(dist, 6), 'k': known})
    dists.sort(key=lambda x: x['d'])
    mut_dists[m_name] = dists

# Known targets (for display)
known_display = {}
for d_name in all_drug_classes:
    if d_name in known_map:
        known_display[d_name] = known_map[d_name]

# ---------------------------------------------------------------------------
# Build full DATA dict
# ---------------------------------------------------------------------------
# Exclude controls from class lists for cleaner dropdown
drug_classes = sorted(d for d in all_drug_classes if d != 'control')
mutant_classes = sorted(m for m in all_mutant_classes if not m.startswith('NC_') and not m.startswith('WT NC_'))

DATA = {
    'points': points_data,
    'drugC': drug_classes,
    'mutC': mutant_classes,
    'drugDists': drug_dists,
    'mutDists': mut_dists,
    'known': known_display,
}

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
json_path = os.path.join(OUTPUT_DIR, 'contour_data_v2.json')
with open(json_path, 'w') as f:
    json.dump(DATA, f)
print(f"Saved JSON: {json_path} ({os.path.getsize(json_path) / 1024 / 1024:.1f} MB)")

# ---------------------------------------------------------------------------
# Generate standalone HTML
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VAE Latent Space - UMAP Contour (v2)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system, BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#1a1a2e; color:#e0e0e0; }
.container { max-width:1600px; margin:0 auto; padding:16px; }
h1 { font-size:22px; color:#fff; margin-bottom:2px; }
.subtitle { color:#aaa; font-size:13px; margin-bottom:12px; }
.row { display:flex; gap:16px; flex-wrap:wrap; }
.left { flex:1; min-width:600px; }
.right { width:380px; min-width:320px; }
.controls { display:flex; gap:12px; margin-bottom:10px; flex-wrap:wrap; align-items:center; }
.controls label { font-size:12px; color:#ccc; }
.controls select, .controls input {
  background:#16213e; color:#e0e0e0; border:1px solid #333; padding:4px 10px;
  border-radius:5px; font-size:12px; cursor:pointer;
}
.stat-box { background:#16213e; border-radius:6px; padding:8px 14px; border:1px solid #333; font-size:13px; }
.stat-box span { color:#4fc3f7; font-weight:bold; }
#plot { width:100%; height:700px; border-radius:8px; overflow:hidden; }
.matches { background:#16213e; border-radius:8px; border:1px solid #333; padding:12px; height:700px; overflow-y:auto; }
.matches h3 { font-size:14px; color:#4fc3f7; margin-bottom:8px; }
.match-row { display:flex; justify-content:space-between; padding:3px 6px; border-radius:4px; font-size:12px; margin:2px 0; }
.match-row:hover { background:#1a1a3e; }
.match-known { border-left:3px solid #4caf50; background:#1a3a1a; }
.match-unknown { border-left:3px solid #555; }
.match-name { flex:1; }
.match-dist { color:#888; width:60px; text-align:right; }
.match-badge { font-size:10px; padding:1px 5px; border-radius:3px; margin-left:6px; }
.badge-known { background:#4caf50; color:#fff; }
.select-row { display:flex; gap:8px; margin-bottom:10px; align-items:center; }
.select-row select { flex:1; }
.search-mode { display:flex; gap:6px; margin-bottom:10px; }
.search-mode button {
  flex:1; padding:6px; border-radius:5px; border:1px solid #444;
  background:#16213e; color:#ccc; cursor:pointer; font-size:12px;
}
.search-mode button.active { background:#0d47a1; border-color:#4fc3f7; color:#fff; }
.info-row { font-size:11px; color:#888; margin-top:6px; }
#selectedLabel { color:#4fc3f7; font-weight:bold; }
.legend-grid { display:flex; flex-wrap:wrap; gap:3px; margin-top:8px; max-height:120px; overflow-y:auto; }
.legend-item { font-size:10px; padding:1px 6px; border-radius:3px; background:#16213e; border:1px solid #333; cursor:pointer; white-space:nowrap; }
.legend-item:hover { border-color:#4fc3f7; }
</style>
</head>
<body>
<div class="container">
  <h1>&#x3B2;-VAE Latent Space &mdash; UMAP Phenotype Matching (v2 &mdash; No Cross-Labeling)</h1>
  <div class="subtitle">
    UMAP of 32-dim latents | Fold: Plate_1 |
    <span id="ptCount">0</span> points |
    <span id="drugCount">0</span> drugs &times; <span id="mutCount">0</span> mutants
    &nbsp;|&nbsp; <span style="color:#4caf50">&#x25a0; known</span> <span style="color:#888">&#x25a0; unknown</span>
  </div>

  <div class="row">
    <div class="left">
      <div class="controls">
        <div>
          <label>Color: </label>
          <select id="colorMode">
            <option value="mutant">Mutant class</option>
            <option value="drug">Drug class</option>
            <option value="source">Source (drug/mutant)</option>
            <option value="density">Density only</option>
          </select>
        </div>
        <div>
          <label>Opacity: </label>
          <input type="range" id="opacity" min="0.1" max="1" step="0.05" value="0.5">
        </div>
        <div>
          <label>Size: </label>
          <input type="range" id="pointSize" min="2" max="10" step="1" value="4">
        </div>
      </div>
      <div id="plot"></div>
      <div class="section-title" id="legendTitle">Legend</div>
      <div class="legend-grid" id="legendGrid"></div>
    </div>

    <div class="right">
      <div class="matches">
        <h3>&#x2697; Phenotype Matching</h3>
        <div class="search-mode">
          <button id="modeDrug" class="active">Drug &rarr; Mutant</button>
          <button id="modeMutant">Mutant &rarr; Drug</button>
        </div>
        <div class="select-row">
          <select id="classSelect"><option>Select a class...</option></select>
        </div>
        <div class="info-row">
          Showing closest matches by centroid distance in 32-dim latent space<br>
          <span style="color:#4caf50">&#x25a0;</span> = known drug-target match
        </div>
        <div id="matchList"></div>
        <div id="matchInfo" style="margin-top:8px; font-size:11px; color:#666;"></div>
      </div>
    </div>
  </div>
</div>

<script>
var DATA = ''' + json.dumps(DATA) + r''';

var pts = DATA.points;
document.getElementById('ptCount').textContent = pts.length;
document.getElementById('drugCount').textContent = DATA.drugC.length;
document.getElementById('mutCount').textContent = DATA.mutC.length;

var plotDiv = document.getElementById('plot');
var selectedClass = null;
var searchMode = 'drug';

function getColorMap(values) {
  var unique = [...new Set(values)].sort();
  var colors = {};
  var n = unique.length;
  var hues = n <= 20
    ? ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf',
       '#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5','#c49c94','#f7b6d2','#c7c7c7','#dbdb8d','#9edae5']
    : Array.from({length:n},function(_,i){return 'hsl('+(i*360/n)+',70%,55%)';});
  unique.forEach(function(v,i){colors[v]=hues[i%hues.length];});
  return colors;
}

function render(colorMode, opacity, size) {
  var xs = pts.map(function(p){return p.x;});
  var ys = pts.map(function(p){return p.y;});

  var colorValues;
  var legendTitle = '';
  if (colorMode === 'drug') {
    colorValues = pts.map(function(p){return p.d || '(none)';});
    legendTitle = 'Legend &mdash; Drug classes';
  } else if (colorMode === 'mutant') {
    colorValues = pts.map(function(p){return p.m || '(none)';});
    legendTitle = 'Legend &mdash; Mutant classes';
  } else if (colorMode === 'source') {
    colorValues = pts.map(function(p){return p.s;});
    legendTitle = 'Legend &mdash; Source';
  } else {
    colorValues = null;
  }
  document.getElementById('legendTitle').innerHTML = legendTitle;

  var markerColors;
  if (selectedClass && colorMode !== 'source' && colorMode !== 'density') {
    markerColors = pts.map(function(p) {
      var val = colorMode === 'drug' ? p.d : p.m;
      return val === selectedClass ? '#ffeb3b' : 'rgba(100,140,200,0.3)';
    });
  } else if (colorValues) {
    var cmap = getColorMap(colorValues);
    markerColors = colorValues.map(function(v){return cmap[v];});
  } else {
    markerColors = 'rgba(100,180,255,0.4)';
  }

  var contour = {
    x: xs, y: ys, type: 'histogram2dcontour',
    colorscale: [[0,'rgba(20,20,50,0)'],[0.3,'rgba(50,80,140,0.12)'],[0.6,'rgba(70,130,200,0.2)'],[1,'rgba(180,220,255,0.3)']],
    contours: {showlabels:false,coloring:'fill'}, showscale:false, ncontours:12, hoverinfo:'skip'
  };

  var scatter = {
    x: xs, y: ys, type: 'scattergl', mode: 'markers',
    marker: {color:markerColors, size:size, opacity:opacity, line:{width:0.2,color:'rgba(0,0,0,0.2)'}},
    text: pts.map(function(p){
      var txt = '<b>Well:</b> '+p.w+'<br><b>Source:</b> '+p.s;
      if (p.d) txt += '<br><b>Drug:</b> '+p.d;
      if (p.m) txt += '<br><b>Mutant:</b> '+p.m;
      return txt;
    }),
    hoverinfo:'text',
    hoverlabel:{bgcolor:'#16213e',font:{color:'#e0e0e0',size:11},bordercolor:'#333'}
  };

  var layout = {
    paper_bgcolor:'#1a1a2e', plot_bgcolor:'#1a1a2e',
    margin:{l:50,r:20,t:10,b:50},
    xaxis:{title:'UMAP 1',color:'#888',gridcolor:'#2a2a4e',zerolinecolor:'#333',showgrid:true,zeroline:false},
    yaxis:{title:'UMAP 2',color:'#888',gridcolor:'#2a2a4e',zerolinecolor:'#333',showgrid:true,zeroline:false},
    hovermode:'closest', dragmode:'pan', showlegend:false
  };

  var config = {responsive:true,displayModeBar:true,modeBarButtonsToRemove:['autoScale2d','lasso2d','select2d'],displaylogo:false};

  Plotly.newPlot(plotDiv, [contour, scatter], layout, config).then(function(){
    updateLegend(colorValues ? getColorMap(colorValues) : null, colorMode);
  });
}

function updateLegend(colorMap, colorMode) {
  var grid = document.getElementById('legendGrid');
  grid.innerHTML = '';
  if (!colorMap) return;
  var entries = Object.entries(colorMap).sort(function(a,b){return a[0].localeCompare(b[0]);});
  entries.forEach(function(e) {
    var el = document.createElement('span');
    el.className = 'legend-item';
    el.style.borderLeft = '3px solid ' + e[1];
    el.textContent = e[0];
    el.onclick = function() {
      selectedClass = e[0];
      var sel = document.getElementById('classSelect');
      for (var i = 0; i < sel.options.length; i++) {
        if (sel.options[i].value === e[0]) { sel.selectedIndex = i; break; }
      }
      showMatches(e[0]);
      render(document.getElementById('colorMode').value,
        parseFloat(document.getElementById('opacity').value),
        parseFloat(document.getElementById('pointSize').value));
    };
    grid.appendChild(el);
  });
}

function showMatches(cls) {
  var list = document.getElementById('matchList');
  var info = document.getElementById('matchInfo');
  selectedClass = cls;
  document.getElementById('selectedLabel').textContent = cls;
  
  var matches, sourceLabel, targetLabel, allDists;
  if (searchMode === 'drug') {
    allDists = DATA.drugDists[cls];
    sourceLabel = 'Drug';
    targetLabel = 'Mutant';
  } else {
    allDists = DATA.mutDists[cls];
    sourceLabel = 'Mutant';
    targetLabel = 'Drug';
  }

  if (!allDists) {
    list.innerHTML = '<div style="color:#666;padding:8px;">No matches found</div>';
    return;
  }

  var knownInfo = '';
  if (searchMode === 'drug' && DATA.known[cls] && DATA.known[cls].length > 0) {
    knownInfo = '<div style="font-size:11px;color:#4caf50;margin-bottom:6px;">&#x2713; Known targets: ' + DATA.known[cls].join(', ') + '</div>';
  }

  var html = '<div style="font-size:12px;color:#4fc3f7;margin-bottom:6px;">Closest to <b>' + cls + '</b>:</div>' + knownInfo;
  allDists.forEach(function(m, i) {
    var known = m.k ? 'match-known' : 'match-unknown';
    var badge = m.k ? '<span class="match-badge badge-known">&#x2713;</span>' : '';
    html += '<div class="match-row ' + known + '" onclick="highlightClass(\'' + m.n + '\')">' +
      '<span class="match-name">' + (i+1) + '. ' + m.n + '</span>' +
      '<span class="match-dist">' + m.d + '</span>' + badge + '</div>';
  });

  info.innerHTML = 'Click a match to highlight it on the plot | Centroid distance (lower = closer)';
  list.innerHTML = html;
  render(document.getElementById('colorMode').value,
    parseFloat(document.getElementById('opacity').value),
    parseFloat(document.getElementById('pointSize').value));
}

function highlightClass(cls) {
  selectedClass = cls;
  var sel = document.getElementById('classSelect');
  for (var i = 0; i < sel.options.length; i++) {
    if (sel.options[i].value === cls) { sel.selectedIndex = i; break; }
  }
  if (searchMode === 'drug' && DATA.mutDists[cls]) {
    document.getElementById('modeMutant').click();
  } else if (searchMode === 'mutant' && DATA.drugDists[cls]) {
    document.getElementById('modeDrug').click();
  }
  showMatches(cls);
}

function populateDropdown(mode) {
  var sel = document.getElementById('classSelect');
  sel.innerHTML = '<option value="">Select a class...</option>';
  var classes = mode === 'drug' ? DATA.drugC : DATA.mutC;
  classes.sort().forEach(function(c) {
    var opt = document.createElement('option');
    opt.value = c;
    opt.textContent = c;
    sel.appendChild(opt);
  });
}

document.getElementById('colorMode').addEventListener('change', function() {
  render(this.value, parseFloat(document.getElementById('opacity').value), parseFloat(document.getElementById('pointSize').value));
});
document.getElementById('opacity').addEventListener('input', function() {
  render(document.getElementById('colorMode').value, parseFloat(this.value), parseFloat(document.getElementById('pointSize').value));
});
document.getElementById('pointSize').addEventListener('input', function() {
  render(document.getElementById('colorMode').value, parseFloat(document.getElementById('opacity').value), parseFloat(this.value));
});

document.getElementById('classSelect').addEventListener('change', function() {
  if (this.value) showMatches(this.value);
});

document.getElementById('modeDrug').addEventListener('click', function() {
  searchMode = 'drug';
  this.className = 'active';
  document.getElementById('modeMutant').className = '';
  populateDropdown('drug');
  document.getElementById('matchList').innerHTML = '';
  document.getElementById('matchInfo').innerHTML = '';
  selectedClass = null;
  render(document.getElementById('colorMode').value, parseFloat(document.getElementById('opacity').value), parseFloat(document.getElementById('pointSize').value));
});

document.getElementById('modeMutant').addEventListener('click', function() {
  searchMode = 'mutant';
  this.className = 'active';
  document.getElementById('modeDrug').className = '';
  populateDropdown('mutant');
  document.getElementById('matchList').innerHTML = '';
  document.getElementById('matchInfo').innerHTML = '';
  selectedClass = null;
  render(document.getElementById('colorMode').value, parseFloat(document.getElementById('opacity').value), parseFloat(document.getElementById('pointSize').value));
});

populateDropdown('drug');
render('mutant', 0.5, 4);
</script>
</body>
</html>'''

html_path = os.path.join(OUTPUT_DIR, 'latent_contour_v2.html')
with open(html_path, 'w') as f:
    f.write(HTML_TEMPLATE)
print(f"Saved HTML: {html_path} ({os.path.getsize(html_path) / 1024 / 1024:.1f} MB)")
print("Done.")
