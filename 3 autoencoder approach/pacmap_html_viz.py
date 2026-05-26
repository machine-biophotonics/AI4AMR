#!/usr/bin/env python3
"""Interactive HTML: PaCMAP on all records before/after PCA domain removal."""
import os, sys, warnings, re, json
warnings.filterwarnings("ignore")
import numpy as np
import torch
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pacmap

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/3 autoencoder approach/mil_vae_both/fold_P1'
LATENTS_PATH = os.path.join(OUTPUT_DIR, 'test_latents_P1_20260523_222527.pt')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MOA_GROUPS = {
    'Fluoroquinolones': ['ciprofloxacin','levofloxacin','norfloxacin'],
    'Rifamycins': ['rifampicin'],
    'Folate_inhibitors': ['trimethoprim'],
    'Ribosome_50S': ['chloramphenicol','clarithromycin'],
    'Ribosome_30S': ['doxicyclin','kanamycin'],
    'Penems': ['penicillin','mecillinam','meropenem'],
    'Cephalosporins': ['cefepim','cefsulodin','ceftriaxone'],
    'Polymyxins': ['polymyxin_b','colistin'],
    'BLactamase_inhibitors': ['avibactam','relebactam','sulbactam','clavulanic_acid'],
    'Other': ['aztreonam'],
}
DRUG_TO_MOA = {}
for g, drugs in MOA_GROUPS.items():
    for d in drugs:
        DRUG_TO_MOA[d] = g

def drug_base(name):
    m = re.match(r'^(.+)_(\d+(?:\.\d+)?x)$', name)
    return m.group(1).lower() if m else name.lower()

print("=" * 60)
print("PaCMAP: before/after PCA domain removal")
print("=" * 60)

pt = torch.load(LATENTS_PATH, map_location='cpu', weights_only=False)
records = pt['records']

# Record-level: mean over 100 positions
n_rec = len(records)
X = np.zeros((n_rec, 1280), dtype=np.float64)
labels = []; sources = []; wells = []
for i, r in enumerate(records):
    X[i] = r['bag'].mean(axis=0).astype(np.float64)
    labels.append(r['true_label'])
    sources.append(r['source'])
    wells.append(r['well'])
labels = np.array(labels); sources = np.array(sources); wells = np.array(wells)
print(f"Loaded {n_rec} records × 1280-dim")

# ---- Class-level PCA to find domain PCs ----
class_bags = defaultdict(list)
for i, lbl in enumerate(labels):
    class_bags[lbl].append(X[i])
classes = sorted(class_bags.keys())
X_cls = np.zeros((len(classes), 1280))
for i, c in enumerate(classes):
    X_cls[i] = np.mean(class_bags[c], axis=0)
src_map = {}
for r in records:
    src_map[r['true_label']] = r['source']
cls_src = np.array([src_map[c] for c in classes])

pca = PCA(n_components=20)
pca.fit(X_cls)
loadings = pca.components_

sep_pcs = []
for k in range(20):
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(pca.transform(X_cls)[:, k:k+1], (cls_src == 'drug').astype(int))
    auc = roc_auc_score((cls_src == 'drug').astype(int), lr.predict_proba(pca.transform(X_cls)[:, k:k+1])[:, 1])
    if auc > 0.7:
        sep_pcs.append(k)

n_remove = len(sep_pcs)
P = np.eye(1280) - loadings[:n_remove].T @ loadings[:n_remove]
X_clean = X @ P.T
print(f"Removing {n_remove} domain PCs: {[f'PC{p+1}' for p in sep_pcs]}")

# ---- PaCMAP ----
print("\nComputing PaCMAP on original 1280-dim ...")
mapper = pacmap.PaCMAP(n_components=2, n_neighbors=30, MN_ratio=0.5, FP_ratio=2.0, random_state=SEED)
pac_orig = mapper.fit_transform(X)
print(f"  Done: {pac_orig.shape}")

print("Computing PaCMAP on corrected 1280-dim ...")
mapper2 = pacmap.PaCMAP(n_components=2, n_neighbors=30, MN_ratio=0.5, FP_ratio=2.0, random_state=SEED)
pac_clean = mapper2.fit_transform(X_clean)
print(f"  Done: {pac_clean.shape}")

# ---- Color map ----
SRC_COLORS = {'drug': '#E41A1C', 'mutant': '#4DAF4A'}
drug_moa_map = {}
for d_name in set(labels[sources == 'drug']):
    drug_moa_map[d_name] = 'Control (water)' if d_name == 'control' else DRUG_TO_MOA.get(drug_base(d_name), 'Other')
mutant_gene_map = {}
for m_name in set(labels[sources == 'mutant']):
    g = re.match(r'^([a-zA-Z]+)', m_name)
    mutant_gene_map[m_name] = g.group(1).upper() if g else m_name

# Subsample for responsive plot
rng = np.random.RandomState(SEED)
drug_idx = np.where(sources == 'drug')[0]
mut_idx = np.where(sources == 'mutant')[0]
n_d = min(1500, len(drug_idx))
n_m = min(1500, len(mut_idx))
sel_d = rng.choice(drug_idx, n_d, replace=False)
sel_m = rng.choice(mut_idx, n_m, replace=False)
sel = np.concatenate([sel_d, sel_m])
sel.sort()

orig_json = json.dumps({
    'x': pac_orig[sel, 0].tolist(), 'y': pac_orig[sel, 1].tolist(),
    'labels': labels[sel].tolist(), 'sources': sources[sel].tolist(),
    'wells': wells[sel].tolist(),
    'moa': [drug_moa_map.get(l, mutant_gene_map.get(l, '')) for l in labels[sel]],
})
clean_json = json.dumps({
    'x': pac_clean[sel, 0].tolist(), 'y': pac_clean[sel, 1].tolist(),
    'labels': labels[sel].tolist(), 'sources': sources[sel].tolist(),
    'wells': wells[sel].tolist(),
    'moa': [drug_moa_map.get(l, mutant_gene_map.get(l, '')) for l in labels[sel]],
})

# ---- HTML ----
html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>PaCMAP: 1280-dim bag — Before vs After domain removal</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{font-family:'Segoe UI',Arial,sans-serif;margin:20px;background:#111;color:#ddd;}}
h1,h2 {{color:#fff;}}
.container {{max-width:1400px;margin:0 auto;}}
.plot {{margin-bottom:20px;}}
.stats {{background:#1e1e1e;border-radius:8px;padding:15px;margin:10px 0;}}
</style></head><body>
<div class="container">
<h1>PaCMAP on 1280-dim bag features</h1>
<p style="color:#888">{n_rec} records ({len(set(labels))} classes, {n_d} drug + {n_m} mutant shown)</p>

<div class="stats">
<p><b>Removed PCs:</b> {n_remove} domain-separating PCs (PC{', '.join(f'PC{p+1}' for p in sep_pcs)})</p>
</div>

<h2>Original 1280-dim</h2>
<div id="plot1" class="plot"></div>

<h2>After removing {n_remove} domain PCs</h2>
<div id="plot2" class="plot"></div>
</div>

<script>
const origData = {orig_json};
const cleanData = {clean_json};
const dc = '#E41A1C', mc = '#4DAF4A';

function makePlot(data, title) {{
    const dt = {{x:[], y:[], text:[], mode:'markers', type:'scattergl',
        marker:{{size:3, color:dc, opacity:0.4}}, name:'Drug',
        hovertemplate:'%{{text}}<extra></extra>'}};
    const mt = {{x:[], y:[], text:[], mode:'markers', type:'scattergl',
        marker:{{size:3, color:mc, opacity:0.4}}, name:'Mutant',
        hovertemplate:'%{{text}}<extra></extra>'}};
    for (let i = 0; i < data.labels.length; i++) {{
        const h = data.labels[i]+'<br>Well: '+data.wells[i]+'<br>'+data.moa[i]+'<br>Src: '+data.sources[i];
        if (data.sources[i] === 'drug') {{ dt.x.push(data.x[i]); dt.y.push(data.y[i]); dt.text.push(h); }}
        else {{ mt.x.push(data.x[i]); mt.y.push(data.y[i]); mt.text.push(h); }}
    }}
    return {{
        data:[dt,mt],
        layout:{{
            title:{{text:title, font:{{size:14,color:'#fff'}}}},
            xaxis:{{title:'PaCMAP 1', color:'#888', gridcolor:'#333', zerolinecolor:'#444', scaleanchor:'y'}},
            yaxis:{{title:'PaCMAP 2', color:'#888', gridcolor:'#333', zerolinecolor:'#444'}},
            plot_bgcolor:'#1a1a1a', paper_bgcolor:'#1a1a1a',
            font:{{color:'#ccc'}}, width:null, height:550,
            margin:{{l:60,r:20,t:50,b:50}}, hovermode:'closest',
        }}
    }};
}}

Plotly.newPlot('plot1', makePlot(origData, 'PaCMAP: Original 1280-dim bag features').data,
    makePlot(origData, 'PaCMAP: Original 1280-dim bag features').layout);
Plotly.newPlot('plot2', makePlot(cleanData, 'PaCMAP: After removing {n_remove} domain PCs').data,
    makePlot(cleanData, 'PaCMAP: After removing {n_remove} domain PCs').layout);
</script>
</body></html>
"""

html_path = os.path.join(OUTPUT_DIR, 'pacmap_before_after_interactive.html')
with open(html_path, 'w') as f:
    f.write(html)
print(f"\nHTML: {html_path}")
print(f"  Size: {os.path.getsize(html_path)/1e6:.1f} MB ({len(sel)} points)")
print(f"{'=' * 60}")
