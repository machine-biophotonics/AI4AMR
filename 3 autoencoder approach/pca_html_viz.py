#!/usr/bin/env python3
"""Interactive HTML: PCA on all records (4032 points) before/after domain removal."""
import os, sys, warnings, re, json
warnings.filterwarnings("ignore")
import numpy as np
import torch
from collections import defaultdict
from sklearn.decomposition import PCA

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
print("PCA HTML visualization: 4032 records")
print("=" * 60)

pt = torch.load(LATENTS_PATH, map_location='cpu', weights_only=False)
records = pt['records']

# Record-level: mean over 100 positions
n_rec = len(records)
X = np.zeros((n_rec, 1280), dtype=np.float64)
labels = []
sources = []
wells = []
for i, r in enumerate(records):
    X[i] = r['bag'].mean(axis=0).astype(np.float64)
    labels.append(r['true_label'])
    sources.append(r['source'])
    wells.append(r['well'])

labels = np.array(labels)
sources = np.array(sources)
wells = np.array(wells)
print(f"Loaded {n_rec} records × 1280-dim")

# ---- Class-level PCA ----
class_bags = defaultdict(list)
for i, lbl in enumerate(labels):
    class_bags[lbl].append(X[i])
classes = sorted(class_bags.keys())
N_cls = len(classes)
X_cls = np.zeros((N_cls, 1280))
for i, c in enumerate(classes):
    X_cls[i] = np.mean(class_bags[c], axis=0)
src_map = {}
for r in records:
    src_map[r['true_label']] = r['source']
cls_src = np.array([src_map[c] for c in classes])

pca = PCA(n_components=20)
pca.fit(X_cls)
loadings = pca.components_
var_exp = pca.explained_variance_ratio_

# Domain-separating PCs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

# Project both to first 3 PCs
proj_orig = X @ loadings[:3].T
proj_clean = X_clean @ loadings[:3].T

# Color mapping
SRC_COLORS = {'drug': '#E41A1C', 'mutant': '#4DAF4A'}

# Build hover data
drug_moa_map = {}
for d_name in set(labels[sources == 'drug']):
    if d_name == 'control':
        drug_moa_map[d_name] = 'Control (water)'
    else:
        db = drug_base(d_name)
        drug_moa_map[d_name] = DRUG_TO_MOA.get(db, 'Other')

mutant_gene_map = {}
for m_name in set(labels[sources == 'mutant']):
    g = re.match(r'^([a-zA-Z]+)', m_name)
    mutant_gene_map[m_name] = g.group(1).upper() if g else m_name

# Subsample for responsive plot (max 3000 per type)
rng = np.random.RandomState(SEED)
drug_idx = np.where(sources == 'drug')[0]
mut_idx = np.where(sources == 'mutant')[0]
n_d = min(3000, len(drug_idx))
n_m = min(3000, len(mut_idx))
sel_d = rng.choice(drug_idx, n_d, replace=False)
sel_m = rng.choice(mut_idx, n_m, replace=False)
sel = np.concatenate([sel_d, sel_m])
sel.sort()

# ---- Generate HTML ----
html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>PCA: 1280-dim bag — Before vs After domain removal</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #111; color: #ddd; }}
h1, h2 {{ color: #fff; }}
.container {{ max-width: 1600px; margin: 0 auto; }}
.plot {{ margin-bottom: 30px; }}
.row {{ display: flex; gap: 20px; }}
.col {{ flex: 1; }}
.stats {{ background: #1e1e1e; border-radius: 8px; padding: 15px; margin: 10px 0; }}
.stats table {{ border-collapse: collapse; width: 100%; }}
.stats td, .stats th {{ padding: 4px 10px; text-align: left; border-bottom: 1px solid #333; }}
.stats th {{ color: #aaa; font-size: 12px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 4px 12px; margin: 10px 0; }}
.legend-item {{ font-size: 11px; display: flex; align-items: center; gap: 4px; }}
.color-box {{ width: 10px; height: 10px; display: inline-block; border-radius: 2px; }}
</style></head><body>
<div class="container">
<h1>PCA on 1280-dim bag features</h1>
<p style="color:#888">{n_rec} records ({len(set(labels))} classes, {n_d} drug + {n_m} mutant shown)</p>

<div class="stats">
<table>
<tr><th>PC</th><th>Variance</th><th>Cumulative</th><th>Domain AUC</th></tr>
"""

for k in range(10):
    auc = 0
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(pca.transform(X_cls)[:, k:k+1], (cls_src == 'drug').astype(int))
    auc = roc_auc_score((cls_src == 'drug').astype(int), lr.predict_proba(pca.transform(X_cls)[:, k:k+1])[:, 1])
    html += f'<tr><td>PC{k+1}</td><td>{var_exp[k]*100:.1f}%</td><td>{var_exp[:k+1].sum()*100:.1f}%</td><td>{auc:.4f}</td></tr>\n'

sep_str = ','.join(str(p+1) for p in sep_pcs)

# Build JSON data outside f-string to avoid brace nesting issue
orig_json = json.dumps({
    'pc1': proj_orig[sel, 0].tolist(),
    'pc2': proj_orig[sel, 1].tolist(),
    'pc3': proj_orig[sel, 2].tolist(),
    'labels': labels[sel].tolist(),
    'sources': sources[sel].tolist(),
    'wells': wells[sel].tolist(),
    'moa': [drug_moa_map.get(l, mutant_gene_map.get(l, '')) for l in labels[sel]],
})
clean_json = json.dumps({
    'pc1': proj_clean[sel, 0].tolist(),
    'pc2': proj_clean[sel, 1].tolist(),
    'pc3': proj_clean[sel, 2].tolist(),
    'labels': labels[sel].tolist(),
    'sources': sources[sel].tolist(),
    'wells': wells[sel].tolist(),
    'moa': [drug_moa_map.get(l, mutant_gene_map.get(l, '')) for l in labels[sel]],
})

html += f"""</table>
<p style="margin-top:10px; color:#ff9">
Removed {n_remove} domain-separating PCs (PC{sep_str})
</p>
</div>

<h2>Before domain removal</h2>
<div class="row">
<div class="col"><div id="plot1" class="plot"></div></div>
<div class="col"><div id="plot2" class="plot"></div></div>
</div>

<h2>After removing {n_remove} domain PCs</h2>
<div class="row">
<div class="col"><div id="plot3" class="plot"></div></div>
<div class="col"><div id="plot4" class="plot"></div></div>
</div>
</div>

<script>
const origData = {orig_json};
const cleanData = {clean_json};

const drugColor = '#E41A1C';
const mutColor = '#4DAF4A';

function makeScatter(data, title, pcx, pcy, showLegend) {{
    const drugTraces = {{x:[], y:[], text:[], mode:'markers', type:'scattergl',
        marker: {{size:3, color:drugColor, opacity:0.5}}, name:'Drug',
        hovertemplate:'%{{text}}<extra></extra>'}};
    const mutTraces = {{x:[], y:[], text:[], mode:'markers', type:'scattergl',
        marker: {{size:3, color:mutColor, opacity:0.5}}, name:'Mutant',
        hovertemplate:'%{{text}}<extra></extra>'}};

    for (let i = 0; i < data.labels.length; i++) {{
        const hover = "${{data.labels[i]}}<br>Well: ${{data.wells[i]}}<br>${{data.moa[i]}}<br>Src: ${{data.sources[i]}}";
        if (data.sources[i] === 'drug') {{
            drugTraces.x.push(data[pcx][i]);
            drugTraces.y.push(data[pcy][i]);
            drugTraces.text.push(hover);
        }} else {{
            mutTraces.x.push(data[pcx][i]);
            mutTraces.y.push(data[pcy][i]);
            mutTraces.text.push(hover);
        }}
    }}
    return {{
        data: [drugTraces, mutTraces],
        layout: {{
            title: {{text: title, font: {{size: 13, color: '#fff'}}}},
            xaxis: {{title: pcx.toUpperCase().replace('pc', 'PC '), color: '#888', gridcolor: '#333', zerolinecolor: '#444'}},
            yaxis: {{title: pcy.toUpperCase().replace('pc', 'PC '), color: '#888', gridcolor: '#333', zerolinecolor: '#444'}},
            plot_bgcolor: '#1a1a1a', paper_bgcolor: '#1a1a1a',
            font: {{color: '#ccc'}},
            width: null, height: 500,
            margin: {{l:60, r:20, t:50, b:50}},
            showlegend: showLegend,
            hovermode: 'closest',
        }}
    }};
}}

Plotly.newPlot('plot1', makeScatter(origData, 'PC1 vs PC2 (original)', 'pc1', 'pc2', true).data,
    makeScatter(origData, 'PC1 vs PC2 (original)', 'pc1', 'pc2', true).layout);
Plotly.newPlot('plot2', makeScatter(origData, 'PC1 vs PC3 (original)', 'pc1', 'pc3', false).data,
    makeScatter(origData, 'PC1 vs PC3 (original)', 'pc1', 'pc3', false).layout);
Plotly.newPlot('plot3', makeScatter(cleanData, 'PC1 vs PC2 (cleaned)', 'pc1', 'pc2', true).data,
    makeScatter(cleanData, 'PC1 vs PC2 (cleaned)', 'pc1', 'pc2', true).layout);
Plotly.newPlot('plot4', makeScatter(cleanData, 'PC1 vs PC3 (cleaned)', 'pc1', 'pc3', false).data,
    makeScatter(cleanData, 'PC1 vs PC3 (cleaned)', 'pc1', 'pc3', false).layout);
</script>
</body></html>
"""

html_path = os.path.join(OUTPUT_DIR, 'pca_before_after_interactive.html')
with open(html_path, 'w') as f:
    f.write(html)
print(f"\nHTML: {html_path}")
fsize = os.path.getsize(html_path) / 1e6
print(f"  Size: {fsize:.1f} MB ({len(sel)} points)")
print(f"{'=' * 60}")
