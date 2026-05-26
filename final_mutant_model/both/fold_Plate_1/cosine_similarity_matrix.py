#!/usr/bin/env python3
"""
Cosine Similarity Matrix Visualization - Research Paper Style
Includes both drugs and mutants with proper labels
"""

import numpy as np
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, leaves_list
import os

embeddings_path = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/final_mutant_model/both/fold_Plate_1/embeddings_Plate_1_mil_n3.npz'
output_dir = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/final_mutant_model/both/fold_Plate_1'
ic50_path = '/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging/final_mutant_model/plate_well_ic50_mapping.json'

print("Loading embeddings...")
data = np.load(embeddings_path, allow_pickle=True)
embeddings = data['embeddings']
labels = data['labels']
paths = data['paths']

print(f"Embeddings shape: {embeddings.shape}")

with open(ic50_path, 'r') as f:
    ic50_mapping = json.load(f)

def get_label(path, default_label):
    path_lower = path.lower()
    well_match = re.search(r'Well([A-H]\d+)_', os.path.basename(path))
    if not well_match:
        return default_label
    well = well_match.group(1)
    
    if '/drugs_data/' in path_lower or 'drugs_data' in path_lower:
        for plate, wells in ic50_mapping.items():
            if well in wells:
                info = wells[well]
                ab = info.get('antibiotic', '')
                ic = info.get('ic50_multiple', '')
                if ab and ic:
                    if ic == 'control':
                        return 'DMSO'
                    return f"{ab.replace(' ', '_')}_{ic}"
        return default_label
    
    elif '/mutants_data/' in path_lower or 'mutants_data' in path_lower:
        return default_label
    
    return default_label

print("Creating proper labels...")
new_labels = []
for i, (path, lbl) in enumerate(zip(paths, labels)):
    new_lbl = get_label(path, lbl)
    new_labels.append(new_lbl)
new_labels = np.array(new_labels)

print(f"Unique labels: {len(np.unique(new_labels))}")
print(f"Sample labels: {list(np.unique(new_labels))[:30]}")

label_counts = {}
for lbl in new_labels:
    label_counts[lbl] = label_counts.get(lbl, 0) + 1
print(f"Label distribution (top 10): {sorted(label_counts.items(), key=lambda x: -x[1])[:10]}")

unique_labels = np.unique(new_labels)
label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

grouped_indices = []
group_labels = []
for lbl in unique_labels:
    indices = np.where(new_labels == lbl)[0]
    grouped_indices.extend(indices)
    group_labels.extend([lbl] * len(indices))

grouped_indices = np.array(grouped_indices)
grouped_labels = np.array(group_labels)

print("Sorting labels alphabetically...")

# Get drug names from ic50 mapping
drug_names = set()
for plate, wells in ic50_mapping.items():
    for well, info in wells.items():
        ab = info.get('antibiotic', '')
        if ab:
            drug_names.add(ab.replace(' ', '_'))

drug_labels = sorted([l for l in unique_labels if any(d in l for d in drug_names)])
mutant_labels = sorted([l for l in unique_labels if l not in drug_labels])

print(f"Drugs: {len(drug_labels)}, Mutants: {len(mutant_labels)}")
print(f"Sample drugs: {drug_labels[:5]}")
print(f"Sample mutants: {mutant_labels[:5]}")

sorted_labels = drug_labels + mutant_labels

grouped_indices = []
group_labels = []
for lbl in sorted_labels:
    indices = np.where(new_labels == lbl)[0]
    grouped_indices.extend(indices)
    group_labels.extend([lbl] * len(indices))

grouped_indices = np.array(grouped_indices)
grouped_labels = np.array(group_labels)

print("Computing cosine similarity...")
cos_sim = cosine_similarity(embeddings[grouped_indices])
print(f"Similarity matrix shape: {cos_sim.shape}")

cos_sim_ordered = cos_sim
clustered_labels = grouped_labels

unique_clustered = sorted_labels

print("Plotting...")
fig, ax = plt.subplots(figsize=(16, 14))

cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)

im = ax.imshow(cos_sim_ordered, cmap=cmap, vmin=-0.2, vmax=1.0, aspect='equal')

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Cosine Similarity', fontsize=12, fontweight='bold')

unique_labels_list = list(unique_clustered)
n_labels = len(unique_labels_list)

tick_positions = []
tick_labels = []
prev_pos = 0
for lbl in unique_labels_list:
    count = np.sum(grouped_labels == lbl)
    tick_positions.append(prev_pos + count // 2)
    short_label = lbl.replace('_', ' ')[:20]
    tick_labels.append(short_label)
    prev_pos += count

ax.set_xticks(tick_positions)
ax.set_yticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=90, fontsize=6, ha='center')
ax.set_yticklabels(tick_labels, fontsize=6)

ax.set_xlabel('Samples (Hierarchical Clustering)', fontsize=12, fontweight='bold')
ax.set_ylabel('Samples (Hierarchical Clustering)', fontsize=12, fontweight='bold')
ax.set_title('Cosine Similarity Matrix of Drug and Mutant Embeddings\n(Ward Linkage Clustering)', fontsize=14, fontweight='bold', pad=20)

for pos in np.cumsum([np.sum(grouped_labels == l) for l in unique_labels_list[:-1]]):
    ax.axhline(pos - 0.5, color='white', linewidth=1.5)
    ax.axvline(pos - 0.5, color='white', linewidth=1.5)

plt.tight_layout()

output_path = os.path.join(output_dir, 'cosine_similarity_matrix_full.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")

plt.close()

print("Done!")
print(f"Total samples: {len(paths)}")
print(f"Drugs: {len([p for p in paths if 'drug' in p.lower()])}")
print(f"Mutants: {len([p for p in paths if 'mutant' in p.lower()])}")