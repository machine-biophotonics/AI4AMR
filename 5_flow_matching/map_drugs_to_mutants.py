#!/usr/bin/env python3
"""
Map drugs → mutants based on shared GMM clusters.
Find which drugs SHARE a phenotype cluster with mutants
and which are ORPHAN (no mutant co-clustering).
"""
import numpy as np
from collections import defaultdict, Counter
import json, re, os

def infer_type(name):
    name = str(name)
    if name == 'control' or 'NC_' in name or 'WT NC' in name:
        return 'Control'
    if re.search(r'_\d+\.?\d*x$', name):
        return 'Drug'
    return 'Mutant'

OUT_DIR = "interpretable_directions/pc1_removed"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load ──
feats_pca = np.load(f"{OUT_DIR}/../latent_analysis_pacmap/feats_t10_cond.npy").astype(np.float64)
labels = np.load("latent_analysis_pacmap/labels.npy")
class_names = np.load("latent_analysis_pacmap/class_names.npy", allow_pickle=True)
gmm_labels = np.load(f"{OUT_DIR}/cluster_labels_gmm.npy")

N = len(labels)
# Build per-class type info
class_type_map = {i: infer_type(n) for i, n in enumerate(class_names)}
class_name_map = {i: str(n) for i, n in enumerate(class_names)}

# Classes by type
drug_classes = [i for i in range(185) if class_type_map[i] == 'Drug']
mutant_classes = [i for i in range(185) if class_type_map[i] == 'Mutant']
control_classes = [i for i in range(185) if class_type_map[i] == 'Control']
print(f"Drugs: {len(drug_classes)}, Mutants: {len(mutant_classes)}, Controls: {len(control_classes)}")

# ── For each class, count images per GMM cluster ──
# class_cluster_dist[class_idx][cluster_id] = count
class_cluster_dist = defaultdict(lambda: defaultdict(int))
cluster_composition = defaultdict(lambda: {"Drug": Counter(), "Mutant": Counter(), "Control": Counter(), "total": 0})

for i in range(N):
    cls_idx = int(labels[i])
    cl = int(gmm_labels[i])
    ctype = class_type_map[cls_idx]
    name = class_name_map[cls_idx]
    class_cluster_dist[cls_idx][cl] += 1
    cluster_composition[cl][ctype][name] += 1
    cluster_composition[cl]["total"] += 1

# ── Identify "specific" vs "promiscuous" clusters ──
# A cluster is promiscuous if it contains > 50% of all classes of any type
total_drugs = len(drug_classes)
total_mutants = len(mutant_classes)

specific_clusters = []
promiscuous_clusters = []
for cl in sorted(cluster_completion.keys()):
    nd = len(cluster_composition[cl]["Drug"])
    nm = len(cluster_composition[cl]["Mutant"])
    is_promiscuous = (nd > 0.5 * total_drugs) and (nm > 0.5 * total_mutants)
    if is_promiscuous:
        promiscuous_clusters.append(cl)
    else:
        specific_clusters.append(cl)

# For each drug: find top-3 clusters (by image count), excluding promiscuous
# Then find mutants in those clusters
drug_mutant_map = {}  # drug_name -> {clusters: [...], mutants: [...]}
orphan_drugs = []

for cls_idx in drug_classes:
    dname = class_name_map[cls_idx]
    clusters = class_cluster_dist[cls_idx]
    total_imgs = sum(clusters.values())
    
    # Sort clusters by image count
    sorted_clusters = sorted(clusters.items(), key=lambda x: -x[1])
    
    # Get specific clusters this drug is in (exclude promiscuous)
    specific_here = [(cl, cnt) for cl, cnt in sorted_clusters if cl in specific_clusters]
    
    # For each specific cluster, find co-occurring mutants
    mutant_matches = {}  # cluster -> list of mutants
    for cl, cnt in specific_here:
        mutants_in_cluster = list(cluster_composition[cl]["Mutant"].keys())
        if mutants_in_cluster:
            mutant_matches[cl] = mutants_in_cluster
    
    all_mutants = set()
    for cl, mlist in mutant_matches.items():
        all_mutants.update(mlist)
    
    if all_mutants:
        drug_mutant_map[dname] = {
            "clusters": {cl: cnt for cl, cnt in specific_here if cl in mutant_matches},
            "mutants": sorted(all_mutants)
        }
    else:
        orphan_drugs.append(dname)

# ── Now do a finer-grained: for each drug, find mutant in its top-3 clusters (any cluster) ──
# This catches everything including promiscuous clusters
drug_mutant_map_all = {}
for cls_idx in drug_classes:
    dname = class_name_map[cls_idx]
    clusters = class_cluster_dist[cls_idx]
    total_imgs = sum(clusters.values())
    sorted_clusters = sorted(clusters.items(), key=lambda x: -x[1])
    
    all_mutants = set()
    cluster_details = {}
    for cl, cnt in sorted_clusters[:5]:  # top 5 clusters
        mutants_in_cluster = list(cluster_composition[cl]["Mutant"].keys())
        if mutants_in_cluster:
            all_mutants.update(mutants_in_cluster)
            cluster_details[cl] = {
                "cnt": cnt,
                "pct": cnt / total_imgs * 100,
                "mutants": mutants_in_cluster[:10]  # show top 10
            }
    
    drug_mutant_map_all[dname] = {
        "clusters": cluster_details,
        "all_mutants": sorted(all_mutants) if all_mutants else []
    }

# ── Report ──
print("\n" + "="*80)
print("DRUG → MUTANT MAPPING (via shared GMM clusters)")
print("="*80)

# Group drugs by their mutant associations
# First, find unique mutant groups
mutant_groups = defaultdict(list)  # frozenset of mutants -> list of drugs
for dname, info in drug_mutant_map_all.items():
    mutants = tuple(info["all_mutants"])
    if mutants:
        mutant_groups[mutants].append(dname)

print(f"\nTotal drugs with any mutant association: {len(drug_mutant_map_all) - len(orphan_drugs)}")
print(f"Orphan drugs (no mutant in any cluster): {len(orphan_drugs)}")

# Print orphan drugs
print(f"\n{'─'*80}")
print("ORPHAN DRUGS (no mutant co-clustering)")
print(f"{'─'*80}")
for d in sorted(orphan_drugs):
    print(f"  • {d}")

# Print drug→mutant mapping by functional group (most specific first)
print(f"\n{'─'*80}")
print("DRUG → MUTANT MAPPINGS")
print(f"{'─'*80}")

# Show drugs grouped by which mutants they associate with
# Sort by specificity (fewer mutants = more specific)
for mutants_key, drugs in sorted(mutant_groups.items(), key=lambda x: (len(x[0]), -len(x[1]))):
    if not mutants_key or len(mutants_key) > 40:  # skip empty and very promiscuous
        continue
    n_d = len(drugs)
    n_m = len(mutants_key)
    specificity = "★ SPECIFIC" if n_m <= 5 and n_d <= 10 else ("medium" if n_m <= 20 else "broad")
    print(f"\n  Group ({n_d} drugs ↔ {n_m} mutants) [{specificity}]:")
    print(f"    Drugs: {', '.join(sorted(drugs))}")
    print(f"    Mutants: {', '.join(mutants_key)}")

# ── Summary statistics ──
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total drugs: {len(drug_classes)}")
drugs_mapped = len([d for d, info in drug_mutant_map_all.items() if info['all_mutants']])
print(f"Drugs with mutant phenotype: {drugs_mapped}")
print(f"Orphan drugs (drug-only phenotype): {len(orphan_drugs)}")
print(f"Total mutants: {len(mutant_classes)}")
mutants_mapped = len(set(m for info in drug_mutant_map_all.values() for m in info['all_mutants']))
print(f"Mutants appearing in drug clusters: {mutants_mapped}/{len(mutant_classes)}")

# Save
output = {
    "drug_mutant_map": {d: list(info['all_mutants']) for d, info in drug_mutant_map_all.items()},
    "orphan_drugs": orphan_drugs,
    "mutants_per_drug": {d: len(info['all_mutants']) for d, info in drug_mutant_map_all.items()}
}
with open(f"{OUT_DIR}/drug_mutant_mapping.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to {OUT_DIR}/drug_mutant_mapping.json")
