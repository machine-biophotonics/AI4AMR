#!/usr/bin/env python3
"""
Derive MOA->Pathway shared group mapping from existing biological knowledge.

Chain of derivation:
  MOA -(contains)-> antibiotic -(known_target)-> gene -(belongs_to)-> Pathway

Output: group_mapping.json  (used by GroupCLIP training)
"""

import json, re

# ============================================================
# SOURCE DATA
# ============================================================

# Drug MOA groups (from generate_drug_confusion.py)
MOA_GROUPS = {
    "Cell wall (PBP 2)": ["Avibactam", "Clavulanic_Acid", "Meropenem", "Mecillinam", "Relebactam"],
    "Cell wall (PBP 3)": ["Aztreonam", "Ceftriaxone", "Cefepim"],
    "Cell wall (PBP 1)": ["Sulbactam", "Penicillin", "Cefsulodin"],
    "Ribosome": ["Doxicyclin", "Chloramphenicol", "Clarithromycin", "Kanamycin"],
    "Gyrase": ["Ciprofloxacin", "Norfloxacin", "Levofloxacin"],
    "Membrane integrity": ["Polymyxin_B", "Colistin"],
    "RNA polymerase": ["Rifampicin"],
    "DNA synthesis": ["Trimethoprim"],
}

# Gene -> Pathway mapping (from generate_mutant_confusion.py / phase_a_analysis.py)
GENE_TO_PATHWAY = {
    'folP': 'Folic acid biosynthesis', 'folA': 'Folic acid biosynthesis',
    'secY': 'Protein transport', 'secA': 'Protein transport',
    'rpoB': 'Transcription elongation', 'rpoA': 'Transcription elongation',
    'lptC': 'Cell envelope organization', 'lptA': 'Cell envelope organization',
    'msbA': 'Cell envelope organization',
    'ftsZ': 'Division septum assembly',
    'rplC': 'Translation initiation', 'rplA': 'Translation initiation',
    'rpsA': 'Translation initiation', 'rpsL': 'Translation initiation',
    'murC': 'Aminoglycan biosynthesis', 'murA': 'Aminoglycan biosynthesis',
    'mrcB': 'Aminoglycan biosynthesis',
    'mrdA': 'Cell shape regulation', 'mrcA': 'Cell shape regulation', 'ftsI': 'Cell shape regulation',
    'lpxC': 'Lipid A biosynthesis', 'lpxA': 'Lipid A biosynthesis',
    'gyrB': 'Chromosome organization', 'gyrA': 'Chromosome organization',
    'dnaB': 'Chromosome organization', 'parE': 'Chromosome organization',
    'parC': 'Chromosome organization', 'dnaE': 'Chromosome organization',
}

# Known drug -> target gene matches (from phase_a_analysis.py)
EXPECTED_MATCHES = {
    'Cefsulodin': {'mrcA', 'mrcB'}, 'Penicillin': {'mrcA', 'mrcB', 'ftsI'},
    'Sulbactam': {'mrcA', 'mrcB', 'ftsI'}, 'Mecillinam': {'mrdA'},
    'Meropenem': {'mrdA', 'ftsI', 'mrcA', 'mrcB'}, 'Aztreonam': {'ftsI'},
    'Cefepim': {'ftsI', 'mrcA', 'mrcB', 'mrdA'}, 'Ceftriaxone': {'ftsI', 'mrcA', 'mrcB'},
    'Chloramphenicol': {'rplA', 'rplC'}, 'Clarithromycin': {'rplA', 'rplC'},
    'Doxicyclin': {'rpsA', 'rpsL'}, 'Kanamycin': {'rpsA', 'rpsL'},
    'Ciprofloxacin': {'gyrA', 'gyrB', 'parC', 'parE'},
    'Levofloxacin': {'gyrA', 'gyrB', 'parC', 'parE'},
    'Norfloxacin': {'gyrA', 'gyrB', 'parC', 'parE'},
    'Rifampicin': {'rpoA', 'rpoB'}, 'Trimethoprim': {'folA', 'folP'},
    'Colistin': {'lpxA', 'lpxC', 'lptA', 'lptC'},
    'Polymyxin_B': {'lpxA', 'lpxC', 'lptA', 'lptC'},
}

# ============================================================
# STEP 1: Antibiotic -> MOA lookup
# ============================================================

ANTIBIOTIC_TO_MOA = {}
for moa, drugs in MOA_GROUPS.items():
    for d in drugs:
        ANTIBIOTIC_TO_MOA[d] = moa

# ============================================================
# STEP 2: Derive MOA -> Pathway via antibiotic -> gene bridge
# ============================================================

# For each MOA, collect all pathways reachable through known drug targets
moa_to_pathways = {}
for drug_name in sorted(EXPECTED_MATCHES.keys()):
    moa = ANTIBIOTIC_TO_MOA.get(drug_name)
    if moa is None:
        print(f"  WARNING: {drug_name} has no MOA mapping, skipping")
        continue
    if moa not in moa_to_pathways:
        moa_to_pathways[moa] = set()
    target_genes = EXPECTED_MATCHES[drug_name]
    for gene in target_genes:
        pathway = GENE_TO_PATHWAY.get(gene)
        if pathway:
            moa_to_pathways[moa].add(pathway)
        else:
            print(f"  WARNING: gene {gene} (target of {drug_name}) has no pathway mapping")

print("\n" + "="*80)
print("STEP 2: MOA -> Pathway derivation")
print("="*80)
for moa in sorted(moa_to_pathways.keys()):
    paths = sorted(moa_to_pathways[moa])
    print(f"  {moa:28s} -> {paths}")

# ============================================================
# STEP 3: Build reverse lookup: Pathway -> MOA
# ============================================================

pathway_to_moas = {}
for moa, paths in moa_to_pathways.items():
    for p in paths:
        if p not in pathway_to_moas:
            pathway_to_moas[p] = set()
        pathway_to_moas[p].add(moa)

print("\n" + "="*80)
print("STEP 3: Pathway -> MOA (reverse lookup)")
print("="*80)
for p in sorted(pathway_to_moas.keys()):
    moas = sorted(pathway_to_moas[p])
    print(f"  {p:35s} <- {moas}")

# ============================================================
# STEP 4: Define shared GroupCLIP groups
# ============================================================
# A shared group = connected component between MOA and Pathway

L3_GROUPS = [
    {
        "name": "Gyrase_DNA",
        "group_id": 0,
        "moas": ["Gyrase"],
        "pathways": ["Chromosome organization"],
    },
    {
        "name": "Ribosome_Translation",
        "group_id": 1,
        "moas": ["Ribosome"],
        "pathways": ["Translation initiation"],
    },
    {
        "name": "Cell_Wall",
        "group_id": 2,
        "moas": ["Cell wall (PBP 1)", "Cell wall (PBP 2)", "Cell wall (PBP 3)"],
        "pathways": ["Aminoglycan biosynthesis", "Cell shape regulation"],
    },
    {
        "name": "Membrane",
        "group_id": 3,
        "moas": ["Membrane integrity"],
        "pathways": ["Cell envelope organization", "Lipid A biosynthesis"],
    },
    {
        "name": "RNA_Polymerase",
        "group_id": 4,
        "moas": ["RNA polymerase"],
        "pathways": ["Transcription elongation"],
    },
    {
        "name": "DNA_Synthesis_Folate",
        "group_id": 5,
        "moas": ["DNA synthesis"],
        "pathways": ["Folic acid biosynthesis"],
    },
    {
        "name": "Control_WT",
        "group_id": 6,
        "moas": ["Control"],
        "pathways": ["WT/NC"],
    },
    {
        "name": "Protein_Transport",
        "group_id": 7,
        "moas": [],
        "pathways": ["Protein transport"],
    },
    {
        "name": "Division_Septum",
        "group_id": 8,
        "moas": [],
        "pathways": ["Division septum assembly"],
    },
]

# Build lookup tables
MOA_TO_GROUP = {}
PATHWAY_TO_GROUP = {}
for g in L3_GROUPS:
    for m in g["moas"]:
        MOA_TO_GROUP[m] = g["group_id"]
    for p in g["pathways"]:
        PATHWAY_TO_GROUP[p] = g["group_id"]

# ============================================================
# STEP 5: Build class_name -> group_id mapping
# ============================================================

# For drugs: class name is like "Ciprofloxacin_2x" or "control"
def get_group_from_drug_label(label):
    if label == 'control':
        return MOA_TO_GROUP.get("Control", -1)
    parts = label.rsplit('_', 1)
    if len(parts) == 2 and parts[1].replace('x', '').replace('.', '').isdigit():
        ab_name = parts[0]
        moa = ANTIBIOTIC_TO_MOA.get(ab_name)
        if moa and moa in MOA_TO_GROUP:
            return MOA_TO_GROUP[moa]
    return -1

# For mutants: class name is like "gyrA_1" or "dnaB_3" or "WT NC_1"
def get_group_from_mutant_label(label):
    if label.upper().startswith('WT') or label.upper().startswith('NC'):
        return MOA_TO_GROUP.get("Control", -1)
    parts = label.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        base_gene = parts[0]
    else:
        base_gene = label
    pathway = GENE_TO_PATHWAY.get(base_gene)
    if pathway and pathway in PATHWAY_TO_GROUP:
        return PATHWAY_TO_GROUP[pathway]
    return -1

# ============================================================
# STEP 6: Save mapping JSON
# ============================================================

mapping = {
    "L3_GROUPS": L3_GROUPS,
    "MOA_TO_GROUP": MOA_TO_GROUP,
    "PATHWAY_TO_GROUP": PATHWAY_TO_GROUP,
    "ANTIBIOTIC_TO_MOA": ANTIBIOTIC_TO_MOA,
    "GENE_TO_PATHWAY": GENE_TO_PATHWAY,
    "EXPECTED_MATCHES": {k: list(v) for k, v in EXPECTED_MATCHES.items()},
}

output_path = "group_mapping.json"
with open(output_path, "w") as f:
    json.dump(mapping, f, indent=2)

print(f"\nSaved mapping to {output_path}")
print()

# ============================================================
# COVERAGE CHECK
# ============================================================

# Load actual drug and mutant labels from JSONs
with open("plate_well_ic50_mapping.json") as f:
    ic50_data = json.load(f)
with open("plate_well_id_path.json") as f:
    mutant_data = json.load(f)

# All unique drug class names
all_drug_classes = set()
for plate, wells in ic50_data.items():
    for well, info in wells.items():
        ab = info.get('antibiotic', '')
        ic = info.get('ic50_multiple', '')
        if ab and ic:
            if ic == 'control':
                all_drug_classes.add('control')
            else:
                ic_str = ic if 'x' in ic else f"{ic}x"
                ab_clean = ab.replace(' ', '_')
                all_drug_classes.add(f"{ab_clean}_{ic_str}")

# All unique mutant class names (guide-level)
all_mutant_classes = set()
for plate, rows in mutant_data.items():
    for row, cols in rows.items():
        for col, info in cols.items():
            if 'id' in info and info['id']:
                all_mutant_classes.add(info['id'])

print("="*80)
print("COVERAGE CHECK")
print("="*80)

drug_mapped = sum(1 for c in all_drug_classes if get_group_from_drug_label(c) >= 0)
drug_total = len(all_drug_classes)
print(f"  Drugs:   {drug_mapped}/{drug_total} mapped ({100*drug_mapped/drug_total:.0f}%)")

mut_mapped = sum(1 for c in all_mutant_classes if get_group_from_mutant_label(c) >= 0)
mut_total = len(all_mutant_classes)
print(f"  Mutants: {mut_mapped}/{mut_total} mapped ({100*mut_mapped/mut_total:.0f}%)")

# Show unmapped if any
drug_unmapped = sorted([c for c in all_drug_classes if get_group_from_drug_label(c) < 0])
mut_unmapped = sorted([c for c in all_mutant_classes if get_group_from_mutant_label(c) < 0])
if drug_unmapped:
    print(f"  Unmapped drugs: {drug_unmapped}")
if mut_unmapped:
    print(f"  Unmapped mutants: {mut_unmapped}")

# ============================================================
# SUMMARY TABLE
# ============================================================
print()
print("="*80)
print("FINAL: Shared Group Labels for GroupCLIP")
print("="*80)

print(f"  {'Group ID':8s} {'Group Name':25s} {'MOAs':35s} {'Pathways':40s}")
print(f"  {'-'*8} {'-'*25} {'-'*35} {'-'*40}")
for g in L3_GROUPS:
    gid = g["group_id"]
    name = g["name"]
    moas = ', '.join(g["moas"]) if g["moas"] else '(mutant-only)'
    paths = ', '.join(g["pathways"])
    print(f"  {gid:<8d} {name:25s} {moas:35s} {paths:40s}")

print(f"\n  -1 (Unknown): samples with no known MOA or Pathway mapping")
print(f"                excluded from contrastive loss, still trained via classification")
