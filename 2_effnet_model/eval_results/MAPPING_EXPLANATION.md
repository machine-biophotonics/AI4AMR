# Classification Hierarchy and Mappings

## Overview

This document explains the hierarchical classification system used in the CRISPRi screening analysis, from the most granular (individual guide RNAs) to the broadest (biological pathways).

---

## Level 1: Full 85 Classes (Guide-Level with Subgroups)

**Total: 85 classes**

- **WT (Wild Type)**: Control wells with no gene knockdown
- **28 Genes × 3 Guides each**: Each gene has 3 different guide RNAs (sgRNA1, sgRNA2, sgRNA3)

Example: `mrcA_1`, `mrcA_2`, `mrcA_3` = mrcA gene with 3 different guide RNAs

### Complete Gene List (28 genes):
| Gene | Full Name |
|------|-----------|
| mrcA | Murein transglycosylase A |
| mrcB | Murein transglycosylase B |
| mrdA | Penicillin-binding protein 2 |
| ftsI | Penicillin-binding protein 3 |
| murA | UDP-N-acetylglucosamine enolpyruvyl transferase |
| murC | UDP-N-acetylmuramate-L-alanine ligase |
| lpxA | UDP-N-acetylglucosamine acyltransferase |
| lpxC | UDP-3-O-acyl-N-acetylglucosamine deacetylase |
| lptA | Lipopolysaccharide transport protein A |
| lptC | Lipopolysaccharide transport protein C |
| msbA | ABC transporter |
| gyrA | DNA gyrase subunit A |
| gyrB | DNA gyrase subunit B |
| parC | DNA topoisomerase IV subunit A |
| parE | DNA topoisomerase IV subunit B |
| dnaB | DNA replication protein |
| dnaE | DNA polymerase III subunit alpha |
| rpoA | RNA polymerase subunit alpha |
| rpoB | RNA polymerase subunit beta |
| rpsA | 30S ribosomal protein S1 |
| rpsL | 30S ribosomal protein S12 |
| rplA | 50S ribosomal protein L1 |
| rplC | 50S ribosomal protein L3 |
| folA | Dihydrofolate reductase |
| folP | Dihydropteroate synthase |
| secY | Protein translocase subunit |
| secA | Protein translocase ATPase |
| ftsZ | Cell division protein |

---

## Level 2: Guide-Level (Gene-Level)

**Total: 29 classes** (WT + 28 genes)

The 3 guide RNAs for each gene are combined into a single class:
- `mrcA_1`, `mrcA_2`, `mrcA_3` → `mrcA`
- `mrcB_1`, `mrcB_2`, `mrcB_3` → `mrcB`
- etc.

---

## Level 3: Family-Level

**Total: 16 classes**

Genes are grouped by protein families/functional groups:

| Family | Genes | Description |
|--------|-------|-------------|
| mrc | mrcA, mrcB | Murein transglycosylases |
| mrd | mrdA | Penicillin-binding proteins |
| mur | murA, murC | Peptidoglycan synthesis |
| fts | ftsI, ftsZ | Cell division |
| lpx | lpxA, lpxC | LPS biosynthesis (early) |
| lpt | lptA, lptC | LPS transport |
| msbA | msbA | LPS transport (ABC transporter) |
| gyr | gyrA, gyrB | DNA gyrase |
| par | parC, parE | Topoisomerase IV |
| dna | dnaB, dnaE | DNA replication |
| rpo | rpoA, rpoB | RNA polymerase |
| rpl | rplA, rplC | Ribosomal proteins (large) |
| rps | rpsA, rpsL | Ribosomal proteins (small) |
| fol | folA, folP | Folate metabolism |
| sec | secA, secY | Protein secretion |
| WT | WT | Wild type control |

---

## Level 4: Pathway-Level (Biological Process)

**Total: 7 classes**

Genes are grouped by their biological pathway/function:

| Pathway | Genes | Description |
|---------|-------|-------------|
| Cell wall | mrcA, mrcB, mrdA, ftsI, murA, murC | Peptidoglycan synthesis and cell wall construction |
| LPS | lpxA, lpxC, lptA, lptC, msbA | Lipopolysaccharide synthesis and transport |
| DNA | gyrA, gyrB, parC, parE, dnaB, dnaE | DNA replication, topology, repair |
| Transcription | rpoA, rpoB, rplA, rplC, rpsA, rpsL | Transcription and translation machinery |
| Metabolism | folA, folP, secA, secY | Folate metabolism and protein export |
| Cell division | ftsZ | Cell division machinery |
| WT | WT | Wild type control |

---

## Confusion Matrix Naming Convention

Files are named following this pattern:
```
confusion_matrix_[LEVEL]_[VOTING]_[CLASSES].png
```

### Level Codes:
- `01`: Crop level (no voting) - individual crop predictions
- `02`: Image level (majority vote of 144 crops per image)
- `03`: Well level (majority vote of all images per well)

### Class Codes:
- `all_85`: Full 85 classes (WT + 28 genes × 3 guides)
- `guide`: 29 classes (WT + 28 genes)
- `family`: 16 classes (family groupings)
- `pathway`: 7 classes (biological pathway groupings)

### Example:
- `confusion_matrix_02_image_level_guide.png` = Image-level majority voting, 29 gene classes

---

## Accuracy Improvement with Majority Voting

| Level | 85 Classes | Guide (29) | Family (16) | Pathway (7) |
|-------|------------|------------|-------------|-------------|
| Crop (no vote) | 18.02% | 35.82% | 43.98% | 55.23% |
| Image (vote) | 23.41% | 43.75% | 52.18% | 62.75% |
| Well (vote) | 29.17% | 51.04% | 57.29% | 67.71% |

### Improvement:
- Crop → Image: +5.39% to +8.20% depending on class level
- Crop → Well: +11.15% to +15.22% depending on class level
- Image → Well: +4.96% to +7.29% depending on class level
