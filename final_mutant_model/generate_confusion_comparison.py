#!/usr/bin/env python3
"""
Generate side-by-side confusion matrix comparison between trial_daniel and our model
for each dose level (with DMSO included = 168 + 462 = 630 samples per dose)

Run this script on the remote machine in final_mutant_model folder:
    python3 ../generate_confusion_comparison.py
"""
import numpy as np
import pandas as pd
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')

BASE = "/media/student/Data_SSD_1-TB/2025_12_19 CRISPRi Reference Plate Imaging"
TD_PATH = BASE + "/trial_daniel/models/train_1-4_val_5_test_6/Plate_6_260501_2136"
OUR_PATH = BASE + "/final_mutant_model/fold_Plate_6"
PARAMS_PATH = BASE + "/trial_daniel/analysis_results/E_coli_params"
OUTPUT_PATH = BASE + "/trial_daniel/analysis_results/confusion_matrices_per_dose"

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("Loading data...")
with open(PARAMS_PATH + "/classes.json", "r") as f:
    classes_list = json.load(f)
classes = {str(i): c for i, c in enumerate(classes_list)}

td_preds = np.genfromtxt(TD_PATH + "/preds.txt", dtype=float).astype(int)
td_labels = np.genfromtxt(TD_PATH + "/labels.txt", dtype=float).astype(int)

td_pred_drugs = []
td_true_drugs = []
td_pred_doses = []
td_true_doses = []

for l in td_labels:
    cls = classes[str(l)]
    if "xIC50" in cls:
        m = re.match(r"(.*?)_(0\.\d+x|1x)IC50", cls)
        if m:
            td_true_drugs.append(m.group(1))
            td_true_doses.append(m.group(2))
        else:
            td_true_drugs.append(cls)
            td_true_doses.append("DMSO")
    else:
        td_true_drugs.append(cls)
        td_true_doses.append("DMSO")

for p in td_preds:
    cls = classes[str(p)]
    if "xIC50" in cls:
        m = re.match(r"(.*?)_(0\.\d+x|1x)IC50", cls)
        if m:
            td_pred_drugs.append(m.group(1))
            td_pred_doses.append(m.group(2))
        else:
            td_pred_drugs.append(cls)
            td_pred_doses.append("DMSO")
    else:
        td_pred_drugs.append(cls)
        td_pred_doses.append("DMSO")

our_df = pd.read_csv(OUR_PATH + "/predictions_drug_per_image_best_model_acc.csv")

OUR_TO_TD = {
    "Cefepim": "Cefepime", 
    "Penicillin": "PenicillinG", 
    "Doxicyclin": "Doxycycline",
    "Polymyxin_B": "PolymyxinB", 
    "Clavulanic_Acid": "Clavulanate", 
    "DMSO_control": "DMSO"
}

def get_drug(label):
    if label == "DMSO_control":
        return "DMSO"
    for d in ["_0.25x", "_0.5x", "_1x", "_2x"]:
        if label.endswith(d):
            return label[:-len(d)]
    return label

def get_dose(label):
    if label == "DMSO_control":
        return "DMSO"
    for d in ["_0.25x", "_0.5x", "_1x", "_2x"]:
        if label.endswith(d):
            return d.replace("_", "")
    return "DMSO"

our_df["drug"] = our_df["ground_truth"].apply(get_drug)
our_df["dose"] = our_df["ground_truth"].apply(get_dose)
our_df["drug_mapped"] = our_df["drug"].map(OUR_TO_TD).fillna(our_df["drug"])
our_df["pred_drug"] = our_df["predicted_class"].apply(get_drug)
our_df["pred_drug_mapped"] = our_df["pred_drug"].map(OUR_TO_TD).fillna(our_df["pred_drug"])

all_drugs = sorted(set(td_true_drugs + list(our_df["drug_mapped"].unique())))
print("Unique drugs:", len(all_drugs))

dose_pairs = [("0.25x", "0.125x"), ("0.5x", "0.25x"), ("1x", "0.5x"), ("2x", "1x")]

# Create 4x2 comparison figure (4 doses, 2 columns)
fig, axes = plt.subplots(4, 2, figsize=(16, 32))
plt.subplots_adjust(hspace=0.3)

for idx, (our_d, td_d) in enumerate(dose_pairs):
    # trial_daniel (left column)
    td_mask = (np.array(td_true_doses) == td_d) | (np.array(td_true_doses) == "DMSO")
    if td_mask.sum() > 0:
        td_cm = confusion_matrix(np.array(td_true_drugs)[td_mask], np.array(td_pred_drugs)[td_mask], labels=all_drugs)
        td_acc = 100 * np.mean(np.array(td_pred_drugs)[td_mask] == np.array(td_true_drugs)[td_mask])
        im1 = axes[idx, 0].imshow(td_cm, cmap="Blues", aspect="auto")
        axes[idx, 0].set_title("trial_daniel: " + our_d + " + DMSO (n=" + str(td_mask.sum()) + "), Acc=" + str(round(td_acc,1)) + "%", fontsize=10)
    axes[idx, 0].set_xticks([])
    axes[idx, 0].set_yticks([])
    
    # Our model (right column)
    our_mask = (our_df["dose"] == our_d) | (our_df["dose"] == "DMSO")
    if our_mask.sum() > 0:
        our_cm = confusion_matrix(our_df[our_mask]["drug_mapped"], our_df[our_mask]["pred_drug_mapped"], labels=all_drugs)
        our_acc = 100 * np.mean(our_df[our_mask]["drug_mapped"] == our_df[our_mask]["pred_drug_mapped"])
        im2 = axes[idx, 1].imshow(our_cm, cmap="Greens", aspect="auto")
        axes[idx, 1].set_title("Our model: " + our_d + " + DMSO (n=" + str(our_mask.sum()) + "), Acc=" + str(round(our_acc,1)) + "%", fontsize=10)
    axes[idx, 1].set_xticks([])
    axes[idx, 1].set_yticks([])

axes[0, 0].annotate("trial_daniel", xy=(0.5, 1.15), xycoords="axes fraction", ha="center", fontsize=12, fontweight="bold")
axes[0, 1].annotate("Our model", xy=(0.5, 1.15), xycoords="axes fraction", ha="center", fontsize=12, fontweight="bold")

plt.suptitle("Confusion Matrix Comparison: trial_daniel (left) vs Our model (right)", fontsize=14, y=0.99)
plt.savefig(OUTPUT_PATH + "/confusion_matrices_sidebyside.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: confusion_matrices_sidebyside.png")

# Print summary
print("\n=== ACCURACY SUMMARY (with DMSO included: 168 DMSO + 462 per dose = 630) ===")
for our_d, td_d in dose_pairs:
    td_mask = (np.array(td_true_doses) == td_d) | (np.array(td_true_doses) == "DMSO")
    our_mask = (our_df["dose"] == our_d) | (our_df["dose"] == "DMSO")
    td_acc = 100 * np.mean(np.array(td_pred_drugs)[td_mask] == np.array(td_true_drugs)[td_mask])
    our_acc = 100 * np.mean(our_df[our_mask]["drug_mapped"] == our_df[our_mask]["pred_drug_mapped"])
    print(our_d + " + DMSO: TD=" + str(round(td_acc,1)) + "% Ours=" + str(round(our_acc,1)) + "% (n=" + str(td_mask.sum()) + "/" + str(our_mask.sum()) + ")")

print("\nDone!")