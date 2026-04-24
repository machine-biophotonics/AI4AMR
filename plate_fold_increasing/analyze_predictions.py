#!/usr/bin/env python3
"""Generate accuracy analysis from crop predictions."""

import os
import argparse
import glob
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Generate accuracy analysis')
parser.add_argument('--case', type=int, choices=[1,2,3,4], help='Case number (1-4)')
parser.add_argument('--run', type=int, choices=[0,1,2,3], help='Run number (0-3)')
parser.add_argument('--all', action='store_true', help='Run all cases for current val/test')
parser.add_argument('--all_experiments', action='store_true', help='Run all 6 val/test experiments')
parser.add_argument('--val_plate', type=str, default='P5', help='Validation plate (P1-P6)')
parser.add_argument('--test_plate', type=str, default='P6', help='Test plate (P1-P6)')
args = parser.parse_args()

def aggregate_crop_to_image(df):
    image_results = []
    for img_name, group in df.groupby('image_name'):
        true_label = group['ground_truth_label'].iloc[0]
        pred_counts = Counter(group['predicted_class_name'].values)
        majority_pred = pred_counts.most_common(1)[0][0]
        well = group['well'].iloc[0] if 'well' in group.columns else None
        image_results.append({'image_name': img_name, 'well': well, 'true_label': true_label, 'pred_majority': majority_pred})
    return pd.DataFrame(image_results)

def aggregate_image_to_well(image_df):
    well_results = []
    for well, group in image_df.groupby('well'):
        if pd.isna(well):
            continue
        true_label = group['true_label'].iloc[0]
        pred_counts = Counter(group['pred_majority'].values)
        majority_pred = pred_counts.most_common(1)[0][0]
        well_results.append({'well': well, 'true_label': true_label, 'pred_majority': majority_pred})
    return pd.DataFrame(well_results)

def compute_accuracy(df, true_col, pred_col):
    if len(df) == 0:
        return 0.0
    return 100.0 * (df[true_col] == df[pred_col]).sum() / len(df)

# Get accuracy.csv files based on args
exp_name = f"val{args.val_plate}{args.test_plate}"
case_combo_names = {
    1: ['P1', 'P2', 'P3', 'P4'],
    2: ['P1P2', 'P2P3', 'P3P4', 'P4P1'],
    3: ['P1P2P3', 'P2P3P4', 'P3P4P1', 'P4P1P2'],
    4: ['P1P2P3P4', 'P1P2P3P4', 'P1P2P3P4', 'P1P2P3P4'],
}

csv_files = []

if args.all_experiments:
    # Find all accuracy.csv files for all 6 experiments
    for exp_n in ['val1_test2', 'val2_test3', 'val3_test4', 'val4_test5', 'val5_test6', 'val6_test1']:
        for case_num in [1, 2, 3, 4]:
            combo_names = case_combo_names[case_num]
            for run_idx in range(4):
                csv_path = os.path.join(SCRIPT_DIR, f'case_{case_num}_{combo_names[run_idx]}_{exp_n}', f'run_{run_idx}', 'accuracy.csv')
                if os.path.exists(csv_path):
                    csv_files.append((csv_path, exp_n, case_num, run_idx))
elif args.all:
    # All cases and runs for given val/test
    for case_num in [1, 2, 3, 4]:
        combo_names = case_combo_names[case_num]
        for run_idx in range(4):
            csv_path = os.path.join(SCRIPT_DIR, f'case_{case_num}_{combo_names[run_idx]}_{exp_name}', f'run_{run_idx}', 'accuracy.csv')
            if os.path.exists(csv_path):
                csv_files.append((csv_path, exp_name, case_num, run_idx))
elif args.case is not None:
    if args.run is not None:
        csv_path = os.path.join(SCRIPT_DIR, f'case_{args.case}_{case_combo_names[args.case][args.run]}_{exp_name}', f'run_{args.run}', 'accuracy.csv')
        if os.path.exists(csv_path):
            csv_files.append((csv_path, exp_name, args.case, args.run))
    else:
        for run_idx in range(4):
            csv_path = os.path.join(SCRIPT_DIR, f'case_{args.case}_{case_combo_names[args.case][run_idx]}_{exp_name}', f'run_{run_idx}', 'accuracy.csv')
            if os.path.exists(csv_path):
                csv_files.append((csv_path, exp_name, args.case, run_idx))
else:
    print("Use: --case N --run N, --case N, --all, or --all_experiments")
    exit(1)

print(f"Found {len(csv_files)} accuracy.csv files")

if len(csv_files) == 0:
    print("No accuracy.csv files found!")
    exit(1)

# Load all accuracy data
all_data = []
for csv_path, exp_n, case_num, run_idx in csv_files:
    df = pd.read_csv(csv_path)
    df['experiment'] = exp_n
    all_data.append(df)

results_df = pd.concat(all_data, ignore_index=True)

# Group by case and compute mean ± std
summary = []
for case_num in sorted(results_df['case'].unique()):
    case_df = results_df[results_df['case'] == case_num]
    n_runs = len(case_df)
    summary.append({
        'case': case_num,
        'n_runs': n_runs,
        'crop_acc_mean': case_df['crop_acc'].mean(),
        'crop_acc_std': case_df['crop_acc'].std() if n_runs > 1 else 0,
        'image_acc_mean': case_df['image_acc'].mean(),
        'image_acc_std': case_df['image_acc'].std() if n_runs > 1 else 0,
        'well_acc_mean': case_df['well_acc'].mean(),
        'well_acc_std': case_df['well_acc'].std() if n_runs > 1 else 0,
    })

summary_df = pd.DataFrame(summary)

# Print summary
print("\n=== ACCURACY BY CASE (mean ± std) ===")
for _, row in summary_df.iterrows():
    print(f"Case {int(row['case'])}: Crop={row['crop_acc_mean']:.2f}±{row['crop_acc_std']:.2f}, Image={row['image_acc_mean']:.2f}±{row['image_acc_std']:.2f}, Well={row['well_acc_mean']:.2f}±{row['well_acc_std']:.2f}")

# Create bar plot
cases = summary_df['case'].values
x = np.arange(len(cases))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, summary_df['crop_acc_mean'], width, yerr=summary_df['crop_acc_std'], label='Crop', color='#2ecc71', capsize=4, alpha=0.85)
ax.bar(x, summary_df['image_acc_mean'], width, yerr=summary_df['image_acc_std'], label='Image', color='#3498db', capsize=4, alpha=0.85)
ax.bar(x + width, summary_df['well_acc_mean'], width, yerr=summary_df['well_acc_std'], label='Well', color='#e74c3c', capsize=4, alpha=0.85)

ax.set_xlabel('Case', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title(f'Accuracy: {exp_name}', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Case {c}' for c in cases])
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, f'accuracy_summary_{exp_name}.png'), dpi=150)
print(f"Saved: accuracy_summary_{exp_name}.png")

# Save summary CSV
summary_df.to_csv(os.path.join(SCRIPT_DIR, f'accuracy_summary_{exp_name}.csv'), index=False)
print(f"Saved: accuracy_summary_{exp_name}.csv")