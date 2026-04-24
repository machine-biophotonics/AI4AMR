#!/usr/bin/env python3
"""Generate summary bar plot from existing accuracy.csv files."""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Find all accuracy.csv files
csv_files = glob.glob(os.path.join(SCRIPT_DIR, 'case_*/run_*/accuracy.csv'))
print(f"Found {len(csv_files)} accuracy.csv files")

if len(csv_files) == 0:
    print("No accuracy.csv files found!")
    exit(1)

# Load all accuracy data
all_data = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    # Extract case and run from path
    path_parts = csv_file.split('/')
    case_dir = path_parts[-3]  # case_N_XXX
    run_dir = path_parts[-2]   # run_N
    df['case_name'] = case_dir
    df['run_name'] = run_dir
    all_data.append(df)
    print(f"Loaded: {case_dir}/{run_dir}")

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
print(f"{'Case':<6} {'n_runs':<8} {'Crop %':<20} {'Image %':<20} {'Well %':<20}")
print("-" * 76)
for _, row in summary_df.iterrows():
    crop_str = f"{row['crop_acc_mean']:.2f} ± {row['crop_acc_std']:.2f}"
    image_str = f"{row['image_acc_mean']:.2f} ± {row['image_acc_std']:.2f}"
    well_str = f"{row['well_acc_mean']:.2f} ± {row['well_acc_std']:.2f}"
    print(f"{int(row['case']):<6} {int(row['n_runs']):<8} {crop_str:<20} {image_str:<20} {well_str:<20}")

# Create bar plot
cases = summary_df['case'].values
x = np.arange(len(cases))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))

# Crop bar
crop_bars = ax.bar(x - width, summary_df['crop_acc_mean'], width, 
                 yerr=summary_df['crop_acc_std'], label='Crop', 
                 color='#2ecc71', capsize=5, alpha=0.85)

# Image bar
image_bars = ax.bar(x, summary_df['image_acc_mean'], width,
                   yerr=summary_df['image_acc_std'], label='Image',
                   color='#3498db', capsize=5, alpha=0.85)

# Well bar
well_bars = ax.bar(x + width, summary_df['well_acc_mean'], width,
                  yerr=summary_df['well_acc_std'], label='Well',
                  color='#e74c3c', capsize=5, alpha=0.85)

ax.set_xlabel('Case (Number of Training Plates)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Accuracy by Case (Crop/Image/Well Level, mean ± std)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'Case {c}' for c in cases])
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(summary_df[['crop_acc_mean', 'image_acc_mean', 'well_acc_mean']].max() * 1.4))

# Add value labels
for bars in [crop_bars, image_bars, well_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 3),
                  textcoords="offset points",
                  ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# Save plot
plot_file = os.path.join(SCRIPT_DIR, 'accuracy_summary_barplot.png')
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"\nSaved: {plot_file}")

# Save summary CSV
summary_file = os.path.join(SCRIPT_DIR, 'accuracy_summary.csv')
summary_df.to_csv(summary_file, index=False)
print(f"Saved: {summary_file}")

# Save per-run results
results_file = os.path.join(SCRIPT_DIR, 'accuracy_all_runs.csv')
results_df.to_csv(results_file, index=False)
print(f"Saved: {results_file}")