#!/usr/bin/env python3
"""Generate combined accuracy plot across all experiments.

Scans both local SSD and external HDD for accuracy.csv files,
combines all runs by case (1-4 plates), and creates a bar plot
showing crop, image, well accuracy with mean ± std.
"""

from __future__ import annotations

import os
import glob
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = SCRIPT_DIR
EXTERNAL_DIR = "/media/student/Data_HDD_12-TB/plate_fold_increasing"

EXPERIMENTS = [
    'valP1testP2',
    'valP2testP3', 
    'valP3testP4',
    'valP4testP5',
    'valP5testP6',
    'valP6testP1',
]


def get_experiment_dir(exp_name: str) -> Optional[str]:
    """Get experiment directory path from either local or external location."""
    local_path = os.path.join(LOCAL_DIR, exp_name)
    external_path = os.path.join(EXTERNAL_DIR, exp_name)
    
    if os.path.exists(local_path):
        return local_path
    elif os.path.exists(external_path):
        return external_path
    return None


def scan_accuracy_csv_files() -> list[dict]:
    """Scan all experiments for accuracy.csv files."""
    all_files = []
    
    for exp_name in EXPERIMENTS:
        exp_dir = get_experiment_dir(exp_name)
        if exp_dir is None:
            print(f"WARNING: Experiment not found: {exp_name}")
            continue
        
        pattern = os.path.join(exp_dir, 'case_*', 'run_*', 'accuracy.csv')
        files = glob.glob(pattern)
        
        for f in files:
            rel_path = os.path.relpath(f, exp_dir)
            parts = rel_path.split(os.sep)
            case_name = parts[0]
            run_name = parts[1]
            
            all_files.append({
                'path': f,
                'experiment': exp_name,
                'case_name': case_name,
                'run_name': run_name,
            })
    
    return all_files


def main() -> None:
    print("Scanning for accuracy.csv files...")
    files = scan_accuracy_csv_files()
    print(f"Found {len(files)} accuracy.csv files")
    
    if len(files) == 0:
        print("No files found!")
        return
    
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f['path'])
            df['experiment'] = f['experiment']
            df['case_name'] = f['case_name']
            df['run_name'] = f['run_name']
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {f['path']}: {e}")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nLoaded {len(combined_df)} runs")
    
    summary = []
    for case_num in [1, 2, 3, 4]:
        case_df = combined_df[combined_df['case'] == case_num]
        n_runs = len(case_df)
        
        if n_runs == 0:
            continue
            
        summary.append({
            'case': case_num,
            'n_runs': n_runs,
            'crop_mean': case_df['crop_acc'].mean(),
            'crop_std': case_df['crop_acc'].std() if n_runs > 1 else 0,
            'image_mean': case_df['image_acc'].mean(),
            'image_std': case_df['image_acc'].std() if n_runs > 1 else 0,
            'well_mean': case_df['well_acc'].mean(),
            'well_std': case_df['well_acc'].std() if n_runs > 1 else 0,
        })
    
    summary_df = pd.DataFrame(summary)
    print("\n=== SUMMARY ===")
    for _, row in summary_df.iterrows():
        print(f"Case {int(row['case'])} ({int(row['n_runs'])} runs): "
              f"Crop={row['crop_mean']:.2f}±{row['crop_std']:.2f}, "
              f"Image={row['image_mean']:.2f}±{row['image_std']:.2f}, "
              f"Well={row['well_mean']:.2f}±{row['well_std']:.2f}")
    
    cases = summary_df['case'].values
    x = np.arange(len(cases))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width, summary_df['crop_mean'], width, yerr=summary_df['crop_std'],
           label='Crop', color='#2ecc71', capsize=5, alpha=0.85, ecolor='black', linewidth=1.5)
    ax.bar(x, summary_df['image_mean'], width, yerr=summary_df['image_std'],
           label='Image', color='#3498db', capsize=5, alpha=0.85, ecolor='black', linewidth=1.5)
    ax.bar(x + width, summary_df['well_mean'], width, yerr=summary_df['well_std'],
           label='Well', color='#e74c3c', capsize=5, alpha=0.85, ecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Number of Training Plates', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Accuracy by Training Plate Count\n(Mean ± Std across 6 experiments)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(c)} plate{"s" if c > 1 else ""}' for c in cases], fontsize=12)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(summary_df[['crop_mean', 'image_mean', 'well_mean']].max()) + 10)
    
    plt.tight_layout()
    
    output_path = os.path.join(SCRIPT_DIR, 'accuracy_combined_by_case.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    
    csv_path = os.path.join(SCRIPT_DIR, 'accuracy_combined_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == '__main__':
    main()