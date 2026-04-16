#!/usr/bin/env python3
"""
Quantitative Comparison: EffNet vs DINO (Logistic Regression) MOA Discovery
"""

import numpy as np
import json
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = '/home/student/Desktop/2025_12_19 CRISPRi Reference Plate Imaging'

# Load cluster analyses
effnet_clusters = pd.read_csv(os.path.join(BASE_DIR, 'moa_k19', 'moa_cluster_analysis.csv'))
dino_clusters = pd.read_csv(os.path.join(BASE_DIR, 'dino_moa', 'moa_cluster_analysis.csv'))

# Load guide consistency
effnet_guide = pd.read_csv(os.path.join(BASE_DIR, 'moa_k19', 'guide_consistency.csv'))
dino_guide = pd.read_csv(os.path.join(BASE_DIR, 'dino_moa', 'guide_consistency.csv'))

print("="*70)
print("QUANTITATIVE COMPARISON: EffNet vs DINO (Logistic Regression)")
print("="*70)

# =============================================================================
# METRIC 1: Silhouette Score (from k optimization)
# =============================================================================
print("\n1. SILHOUETTE SCORES")
print("-" * 40)

# Silhouette scores from the analyses
effnet_sil = 0.2442  # From moa_k19 documentation
dino_sil = 0.3135   # From dino_moa run output

print(f"EffNet (k=19): {effnet_sil:.4f}")
print(f"DINO (k=5):    {dino_sil:.4f}")
print(f"Improvement:   {((dino_sil - effnet_sil) / effnet_sil * 100):+.1f}%")

# =============================================================================
# METRIC 2: Cluster Purity
# =============================================================================
print("\n2. CLUSTER PURITY ANALYSIS")
print("-" * 40)

def analyze_purity(df, name):
    high = len(df[df['Purity'] >= 0.8])
    medium = len(df[(df['Purity'] >= 0.5) & (df['Purity'] < 0.8)])
    low = len(df[df['Purity'] < 0.5])
    avg_purity = df['Purity'].mean()
    return {'high': high, 'medium': medium, 'low': low, 'avg': avg_purity}

effnet_purity = analyze_purity(effnet_clusters, 'EffNet')
dino_purity = analyze_purity(dino_clusters, 'DINO')

print(f"\n{'Metric':<25} {'EffNet (k=19)':<15} {'DINO (k=5)':<15}")
print("-" * 55)
print(f"{'High purity (>=0.8)':<25} {effnet_purity['high']:<15} {dino_purity['high']:<15}")
print(f"{'Medium purity (0.5-0.8)':<25} {effnet_purity['medium']:<15} {dino_purity['medium']:<15}")
print(f"{'Low purity (<0.5)':<25} {effnet_purity['low']:<15} {dino_purity['low']:<15}")
print(f"{'Average purity':<25} {effnet_purity['avg']:<15.4f} {dino_purity['avg']:<15.4f}")

purity_diff = ((effnet_purity['avg'] - dino_purity['avg']) / dino_purity['avg'] * 100)
print(f"\nEffNet avg purity is {purity_diff:+.1f}% vs DINO")

# =============================================================================
# METRIC 3: Guide RNA Consistency
# =============================================================================
print("\n3. GUIDE RNA CONSISTENCY")
print("-" * 40)

effnet_same = len(effnet_guide[effnet_guide['Consistency'] == 'Same'])
effnet_diff = len(effnet_guide[effnet_guide['Consistency'] == 'Different'])
effnet_consistency = effnet_same / (effnet_same + effnet_diff) * 100 if (effnet_same + effnet_diff) > 0 else 0

dino_same = len(dino_guide[dino_guide['Consistency'] == 'Same'])
dino_diff = len(dino_guide[dino_guide['Consistency'] == 'Different'])
dino_consistency = dino_same / (dino_same + dino_diff) * 100 if (dino_same + dino_diff) > 0 else 0

print(f"\n{'Metric':<25} {'EffNet (k=19)':<15} {'DINO (k=5)':<15}")
print("-" * 55)
print(f"{'Genes with same cluster':<25} {effnet_same:<15} {dino_same:<15}")
print(f"{'Genes diff clusters':<25} {effnet_diff:<15} {dino_diff:<15}")
print(f"{'Consistency rate':<25} {effnet_consistency:<14.1f}% {dino_consistency:<14.1f}%")

consistency_diff = effnet_consistency - dino_consistency
print(f"\nEffNet guide consistency is {consistency_diff:+.1f}% points vs DINO")

# =============================================================================
# METRIC 4: Pathway Coverage per Cluster
# =============================================================================
print("\n4. PATHWAY COVERAGE")
print("-" * 40)

PATHWAYS = ['Cell wall', 'LPS', 'DNA', 'Transcription', 'Metabolism', 'Cell division']

def pathway_coverage(df, name):
    coverage = {}
    for _, row in df.iterrows():
        cluster = f"MOA-{row['Cluster']}"
        pathways_in = eval(row['Pathway_Distribution']) if isinstance(row['Pathway_Distribution'], str) else row['Pathway_Distribution']
        coverage[cluster] = len(pathways_in)
    return coverage

effnet_coverage = pathway_coverage(effnet_clusters, 'EffNet')
dino_coverage = pathway_coverage(dino_clusters, 'DINO')

print(f"\nPathways per cluster:")
print(f"  EffNet (k=19): mean={np.mean(list(effnet_coverage.values())):.2f}, std={np.std(list(effnet_coverage.values())):.2f}")
print(f"  DINO (k=5):    mean={np.mean(list(dino_coverage.values())):.2f}, std={np.std(list(dino_coverage.values())):.2f}")

# =============================================================================
# METRIC 5: Cluster Size Balance
# =============================================================================
print("\n5. CLUSTER SIZE BALANCE")
print("-" * 40)

effnet_sizes = effnet_clusters['Size'].values
dino_sizes = dino_clusters['Size'].values

print(f"\n{'Metric':<25} {'EffNet (k=19)':<15} {'DINO (k=5)':<15}")
print("-" * 55)
print(f"{'Min cluster size':<25} {effnet_sizes.min():<15} {dino_sizes.min():<15}")
print(f"{'Max cluster size':<25} {effnet_sizes.max():<15} {dino_sizes.max():<15}")
print(f"{'Mean cluster size':<25} {effnet_sizes.mean():<15.2f} {dino_sizes.mean():<15.2f}")
print(f"{'Size std dev':<25} {effnet_sizes.std():<15.2f} {dino_sizes.std():<15.2f}")
print(f"{'CV (std/mean)':<25} {(effnet_sizes.std()/effnet_sizes.mean()):<15.2f} {(dino_sizes.std()/dino_sizes.mean()):<15.2f}")

# =============================================================================
# SUMMARY COMPARISON
# =============================================================================
print("\n" + "="*70)
print("SUMMARY: HEAD-TO-HEAD COMPARISON")
print("="*70)

# Score each model (higher is better)
scores = {
    'Silhouette Score': (effnet_sil, dino_sil, 'Higher is better'),
    'Avg Purity': (effnet_purity['avg'], dino_purity['avg'], 'Higher is better'),
    'Guide Consistency': (effnet_consistency, dino_consistency, 'Higher is better'),
    'High Purity Clusters': (effnet_purity['high'], dino_purity['high'], 'Higher is better'),
    'Cluster Balance (CV)': (effnet_sizes.std()/effnet_sizes.mean(), dino_sizes.std()/dino_sizes.mean(), 'Lower is better'),
}

print(f"\n{'Metric':<25} {'EffNet':<12} {'DINO':<12} {'Winner':<12}")
print("-" * 60)

effnet_wins = 0
dino_wins = 0

for metric, (e_val, d_val, direction) in scores.items():
    if direction == 'Higher is better':
        winner = 'EffNet' if e_val > d_val else 'DINO'
    else:  # Lower is better
        winner = 'EffNet' if e_val < d_val else 'DINO'
    
    if winner == 'EffNet':
        effnet_wins += 1
    else:
        dino_wins += 1
    
    print(f"{metric:<25} {e_val:<12.4f} {d_val:<12.4f} {winner:<12}")

print("-" * 60)
print(f"Overall wins: EffNet={effnet_wins}, DINO={dino_wins}")

# =============================================================================
# VISUALIZATION: Side-by-side Comparison
# =============================================================================
print("\nGenerating comparison visualization...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Silhouette scores
ax = axes[0, 0]
models = ['EffNet\n(k=19)', 'DINO\n(k=5)']
sil_scores = [effnet_sil, dino_sil]
bars = ax.bar(models, sil_scores, color=['#3498db', '#e74c3c'])
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score Comparison')
for bar, val in zip(bars, sil_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
            f'{val:.4f}', ha='center', fontsize=10)
ax.set_ylim(0, max(sil_scores) * 1.2)

# 2. Purity distribution
ax = axes[0, 1]
x = np.arange(3)
width = 0.35
ax.bar(x - width/2, [effnet_purity['high'], effnet_purity['medium'], effnet_purity['low']], 
       width, label='EffNet', color='#3498db')
ax.bar(x + width/2, [dino_purity['high'], dino_purity['medium'], dino_purity['low']], 
       width, label='DINO', color='#e74c3c')
ax.set_xlabel('Purity Category')
ax.set_ylabel('Number of Clusters')
ax.set_title('Cluster Purity Distribution')
ax.set_xticks(x)
ax.set_xticklabels(['High\n(>=0.8)', 'Medium\n(0.5-0.8)', 'Low\n(<0.5)'])
ax.legend()

# 3. Guide consistency
ax = axes[0, 2]
consistency_data = [effnet_consistency, dino_consistency]
bars = ax.bar(models, consistency_data, color=['#3498db', '#e74c3c'])
ax.set_ylabel('Consistency Rate (%)')
ax.set_title('Guide RNA Consistency')
for bar, val in zip(bars, consistency_data):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{val:.1f}%', ha='center', fontsize=10)
ax.set_ylim(0, 100)

# 4. Average purity
ax = axes[1, 0]
purity_vals = [effnet_purity['avg'], dino_purity['avg']]
bars = ax.bar(models, purity_vals, color=['#3498db', '#e74c3c'])
ax.set_ylabel('Average Purity')
ax.set_title('Average Cluster Purity')
for bar, val in zip(bars, purity_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.3f}', ha='center', fontsize=10)
ax.set_ylim(0, 1)

# 5. Cluster sizes boxplot
ax = axes[1, 1]
bp = ax.boxplot([effnet_sizes, dino_sizes], labels=['EffNet\n(k=19)', 'DINO\n(k=5)'])
ax.set_ylabel('Cluster Size')
ax.set_title('Cluster Size Distribution')
for patch in bp['boxes']:
    patch.set_color(['#3498db', '#e74c3c'][bp['boxes'].index(patch)])

# 6. Win summary
ax = axes[1, 2]
categories = ['Silhouette', 'Purity', 'Guide\nConsistency', 'High Purity\nClusters', 'Balance']
effnet_wins_list = [1, 1, 1, 1, 1]  # Will calculate
dino_wins_list = [0, 0, 0, 0, 0]

# Calculate actual wins
for metric, (e_val, d_val, direction) in scores.items():
    idx = list(scores.keys()).index(metric)
    if direction == 'Higher is better':
        if e_val > d_val:
            effnet_wins_list[idx] = 1
            dino_wins_list[idx] = 0
        else:
            effnet_wins_list[idx] = 0
            dino_wins_list[idx] = 1
    else:
        if e_val < d_val:
            effnet_wins_list[idx] = 1
            dino_wins_list[idx] = 0
        else:
            effnet_wins_list[idx] = 0
            dino_wins_list[idx] = 1

x = np.arange(len(categories))
ax.bar(x - 0.2, effnet_wins_list, 0.4, label='EffNet', color='#3498db')
ax.bar(x + 0.2, dino_wins_list, 0.4, label='DINO', color='#e74c3c')
ax.set_ylabel('Win (1) vs Loss (0)')
ax.set_title('Head-to-Head Comparison')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8)
ax.legend()
ax.set_ylim(0, 1.2)

plt.suptitle('EffNet vs DINO: Quantitative MOA Discovery Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'dino_moa', 'effnet_vs_dino_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved: effnet_vs_dino_comparison.png")

# =============================================================================
# FINAL VERDICT
# =============================================================================
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

if effnet_wins > dino_wins:
    winner = "EffNet"
    margin = effnet_wins - dino_wins
else:
    winner = "DINO"
    margin = dino_wins - effnet_wins

print(f"""
RECOMMENDATION: {winner} (wins {effnet_wins} vs {dino_wins} metrics)

Key Trade-offs:

EffNet (k=19):
+ More granular clusters ({len(effnet_clusters)} vs {len(dino_clusters)})
+ Higher pathway purity ({effnet_purity['avg']:.3f} vs {dino_purity['avg']:.3f})
+ More high-purity clusters ({effnet_purity['high']} vs {dino_purity['high']})
- Lower silhouette score ({effnet_sil:.4f} vs {dino_sil:.4f})
- More clusters to interpret

DINO (k=5):
+ Better silhouette score ({dino_sil:.4f} vs {effnet_sil:.4f})
+ Fewer, more distinct clusters
+ Easier interpretation with 5 groups
- Lower pathway purity
- Less granular separation

If you want detailed MOA separation: Use EffNet (k=19)
If you want broad functional grouping: Use DINO (k=5)
""")

print("="*70)
print("COMPARISON COMPLETE")
print("="*70)
