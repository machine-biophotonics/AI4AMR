"""
Align drug_control and mutant NC controls to find transformation vector.
Implements: mean shift (simple) and CORAL (mean + covariance alignment).
"""
import numpy as np, csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm, inv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(BASE_DIR, "features_all.npz")
CSV_PATH = os.path.join(BASE_DIR, "features_metadata.csv")
OUT_DIR = os.path.join(BASE_DIR, "analysis_figures")
MISC_DIR = os.path.join(BASE_DIR, "analysis_figures")
os.makedirs(OUT_DIR, exist_ok=True)

MUTANT_NC_LABELS = {f'NC_{i}' for i in range(1, 7)}

data = np.load(NPZ_PATH)
embeddings = data["embeddings"]
with open(CSV_PATH) as f:
    metadata = list(csv.DictReader(f))

sources = np.array([m["source"] for m in metadata])
labels_arr = np.array([m["label"] for m in metadata])

drug_mask = (sources == 'drug') & (labels_arr == 'drug_control')
nc_mask = (sources == 'mutant') & np.array([m['label'] in MUTANT_NC_LABELS for m in metadata])

drug_ctrl = embeddings[drug_mask]
nc_ctrl = embeddings[nc_mask]

print(f"Drug control: {len(drug_ctrl)} pts")
print(f"NC controls (mutant): {len(nc_ctrl)} pts")

# ============ MEAN SHIFT ============
mu_drug = drug_ctrl.mean(axis=0)
mu_nc = nc_ctrl.mean(axis=0)
shift_vector = mu_nc - mu_drug  # Add this to drug samples to align to NC space
drug_shifted = drug_ctrl + shift_vector

# ============ CORAL ============
# 1. Center both
drug_centered = drug_ctrl - mu_drug
nc_centered = nc_ctrl - mu_nc

# 2. Covariances
C_drug = np.cov(drug_centered, rowvar=False) + 1e-6 * np.eye(drug_ctrl.shape[1])
C_nc = np.cov(nc_centered, rowvar=False) + 1e-6 * np.eye(nc_ctrl.shape[1])

# 3. CORAL transform: X_aligned = X_src * (C_src^{-1/2} * C_tgt^{1/2})
sqrt_C_drug = sqrtm(C_drug)
inv_sqrt_C_drug = inv(sqrt_C_drug)
sqrt_C_nc = sqrtm(C_nc)
W_coral = inv_sqrt_C_drug @ sqrt_C_nc  # transformation matrix

# Apply: center, transform, then add target mean
drug_coral = drug_centered @ W_coral + mu_nc

# ============ t-SNE visualization ============
all_emb = np.vstack([drug_ctrl, nc_ctrl, drug_shifted, drug_coral])
groups = np.array(['drug_ctrl'] * len(drug_ctrl) +
                   ['nc_ctrl'] * len(nc_ctrl) +
                   ['drug_shifted'] * len(drug_ctrl) +
                   ['drug_coral'] * len(drug_ctrl))

pca = PCA(n_components=min(50, len(all_emb) - 1), random_state=42)
emb_pca = pca.fit_transform(all_emb)

tsne = TSNE(n_components=2, perplexity=30, random_state=42, method='barnes_hut', verbose=0)
emb_2d = tsne.fit_transform(emb_pca)

# ---- Figure 1: Before alignment ----
fig, ax = plt.subplots(figsize=(14, 12))
mask_orig = (groups == 'drug_ctrl') | (groups == 'nc_ctrl')
plot_emb = emb_2d[mask_orig]
plot_grp = groups[mask_orig]
colors_map = {'drug_ctrl': '#e74c3c', 'nc_ctrl': '#3498db'}
labels_map = {'drug_ctrl': 'Drug control', 'nc_ctrl': 'NC controls (mutant)'}
for key in ['drug_ctrl', 'nc_ctrl']:
    m = plot_grp == key
    ax.scatter(plot_emb[m, 0], plot_emb[m, 1],
               c=colors_map[key], label=labels_map[key],
               s=10, alpha=0.6, linewidths=0)
ax.set_title('Before Alignment: Drug Control vs NC Controls', fontsize=16, fontweight='bold')
ax.legend(fontsize=14, markerscale=4)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tsne_alignment_before.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: tsne_alignment_before.png")

# ---- Figure 2: After mean shift ----
fig, ax = plt.subplots(figsize=(14, 12))
mask_shift = (groups == 'drug_shifted') | (groups == 'nc_ctrl')
plot_emb = emb_2d[mask_shift]
plot_grp = groups[mask_shift]
colors_map2 = {'drug_shifted': '#e74c3c', 'nc_ctrl': '#3498db'}
labels_map2 = {'drug_shifted': 'Drug control (mean-shifted)', 'nc_ctrl': 'NC controls (mutant)'}
for key in ['drug_shifted', 'nc_ctrl']:
    m = plot_grp == key
    ax.scatter(plot_emb[m, 0], plot_emb[m, 1],
               c=colors_map2[key], label=labels_map2[key],
               s=10, alpha=0.6, linewidths=0)
ax.set_title('After Mean Shift Alignment', fontsize=16, fontweight='bold')
ax.legend(fontsize=14, markerscale=4)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tsne_alignment_mean_shift.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: tsne_alignment_mean_shift.png")

# ---- Figure 3: After CORAL ----
fig, ax = plt.subplots(figsize=(14, 12))
mask_coral = (groups == 'drug_coral') | (groups == 'nc_ctrl')
plot_emb = emb_2d[mask_coral]
plot_grp = groups[mask_coral]
colors_map3 = {'drug_coral': '#e74c3c', 'nc_ctrl': '#3498db'}
labels_map3 = {'drug_coral': 'Drug control (CORAL)', 'nc_ctrl': 'NC controls (mutant)'}
for key in ['drug_coral', 'nc_ctrl']:
    m = plot_grp == key
    ax.scatter(plot_emb[m, 0], plot_emb[m, 1],
               c=colors_map3[key], label=labels_map3[key],
               s=10, alpha=0.6, linewidths=0)
ax.set_title('After CORAL Alignment (mean + covariance)', fontsize=16, fontweight='bold')
ax.legend(fontsize=14, markerscale=4)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tsne_alignment_coral.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: tsne_alignment_coral.png")

# ============ SAVE TRANSFORMATIONS ============
np.save(os.path.join(MISC_DIR, 'alignment_mean_shift_vector.npy'), shift_vector)
np.save(os.path.join(MISC_DIR, 'alignment_coral_matrix.npy'), W_coral)
np.save(os.path.join(MISC_DIR, 'alignment_drug_mean.npy'), mu_drug)
np.save(os.path.join(MISC_DIR, 'alignment_target_mean.npy'), mu_nc)

# ============ METRICS ============
def cosine_sim(a, b):
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a_n * b_n).sum(axis=1)

# Compute alignment quality: mean cosine similarity between each drug and nearest NC
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1).fit(nc_ctrl)

dists_orig, _ = nn.kneighbors(drug_ctrl)
dists_shift, _ = nn.kneighbors(drug_shifted)
dists_coral, _ = nn.kneighbors(drug_coral)

print(f"\n=== Alignment Metrics ===")
print(f"{'Method':<20} {'Mean dist to nearest NC':>25}")
print(f"{'-'*45}")
print(f"{'No alignment':<20} {dists_orig.mean():>25.4f}")
print(f"{'Mean shift':<20} {dists_shift.mean():>25.4f}")
print(f"{'CORAL':<20} {dists_coral.mean():>25.4f}")

print(f"\nVector/norm of mean shift: {np.linalg.norm(shift_vector):.4f}")
print(f"Saved: alignment_mean_shift_vector.npy, alignment_coral_matrix.npy")
