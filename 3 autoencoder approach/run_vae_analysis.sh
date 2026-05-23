#!/bin/bash
# =============================================================================
# VAE Analysis Pipeline for CRISPRi Reference Plate Imaging
# =============================================================================
# This script:
#   1. Trains VAE on drug + mutant images
#   2. Runs full analysis (Wasserstein, t-SNE, concentration effects, etc.)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_MODE="${1:-both}"       # drug, mutant, or both
TEST_PLATE="${2:-Plate_6}"   # Plate_1 through Plate_6
EPOCHS="${3:-100}"           # training epochs
LATENT_DIM="${4:-128}"       # latent dimension
BETA="${5:-1.0}"             # beta-VAE weight

echo "=============================================="
echo "VAE Analysis Pipeline"
echo "=============================================="
echo "Data mode:          $DATA_MODE"
echo "Test plate:         $TEST_PLATE"
echo "Epochs:             $EPOCHS"
echo "Latent dimension:   $LATENT_DIM"
echo "Beta (KL weight):   $BETA"
echo "=============================================="

# Step 1: Train VAE
echo ""
echo "[Step 1] Training VAE..."
python3 train_vae.py \
    --data_mode "$DATA_MODE" \
    --test_plate "$TEST_PLATE" \
    --epochs "$EPOCHS" \
    --latent_dim "$LATENT_DIM" \
    --beta "$BETA" \
    --batch_size 128 \
    --crops_per_image 10

# Step 2: Find the checkpoint
VAE_DIR="vae_${DATA_MODE}/fold_${TEST_PLATE}"
CHECKPOINT="${VAE_DIR}/best_vae.pth"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Make sure training completed successfully."
    exit 1
fi

echo ""
echo "[Step 2] Running analysis..."
python3 analyze_vae.py \
    --checkpoint "$CHECKPOINT" \
    --data_mode "$DATA_MODE" \
    --test_plate "$TEST_PLATE" \
    --img_size 224 \
    --output_dir "${VAE_DIR}/analysis"

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "=============================================="
echo "Model:        ${VAE_DIR}/best_vae.pth"
echo "Analysis:     ${VAE_DIR}/analysis/"
echo ""
echo "Key outputs:"
echo "  - ${VAE_DIR}/analysis/tsne_latent.png"
echo "  - ${VAE_DIR}/analysis/wasserstein_heatmap.png"
echo "  - ${VAE_DIR}/analysis/drug_mutant_wasserstein.png (if both)"
echo "  - ${VAE_DIR}/analysis/drug_concentration_effect.png (if drug/both)"
echo "  - ${VAE_DIR}/analysis/top_similar_classes.txt"
echo "  - ${VAE_DIR}/analysis/interpolation/"
echo "  - ${VAE_DIR}/analysis/traversal/"
echo "=============================================="
