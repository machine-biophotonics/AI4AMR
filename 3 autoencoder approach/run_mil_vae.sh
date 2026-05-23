#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_MODE="${1:-mutant}"
TEST_PLATE="${2:-Plate_6}"
MIL_EPOCHS="${3:-100}"
VAE_EPOCHS="${4:-50}"
LATENT_DIM="${5:-32}"
VAE_BETA="${6:-0.1}"

echo "=============================================="
echo "MIL + VAE Pipeline"
echo "=============================================="
echo "Data mode:     $DATA_MODE"
echo "Test plate:    $TEST_PLATE"
echo "MIL epochs:    $MIL_EPOCHS"
echo "VAE epochs:    $VAE_EPOCHS"
echo "Latent dim:    $LATENT_DIM"
echo "VAE beta:      $VAE_BETA"
echo "=============================================="

# Stage 1: MIL training
echo ""
echo "=== STAGE 1: MIL Training ==="
python3 train_mil_vae.py \
    --data_mode "$DATA_MODE" \
    --test_plate "$TEST_PLATE" \
    --epochs "$MIL_EPOCHS" \
    --batch_size 16 \
    --latent_dim "$LATENT_DIM" \
    --vae_beta "$VAE_BETA"

# Stage 2: VAE training (loads MIL checkpoint)
echo ""
echo "=== STAGE 2: VAE Training ==="
python3 train_mil_vae.py \
    --data_mode "$DATA_MODE" \
    --test_plate "$TEST_PLATE" \
    --latent_dim "$LATENT_DIM" \
    --vae_beta "$VAE_BETA" \
    --vae_epochs "$VAE_EPOCHS" \
    --stage2_only

# Stage 3: Analysis
FOLD_DIR="mil_vae_${DATA_MODE}/fold_${TEST_PLATE}"
CHECKPOINT="${FOLD_DIR}/best_mil_vae.pth"

echo ""
echo "=== STAGE 3: Analysis ==="
python3 analyze_vae.py \
    --checkpoint "$CHECKPOINT" \
    --data_mode "$DATA_MODE" \
    --test_plate "$TEST_PLATE" \
    --output_dir "${FOLD_DIR}/analysis"

echo ""
echo "=============================================="
echo "Pipeline complete!"
echo "Model:    $CHECKPOINT"
echo "Analysis: ${FOLD_DIR}/analysis/"
echo "=============================================="
