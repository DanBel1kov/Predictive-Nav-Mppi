#!/bin/bash
# Train residual predictor on curated 10k dataset

set -e

cd "$(dirname "$0")/src/predictive_nav_mppi"

TRAIN_FILE="../../datasets/curated_training_10k/train_residual_cases.json"
TEST_FILE="../../datasets/curated_training_10k/benchmark_residual_cases.json"
OUTPUT_DIR="../../models/residual_10k_baseline"

echo "=========================================="
echo "Training Residual Predictor (10k dataset)"
echo "=========================================="
echo ""
echo "Train: $TRAIN_FILE"
echo "Test:  $TEST_FILE"
echo "Model: $OUTPUT_DIR"
echo ""

if [ ! -f "$TRAIN_FILE" ]; then
    echo "❌ Train file not found: $TRAIN_FILE"
    echo "Run curate_raw_10k.sh first!"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    echo "Run curate_raw_10k.sh first!"
    exit 1
fi

python3 -m predictive_nav_mppi.train_residual_predictor \
  --train_dataset "$TRAIN_FILE" \
  --val_dataset "$TEST_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --obs_len 8 \
  --pred_len 26 \
  --obs_dt 0.1 \
  --pred_dt 0.1 \
  --batch_size 32 \
  --epochs 100 \
  --lr 0.001 \
  --weight_decay 1e-4 \
  --hidden 64 \
  --social_hidden 32 \
  --k_neighbors 3 \
  --scene_patch_size_m 6.0 \
  --scene_patch_pixels 32 \
  --scene_patch_align_to_heading \
  --include_robot \
  --lambda_jerk 0.05 \
  --lambda_acc_match 0.1 \
  --lambda_vel 0.05 \
  --lambda_res_mag 0.01 \
  --lambda_risk 0.5 \
  --lambda_safety 1.0 \
  --risk_sigma 1.5 \
  --safety_radius 0.5 \
  --seed 42

echo ""
echo "=========================================="
echo "✓ Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
