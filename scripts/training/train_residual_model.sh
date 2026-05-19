#!/bin/bash
# Train residual predictor model on curated benchmark data

set -e

cd "$(dirname "$0")/src/predictive_nav_mppi"

echo "=========================================="
echo "Training Residual Prediction Model"
echo "=========================================="
echo ""

TRAIN_FILE="../../datasets/residual_train.json"
TEST_FILE="../../datasets/residual_test.json"
OUTPUT_DIR="../../models/residual_baseline"

echo "Train file: $TRAIN_FILE"
echo "Test file:  $TEST_FILE"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Training parameters:
# - obs_len=8, pred_len=26 (matches benchmark: 0.8s observation, 2.6s prediction)
# - obs_dt=0.1, pred_dt=0.1 (10Hz frame rate)
# - batch_size=32 (reasonable for ~1160 training cases)
# - epochs=100 (allow model to converge)
# - lr=0.001 (conservative learning rate)
# - k_neighbors=3 (include robot + 2 closest pedestrians)
#
# Loss weights:
# - lambda_jerk=0.05 (light smoothness penalty)
# - lambda_acc_match=0.1 (match acceleration patterns)
# - lambda_vel=0.05 (match velocity)
# - lambda_res_mag=0.01 (keep residuals sparse)
# - lambda_risk=0.5 (weight close encounters)
# - lambda_safety=1.0 (penalize under-predicting danger to robot)

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
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
