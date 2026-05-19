#!/bin/bash
# Curate raw datasets into ~10k train + ~2k test for residual model training

cd "$(dirname "$0")/src/predictive_nav_mppi"

echo "=========================================="
echo "CURATING RAW DATASETS FOR TRAINING"
echo "=========================================="
echo ""

# Target: ~10k train + ~2k test
# With holdout_fraction=0.2, we get 80% train, 20% test
# So we need train_target = 10k / 0.8 = 12500 total cases
# Then test_target would be naturally 2500 at 20% holdout

python3 -m predictive_nav_mppi.curate_people_dataset \
  /home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_long_corridor_react_0p5.json \
  /home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_nonlinear_corridor_react_0p5.json \
  /home/danbel1kov/predictive-nav-mppi/datasets/raw_react_0p5/people_dataset_labyrinth_turns_react_0p5.json \
  --output_dir ../../datasets/curated_training_10k \
  --obs_len 8 \
  --obs_dt 0.1 \
  --pred_len 26 \
  --pred_dt 0.1 \
  --stride 3 \
  --neighbor_radius 6.0 \
  --max_neighbors 16 \
  --holdout_fraction 0.2 \
  --train_target 12500 \
  --benchmark_target 0 \
  --min_gap_frames 10 \
  --max_linear_fraction 1.0 \
  --interaction_dist 1.5 \
  --dense_neighbors_min 4 \
  --turn_threshold_deg 30.0 \
  --stop_speed_thresh 0.10 \
  --moving_speed_min 0.25 \
  --stop_go_delta 0.25 \
  --near_robot_radius 0.0 \
  --near_robot_fov_deg 360.0

echo ""
echo "=========================================="
echo "Curation complete!"
echo "Output directory: datasets/curated_training_10k"
echo "=========================================="
echo ""
echo "Training datasets created:"
echo "  - train_residual_cases.json (~10k cases, 80%)"
echo "  - benchmark_residual_cases.json (test, ~2.5k cases, 20%)"
echo ""
echo "Next step: run training with these datasets"
