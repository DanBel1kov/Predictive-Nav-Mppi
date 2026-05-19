Predictive Nav MPPI + HuNav

Quick start (Humble, Gazebo Classic)

1) Patch HuNav for custom BT path, actor scale, and robot force scaling

**Required patches:**

```bash
cd ~/hunav_ws/src/hunav_gazebo_wrapper
git apply /home/danbel1kov/predictive-nav-mppi/patches/hunav_actor_scale.patch
git apply /home/danbel1kov/predictive-nav-mppi/patches/hunav_behavior_tree_path.patch

cd ../hunav_sim/hunav_agent_manager
git apply /home/danbel1kov/predictive-nav-mppi/patches/hunav_robot_force_scale.patch

cd ~/hunav_ws
colcon build --packages-select hunav_gazebo_wrapper hunav_agent_manager --symlink-install
source install/setup.bash
```

**Patch descriptions:**
- `hunav_actor_scale.patch` — Scale actor masses and inertias for sim stability
- `hunav_behavior_tree_path.patch` — Allow custom behavior tree path via ROS parameter
- `hunav_robot_force_scale.patch` — **CRITICAL**: Fix `robot_force_scale` parameter to actually scale the robot's social force contribution to pedestrians (see robot_force_scale fix notes below)

2) Build this repo and run

```bash
source /opt/ros/humble/setup.bash
source ~/hunav_ws/install/setup.bash

cd ~/predictive-nav-mppi
colcon build --symlink-install
source install/setup.bash

ros2 launch predictive_nav_mppi sim_nav2.launch.py
```

Notes

**HuNav robot_force_scale fix:**
The `hunav_robot_force_scale.patch` corrects a critical bug in `hunav_agent_manager/src/agent_manager.cpp` where the `robot_force_scale` parameter had **zero effect** on pedestrian behavior. 

**Root cause:** The Social Force Model (SFM) formula at line 307 of `lightsfm/include/sfm.hpp`:
```cpp
me.forces.socialForce += me.params.forceFactorSocial * (forceVelocity + forceAngle);
```
uses the **observing agent's** `forceFactorSocial` (pedestrian's own ≈3.5), not the other agent's. The old code tried to scale the robot's parameter:
```cpp
scaledRobot.params.forceFactorSocial *= robot_force_scale_;  // WRONG: SFM ignores this
```

The fix (lines 1395–1414) computes robot and pedestrian forces separately, then scales the resulting force vector:
```cpp
// 1. Compute pedestrian-pedestrian forces only
sfm::SFM.computeForces(agents_[id].sfmAgent, otherAgents);

// 2. Compute robot contribution in isolation
utils::Vector2d robotContrib(0.0, 0.0);
{
  sfm::Agent tmp = agents_[id].sfmAgent;
  tmp.forces.socialForce.set(0.0, 0.0);
  std::vector<sfm::Agent> robotOnly = {robot_.sfmAgent};
  sfm::SFM.computeForces(tmp, robotOnly);
  robotContrib = tmp.forces.socialForce * robot_force_scale_;  // Scale RESULT, not parameter
}

// 3. Add scaled robot contribution
agents_[id].sfmAgent.forces.socialForce += robotContrib;
```

**Verification:** Without the patch, `robot_force_scale=0.0` and `robot_force_scale=1.0` produce identical metrics (min_dist, viol_time, avg_robot_influence). With the patch, `force=1.0` shows significantly better navigation (lower min_dist and viol_time) compared to `force=0.0`.

**Other notes:**
- Custom BTs live in `src/hunav_extension/behavior_trees`.
- The launch passes `behavior_tree_path` to `hunav_agent_manager`.
- People predictor backend:
  - `predictor_type:=kalman` (CV filter),
  - `predictor_type:=model` (internal Social GRU),
  - `predictor_type:=social_vae` (external SocialVAE repo + checkpoint).

Example SocialVAE launch:

```bash
ros2 launch predictive_nav_mppi sim_nav2.launch.py \
  predictor_type:=social_vae \
  social_vae_repo_path:=/path/to/SocialVAE \
  social_vae_ckpt_path:=/path/to/SocialVAE/models/hotel \
  social_vae_config_path:=/path/to/SocialVAE/config/hotel.py
```

Robot force sweep benchmark

Measure how trajectory category distributions change with robot force scaling:

```bash
ros2 run predictive_nav_mppi run_robot_force_sweep \
  --config src/predictive_nav_mppi/config/benchmark_config.yaml \
  --forces 0.0,0.25,0.5,1.0 \
  --episode-count 3 \
  --predictor kalman \
  --study-name kalman_force_sweep \
  --output-root benchmark_force_sweep
```

This automatically generates:
- `force_interaction_report.txt` — metrics aggregated across episodes
- `force_interaction_counts.png` — absolute interaction category counts
- `category_vs_radius_by_force.png` — how category share changes with near-robot radius (r=1,2,3,6m) for each force
- `category_by_radius_grouped.png` — category distributions grouped by radius
- `metrics_boxplot_by_force.png` — boxplots of time_to_goal, path_length, min_dist, viol_time, avg_robot_influence by force

All graphs are saved in `benchmark_force_sweep/<study_name>/`.

To manually re-analyze an existing sweep:
```bash
python3 src/predictive_nav_mppi/predictive_nav_mppi/analyze_force_radius.py \
  benchmark_force_sweep/kalman_force_sweep
```

Offline predictor benchmark (without robot navigation)

1) Record people trajectories into a dataset:

```bash
ros2 run predictive_nav_mppi record_people_dataset --ros-args \
  -p input_topic:=/people \
  -p output_path:=/tmp/people_dataset.json
```

2) Run offline benchmark for Kalman / SocialGRU / SocialVAE:

```bash
ros2 run predictive_nav_mppi benchmark_people_predictors -- \
  --dataset /tmp/people_dataset.json \
  --output_dir /tmp/predictor_bench \
  --obs_len 8 --obs_dt 0.4 --pred_dt 0.1 --pred_steps 12,26 \
  --social_gru_weights /home/danbel1kov/predictive-nav-mppi/src/predictive_nav_mppi/predictive_nav_mppi/models/best_model.pt \
  --social_vae_repo_path /home/danbel1kov/SocialVAE \
  --social_vae_ckpt_path /home/danbel1kov/predictive-nav-mppi/src/predictive_nav_mppi/predictive_nav_mppi/models/vae_hotel \
  --social_vae_config_path /home/danbel1kov/SocialVAE/config/hotel.py \
  --social_vae_samples 20 \
  --n_permutations 3000
```

Outputs:
- `/tmp/predictor_bench/report.txt` with ADE/FDE summary by horizon
- `/tmp/predictor_bench/summary.json` with full metrics and paired permutation-test p-values

Force 1.0 data, training, offline tests, and navigation benchmarks

This workflow records human trajectories with the current clean semantics:
`robot_force_scale=1.0` means full robot social-force reaction. The robot is
not additionally injected as an unscaled Gazebo obstacle for pedestrians.

## 1) Manual data recording in three scenes

For each scene, use two terminals. In Terminal 1 launch the simulation, then set
robot goals manually in RViz/Nav2. In Terminal 2 record `/people`. Stop the
recorder with `Ctrl-C` when the scene is done so it writes the JSON file.

Common setup:

```bash
cd ~/predictive-nav-mppi
source /opt/ros/humble/setup.bash
source ~/hunav_ws/install/setup.bash
source install/setup.bash
mkdir -p datasets/people_force1p0
```

### long_corridor

Terminal 1:

```bash
ros2 launch predictive_nav_mppi sim_nav2.launch.py \
  scenario:=long_corridor \
  mppi_mode:=standard \
  predictor_type:=kalman \
  use_hunav:=True \
  humans_ignore_robot:=False \
  robot_force_scale:=1.0 \
  gui:=True
```

Terminal 2:

```bash
ros2 run predictive_nav_mppi record_people_dataset --ros-args \
  -p input_topic:=/people \
  -p output_path:=$HOME/predictive-nav-mppi/datasets/people_force1p0/long_corridor.json \
  -p map_name:=long_corridor \
  -p robot_force_scale:=1.0 \
  -p visibility_radius:=15.0 \
  -p visibility_fov_deg:=360.0 \
  -p require_robot_pose:=true
```

### nonlinear_corridor

Terminal 1:

```bash
ros2 launch predictive_nav_mppi sim_nav2.launch.py \
  scenario:=nonlinear_corridor \
  mppi_mode:=standard \
  predictor_type:=kalman \
  use_hunav:=True \
  humans_ignore_robot:=False \
  robot_force_scale:=1.0 \
  gui:=True
```

Terminal 2:

```bash
ros2 run predictive_nav_mppi record_people_dataset --ros-args \
  -p input_topic:=/people \
  -p output_path:=$HOME/predictive-nav-mppi/datasets/people_force1p0/nonlinear_corridor.json \
  -p map_name:=nonlinear_corridor \
  -p robot_force_scale:=1.0 \
  -p visibility_radius:=15.0 \
  -p visibility_fov_deg:=360.0 \
  -p require_robot_pose:=true
```

### labyrinth_turns

Terminal 1:

```bash
ros2 launch predictive_nav_mppi sim_nav2.launch.py \
  scenario:=labyrinth_turns \
  mppi_mode:=standard \
  predictor_type:=kalman \
  use_hunav:=True \
  humans_ignore_robot:=False \
  robot_force_scale:=1.0 \
  gui:=True
```

Terminal 2:

```bash
ros2 run predictive_nav_mppi record_people_dataset --ros-args \
  -p input_topic:=/people \
  -p output_path:=$HOME/predictive-nav-mppi/datasets/people_force1p0/labyrinth_turns.json \
  -p map_name:=labyrinth_turns \
  -p robot_force_scale:=1.0 \
  -p visibility_radius:=15.0 \
  -p visibility_fov_deg:=360.0 \
  -p require_robot_pose:=true
```

## 2) Curate the training dataset

```bash
cd ~/predictive-nav-mppi
source /opt/ros/humble/setup.bash
source ~/hunav_ws/install/setup.bash
source install/setup.bash

rm -rf /home/danbel1kov/predictive-nav-mppi/datasets/curated_force1p0
ros2 run predictive_nav_mppi curate_people_dataset \
  --datasets \
    /home/danbel1kov/predictive-nav-mppi/datasets/people_force1p0/long_corridor.json \
    /home/danbel1kov/predictive-nav-mppi/datasets/people_force1p0/nonlinear_corridor.json \
    /home/danbel1kov/predictive-nav-mppi/datasets/people_force1p0/labyrinth_turns.json \
  --output_dir /home/danbel1kov/predictive-nav-mppi/datasets/curated_force1p0 \
  --near_robot_radius 6.0 \
  --near_robot_fov_deg 360.0 \
  --pred_len 12 \
  --pred_dt 0.4
```

## 3) Train predictors

KalmanResidualNet:

```bash
cd ~/predictive-nav-mppi
source /opt/ros/humble/setup.bash
source ~/hunav_ws/install/setup.bash
source install/setup.bash

ros2 run predictive_nav_mppi train_residual_predictor \
  --train_dataset /home/danbel1kov/predictive-nav-mppi/datasets/curated_force1p0/train_residual_cases.json \
  --val_dataset   /home/danbel1kov/predictive-nav-mppi/datasets/curated_force1p0/benchmark_residual_cases.json \
  --output_dir    /home/danbel1kov/predictive-nav-mppi/models/kalman_residual_force1p0 \
  --obs_len 8 --pred_len 12 --obs_dt 0.4 --pred_dt 0.4 \
  --batch_size 64 --epochs 40 --lr 5e-4 --weight_decay 1e-4 \
  --hidden 64 --social_hidden 32 --k_neighbors 3 \
  --scene_patch_size_m 6.0 --scene_patch_pixels 32 --scene_hidden 32 \
  --lambda_vel 0.2 --lambda_jerk 0.05 --seed 42
```

SocialVAE with velocity/jerk regularization:

```bash
ros2 run predictive_nav_mppi finetune_social_vae \
  --social_vae_repo  /home/danbel1kov/SocialVAE \
  --social_vae_ckpt  /home/danbel1kov/SocialVAE/models/hotel \
  --train_dataset    /home/danbel1kov/predictive-nav-mppi/datasets/curated_force1p0/train_residual_cases.json \
  --val_dataset      /home/danbel1kov/predictive-nav-mppi/datasets/curated_force1p0/benchmark_residual_cases.json \
  --output_dir       /home/danbel1kov/predictive-nav-mppi/models/social_vae_finetuned_force1p0_stable \
  --obs_len 8 --pred_len 12 --obs_dt 0.4 --ob_radius 2.0 --hidden_dim 256 \
  --max_neighbors 8 --batch_size 32 --epochs 8 --lr 3e-5 --weight_decay 1e-4 \
  --kl_weight 0.2 --kl_warmup_epochs 1 --early_stop_patience 2 \
  --pred_samples_eval 20 --grad_clip 1.0 \
  --lambda_vel 0.2 --lambda_jerk 0.05 \
  --seed 42
```

SocialVAE without velocity/jerk regularization:

```bash
ros2 run predictive_nav_mppi finetune_social_vae \
  --social_vae_repo  /home/danbel1kov/SocialVAE \
  --social_vae_ckpt  /home/danbel1kov/SocialVAE/models/hotel \
  --train_dataset    /home/danbel1kov/predictive-nav-mppi/datasets/curated_force1p0/train_residual_cases.json \
  --val_dataset      /home/danbel1kov/predictive-nav-mppi/datasets/curated_force1p0/benchmark_residual_cases.json \
  --output_dir       /home/danbel1kov/predictive-nav-mppi/models/social_vae_finetuned_force1p0 \
  --obs_len 8 --pred_len 12 --obs_dt 0.4 --ob_radius 2.0 --hidden_dim 256 \
  --max_neighbors 8 --batch_size 32 --epochs 8 --lr 3e-5 --weight_decay 1e-4 \
  --kl_weight 0.2 --kl_warmup_epochs 1 --early_stop_patience 2 \
  --pred_samples_eval 20 --grad_clip 1.0 --seed 42
```

## 4) Offline predictor tests

Residual:

```bash
ros2 run predictive_nav_mppi benchmark_people_predictors -- \
  --dataset /home/danbel1kov/predictive-nav-mppi/datasets/people_force1p0/long_corridor.json \
  --output_dir /home/danbel1kov/predictive-nav-mppi/benchmark_people_predictors/force1p0_residual_long_corridor \
  --obs_len 8 --obs_dt 0.4 --pred_dt 0.4 --pred_steps 12 \
  --residual_scene_model_weights /home/danbel1kov/predictive-nav-mppi/models/kalman_residual_force1p0/best_residual_model.pt \
  --scene_patch_size_m 6.0 --scene_patch_pixels 32 \
  --streaming_metrics \
  --stride 1 \
  --n_permutations 3000
```

SocialVAE without aux regularization:

```bash
ros2 run predictive_nav_mppi benchmark_people_predictors -- \
  --dataset /home/danbel1kov/predictive-nav-mppi/datasets/people_force1p0/long_corridor.json \
  --output_dir /home/danbel1kov/predictive-nav-mppi/benchmark_people_predictors/force1p0_vae_long_corridor \
  --obs_len 8 --obs_dt 0.4 --pred_dt 0.4 --pred_steps 12 \
  --social_vae_repo_path /home/danbel1kov/SocialVAE \
  --social_vae_ckpt_path /home/danbel1kov/predictive-nav-mppi/models/social_vae_finetuned_force1p0 \
  --social_vae_config_path /home/danbel1kov/SocialVAE/config/hotel.py \
  --social_vae_samples 20 \
  --streaming_metrics \
  --stride 1 \
  --n_permutations 3000
```

SocialVAE with velocity/jerk regularization:

```bash
ros2 run predictive_nav_mppi benchmark_people_predictors -- \
  --dataset /home/danbel1kov/predictive-nav-mppi/datasets/people_force1p0/long_corridor.json \
  --output_dir /home/danbel1kov/predictive-nav-mppi/benchmark_people_predictors/force1p0_vae_stable_long_corridor \
  --obs_len 8 --obs_dt 0.4 --pred_dt 0.4 --pred_steps 12 \
  --social_vae_repo_path /home/danbel1kov/SocialVAE \
  --social_vae_ckpt_path /home/danbel1kov/predictive-nav-mppi/models/social_vae_finetuned_force1p0_stable \
  --social_vae_config_path /home/danbel1kov/SocialVAE/config/hotel.py \
  --social_vae_samples 20 \
  --streaming_metrics \
  --stride 1 \
  --n_permutations 3000
```

## 5) Navigation benchmark with force 1.0 and trained models

```bash
cd ~/predictive-nav-mppi
source /opt/ros/humble/setup.bash
source ~/hunav_ws/install/setup.bash
source install/setup.bash

STUDY=three_way_force1p0_long_corridor_standard_retrained
EPISODES=20
```

Kalman vs residual:

```bash
ros2 run predictive_nav_mppi run_paired_benchmark \
  --config src/predictive_nav_mppi/config/benchmark_config.yaml \
  --study-name $STUDY --episodes $EPISODES \
  --humans-react-to-robot --robot-force-scale 1.0 \
  --left-mode standard --left-predictor kalman \
  --right-mode standard --right-predictor residual \
  --residual-weights /home/danbel1kov/predictive-nav-mppi/models/kalman_residual_force1p0/best_residual_model.pt \
  --residual-alpha 1.0 --residual-beta 0.0 --residual-clip 0.0 \
  --no-residual-turn-gate \
  --residual-tag force1p0_honest
```

Kalman vs SocialVAE stable:

```bash
ros2 run predictive_nav_mppi run_paired_benchmark \
  --config src/predictive_nav_mppi/config/benchmark_config.yaml \
  --study-name $STUDY --episodes $EPISODES \
  --humans-react-to-robot --robot-force-scale 1.0 \
  --left-mode standard --left-predictor kalman \
  --right-mode standard --right-predictor social_vae \
  --social-vae-repo /home/danbel1kov/SocialVAE \
  --social-vae-ckpt /home/danbel1kov/predictive-nav-mppi/models/social_vae_finetuned_force1p0_stable \
  --social-vae-config /home/danbel1kov/SocialVAE/config/hotel.py \
  --social-vae-tag force1p0_stable
```

Residual vs SocialVAE stable:

```bash
ros2 run predictive_nav_mppi run_paired_benchmark \
  --config src/predictive_nav_mppi/config/benchmark_config.yaml \
  --study-name $STUDY --episodes $EPISODES \
  --humans-react-to-robot --robot-force-scale 1.0 \
  --left-mode standard --left-predictor residual \
  --right-mode standard --right-predictor social_vae \
  --residual-weights /home/danbel1kov/predictive-nav-mppi/models/kalman_residual_force1p0/best_residual_model.pt \
  --residual-alpha 1.0 --residual-beta 0.0 --residual-clip 0.0 \
  --no-residual-turn-gate \
  --residual-tag force1p0_honest \
  --social-vae-repo /home/danbel1kov/SocialVAE \
  --social-vae-ckpt /home/danbel1kov/predictive-nav-mppi/models/social_vae_finetuned_force1p0_stable \
  --social-vae-config /home/danbel1kov/SocialVAE/config/hotel.py \
  --social-vae-tag force1p0_stable
```

Summary plots and paired tables:

```bash
ros2 run predictive_nav_mppi plot_paired_compare \
  --study-dir benchmark_paired/$STUDY
```
