Predictive Nav MPPI + HuNav

Quick start (Humble, Gazebo Classic)

1) Patch HuNav to allow custom BT path + actor scale

```bash
cd ~/hunav_ws/src/hunav_sim

cd ~/hunav_ws/src/hunav_gazebo_wrapper
git apply /home/danbel1kov/predictive-nav-mppi/patches/hunav_actor_scale.patch
git apply /home/danbel1kov/predictive-nav-mppi/patches/hunav_behavior_tree_path.patch
cd ~/hunav_ws
colcon build --symlink-install
source install/setup.bash
```

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
