Predictive Nav MPPI + HuNav

Quick start (Humble, Gazebo Classic)

1) Patch HuNav to allow custom BT path + actor scale

```bash
cd ~/hunav_ws/src/hunav_sim
git apply /home/danbel1kov/predictive-nav-mppi/patches/hunav_behavior_tree_path.patch

cd ~/hunav_ws/src/hunav_gazebo_wrapper
git apply /home/danbel1kov/predictive-nav-mppi/patches/hunav_actor_scale.patch
git apply /home/danbel1kov/predictive-nav-mppi/patches/hunav_actor_z_offset.patch
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
