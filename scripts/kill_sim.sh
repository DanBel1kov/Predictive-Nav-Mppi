#!/bin/bash
# Полная очистка всех процессов симуляции.
# Запускай перед новым стартом если был Ctrl+C или ошибка.

echo "=== Killing everything ==="

echo "Gazebo..."
pkill -9 -f gzserver 2>/dev/null || true
pkill -9 -f gzclient 2>/dev/null || true

echo "HuNav..."
pkill -9 -f hunav_gazebo_world_generator 2>/dev/null || true
pkill -9 -f hunav_loader 2>/dev/null || true
pkill -9 -f hunav_agent_manager 2>/dev/null || true

echo "Nav2..."
pkill -9 -f controller_server 2>/dev/null || true
pkill -9 -f planner_server 2>/dev/null || true
pkill -9 -f behavior_server 2>/dev/null || true
pkill -9 -f bt_navigator 2>/dev/null || true
pkill -9 -f waypoint_follower 2>/dev/null || true
pkill -9 -f velocity_smoother 2>/dev/null || true
pkill -9 -f smoother_server 2>/dev/null || true
pkill -9 -f lifecycle_manager 2>/dev/null || true
pkill -9 -f amcl 2>/dev/null || true
pkill -9 -f map_server 2>/dev/null || true
pkill -9 -f component_container_isolated 2>/dev/null || true
pkill -9 -f component_container 2>/dev/null || true

echo "Other ROS2 nodes..."
pkill -9 -f robot_state_publisher 2>/dev/null || true
pkill -9 -f rviz2 2>/dev/null || true
pkill -9 -f people_kf_predictor 2>/dev/null || true
pkill -9 -f compute_agents_proxy 2>/dev/null || true
pkill -9 -f publish_initial_pose 2>/dev/null || true
pkill -9 -f benchmark_episode 2>/dev/null || true
pkill -9 -f run_benchmark 2>/dev/null || true
pkill -9 -f spawn_entity 2>/dev/null || true

echo "Any remaining ros2 processes..."
pkill -9 -f "ros2" 2>/dev/null || true

sleep 3

echo "Restarting ROS2 daemon..."
ros2 daemon stop 2>/dev/null || true
sleep 1
ros2 daemon start 2>/dev/null || true

echo "=== Done. Wait 2-3 sec, then launch again. ==="
