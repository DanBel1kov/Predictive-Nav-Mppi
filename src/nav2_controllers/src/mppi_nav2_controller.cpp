#include "nav2_controllers/mppi_nav2_controller.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_util/node_utils.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker.hpp"

namespace nav2_controllers
{

namespace
{

std::array<double, 2> pickLookaheadGoal(
  const nav_msgs::msg::Path & plan,
  const std::array<double, 3> & x0,
  double lookahead_dist)
{
  const size_t n = plan.poses.size();
  if (n == 0) {
    return {x0[0], x0[1]};
  }
  if (n == 1 || lookahead_dist <= 0.0) {
    return {
      plan.poses.back().pose.position.x,
      plan.poses.back().pose.position.y
    };
  }

  size_t nearest_idx = 0;
  double best_d2 = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < n; ++i) {
    const double dx = plan.poses[i].pose.position.x - x0[0];
    const double dy = plan.poses[i].pose.position.y - x0[1];
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      nearest_idx = i;
    }
  }

  double acc = 0.0;
  for (size_t i = nearest_idx; i + 1 < n; ++i) {
    const auto & p0 = plan.poses[i].pose.position;
    const auto & p1 = plan.poses[i + 1].pose.position;
    const double dx = p1.x - p0.x;
    const double dy = p1.y - p0.y;
    acc += std::hypot(dx, dy);
    if (acc >= lookahead_dist) {
      return {p1.x, p1.y};
    }
  }

  return {
    plan.poses.back().pose.position.x,
    plan.poses.back().pose.position.y
  };
}

}  // namespace

double MppiNav2Controller::yawFromQuat(const geometry_msgs::msg::Quaternion & q) const
{
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

void MppiNav2Controller::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();

  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("MppiNav2Controller: invalid lifecycle node");
  }

  global_frame_ = costmap_ros_->getGlobalFrameID();
  robot_base_frame_ = costmap_ros_->getBaseFrameID();

  const std::string prefix = name_ + ".";
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "dt", rclcpp::ParameterValue(params_.dt));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "horizon_steps", rclcpp::ParameterValue(params_.horizon_steps));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "n_rollouts", rclcpp::ParameterValue(params_.n_rollouts));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "v_mean", rclcpp::ParameterValue(params_.v_mean));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "omega_mean", rclcpp::ParameterValue(params_.omega_mean));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "v_std", rclcpp::ParameterValue(params_.v_std));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "omega_std", rclcpp::ParameterValue(params_.omega_std));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "v_min", rclcpp::ParameterValue(params_.v_min));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "v_max", rclcpp::ParameterValue(params_.v_max));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "omega_min", rclcpp::ParameterValue(params_.omega_min));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "omega_max", rclcpp::ParameterValue(params_.omega_max));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "w_goal", rclcpp::ParameterValue(params_.w_goal));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "w_obs", rclcpp::ParameterValue(params_.w_obs));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "w_ctrl", rclcpp::ParameterValue(params_.w_ctrl));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "w_speed", rclcpp::ParameterValue(params_.w_speed));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "w_path", rclcpp::ParameterValue(params_.w_path));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "w_dyn_obs", rclcpp::ParameterValue(params_.w_dyn_obs));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "dyn_risk_delta", rclcpp::ParameterValue(params_.dyn_risk_delta));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "dyn_risk_beta", rclcpp::ParameterValue(params_.dyn_risk_beta));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "dyn_robot_var", rclcpp::ParameterValue(params_.dyn_robot_var));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "lambda", rclcpp::ParameterValue(params_.lambda));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "goal_lookahead_dist", rclcpp::ParameterValue(goal_lookahead_dist_));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "debug_rollouts_enabled",
    rclcpp::ParameterValue(debug_rollouts_enabled_));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "debug_rollouts_top_n",
    rclcpp::ParameterValue(debug_rollouts_top_n_));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "debug_rollouts_line_width",
    rclcpp::ParameterValue(debug_rollouts_line_width_));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "debug_rollouts_topic",
    rclcpp::ParameterValue(debug_rollouts_topic_));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "dyn_people_enabled",
    rclcpp::ParameterValue(dyn_people_enabled_));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "dyn_people_topic",
    rclcpp::ParameterValue(dyn_people_topic_));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "dyn_people_cov_n_sigma",
    rclcpp::ParameterValue(dyn_people_cov_n_sigma_));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "dyn_people_default_std",
    rclcpp::ParameterValue(dyn_people_default_std_));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "dyn_people_cov_std_cap",
    rclcpp::ParameterValue(dyn_people_cov_std_cap_));

  node->get_parameter(prefix + "dt", params_.dt);
  node->get_parameter(prefix + "horizon_steps", params_.horizon_steps);
  node->get_parameter(prefix + "n_rollouts", params_.n_rollouts);
  node->get_parameter(prefix + "v_mean", params_.v_mean);
  node->get_parameter(prefix + "omega_mean", params_.omega_mean);
  node->get_parameter(prefix + "v_std", params_.v_std);
  node->get_parameter(prefix + "omega_std", params_.omega_std);
  node->get_parameter(prefix + "v_min", params_.v_min);
  node->get_parameter(prefix + "v_max", params_.v_max);
  node->get_parameter(prefix + "omega_min", params_.omega_min);
  node->get_parameter(prefix + "omega_max", params_.omega_max);
  node->get_parameter(prefix + "w_goal", params_.w_goal);
  node->get_parameter(prefix + "w_obs", params_.w_obs);
  node->get_parameter(prefix + "w_ctrl", params_.w_ctrl);
  node->get_parameter(prefix + "w_speed", params_.w_speed);
  node->get_parameter(prefix + "w_path", params_.w_path);
  node->get_parameter(prefix + "w_dyn_obs", params_.w_dyn_obs);
  node->get_parameter(prefix + "dyn_risk_delta", params_.dyn_risk_delta);
  node->get_parameter(prefix + "dyn_risk_beta", params_.dyn_risk_beta);
  node->get_parameter(prefix + "dyn_robot_var", params_.dyn_robot_var);
  node->get_parameter(prefix + "lambda", params_.lambda);
  node->get_parameter(prefix + "goal_lookahead_dist", goal_lookahead_dist_);
  node->get_parameter(prefix + "debug_rollouts_enabled", debug_rollouts_enabled_);
  node->get_parameter(prefix + "debug_rollouts_top_n", debug_rollouts_top_n_);
  node->get_parameter(prefix + "debug_rollouts_line_width", debug_rollouts_line_width_);
  node->get_parameter(prefix + "debug_rollouts_topic", debug_rollouts_topic_);
  node->get_parameter(prefix + "dyn_people_enabled", dyn_people_enabled_);
  node->get_parameter(prefix + "dyn_people_topic", dyn_people_topic_);
  node->get_parameter(prefix + "dyn_people_cov_n_sigma", dyn_people_cov_n_sigma_);
  node->get_parameter(prefix + "dyn_people_default_std", dyn_people_default_std_);
  node->get_parameter(prefix + "dyn_people_cov_std_cap", dyn_people_cov_std_cap_);

  mppi_ = MPPIController(params_);

  if (debug_rollouts_enabled_) {
    rollouts_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
      debug_rollouts_topic_, rclcpp::QoS(10));
  }
  if (dyn_people_enabled_) {
    dyn_people_sub_ = node->create_subscription<visualization_msgs::msg::MarkerArray>(
      dyn_people_topic_, rclcpp::QoS(10),
      std::bind(&MppiNav2Controller::predictedPeopleCallback, this, std::placeholders::_1));
  }
}

void MppiNav2Controller::cleanup()
{
  rollouts_pub_.reset();
  dyn_people_sub_.reset();
  costmap_ros_.reset();
  tf_.reset();
  costmap_ = nullptr;
  std::lock_guard<std::mutex> lock(dyn_people_mutex_);
  dyn_people_trajs_.clear();
}

void MppiNav2Controller::activate()
{
  // ничего особенного
}

void MppiNav2Controller::deactivate()
{
  // ничего особенного
}

void MppiNav2Controller::setPlan(const nav_msgs::msg::Path & path)
{
  global_plan_ = path;
}

geometry_msgs::msg::TwistStamped MppiNav2Controller::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & /*velocity*/,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  geometry_msgs::msg::TwistStamped cmd;

  if (global_plan_.poses.empty() || !costmap_) {
    // если нет плана или нет costmap — просто нули
    cmd.twist.linear.x = 0.0;
    cmd.twist.angular.z = 0.0;
    return cmd;
  }

  std::array<double, 3> x0;
  x0[0] = pose.pose.position.x;
  x0[1] = pose.pose.position.y;
  x0[2] = yawFromQuat(pose.pose.orientation);

  nav_msgs::msg::Path plan_local = global_plan_;
  try {
    if (global_plan_.header.frame_id != global_frame_) {
      const auto tf = tf_->lookupTransform(
        global_frame_, global_plan_.header.frame_id,
        tf2::TimePointZero);
      plan_local.poses.clear();
      plan_local.poses.reserve(global_plan_.poses.size());
      for (const auto & pose_stamped : global_plan_.poses) {
        geometry_msgs::msg::PoseStamped pose_out;
        tf2::doTransform(pose_stamped, pose_out, tf);
        plan_local.poses.push_back(pose_out);
      }
      plan_local.header.frame_id = global_frame_;
    }
  } catch (const tf2::TransformException & ex) {
    cmd.twist.linear.x = 0.0;
    cmd.twist.angular.z = 0.0;
    return cmd;
  }

  std::array<double, 2> goal_xy =
    pickLookaheadGoal(plan_local, x0, goal_lookahead_dist_);

  std::vector<std::array<double, 2>> path_xy;
  path_xy.reserve(plan_local.poses.size());
  for (const auto & pose_stamped : plan_local.poses) {
    path_xy.push_back({
      pose_stamped.pose.position.x,
      pose_stamped.pose.position.y
    });
  }

  const nav2_costmap_2d::Costmap2D * cm = costmap_;
  std::vector<RolloutDebug> rollout_debug;
  auto * rollout_ptr = debug_rollouts_enabled_ ? &rollout_debug : nullptr;
  std::vector<std::vector<DynPredictionStep>> dyn_people_snapshot;
  if (dyn_people_enabled_) {
    std::lock_guard<std::mutex> lock(dyn_people_mutex_);
    dyn_people_snapshot = dyn_people_trajs_;
  }
  auto * dyn_ptr = dyn_people_enabled_ ? &dyn_people_snapshot : nullptr;
  auto u = mppi_.computeControl(x0, goal_xy, cm, &path_xy, dyn_ptr, rollout_ptr);

  cmd.twist.linear.x = static_cast<float>(u[0]);
  cmd.twist.angular.z = static_cast<float>(u[1]);
  cmd.header.stamp = node_.lock()->now();
  cmd.header.frame_id = robot_base_frame_;   // base_link

  if (debug_rollouts_enabled_) {
    publishRollouts(rollout_debug, cmd.header.stamp);
  }

  return cmd;
}

void MppiNav2Controller::publishRollouts(
  const std::vector<RolloutDebug> & rollouts,
  const rclcpp::Time & stamp)
{
  if (!rollouts_pub_) {
    return;
  }

  visualization_msgs::msg::MarkerArray msg;
  visualization_msgs::msg::Marker clear;
  clear.header.stamp = stamp;
  clear.header.frame_id = global_frame_;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  msg.markers.push_back(clear);

  if (rollouts.empty()) {
    rollouts_pub_->publish(msg);
    return;
  }

  std::vector<size_t> ids(rollouts.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    ids[i] = i;
  }
  std::sort(ids.begin(), ids.end(), [&rollouts](size_t a, size_t b) {
    return rollouts[a].cost < rollouts[b].cost;
  });

  const int keep_n = std::max(0, std::min(debug_rollouts_top_n_, static_cast<int>(ids.size())));
  for (int rank = 0; rank < keep_n; ++rank) {
    const auto & r = rollouts[ids[rank]];
    visualization_msgs::msg::Marker m;
    m.header.stamp = stamp;
    m.header.frame_id = global_frame_;
    m.ns = "mppi_rollouts";
    m.id = rank;
    m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.orientation.w = 1.0;
    m.scale.x = std::max(0.001, debug_rollouts_line_width_);
    m.lifetime = rclcpp::Duration::from_seconds(0.2);

    const double t = (keep_n > 1) ? static_cast<double>(rank) / static_cast<double>(keep_n - 1) : 0.0;
    m.color.r = static_cast<float>(0.1 + 0.9 * t);
    m.color.g = static_cast<float>(0.9 - 0.7 * t);
    m.color.b = 0.2f;
    m.color.a = 0.95f;

    m.points.reserve(r.states.size());
    for (const auto & x : r.states) {
      geometry_msgs::msg::Point p;
      p.x = x[0];
      p.y = x[1];
      p.z = 0.03;
      m.points.push_back(p);
    }
    msg.markers.push_back(m);
  }

  rollouts_pub_->publish(msg);
}

void MppiNav2Controller::predictedPeopleCallback(
  const visualization_msgs::msg::MarkerArray::SharedPtr msg)
{
  std::unordered_map<int, std::vector<std::array<double, 2>>> path_by_id;
  std::unordered_map<int, std::unordered_map<int, std::array<double, 4>>> cov_by_id_step;

  for (const auto & m : msg->markers) {
    if (m.action != visualization_msgs::msg::Marker::ADD) {
      continue;
    }
    if (m.ns == "predicted_people_path" &&
      m.type == visualization_msgs::msg::Marker::LINE_STRIP)
    {
      std::vector<std::array<double, 2>> traj;
      traj.reserve(m.points.size());
      for (const auto & p : m.points) {
        traj.push_back({p.x, p.y});
      }
      path_by_id[m.id] = std::move(traj);
      continue;
    }
    if (m.ns != "predicted_people_cov" || m.type != visualization_msgs::msg::Marker::CYLINDER) {
      continue;
    }

    const int person_id = m.id / 100;
    const int step_idx = m.id % 100;
    if (step_idx < 0) {
      continue;
    }

    const double n_sigma = std::max(1e-3, dyn_people_cov_n_sigma_);
    const double l1 = std::pow(std::max(1e-6, m.scale.x) / (2.0 * n_sigma), 2);
    const double l2 = std::pow(std::max(1e-6, m.scale.y) / (2.0 * n_sigma), 2);

    const double qx = m.pose.orientation.x;
    const double qy = m.pose.orientation.y;
    const double qz = m.pose.orientation.z;
    const double qw = m.pose.orientation.w;
    const double siny_cosp = 2.0 * (qw * qz + qx * qy);
    const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    const double yaw = std::atan2(siny_cosp, cosy_cosp);

    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    double s00 = c * c * l1 + s * s * l2;
    double s01 = c * s * (l1 - l2);
    double s11 = s * s * l1 + c * c * l2;

    const double std_cap = std::max(1e-3, dyn_people_cov_std_cap_);
    const double max_std = std::max(std::sqrt(std::max(1e-9, s00)), std::sqrt(std::max(1e-9, s11)));
    if (max_std > std_cap) {
      const double scale = (std_cap * std_cap) / (max_std * max_std);
      s00 *= scale;
      s01 *= scale;
      s11 *= scale;
    }
    cov_by_id_step[person_id][step_idx] = {s00, s01, s01, s11};
  }

  std::vector<std::vector<DynPredictionStep>> parsed;
  parsed.reserve(path_by_id.size());
  for (auto & kv : path_by_id) {
    const int person_id = kv.first;
    auto & path = kv.second;
    std::vector<DynPredictionStep> traj;
    traj.reserve(path.size());
    const double def_var = std::max(1e-6, dyn_people_default_std_ * dyn_people_default_std_);
    std::array<double, 4> last_sigma{def_var, 0.0, 0.0, def_var};
    bool has_last_sigma = false;
    for (size_t step = 0; step < path.size(); ++step) {
      DynPredictionStep dp;
      dp.mu = path[step];
      const auto person_it = cov_by_id_step.find(person_id);
      if (person_it != cov_by_id_step.end()) {
        const auto cov_it = person_it->second.find(static_cast<int>(step));
        if (cov_it != person_it->second.end()) {
          dp.sigma = cov_it->second;
          dp.has_sigma = true;
          last_sigma = dp.sigma;
          has_last_sigma = true;
        }
      }
      if (!dp.has_sigma) {
        dp.sigma = has_last_sigma ? last_sigma : std::array<double, 4>{def_var, 0.0, 0.0, def_var};
        dp.has_sigma = true;
      }
      traj.push_back(dp);
    }
    parsed.push_back(std::move(traj));
  }
  std::lock_guard<std::mutex> lock(dyn_people_mutex_);
  dyn_people_trajs_ = std::move(parsed);
}

void MppiNav2Controller::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  // заглушка: пока не делаем ничего
}

}  // namespace nav2_controllers

PLUGINLIB_EXPORT_CLASS(
  nav2_controllers::MppiNav2Controller,
  nav2_core::Controller)
