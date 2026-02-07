#include "nav2_controllers/mppi_nav2_controller.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_util/node_utils.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

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
    node, prefix + "lambda", rclcpp::ParameterValue(params_.lambda));
  nav2_util::declare_parameter_if_not_declared(
    node, prefix + "goal_lookahead_dist", rclcpp::ParameterValue(goal_lookahead_dist_));

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
  node->get_parameter(prefix + "lambda", params_.lambda);
  node->get_parameter(prefix + "goal_lookahead_dist", goal_lookahead_dist_);

  mppi_ = MPPIController(params_);
}

void MppiNav2Controller::cleanup()
{
  costmap_ros_.reset();
  tf_.reset();
  costmap_ = nullptr;
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
  auto u = mppi_.computeControl(x0, goal_xy, cm, &path_xy);

  cmd.twist.linear.x = static_cast<float>(u[0]);
  cmd.twist.angular.z = static_cast<float>(u[1]);
  cmd.header.stamp = node_.lock()->now();
  cmd.header.frame_id = robot_base_frame_;   // base_link

  return cmd;
}

void MppiNav2Controller::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  // заглушка: пока не делаем ничего
}

}  // namespace nav2_controllers

PLUGINLIB_EXPORT_CLASS(
  nav2_controllers::MppiNav2Controller,
  nav2_core::Controller)
