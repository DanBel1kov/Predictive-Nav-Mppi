#include "nav2_controllers/mppi_nav2_controller.hpp"

#include <cmath>
#include <stdexcept>

#include "pluginlib/class_list_macros.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_util/node_utils.hpp"

namespace nav2_controllers
{

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

  const auto & goal_pose = global_plan_.poses.back();
  std::array<double, 2> goal_xy = {
    goal_pose.pose.position.x,
    goal_pose.pose.position.y
  };

  const nav2_costmap_2d::Costmap2D * cm = costmap_;
  auto u = mppi_.computeControl(x0, goal_xy, cm);

  cmd.twist.linear.x = static_cast<float>(u[0]);
  cmd.twist.angular.z = static_cast<float>(u[1]);
  cmd.header = pose.header;
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
