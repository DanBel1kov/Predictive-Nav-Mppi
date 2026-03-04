#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <array>
#include <vector>

#include "rclcpp_lifecycle/lifecycle_node.hpp"
#include "rclcpp/publisher.hpp"
#include "rclcpp/subscription.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "tf2_ros/buffer.h"
#include "nav2_core/controller.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"

#include "nav2_controllers/mppi_controller.hpp"  // ядро MPPI

namespace nav2_controllers
{

class MppiNav2Controller : public nav2_core::Controller
{
public:
  MppiNav2Controller() = default;
  ~MppiNav2Controller() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;

  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

private:
  double yawFromQuat(const geometry_msgs::msg::Quaternion & q) const;
  void publishRollouts(const std::vector<RolloutDebug> & rollouts, const rclcpp::Time & stamp);
  void predictedPeopleCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg);

  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::string name_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_{nullptr};

  std::string global_frame_;
  std::string robot_base_frame_;

  nav_msgs::msg::Path global_plan_;

  MPPIParams params_;
  double goal_lookahead_dist_{0.6};
  MPPIController mppi_;

  bool debug_rollouts_enabled_{false};
  int debug_rollouts_top_n_{10};
  double debug_rollouts_line_width_{0.02};
  std::string debug_rollouts_topic_{"/mppi_debug_rollouts"};
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr rollouts_pub_;

  bool dyn_people_enabled_{true};
  std::string dyn_people_topic_{"/predicted_people_markers"};
  double dyn_people_cov_n_sigma_{2.0};
  double dyn_people_default_std_{0.35};
  double dyn_people_cov_std_cap_{0.8};
  std::vector<std::vector<DynPredictionStep>> dyn_people_trajs_;
  std::mutex dyn_people_mutex_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr dyn_people_sub_;
};

}  // namespace my_nav2_controllers
