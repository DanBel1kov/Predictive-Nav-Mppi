#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "nav2_mppi_controller/critic_function.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace mppi::critics
{

class PredictedPeopleCritic : public CriticFunction
{
public:
  void initialize() override;
  void score(CriticData & data) override;

private:
  struct Pt2
  {
    float x{0.0f};
    float y{0.0f};
  };

  void predictedPeopleCallback(
    const visualization_msgs::msg::MarkerArray::SharedPtr msg);

  std::mutex people_mutex_;
  std::vector<std::vector<Pt2>> people_trajs_;
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr people_sub_;

  std::string people_topic_{"/predicted_people_markers"};
  std::string people_path_ns_{"predicted_people_path"};
  float cost_weight_{40.0f};
  float sigma_{0.6f};
  float collision_distance_{0.35f};
  float collision_cost_{400.0f};
  float time_decay_{0.98f};
  float max_consider_distance_{2.5f};
  int nearest_people_per_step_{3};
  int max_people_{64};
};

}  // namespace mppi::critics

