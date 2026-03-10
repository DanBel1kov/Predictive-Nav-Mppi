#pragma once

#include <array>
#include <vector>

#include "nav2_costmap_2d/costmap_2d.hpp"


namespace nav2_controllers
{

struct MPPIParams
{
  double dt = 0.1;
  int horizon_steps = 20;
  int n_rollouts = 200;

  double v_mean = 0.2;
  double omega_mean = 0.0;

  double v_std = 0.04;
  double omega_std = 0.25;

  double v_min = -0.05;
  double v_max = 0.15;
  double omega_min = -1.0;
  double omega_max = 1.0;

  double w_goal = 1.0;
  double w_obs = 50.0;
  double w_ctrl = 0.3;
  double w_speed = 0.1;
  double w_path = 1.0;
  double w_reverse = 5.0;   // штраф за езду назад (аналог PreferForwardCritic)
  int path_lookahead_steps = 5;  // PathFollowCritic: offset ahead of nearest path idx
  double w_dyn_obs = 20.0;
  double dyn_risk_delta = 0.05;
  double dyn_risk_beta = 1.0;
  double dyn_robot_var = 0.0;

  double lambda = 1.0;
};

struct DynPredictionStep
{
  std::array<double, 2> mu{0.0, 0.0};
  std::array<double, 4> sigma{0.0, 0.0, 0.0, 0.0};  // row-major 2x2
  bool has_sigma{false};
};

struct RolloutDebug
{
  std::vector<std::array<double, 3>> states;
  double cost{0.0};
};

class MPPIController
{
public:
  explicit MPPIController(const MPPIParams & params = MPPIParams());

  /// x0 = [x, y, theta]
  /// goal = [x_goal, y_goal]
  /// costmap может быть nullptr
  std::array<double, 2> computeControl(
    const std::array<double, 3> & x0,
    const std::array<double, 2> & goal,
    const nav2_costmap_2d::Costmap2D * costmap,
    const std::vector<std::array<double, 2>> * path_xy = nullptr,
    const std::vector<std::vector<DynPredictionStep>> * dyn_predictions = nullptr,
    std::vector<RolloutDebug> * rollout_debug = nullptr);

private:
  std::array<double, 3> dynamics(
    const std::array<double, 3> & x,
    const std::array<double, 2> & u) const;

  double stageCost(
    const std::array<double, 3> & x,
    int step_idx,
    const std::array<double, 2> & u,
    const std::array<double, 2> & goal,
    const nav2_costmap_2d::Costmap2D * costmap,
    const std::vector<std::array<double, 2>> * path_xy,
    const std::vector<std::vector<DynPredictionStep>> * dyn_predictions) const;

  MPPIParams p_;
  std::vector<double> u_seq_;
};

}  // namespace my_nav2_controllers
