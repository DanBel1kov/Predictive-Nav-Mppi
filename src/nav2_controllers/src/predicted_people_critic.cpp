#include "nav2_controllers/predicted_people_critic.hpp"

#include <algorithm>
#include <cmath>

#include "visualization_msgs/msg/marker.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace mppi::critics
{

void PredictedPeopleCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(people_topic_, "people_topic", std::string("/predicted_people_markers"));
  getParam(people_path_ns_, "people_path_ns", std::string("predicted_people_path"));
  getParam(cost_weight_, "cost_weight", 40.0f);
  getParam(sigma_, "sigma", 0.6f);
  getParam(collision_distance_, "collision_distance", 0.35f);
  getParam(collision_cost_, "collision_cost", 400.0f);
  getParam(time_decay_, "time_decay", 0.98f);
  getParam(max_consider_distance_, "max_consider_distance", 2.5f);
  getParam(nearest_people_per_step_, "nearest_people_per_step", 3);
  getParam(max_people_, "max_people", 64);

  auto node = parent_.lock();
  if (!node) {
    return;
  }

  people_sub_ = node->create_subscription<visualization_msgs::msg::MarkerArray>(
    people_topic_, rclcpp::QoS(10),
    std::bind(&PredictedPeopleCritic::predictedPeopleCallback, this, std::placeholders::_1));
}

void PredictedPeopleCritic::predictedPeopleCallback(
  const visualization_msgs::msg::MarkerArray::SharedPtr msg)
{
  std::vector<std::vector<Pt2>> parsed;
  parsed.reserve(std::max(0, max_people_));

  for (const auto & marker : msg->markers) {
    if (static_cast<int>(parsed.size()) >= max_people_) {
      break;
    }
    if (marker.action != visualization_msgs::msg::Marker::ADD) {
      continue;
    }
    if (marker.ns != people_path_ns_) {
      continue;
    }
    if (marker.type != visualization_msgs::msg::Marker::LINE_STRIP) {
      continue;
    }
    if (marker.points.empty()) {
      continue;
    }

    std::vector<Pt2> traj;
    traj.reserve(marker.points.size());
    for (const auto & p : marker.points) {
      traj.push_back({static_cast<float>(p.x), static_cast<float>(p.y)});
    }
    parsed.push_back(std::move(traj));
  }

  std::lock_guard<std::mutex> lock(people_mutex_);
  people_trajs_ = std::move(parsed);
}

void PredictedPeopleCritic::score(CriticData & data)
{
  if (!enabled_) {
    return;
  }

  std::vector<std::vector<Pt2>> people_snapshot;
  {
    std::lock_guard<std::mutex> lock(people_mutex_);
    people_snapshot = people_trajs_;
  }
  if (people_snapshot.empty()) {
    return;
  }

  const auto batch_size = data.trajectories.x.shape(0);
  const auto time_steps = data.trajectories.x.shape(1);
  const float sigma = std::max(1e-3f, sigma_);
  const float sigma_sq = sigma * sigma;
  const float collision_d = std::max(1e-3f, collision_distance_);
  const float collision_d_sq = collision_d * collision_d;
  const float consider_d = std::max(collision_d, max_consider_distance_);
  const float consider_d_sq = consider_d * consider_d;
  const int top_k = std::max(1, nearest_people_per_step_);

  for (size_t b = 0; b < batch_size; ++b) {
    float risk_cost = 0.0f;
    float decay = 1.0f;

    for (size_t t = 0; t < time_steps; ++t) {
      const float rx = data.trajectories.x(b, t);
      const float ry = data.trajectories.y(b, t);
      std::vector<float> local_risks;
      local_risks.reserve(people_snapshot.size());
      float collision_hits = 0.0f;

      for (const auto & person_traj : people_snapshot) {
        if (person_traj.empty()) {
          continue;
        }
        const size_t ti = std::min(t, person_traj.size() - 1);
        const float dx = rx - person_traj[ti].x;
        const float dy = ry - person_traj[ti].y;
        const float d_sq = dx * dx + dy * dy;
        if (d_sq > consider_d_sq) {
          continue;
        }

        const float r = std::exp(-0.5f * d_sq / sigma_sq);
        if (d_sq < collision_d_sq) {
          collision_hits += 1.0f;
        }
        local_risks.push_back(r);
      }

      float step_risk = 0.0f;
      if (!local_risks.empty()) {
        if (static_cast<int>(local_risks.size()) > top_k) {
          std::nth_element(
            local_risks.begin(),
            local_risks.begin() + top_k,
            local_risks.end(),
            std::greater<float>());
          local_risks.resize(top_k);
        }
        for (const float r : local_risks) {
          step_risk += r;
        }
        // Average over selected nearby people to avoid over-penalizing dense distant crowds.
        step_risk /= static_cast<float>(local_risks.size());
      }
      step_risk += collision_hits * collision_cost_;

      risk_cost += decay * step_risk;
      decay *= time_decay_;
    }

    // Average over horizon, so scale does not explode with longer time_steps.
    risk_cost /= static_cast<float>(std::max<size_t>(1, time_steps));
    data.costs(b) += cost_weight_ * risk_cost;
  }
}

}  // namespace mppi::critics

PLUGINLIB_EXPORT_CLASS(mppi::critics::PredictedPeopleCritic, mppi::critics::CriticFunction)

