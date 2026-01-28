#include "nav2_controllers/mppi_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/cost_values.hpp"

namespace nav2_controllers
{

MPPIController::MPPIController(const MPPIParams & params)
: p_(params)
{
  u_seq_.resize(p_.horizon_steps * 2);
  for (int t = 0; t < p_.horizon_steps; ++t) {
    u_seq_[2 * t + 0] = p_.v_mean;
    u_seq_[2 * t + 1] = p_.omega_mean;
  }
}

std::array<double, 3> MPPIController::dynamics(
  const std::array<double, 3> & x,
  const std::array<double, 2> & u) const
{
  const double v = u[0];
  const double w = u[1];
  const double theta = x[2];
  const double dt = p_.dt;

  std::array<double, 3> x_next;
  x_next[0] = x[0] + v * std::cos(theta) * dt;
  x_next[1] = x[1] + v * std::sin(theta) * dt;
  x_next[2] = x[2] + w * dt;
  return x_next;
}

double MPPIController::stageCost(
  const std::array<double, 3> & x,
  const std::array<double, 2> & u,
  const std::array<double, 2> & goal,
  const nav2_costmap_2d::Costmap2D * costmap) const
{
  // 1) расстояние до цели
  const double dx = x[0] - goal[0];
  const double dy = x[1] - goal[1];
  double cost = p_.w_goal * (dx * dx + dy * dy);

  // 2) штраф за управление
  cost += p_.w_ctrl * (u[0] * u[0] + u[1] * u[1]);

  // 3) препятствия через costmap2d (если доступен)
  if (costmap != nullptr) {
    unsigned int mx, my;
    if (costmap->worldToMap(x[0], x[1], mx, my)) {
      const unsigned char c = costmap->getCost(mx, my);

      if (c == nav2_costmap_2d::LETHAL_OBSTACLE ||
          c == nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE ||
          c == nav2_costmap_2d::NO_INFORMATION) {
        // очень сильный штраф за коллизию / неизвестность
        cost += p_.w_obs * 1000.0;
      } else {
        // нормализуем [0..252] → [0..1] и добавляем штраф
        const double norm = static_cast<double>(c) / 252.0;
        cost += p_.w_obs * norm;
      }
    } else {
      // точка вне карты — тоже считаем плохо
      cost += p_.w_obs * 1000.0;
    }
  }

  // 4) награда за движение вперёд
  cost -= p_.w_speed * u[0] * p_.dt;

  return cost;
}

std::array<double, 2> MPPIController::computeControl(
  const std::array<double, 3> & x0,
  const std::array<double, 2> & goal,
  const nav2_costmap_2d::Costmap2D * costmap)
{
  const int K = p_.n_rollouts;
  const int T = p_.horizon_steps;
  const double lambda = std::max(1e-6, p_.lambda);

  // шум (K * T * 2)
  std::vector<double> noise(K * T * 2, 0.0);
  std::vector<double> costs(K, 0.0);

  static thread_local std::mt19937 gen{std::random_device{}()};
  std::normal_distribution<double> dist(0.0, 1.0);

  // генерим шум
  for (int k = 0; k < K; ++k) {
    for (int t = 0; t < T; ++t) {
      const int idx = (k * T + t) * 2;
      const double z_v = dist(gen);
      const double z_w = dist(gen);
      noise[idx + 0] = z_v * p_.v_std;
      noise[idx + 1] = z_w * p_.omega_std;
    }
  }

  // прогоняем траектории
  for (int k = 0; k < K; ++k) {
    std::array<double, 3> x = x0;
    double total_cost = 0.0;

    for (int t = 0; t < T; ++t) {
      const int idx = (k * T + t) * 2;

      double v = u_seq_[2 * t + 0] + noise[idx + 0];
      double w = u_seq_[2 * t + 1] + noise[idx + 1];

      // клип по ограничениям
      v = std::max(p_.v_min, std::min(p_.v_max, v));
      w = std::max(p_.omega_min, std::min(p_.omega_max, w));

      std::array<double, 2> u{v, w};
      x = dynamics(x, u);

      total_cost += stageCost(x, u, goal, costmap);
    }

    costs[k] = total_cost;
  }

  // нормировка стоимостей
  const double beta = *std::min_element(costs.begin(), costs.end());
  std::vector<double> weights(K, 0.0);
  double sum_w = 0.0;
  for (int k = 0; k < K; ++k) {
    const double w = std::exp(-(costs[k] - beta) / lambda);
    weights[k] = w;
    sum_w += w;
  }

  if (sum_w < 1e-9) {
    const double w_eq = 1.0 / static_cast<double>(K);
    for (int k = 0; k < K; ++k) {
      weights[k] = w_eq;
    }
  } else {
    for (int k = 0; k < K; ++k) {
      weights[k] /= sum_w;
    }
  }

  // du = суммарный шум с весами
  std::vector<double> du(T * 2, 0.0);
  for (int k = 0; k < K; ++k) {
    const double wk = weights[k];
    for (int t = 0; t < T; ++t) {
      const int idx = (k * T + t) * 2;
      du[2 * t + 0] += wk * noise[idx + 0];
      du[2 * t + 1] += wk * noise[idx + 1];
    }
  }

  // обновляем u_seq
  for (int t = 0; t < T; ++t) {
    u_seq_[2 * t + 0] += du[2 * t + 0];
    u_seq_[2 * t + 1] += du[2 * t + 1];

    u_seq_[2 * t + 0] =
      std::max(p_.v_min, std::min(p_.v_max, u_seq_[2 * t + 0]));
    u_seq_[2 * t + 1] =
      std::max(p_.omega_min, std::min(p_.omega_max, u_seq_[2 * t + 1]));
  }

  // первое управление
  std::array<double, 2> u0{
    u_seq_[0],
    u_seq_[1]
  };

  // сдвиг горизонта
  for (int t = 0; t < T - 1; ++t) {
    u_seq_[2 * t + 0] = u_seq_[2 * (t + 1) + 0];
    u_seq_[2 * t + 1] = u_seq_[2 * (t + 1) + 1];
  }
  u_seq_[2 * (T - 1) + 0] = p_.v_mean;
  u_seq_[2 * (T - 1) + 1] = p_.omega_mean;

  return u0;
}

}  // namespace nav2_controllers
