#include "nav2_controllers/mppi_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>

#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_costmap_2d/cost_values.hpp"

namespace nav2_controllers
{

MPPIController::MPPIController(const MPPIParams & params)
: p_(params)
{
  // Первый запуск — нули, без bias на v_mean
  u_seq_.resize(p_.horizon_steps * 2, 0.0);
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
  int step_idx,
  const std::array<double, 2> & u,
  const std::array<double, 2> & goal,
  const nav2_costmap_2d::Costmap2D * costmap,
  const std::vector<std::array<double, 2>> * /*path_xy*/,  // не используется — path cost вычисляется в computeControl
  const std::vector<std::vector<DynPredictionStep>> * dyn_predictions) const
{
  // 1) расстояние до цели
  const double dx = x[0] - goal[0];
  const double dy = x[1] - goal[1];
  double cost = p_.w_goal * (dx * dx + dy * dy);

  // 2) штраф за управление
  cost += p_.w_ctrl * (u[0] * u[0] + u[1] * u[1]);

  // 2.5) штраф за езду назад (аналог Nav2 PreferForwardCritic)
  if (u[0] < 0.0) {
    cost += p_.w_reverse * u[0] * u[0];
  }

  // 3) препятствия через costmap2d (если доступен)
  if (costmap != nullptr) {
    unsigned int mx, my;
    if (costmap->worldToMap(x[0], x[1], mx, my)) {
      const unsigned char c = costmap->getCost(mx, my);
      double obstacle_penalty = 0.0;

      if (c == nav2_costmap_2d::LETHAL_OBSTACLE ||
          c == nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
        // Стены — жёсткое ограничение: множитель 20x, всегда дороже людей
        obstacle_penalty = p_.w_obs * 20.0;
      } else if (c == nav2_costmap_2d::NO_INFORMATION) {
        obstacle_penalty = p_.w_obs * 2.0;
      } else {
        const double norm = static_cast<double>(c) / 252.0;
        obstacle_penalty = p_.w_obs * norm * norm;
      }

      cost += obstacle_penalty;
    } else {
      // точка вне карты — тоже стена
      cost += p_.w_obs * 20.0;
    }
  }

  // 3.5) chance-constrained risk для динамических препятствий на шаге t
  // Суммарный штраф от людей ограничен cap'ом — чтобы толпа не была дороже стены
  if (dyn_predictions != nullptr && p_.w_dyn_obs > 0.0) {
    const double delta = std::max(1e-6, std::min(0.49, p_.dyn_risk_delta));
    const double c = -2.0 * std::log(delta);  // chi-square quantile for dof=2
    const double beta = std::max(1e-3, p_.dyn_risk_beta);
    const double robot_var = std::max(0.0, p_.dyn_robot_var);
    double dyn_cost_sum = 0.0;
    for (const auto & traj : *dyn_predictions) {
      if (step_idx < 0 || static_cast<size_t>(step_idx) >= traj.size()) {
        continue;
      }
      const auto & pred = traj[static_cast<size_t>(step_idx)];
      const double dx = x[0] - pred.mu[0];
      const double dy = x[1] - pred.mu[1];

      if (!pred.has_sigma) {
        continue;
      }

      const double s00 = pred.sigma[0] + robot_var;
      const double s01 = pred.sigma[1];
      const double s10 = pred.sigma[2];
      const double s11 = pred.sigma[3] + robot_var;
      const double det = s00 * s11 - s01 * s10;
      if (det < 1e-12) {
        continue;
      }

      const double inv00 = s11 / det;
      const double inv01 = -s01 / det;
      const double inv10 = -s10 / det;
      const double inv11 = s00 / det;
      const double d2 = dx * (inv00 * dx + inv01 * dy) + dy * (inv10 * dx + inv11 * dy);

      const double margin = c - d2;
      const double softplus = (margin > 30.0 / beta) ?
        margin :
        std::log1p(std::exp(beta * margin)) / beta;
      dyn_cost_sum += p_.w_dyn_obs * softplus;
    }
    // Cap: суммарный штраф от людей не превышает половины стоимости летального препятствия
    // Это гарантирует что стена ВСЕГДА хуже любого числа людей
    const double dyn_cap = p_.w_obs * 8.0;
    cost += std::min(dyn_cost_sum, dyn_cap);
  }

  // 4) награда за движение вперёд
  cost -= p_.w_speed * u[0] * p_.dt;

  return cost;
}

std::array<double, 2> MPPIController::computeControl(
  const std::array<double, 3> & x0,
  const std::array<double, 2> & goal,
  const nav2_costmap_2d::Costmap2D * costmap,
  const std::vector<std::array<double, 2>> * path_xy,
  const std::vector<std::vector<DynPredictionStep>> * dyn_predictions,
  std::vector<RolloutDebug> * rollout_debug)
{
  const int K = p_.n_rollouts;
  const int T = p_.horizon_steps;
  const double lambda = std::max(1e-6, p_.lambda);

  // PathFollowCritic: найти ближайшую точку пути к x0, взять target = nearest + lookahead
  // Это тянет робота вперёд по пути, а не просто держит близко к любой точке
  std::array<double, 2> path_target{x0[0], x0[1]};  // fallback = текущая позиция
  if (path_xy != nullptr && !path_xy->empty()) {
    const int N = static_cast<int>(path_xy->size());
    int nearest_idx = 0;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (int i = 0; i < N; ++i) {
      const double dx = x0[0] - (*path_xy)[i][0];
      const double dy = x0[1] - (*path_xy)[i][1];
      const double d2 = dx * dx + dy * dy;
      if (d2 < best_d2) {
        best_d2 = d2;
        nearest_idx = i;
      }
    }
    const int target_idx = std::min(nearest_idx + p_.path_lookahead_steps, N - 1);
    path_target = (*path_xy)[target_idx];
  }

  // шум (K * T * 2)
  std::vector<double> noise(K * T * 2, 0.0);
  std::vector<double> costs(K, 0.0);
  if (rollout_debug != nullptr) {
    rollout_debug->clear();
    rollout_debug->reserve(K);
  }

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
    RolloutDebug dbg;
    if (rollout_debug != nullptr) {
      dbg.states.reserve(static_cast<size_t>(T) + 1U);
      dbg.states.push_back(x0);
    }

    for (int t = 0; t < T; ++t) {
      const int idx = (k * T + t) * 2;

      double v = u_seq_[2 * t + 0] + noise[idx + 0];
      double w = u_seq_[2 * t + 1] + noise[idx + 1];

      // клип по ограничениям
      v = std::max(p_.v_min, std::min(p_.v_max, v));
      w = std::max(p_.omega_min, std::min(p_.omega_max, w));

      std::array<double, 2> u{v, w};
      x = dynamics(x, u);
      if (rollout_debug != nullptr) {
        dbg.states.push_back(x);
      }

      total_cost += stageCost(x, t, u, goal, costmap, nullptr, dyn_predictions);

      // PathFollowCritic: расстояние до lookahead-точки пути (тянет робота вперёд по пути)
      if (p_.w_path > 0.0) {
        const double pdx = x[0] - path_target[0];
        const double pdy = x[1] - path_target[1];
        total_cost += p_.w_path * (pdx * pdx + pdy * pdy);
      }
    }

    costs[k] = total_cost;
    if (rollout_debug != nullptr) {
      dbg.cost = total_cost;
      rollout_debug->push_back(std::move(dbg));
    }
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

  // warm-start: сдвигаем горизонт на один шаг вперёд
  // хвост заполняем нулями (нейтральное управление, без v_mean bias)
  for (int t = 0; t < T - 1; ++t) {
    u_seq_[2 * t + 0] = u_seq_[2 * (t + 1) + 0];
    u_seq_[2 * t + 1] = u_seq_[2 * (t + 1) + 1];
  }
  u_seq_[2 * (T - 1) + 0] = 0.0;
  u_seq_[2 * (T - 1) + 1] = 0.0;

  return u0;
}

}  // namespace nav2_controllers
