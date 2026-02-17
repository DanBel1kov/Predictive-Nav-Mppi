import math
import sys
from pathlib import Path


PKG_ROOT = Path(__file__).resolve().parents[1] / "predictive_nav_mppi"
if str(PKG_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT.parent))

from predictive_nav_mppi.kf_cv import (  # noqa: E402
    TrackState,
    build_f_q,
    predict_state_cov,
    prune_stale_tracks,
    update_state_cov,
)


def test_build_f_q_shapes_and_positive_diagonal():
    f, q = build_f_q(0.1, 0.6)
    assert len(f) == 4
    assert len(q) == 4
    for row in f:
        assert len(row) == 4
    for row in q:
        assert len(row) == 4
    assert q[0][0] > 0.0
    assert q[1][1] > 0.0
    assert q[2][2] > 0.0
    assert q[3][3] > 0.0


def test_linear_motion_filter_beats_raw_measurement():
    dt = 0.1
    true_vx = 1.0
    sigma_meas = 0.08
    sigma_acc = 0.2

    mu = [0.0, 0.0, 0.0, 0.0]
    sigma = [
        [0.25, 0.0, 0.0, 0.0],
        [0.0, 0.25, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    raw_err = []
    kf_err = []
    for k in range(1, 31):
        true_x = true_vx * dt * k
        noise = 0.25 if (k % 2 == 0) else -0.25
        z_x = true_x + noise
        z_y = 0.0
        mu, sigma = predict_state_cov(mu, sigma, dt, sigma_acc)
        mu, sigma = update_state_cov(mu, sigma, z_x, z_y, sigma_meas)
        raw_err.append(abs(z_x - true_x))
        kf_err.append(abs(mu[0] - true_x))

    assert sum(kf_err[-10:]) / 10.0 < sum(raw_err[-10:]) / 10.0
    assert math.isfinite(mu[0])
    assert sigma[0][0] >= 0.0


def test_prune_stale_tracks():
    tracks = {
        1: TrackState(
            mu=[0.0, 0.0, 0.0, 0.0],
            sigma=[[0.0] * 4 for _ in range(4)],
            last_update_sec=0.1,
        ),
        2: TrackState(
            mu=[0.0, 0.0, 0.0, 0.0],
            sigma=[[0.0] * 4 for _ in range(4)],
            last_update_sec=0.9,
        ),
    }
    prune_stale_tracks(tracks, now_sec=1.2, timeout_sec=0.5)
    assert 1 not in tracks
    assert 2 in tracks
