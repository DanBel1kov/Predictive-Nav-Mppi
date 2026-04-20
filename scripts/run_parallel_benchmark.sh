#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_parallel_benchmark.sh --config /tmp/bench.yaml [--workers 3] [--base-domain 11] [--base-port 11345]

Runs the current benchmark config in N parallel workers:
  - splits benchmark.goals across workers
  - assigns unique ROS_DOMAIN_ID / GAZEBO_MASTER_URI per worker
  - waits for all workers
  - merges per-worker summary.json files into one combined result directory

Outputs:
  benchmark_parallel/<timestamp>/
    worker_0.log
    worker_1.log
    ...
    worker_0_config.yaml
    worker_1_config.yaml
    ...
    merged_summary.json
    merged_results.csv
    merged_stats.txt
EOF
}

CONFIG=""
WORKERS=3
BASE_DOMAIN=11
BASE_PORT=11345

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --base-domain)
      BASE_DOMAIN="$2"
      shift 2
      ;;
    --base-port)
      BASE_PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "--config is required" >&2
  usage >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT_DIR/benchmark_parallel/$STAMP"
mkdir -p "$RUN_ROOT"

echo "[parallel] root: $RUN_ROOT"
echo "[parallel] splitting config: $CONFIG into $WORKERS workers"

python3 - "$CONFIG" "$RUN_ROOT" "$WORKERS" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
run_root = Path(sys.argv[2])
workers = int(sys.argv[3])

cfg = yaml.safe_load(config_path.read_text())
bench = cfg["benchmark"]
goals = list(bench.get("goals", []))
if not goals:
    raise SystemExit("benchmark.goals is empty")

workers = max(1, min(workers, len(goals)))
chunks = [goals[i::workers] for i in range(workers)]

for i, chunk in enumerate(chunks):
    worker_cfg = yaml.safe_load(config_path.read_text())
    b = worker_cfg["benchmark"]
    b["goals"] = chunk
    b["output_dir"] = str(run_root / f"worker_{i}_results")
    out = run_root / f"worker_{i}_config.yaml"
    out.write_text(yaml.safe_dump(worker_cfg, sort_keys=False))
    print(out)
PY

PIDS=()
LOGS=()
CONFIGS=()

mapfile -t CONFIGS < <(find "$RUN_ROOT" -maxdepth 1 -name 'worker_*_config.yaml' | sort)

for idx in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$idx]}"
  log="$RUN_ROOT/worker_${idx}.log"
  domain=$((BASE_DOMAIN + idx))
  port=$((BASE_PORT + idx))
  LOGS+=("$log")

  echo "[parallel] worker $idx: ROS_DOMAIN_ID=$domain GAZEBO_MASTER_URI=http://127.0.0.1:$port"
  (
    cd "$ROOT_DIR"
    set +u
    source /opt/ros/humble/setup.bash
    source /home/danbel1kov/hunav_ws/install/setup.bash
    source "$ROOT_DIR/install/setup.bash"
    set -u
    export ROS_DOMAIN_ID="$domain"
    export GAZEBO_MASTER_URI="http://127.0.0.1:$port"
    export PREDICTIVE_NAV_MPPI_DISABLE_GLOBAL_CLEANUP=1
    ros2 run predictive_nav_mppi run_benchmark --config "$cfg"
  ) >"$log" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  if wait "$pid"; then
    echo "[parallel] worker $idx finished"
  else
    echo "[parallel] worker $idx failed, see ${LOGS[$idx]}" >&2
    FAIL=1
  fi
done

python3 - "$RUN_ROOT" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
summary_paths = sorted(run_root.glob("worker_*_results/*/summary.json"))
if not summary_paths:
    print("No worker summary.json files found", file=sys.stderr)
    raise SystemExit(1)

all_results = []
source_dirs = []
for path in summary_paths:
    source_dirs.append(str(path.parent))
    data = json.loads(path.read_text())
    if isinstance(data, list):
        all_results.extend(data)

merged_json = run_root / "merged_summary.json"
merged_json.write_text(json.dumps({
    "worker_output_dirs": source_dirs,
    "results": all_results,
}, indent=2))

if all_results:
    merged_csv = run_root / "merged_results.csv"
    rows = []
    for r in all_results:
      row = dict(r)
      for key in ("goal", "start"):
          if isinstance(row.get(key), dict):
              row[key] = f"({row[key]['x']}, {row[key]['y']})"
      rows.append(row)
    with merged_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

ok = [r for r in all_results if r.get("status") == "SUCCEEDED"]

def stats(vals):
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
    return mean, math.sqrt(var), min(vals), max(vals)

lines = []
lines.append(f"episodes={len(all_results)} succeeded={len(ok)}")
for label, key in [
    ("time_to_goal", "time_to_goal"),
    ("path_length", "path_length"),
    ("min_dist", "min_dist"),
    ("collision_count", "collision_count"),
    ("viol_time", "viol_time"),
]:
    s = stats([float(r[key]) for r in ok]) if ok else None
    if s is None:
        lines.append(f"{label}: n/a")
    else:
        mean, std, vmin, vmax = s
        lines.append(f"{label}: {mean:.3f} ± {std:.3f} [{vmin:.3f} .. {vmax:.3f}]")

(run_root / "merged_stats.txt").write_text("\n".join(lines) + "\n")
print(run_root)
PY

if [[ "$FAIL" -ne 0 ]]; then
  echo "[parallel] completed with failures; see worker logs in $RUN_ROOT" >&2
  exit 1
fi

echo "[parallel] merged results written to $RUN_ROOT"
if [[ -f "$RUN_ROOT/merged_stats.txt" ]]; then
  echo
  echo "================ MERGED STATS ================"
  cat "$RUN_ROOT/merged_stats.txt"
  echo "=============================================="
fi
