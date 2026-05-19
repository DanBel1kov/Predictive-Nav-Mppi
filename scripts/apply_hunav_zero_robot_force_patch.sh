#!/usr/bin/env bash
set -euo pipefail

HUNAV_SIM_REPO="${HUNAV_SIM_REPO:-$HOME/hunav_ws/src/hunav_sim}"
PATCH_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/patches/hunav_zero_robot_force.patch"

if [[ ! -d "$HUNAV_SIM_REPO/.git" ]]; then
  echo "HuNav repo not found: $HUNAV_SIM_REPO" >&2
  exit 1
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "Patch file not found: $PATCH_FILE" >&2
  exit 1
fi

cd "$HUNAV_SIM_REPO"
git apply --check "$PATCH_FILE"
git apply "$PATCH_FILE"
cd "$HOME/hunav_ws"
colcon build --packages-select hunav_agent_manager

echo "Applied $PATCH_FILE and rebuilt hunav_agent_manager"
