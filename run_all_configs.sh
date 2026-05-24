#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

configs=(
  "configs/config_2500_variance_5_fold_x20_run_01.toml"
  "configs/config_2500_variance_5_fold_x20_run_02.toml"
  "configs/config_2500_variance_5_fold_x20_run_03.toml"
  "configs/config_2500_variance_5_fold_x20_run_04.toml"
  "configs/config_2500_variance_5_fold_x20_run_05.toml"
)

if command -v uv >/dev/null 2>&1; then
  runner=(uv run python)
elif [[ -x ".venv/bin/python" ]]; then
  runner=(.venv/bin/python)
else
  runner=(python3)
fi

echo "Using runner: ${runner[*]}"

for config in "${configs[@]}"; do
  config_path="$ROOT_DIR/$config"

  if [[ ! -f "$config_path" ]]; then
    echo "Missing config: $config_path" >&2
    exit 1
  fi

  echo
  echo "============================================================"
  echo "Running single_main.py with $config"
  echo "Started: $(date)"
  echo "============================================================"

  CONFIG_TOML="$config_path" "${runner[@]}" single_main.py

  echo "Finished: $(date)"
done

echo
echo "All config runs completed."
