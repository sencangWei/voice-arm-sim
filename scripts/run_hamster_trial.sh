#!/usr/bin/env bash
# One bounded, CPU-only checkpoint download -> integrity check -> path probe.
# This script never drives a robot or changes an existing model service.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:?Usage: bash scripts/run_hamster_trial.sh NEW_OUTPUT_DIRECTORY}"
mkdir -p logs
status="logs/hamster-trial.status"
phase="starting"
trap 'printf "%s exit=%s\n" "$phase" "$?" > "$status"' EXIT
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
phase="downloading"
printf '%s\n' "$phase" > "$status"
timeout --kill-after=20s 3h .venv-hamster/bin/python -u scripts/fetch_hamster_ranges.py
phase="verifying-checkpoint"
printf '%s\n' "$phase" > "$status"
.venv-hamster/bin/python -u scripts/prepare_hamster.py --verify-only
phase="probing-official-model"
printf '%s\n' "$phase" > "$status"
HF_HUB_OFFLINE=1 timeout --kill-after=20s 2h .venv-hamster/bin/python -u -m vla.hamster \
  --image evidence/hamster/scene-init0.png \
  --commands evidence/hamster/commands.json \
  --output "$output" --threads 8 --max-new-tokens 300
phase="probe-finished-needs-visual-review"
