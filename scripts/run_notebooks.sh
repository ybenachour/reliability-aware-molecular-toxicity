#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
PROFILE="${TOX_SCREEN_PROFILE:-smoke}"
mkdir -p reports/executed_notebooks logs
for nb in notebooks/*.ipynb; do
  base=$(basename "$nb")
  echo "[${PROFILE}] executing ${base}"
  papermill "$nb" "reports/executed_notebooks/${base}" \
    -p execution_profile "$PROFILE" \
    --log-output 2>&1 | tee "logs/${base%.ipynb}.log"
done
