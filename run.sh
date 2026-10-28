#!/usr/bin/env bash
set -euo pipefail
# Tasks: experiment (train/calibrate/evaluate), infer (unlabeled open-set),
# train, predict, profile, discover, cluster. Use TASK --help for options.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec "${PYTHON:-python}" main.py "$@"
