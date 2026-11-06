#!/usr/bin/env bash
# v2 strict burst submission. Usage: bash test2.sh predictions.csv output_dir DAT_root
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: bash test2.sh predictions.csv new_output_dir DAT_root" >&2
  exit 2
fi
python prepare_submission.py --predictions "$1" --output "$2" --input-root "$3" \
  --task2-level burst --frequency-policy error --unresolved error
