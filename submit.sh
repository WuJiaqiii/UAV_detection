R11="$REPO/experiment/open_set_inference_替换为1-1实际时间"
R12="$REPO/experiment/open_set_inference_替换为1-2实际时间"
R13="$REPO/experiment/open_set_inference_替换为1-3实际时间"
R14="$REPO/experiment/open_set_inference_替换为1-4实际时间"
R21="$REPO/experiment/open_set_inference_替换为2-1实际时间"
R22="$REPO/experiment/open_set_inference_替换为2-2实际时间"

JOINT="$REPO/experiment/task1_joint_$(date +%Y%m%d_%H%M%S)"

python scripts/recluster_competition_batch.py \
  --item 1-1 "$R11/unknown_features.npz" "$R11/predictions.csv" \
  --item 1-2 "$R12/unknown_features.npz" "$R12/predictions.csv" \
  --item 1-3 "$R13/unknown_features.npz" "$R13/predictions.csv" \
  --item 1-4 "$R14/unknown_features.npz" "$R14/predictions.csv" \
  --output "$JOINT"

SUBMIT="$REPO/submission/v2_$(date +%Y%m%d_%H%M%S)"

python - "$JOINT/joint_selection.json" \
  "$R21/predictions.csv" "$R22/predictions.csv" \
  "$DATA" "$SUBMIT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

joint_path, csv21, csv22, data_root, output = sys.argv[1:]
joint = json.loads(Path(joint_path).read_text(encoding="utf-8"))

items = ["1-1", "1-2", "1-3", "1-4", "2-1", "2-2"]
csvs = [joint["predictions"][item] for item in items[:4]]
csvs += [csv21, csv22]

subprocess.run([
    sys.executable, "prepare_submission.py",
    "--predictions", *csvs,
    "--input-root", *[str(Path(data_root) / item) for item in items],
    "--output", output,
], check=True)
PY