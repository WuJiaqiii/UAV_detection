REPO=/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code
cd "$REPO"
BATCH="$REPO/experiment/competition_batch_20261107_124532/batch_manifest.json"
TRAIN="$REPO/experiment/all_known_training_20261107_113635"
DATA=/media/kaneki/05E4235537EC7AC2/dataset_wjq

python scripts/export_legacy_submission.py \
  --batch "$BATCH" \
  --data-root "$DATA" \
  --output "$REPO/experiment/legacy_extra_class_submission_20261109"