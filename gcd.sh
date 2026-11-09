REPO=/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code
REFRESH="$REPO/experiment/profile_cover_recovery_20261109"
BATCH="$REPO/experiment/competition_batch_20261107_124532/batch_manifest.json"
TRAIN="$REPO/experiment/all_known_training_20261107_113635"
DATA=/media/kaneki/05E4235537EC7AC2/dataset_wjq

python scripts/refresh_competition_profile.py "$TRAIN" \
  --output "$REFRESH" \
  --tail-policy cover \
  --wait-for-gpu --gpu-id 0 --gpu-min-free-mib 16384

PLAN="$REPO/experiment/full_duration_supplement_20261109"

# python scripts/supplement_full_duration.py prepare \
#   --batch "$BATCH" \
#   --profile-refresh "$REFRESH" \
#   --data-root "$DATA" \
#   --scope pending \
#   --output "$PLAN"

# python scripts/supplement_full_duration.py run \
#   --plan "$PLAN/supplement_plan.json" \
#   --wait-for-gpu --gpu-id 0 --gpu-min-free-mib 16384

# python scripts/supplement_full_duration.py export \
#   --plan "$PLAN/supplement_plan.json" \
#   --output "$REPO/experiment/full_duration_submission_20261109"