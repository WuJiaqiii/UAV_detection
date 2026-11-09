OLD=/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code
# cd "$OLD"
# git fetch origin temp
# git worktree add ../code_spectral_infer origin/temp
# cd ../code_spectral_infer
# git submodule update --init --recursive
# conda activate myconda

TRAIN="$OLD/experiment/all_known_training_20261107_113635"
DATA=/media/kaneki/05E4235537EC7AC2/dataset_wjq
REFRESH="$PWD/experiment/profile_refresh_all20_spectral"

python scripts/refresh_competition_profile.py "$TRAIN" \
  --output "$REFRESH" --dry-run

python -u scripts/refresh_competition_profile.py "$TRAIN" \
  --output "$REFRESH" --wait-for-gpu \
  --gpu-id 0 --gpu-min-free-mib 16384

bash infer.sh "$TRAIN" "$DATA" \
  --profile-refresh "$REFRESH" --cluster-method spectral --dry-run

bash infer.sh "$TRAIN" "$DATA" \
  --profile-refresh "$REFRESH" --cluster-method spectral \
  --wait-for-gpu --gpu-id 0 --gpu-min-free-mib 16384

python scripts/inspect_competition_clusters.py "$BATCH" --projection both