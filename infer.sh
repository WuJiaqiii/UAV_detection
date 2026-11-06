REPO="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code"
DATA="/media/kaneki/05E4235537EC7AC2/dataset_wjq"

# 替换为本次新版训练实际生成的目录
V2_RUN="$REPO/experiment/all_known_training_20261107_033606"

# 必须与本次训练、建立 profile 时使用的检测器一致
YOLO="$REPO/yolov5/runs/train/exp10/weights/best.pt"

cd "$REPO"

infer_item() {
  local item="$1" profile="$2" mode="$3"
  shift 3

  bash run.sh infer \
    --config config/competition_v2_infer.yaml \
    --checkpoint_path "$V2_RUN/training/models/best.pth" \
    --open_set_profile "$V2_RUN/open_set/known_profile.npz" \
    --yolo_weights "$YOLO" \
    --inputs "$DATA/$item" \
    --test_item "$item" \
    --competition_profile "$profile" \
    --unknown_input_mode "$mode" \
    "$@"
}

infer_item 1-1 task1_n1 single
infer_item 1-2 task1_n2 single
infer_item 1-3 task1_n1 single
infer_item 1-4 task1_n2 single

infer_item 2-1 task2 multi
infer_item 2-2 task2 multi