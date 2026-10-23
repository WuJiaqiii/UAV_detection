python scripts/evaluate_single_signal.py \
  --signal_mode single \
  --inputs /media/kaneki/05E4235537EC7AC2/dataset_wjq/1-1 \
  --checkpoint_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/UAV_20261020_202241/models/epoch_40.pth \
  --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/yolov5/runs/train/exp10/weights/best.pt \
  --output_dir /media/kaneki/05E4235537EC7AC2/dataset_wjq/1-1_result \
  --iq_sample_rate 100000000 \
  --iq_stride_ms 50 \
  --vote_method majority