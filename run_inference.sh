python main.py \
  --dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/new_dataset/new_dataset_awgn_space/val/3 \
  --batch_size 256 \
  --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/yolov5/runs/train/exp31/weights/best.pt \
  --run_mode infer \
  --exclude_classes FPV \
  --eval_exclude_classes FPV\
  --checkpoint_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/UAV_20260522_180302/models/best.pth \
  --bbox_cache_mode off