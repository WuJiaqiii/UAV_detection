#!/bin/bash
python main.py \
  --train_dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/new_dataset/new_dataset_awgn_space/train/1 /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/data/background \
  --val_dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/new_dataset/new_dataset_awgn_space/val/1 \
  --epochs 300 \
  --batch_size 128 \
  --lr 1e-4 \
  --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/yolov5/runs/train/exp31/weights/best.pt \
  --run_mode train \
  --exclude_classes FPV \
  --eval_exclude_classes FPV \
  --trainer_type lwf \
  --incremental_split_mode base \
  --incremental_new_classes Skylink1 \
  --bbox_cache_mode readwrite \
  --bbox_cache_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/cache \
  --bbox_cache_dataset_root /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/new_dataset/new_dataset_awgn_space \