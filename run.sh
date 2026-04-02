#!/bin/bash

python main.py \
  --dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/dataset/images \
  --exclude_classes FPV1 \
  --epochs 1000 \
  --batch_size 16 \
  --lr 1e-5 \
  --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/train/exp5/weights/best.pt \
  --bbox_cache_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/cache
# --dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/train_data_space_awgn/1 \