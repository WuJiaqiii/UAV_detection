#!/bin/bash
python main.py \
  --dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/train_data_space_awgn/1 \
  --exclude_classes FPV1 \
  --epochs 300 \
  --batch_size 512 \
  --lr 1e-4 \
  --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/best.pt \
  --bbox_cache_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/cache \
  --cnn_input_mode mask
# --dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/train_data_space_awgn/1 \
# /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/dataset/images
