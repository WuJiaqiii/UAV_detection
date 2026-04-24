#!/bin/bash
python main.py \
  --dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/train_data_space_awgn/1 \
  --epochs 300 \
  --batch_size 256 \
  --lr 1e-4 \
  --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/best.pt \
  --checkpoint_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/UAV_20260424_063033/models/epoch_1.pth \
  --cnn_input_mode mask \
  --run_mode train \
  --exclude_classes FPV1 \
  --eval_exclude_classes FPV1
  #  --bbox_cache_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/cache \

# --dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/train_data_space_awgn/1 \
# /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/dataset/images
# /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/temp_signal2/temp_signal2_awgn_space/train/1

# #  /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/data/background/mats
