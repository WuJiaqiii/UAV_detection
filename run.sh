#!/bin/bash
python main.py \
  --train_dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/new_dataset/new_dataset_awgn_space/train/1 /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/data/background/mats \
  --val_dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/new_dataset/new_dataset_awgn_space/val/1 \
  --epochs 300 \
  --batch_size 256 \
  --lr 1e-4 \
  --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/best.pt \
  --cnn_input_mode mask \
  --run_mode train \
  --exclude_classes FPV1 \
  --eval_exclude_classes FPV1 \
  --bbox_cache_mode refresh \
  --bbox_cache_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/cache \
  --bbox_cache_dataset_root /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/new_dataset/new_dataset_awgn_space

#  --checkpoint_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/UAV_20260428_032408/models/best.pth \

 # --checkpoint_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/UAV_20260424_110316/models/epoch_11.pth \

#  --bbox_cache_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/cache \

# --dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/train_data_space_awgn/1 \
# /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/dataset/images
# /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/temp_signal2/temp_signal2_awgn_space/train/1

# #  /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/data/background/mats
# /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/signals_no_overlap/signals_no_overlap_awgn_space/train/3