#!/usr/bin/env bash

python main.py \
  --run_mode train \
  --trainer_type normal \
  --incremental_split_mode normal \
  --train_signal_mode single \
  --input_type iq_dat \
  --dataset_path /media/kaneki/05E4235537EC7AC2/dataset_wjq/train_dataset \
  --val_ratio 0.2 \
  --iq_sample_rate 100000000 \
  --iq_segment_ms 100 \
  --iq_n_fft 512 \
  --iq_hop_length 1024 \
  --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/yolov5/runs/train/exp31/weights/best.pt \
  --bbox_cache_mode off \
  --backbone resnet18 \
  --no-mask_pretrained \
  --batch_size 128 \
  --num_workers 1 \
  --epochs 500 \
  --lr 1e-4