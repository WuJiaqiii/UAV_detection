#!/usr/bin/env bash
set -euo pipefail
# The manifest can come from MAT or IQ exports: scientific profile is shared.
python main.py \
  --run_mode train --trainer_type normal --incremental_split_mode normal \
  --train_signal_mode single --input_type iq_dat \
  --grouping_mode hopping --hop_association shape \
  --dataset_path /media/kaneki/05E4235537EC7AC2/dataset_wjq/train_dataset \
  --iq_sample_rate 100000000 --iq_segment_ms 50 --iq_n_fft 512 --iq_hop_length 512 \
  --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/yolov5/runs/train/exp9/weights/best.pt \
  --bbox_cache_mode off --backbone resnet18 --no-mask_pretrained \
  --batch_size 32 --num_workers 1 --epochs 500 --lr 1e-5
  
#   --frontend_config "${UAV_FRONTEND_CONFIG:?Set UAV_FRONTEND_CONFIG}" \
