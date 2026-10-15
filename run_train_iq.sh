#!/usr/bin/env bash
set -euo pipefail
# The manifest can come from MAT or IQ exports: scientific profile is shared.
python main.py \
  --run_mode train --trainer_type normal --incremental_split_mode normal \
  --train_signal_mode single --input_type iq_dat \
  --dataset_path "${UAV_IQ_DATASET_PATH:?Set UAV_IQ_DATASET_PATH}" \
  --frontend_config "${UAV_FRONTEND_CONFIG:?Set UAV_FRONTEND_CONFIG}" \
  --iq_sample_rate 100000000 --iq_segment_ms 50 --iq_n_fft 512 --iq_hop_length 512 \
  --yolo_weights "${UAV_YOLO_WEIGHTS:?Set UAV_YOLO_WEIGHTS}" \
  --bbox_cache_mode off --backbone resnet18 --no-mask_pretrained \
  --batch_size 32 --num_workers 1 --epochs 500 --lr 1e-4
