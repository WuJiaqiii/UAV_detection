#!/usr/bin/env bash
set -euo pipefail

# Use the exact manifest exported with the newly annotated detector dataset.
# Example:
# UAV_IQ_DATASET_PATH=/data/iq UAV_YOLO_WEIGHTS=/models/best.pt \
# UAV_FRONTEND_CONFIG=/data/yolo/frontend_config.json bash run_train_iq.sh
python main.py \
  --run_mode train \
  --trainer_type normal \
  --incremental_split_mode normal \
  --train_signal_mode single \
  --input_type iq_dat \
  --dataset_path "${UAV_IQ_DATASET_PATH:?Set UAV_IQ_DATASET_PATH}" \
  --frontend_config "${UAV_FRONTEND_CONFIG:?Set UAV_FRONTEND_CONFIG to the exported JSON}" \
  --val_ratio 0.2 \
  --yolo_weights "${UAV_YOLO_WEIGHTS:?Set UAV_YOLO_WEIGHTS}" \
  --bbox_cache_mode off \
  --backbone resnet18 \
  --no-mask_pretrained \
  --batch_size 4 \
  --num_workers 1 \
  --epochs 500 \
  --lr 1e-4
