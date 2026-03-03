#!/bin/bash

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/sam2lib:$ROOT:$PYTHONPATH"

# python main.py \
#   --dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/train_data_space_awgn/1 \
#   --batch_size 16 \
#   --bbox_cache_mode readwrite \
#   --bbox_viz \
#   --bbox_viz_dir experiments/postfilter_viz/run_A \
#   --bbox_viz_limit 0 \
#   --render_cached_viz \
#   --render_boxes_source filt \
# #   --precompute_boxes

python main.py \
  --dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/train_data_space_awgn/1 \
  --epochs 1000 \
  --batch_size 16 \
  --bbox_cache_mode readwrite \
  --lr 1e-6 \
