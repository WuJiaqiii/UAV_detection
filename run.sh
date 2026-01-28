#!/bin/bash

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/sam2lib:$ROOT:$PYTHONPATH"

# python main.py \
#   --dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/train_data_space_awgn/1 \
#   --epochs 1000 \
#   --batch_size 16 \
#   --token_cache_mode refresh \
#   --precompute_tokens \
#   --lr 1e-7 # \

python main.py \
  --dataset_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/train_data_space_awgn/1 \
  --epochs 1000 \
  --batch_size 16 \
  --token_cache_mode readwrite \
  --lr 1e-5 \
  --checkpoint_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/UAV_20260127_142302/models/best.pth