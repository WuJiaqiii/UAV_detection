#!/bin/bash

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/sam2lib:$ROOT:$PYTHONPATH"

python main.py \
  --dataset_path /home/jiaqi/uav/dataset \
  --epochs 1000 \
  --batch_size 2 \
  --lr 1e-7 # \
#   --checkpoint_path /root/Desktop/AMR_NCD/experiments/RadioF_ResNet_20-5_20251028_154827/models/ResNet_best.pth