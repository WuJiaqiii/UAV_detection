#!/bin/bash

python main.py \
  --dataset_path /root/Desktop/AMR_NCD/data/config_dynamic_cfo_500w_snr.hdf5 \
  --epochs 1000 \
  --batch_size 512 \
  --lr 1e-7 \
  --checkpoint_path /root/Desktop/AMR_NCD/experiments/RadioF_ResNet_20-5_20251028_154827/models/ResNet_best.pth