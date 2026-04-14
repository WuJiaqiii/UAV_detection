# python scripts/infer_multi_signal_mat_dbscan.py \
#   --mat_path /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/data/Ocusync21-[0,49.1,1000,18]-SNR-9-SNRSPACE5.372315e+00-Figure-24.mat  \
#   --out_dir /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments \
#   --weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/best.pt \
#   --device 0

python scripts/debug_mat_pipeline.py \
  --input_dir /path/to/mat_dir \
  --out_dir /path/to/debug_out \
  --yolo_weights /path/to/best.pt \
  --device 0

# python scripts/debug_mat_pipeline.py \
#   --mat_path /path/to/sample.mat \
#   --out_dir /path/to/debug_out \
#   --yolo_weights /path/to/yolo_best.pt \
#   --classifier_checkpoint /path/to/cnn_best.pth \
#   --device 0 \
#   --backbone resnet18 \
#   --num_classes 8 \
#   --cnn_input_mode mask