# python scripts/infer_one_signal_mat.py \
#   --input_dir /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/temp_signal2/temp_signal2_awgn_space/train/1 \
#   --out_dir /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_detection/experiments \
#   --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/best.pt \
#   --classifier_checkpoint /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/UAV_20260413_175709/models/epoch_52.pth \
#   --cnn_input_mode raw

# python scripts/infer_one_signal_mat.py \
#   --input_dir /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/temp_signal2/temp_signal2_awgn_space/train/1 \
#   --out_dir /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_detection/experiments \
#   --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/best.pt \
#   --classifier_checkpoint /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/UAV_20260413_192449/models/epoch_15.pth \
#   --cnn_input_mode raw_with_boxes

# python scripts/infer_one_signal_mat.py \
#   --input_dir /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/temp_signal2/temp_signal2_awgn_space/train/1 \
#   --out_dir /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_detection/experiments \
#   --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/best.pt \
#   --classifier_checkpoint /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/UAV_20260413_170732/models/epoch_27.pth \
#   --cnn_input_mode raw_in_boxes

python scripts/infer_one_signal_mat.py \
  --exclude_classes FPV1 \
  --input_dir /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/temp_signal2/temp_signal2_awgn_space/train/1 \
  --out_dir /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_detection/experiments \
  --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/best.pt \
  --classifier_checkpoint /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/UAV_20260413_162523/models/epoch_23.pth \
  --cnn_input_mode mask
  

# python scripts/infer_one_signal_mat.py \
#   --input_dir /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/train_data_space_awgn/1 \
#   --out_dir /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_detection/test \
#   --yolo_weights /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/best.pt \
#   --classifier_checkpoint /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/experiments/UAV_20260413_162523/models/epoch_23.pth \
#   --cnn_input_mode mask