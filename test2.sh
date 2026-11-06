# python prepare_submission.py \
#   --predictions /path/to/predictions_task1.csv \
#   --output /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/result

python prepare_submission.py \
  --predictions /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/experiment/open_set_inference_20261106_074026/predictions.csv \
  --output /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/result1-2

python prepare_submission.py \
  --predictions /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/experiment/open_set_inference_20261106_084745/predictions.csv \
  --output /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/result1-3

python prepare_submission.py \
  --predictions /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/experiment/open_set_inference_20261106_095707/predictions.csv \
  --output /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/result1-4

python prepare_submission.py \
  --predictions /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/experiment/open_set_inference_20261106_111209/predictions.csv \
  --output /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/result2-1 \
  --task2-level track --frequency-policy clip

python prepare_submission.py \
  --predictions /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/experiment/open_set_inference_20261106_124820/predictions.csv \
  --output /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/result2-2 \
  --task2-level track --frequency-policy clip