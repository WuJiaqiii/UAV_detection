export DATA=/media/kaneki/05E4235537EC7AC2/dataset_wjq/train_dataset
export YOLO=/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/yolov5/runs/train/exp10/weights/best.pt
export SUITE="$PWD/experiment/task12_dev_s42"

python scripts/task12_suite.py prepare \
  --suite "$SUITE" --inputs "$DATA" --yolo-weights "$YOLO" \
  --protocol open_set --seed 42 --epochs 100 \
  --unknown-classes 'DJI_Inspire_2' 'DJI_MATRICE_200' 'DJMiniSE' 'FRSKY X14'

python scripts/task12_suite.py check --suite "$SUITE"
python -u scripts/task12_suite.py run --suite "$SUITE"
python scripts/task12_suite.py compare --suite "$SUITE"