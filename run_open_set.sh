#!/usr/bin/env bash
# Known-only reference/calibration, followed by mixed-signal discovery.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

: "${CLASSIFIER_CHECKPOINT:?Set CLASSIFIER_CHECKPOINT to the trained classifier .pth}"
: "${YOLO_WEIGHTS:?Set YOLO_WEIGHTS to the detector .pt}"
: "${KNOWN_REFERENCE_INPUT:?Set KNOWN_REFERENCE_INPUT to clean known training data}"
: "${KNOWN_CALIBRATION_INPUT:?Set KNOWN_CALIBRATION_INPUT to disjoint known calibration data}"
: "${TEST_INPUT:?Set TEST_INPUT to the mixed test directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new output directory}"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "OUTPUT_ROOT already exists; choose a new directory: $OUTPUT_ROOT" >&2
  exit 2
fi

python_bin="${PYTHON:-python}"
common=(--checkpoint_path "$CLASSIFIER_CHECKPOINT" --yolo_weights "$YOLO_WEIGHTS"
        --signal_mode multi --recursive --device "${DEVICE:-auto}"
        --iq_sample_rate "${IQ_SAMPLE_RATE:-100000000}" --iq_stride_ms "${IQ_STRIDE_MS:-50}"
        --max_slices "${MAX_SLICES:-0}" --min_votes "${MIN_VOTES:-1}")
if [[ -n "${RECORDING_IDS_JSON:-}" ]]; then common+=(--recording_ids_json "$RECORDING_IDS_JSON"); fi
if [[ -n "${GROUPING_MODE:-}" ]]; then common+=(--grouping_mode "$GROUPING_MODE"); fi
if [[ -n "${HOP_ASSOCIATION:-}" ]]; then common+=(--hop_association "$HOP_ASSOCIATION"); fi
if [[ -n "${RF_CENTER_MHZ:-}" ]]; then common+=(--rf_center_mhz "$RF_CENTER_MHZ"); fi
if [[ -n "${RF_CENTERS_JSON:-}" ]]; then common+=(--rf_centers_json "$RF_CENTERS_JSON"); fi

reference_labels=()
calibration_labels=()
case "${LABEL_MODE:-filename}" in
  filename) reference_labels=(--label_from_filename); calibration_labels=(--label_from_filename) ;;
  parent) reference_labels=(--label_from_parent); calibration_labels=(--label_from_parent) ;;
  json)
    : "${REFERENCE_LABELS_JSON:?Set REFERENCE_LABELS_JSON in json mode}"
    : "${CALIBRATION_LABELS_JSON:?Set CALIBRATION_LABELS_JSON in json mode}"
    reference_labels=(--labels_json "$REFERENCE_LABELS_JSON")
    calibration_labels=(--labels_json "$CALIBRATION_LABELS_JSON") ;;
  *) echo "LABEL_MODE must be filename, parent or json" >&2; exit 2 ;;
esac

"$python_bin" scripts/evaluate_single_signal.py "${common[@]}" --inputs "$KNOWN_REFERENCE_INPUT" \
  --output_dir "$OUTPUT_ROOT/reference" --export_features --no-save_visuals "${reference_labels[@]}"
"$python_bin" scripts/evaluate_single_signal.py "${common[@]}" --inputs "$KNOWN_CALIBRATION_INPUT" \
  --output_dir "$OUTPUT_ROOT/calibration" --export_features --no-save_visuals "${calibration_labels[@]}"

ignored=()
# Current main includes Background. Set IGNORE_PROFILE_CLASS="" when it was excluded from training.
if [[ -n "${IGNORE_PROFILE_CLASS-Background}" ]]; then ignored=(--ignore_classes "${IGNORE_PROFILE_CLASS-Background}"); fi
"$python_bin" scripts/build_open_set_profile.py \
  --reference_features "$OUTPUT_ROOT/reference/features.npz" \
  --calibration_features "$OUTPUT_ROOT/calibration/features.npz" --output "$OUTPUT_ROOT/known_profile.npz" \
  --knn_k "${KNN_K:-5}" --known_false_reject_rate "${KNOWN_FALSE_REJECT_RATE:-0.01}" \
  --min_calibration_per_class "${MIN_CALIBRATION_PER_CLASS:-20}" "${ignored[@]}"

"$python_bin" scripts/evaluate_single_signal.py "${common[@]}" --inputs "$TEST_INPUT" \
  --output_dir "$OUTPUT_ROOT/discovery" --open_set_profile "$OUTPUT_ROOT/known_profile.npz" \
  --unknown_cluster_backend "${UNKNOWN_BACKEND:-hdbscan}" \
  --unknown_min_cluster_size "${UNKNOWN_MIN_CLUSTER_SIZE:-5}" \
  --unknown_min_recordings "${UNKNOWN_MIN_RECORDINGS:-2}" \
  --dcg_topk "${DCG_TOPK:-15}" --dcg_cosine_threshold "${DCG_COSINE_THRESHOLD:-0.85}" \
  --dcg_resolution "${DCG_RESOLUTION:-1.0}"
