#!/usr/bin/env python3
"""Classify signals in MAT/PNG/DAT; report time/frequency positions and track votes."""
import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="Files, directories or quoted glob patterns; formats may be mixed")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--checkpoint_path", type=Path, required=True, help="Classifier .pth; effective settings and label IDs are read from config")
    p.add_argument("--yolo_weights", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True, help="Must be absent or empty")
    p.add_argument("--classes_json", type=Path, help="Name-to-ID mapping for legacy checkpoints; must agree with saved metadata if present")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    p.add_argument("--signal_mode", choices=["multi", "single"], default="multi", help="multi classifies every valid group; single retains the previous file-level voter")
    p.add_argument("--frontend_config", type=Path, help="Optional explicit frontend manifest, overriding saved rendering settings")
    p.add_argument("--grouping_mode", choices=["frequency", "hopping"], default=None)
    p.add_argument("--hop_association", choices=["shape", "periodic"], default=None)
    p.add_argument("--group_policy", choices=["main", "union", "all"], default=None, help="Default: all in multi mode, main in single mode; union is single-only")
    p.add_argument("--classifier_batch_size", type=int, default=32)
    p.add_argument("--yolo_conf_thres", type=float, default=None)
    p.add_argument("--iq_sample_rate", type=int, default=None, help="Real capture rate: 100000000 or 122880000")
    p.add_argument("--iq_byte_order", choices=["<", ">"], default=None)
    p.add_argument("--mat_key", default=None)
    p.add_argument("--mat_spec_domain", choices=["amplitude", "power", "db", "image"], default=None)
    p.add_argument("--set", dest="settings", action="append", default=[], metavar="KEY=JSON", help="Override another saved inference setting, e.g. min_group_len=1; repeatable")
    p.add_argument("--iq_stride_ms", type=float, default=50.)
    p.add_argument("--tail_policy", choices=["drop", "cover"], default="drop")
    p.add_argument("--max_slices", type=int, default=0, help="0 uses all; a positive limit samples evenly over the file")
    p.add_argument("--vote_method", choices=["majority", "soft"], default="majority")
    p.add_argument("--min_slice_confidence", type=float, default=0.)
    p.add_argument("--min_votes", type=int, default=1)
    p.add_argument("--min_vote_share", type=float, default=0.)
    p.add_argument("--exclude_vote_classes", nargs="*", default=[], help="Optional abstention classes, e.g. Background; default includes all model classes")
    p.add_argument("--rf_center_mhz", type=float, default=None, help="Optional recording RF center for absolute MHz; otherwise report offsets only")
    p.add_argument("--rf_centers_json", type=Path, help="Optional input-path-to-RF-center-MHz mapping, overriding the global center per file")
    p.add_argument("--track_freq_tolerance_mhz", type=float, default=3.)
    p.add_argument("--track_min_band_iou", type=float, default=.1)
    p.add_argument("--track_max_gap_ms", type=float, default=150.)
    p.add_argument("--save_visuals", action=argparse.BooleanOptionalAction, default=True, help="Multi-mode annotated slice PNGs; HTML timeline is always written")
    p.add_argument("--max_visualizations", type=int, default=24, help="Per file, evenly spaced; 0 saves all slices")
    p.add_argument("--labels_json", type=Path, help="Input-path-to-class-name mapping; multi mode also accepts lists of class names")
    p.add_argument("--label_from_parent", action="store_true", help="Explicitly use parent folder names as truth; never changes model label IDs")
    p.add_argument("--fail_fast", action="store_true")
    return p


def run(args, predictor=None):
    import torch
    from util.single_signal_eval import (
        SingleSignalPredictor, collect_inputs, resolve_config, evaluate_file,
        vote_predictions, load_truth, evaluation_metrics,
    )

    output = args.output_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use an empty output_dir; existing reports are never overwritten")
    inputs = collect_inputs(args.inputs, args.recursive)
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a project classifier checkpoint with model_state_dict")
    classes = json.loads(args.classes_json.read_text(encoding="utf-8")) if args.classes_json else None
    overrides = {}
    for setting in args.settings:
        key, sep, text = setting.partition("=")
        if not sep or not key:
            raise ValueError("--set requires KEY=JSON")
        try:
            overrides[key] = json.loads(text)
        except json.JSONDecodeError:
            overrides[key] = text
    for key in ("grouping_mode", "hop_association", "yolo_conf_thres", "iq_sample_rate",
                "iq_byte_order", "mat_key", "mat_spec_domain"):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    logger = logging.getLogger("single_signal_eval")
    cfg = resolve_config(checkpoint, classes, overrides, args.frontend_config, logger)
    if args.signal_mode == "multi" and args.group_policy not in (None, "all"):
        raise ValueError("Multi mode classifies every group; main/union require --signal_mode single")
    if args.signal_mode == "single" and args.group_policy == "all":
        raise ValueError("--group_policy all requires --signal_mode multi")
    options = dict(method=args.vote_method, min_slice_confidence=args.min_slice_confidence,
                   min_votes=args.min_votes, min_vote_share=args.min_vote_share,
                   exclude_classes=args.exclude_vote_classes)
    vote_predictions([], cfg.classes, **options)  # Validate before creating reports or models.
    if args.max_slices < 0 or not 0 < args.iq_stride_ms < float("inf"):
        raise ValueError("Need max_slices >= 0 and finite iq_stride_ms > 0")
    if args.classifier_batch_size < 1 or args.max_visualizations < 0:
        raise ValueError("Need classifier_batch_size >= 1 and max_visualizations >= 0")
    if args.signal_mode == "multi":
        from util.multi_signal_eval import FrequencyTracker, load_multi_truth, resolve_rf_centers
        FrequencyTracker(args.track_freq_tolerance_mhz, args.track_min_band_iou, args.track_max_gap_ms)
        resolve_rf_centers(args.rf_centers_json, args.rf_center_mhz)
        truth = load_multi_truth(args.labels_json, inputs, cfg.classes, args.label_from_parent)
    else:
        if args.rf_center_mhz is not None or args.rf_centers_json:
            raise ValueError("Frequency reporting requires --signal_mode multi")
        truth = load_truth(args.labels_json, inputs, cfg.classes, args.label_from_parent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    cfg.device = device
    if predictor is None:
        from model.resnet import MaskImageClassifier
        from util.detector import YoloV5Detector
        from util.preprocess import SignalPreprocessor
        from util.checkpoint import _strip_module_prefix

        weights = args.yolo_weights.expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(weights)
        cfg.yolo_weights = str(weights)
        model = MaskImageClassifier(backbone=cfg.backbone, num_classes=cfg.num_classes,
                                    in_chans=cfg.mask_in_chans, pretrained=False,
                                    dropout=cfg.cnn_dropout, freeze_backbone=False)
        # Standalone inference must not silently accept a partly random classifier.
        model.load_state_dict(_strip_module_prefix(checkpoint["model_state_dict"]), strict=True)
        yolo_device = str(device.index or 0) if device.type == "cuda" else str(device)
        if args.signal_mode == "multi":
            from util.multi_signal_eval import MultiSignalPredictor
            predictor = MultiSignalPredictor(model, YoloV5Detector(cfg, yolo_device),
                SignalPreprocessor(cfg, logger), cfg, device, args.classifier_batch_size)
        else:
            predictor = SingleSignalPredictor(model, YoloV5Detector(cfg, yolo_device),
                SignalPreprocessor(cfg, logger), cfg, device, args.group_policy or "main")
    del checkpoint
    output.mkdir(parents=True, exist_ok=True)
    if args.signal_mode == "multi":
        return run_multi(args, predictor, cfg, inputs, output, options, truth, logger)
    results = []
    with (output / "slices.jsonl").open("w", encoding="utf-8") as stream:
        for path in inputs:
            try:
                result, rows = evaluate_file(path, predictor, cfg, options, args.iq_stride_ms,
                                             args.tail_policy, args.max_slices, args.fail_fast)
            except Exception as exc:
                if args.fail_fast:
                    raise
                result = {"path": str(path), "input_type": path.suffix.lower()[1:], "status": "error",
                          "class_name": None, "error": f"{type(exc).__name__}: {exc}",
                          "total_slices": 0, "valid_votes": 0, "error_slices": 0}
                rows = []
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            result["true_class_name"] = truth.get(str(path))
            result["correct"] = (result.get("class_name") == result["true_class_name"]
                                  if result["true_class_name"] is not None else None)
            results.append(result)
            logger.info("%s | %s | %s | votes %s/%s", path.name, result["status"],
                        result.get("class_name"), result["valid_votes"], result["total_slices"])
    metrics = evaluation_metrics(results, cfg.classes)
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "checkpoint_path": str(checkpoint_path), "yolo_weights": str(args.yolo_weights.expanduser().resolve()),
              "effective_config": {key: getattr(cfg, key) for key in vars(cfg) if key != "device"},
              "device": str(device), "arguments": vars(args), "metrics": metrics, "files": results}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["path", "input_type", "status", "class_name", "candidate_class_name", "mean_probability",
               "vote_share", "valid_votes", "total_slices", "error_slices", "true_class_name", "correct", "unknown_reason", "error"]
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return report


def run_multi(args, predictor, cfg, inputs, output, options, truth, logger):
    from util.multi_signal_eval import evaluate_multi_file, multi_evaluation_metrics, resolve_rf_centers
    from util.signal_eval_report import frequency_text, write_html_report

    centers = resolve_rf_centers(args.rf_centers_json, args.rf_center_mhz)
    track_options = dict(frequency_tolerance_mhz=args.track_freq_tolerance_mhz,
                         min_band_iou=args.track_min_band_iou, max_gap_ms=args.track_max_gap_ms)
    results, rows_by_file = [], {}
    with (output/"slices.jsonl").open("w", encoding="utf-8") as stream:
        for path in inputs:
            rf = centers.get(str(path), args.rf_center_mhz)
            try:
                result, rows = evaluate_multi_file(path, predictor, cfg, options, args.iq_stride_ms,
                    args.tail_policy, args.max_slices, args.fail_fast, rf, track_options,
                    output/"visualizations" if args.save_visuals else None, args.max_visualizations)
            except Exception as exc:
                if args.fail_fast:
                    raise
                result = {"path": str(path), "input_type": path.suffix.lower()[1:], "status": "error",
                    "error": f"{type(exc).__name__}: {exc}", "predicted_classes": [], "signals": [],
                    "total_slices": 0, "observation_count": 0, "track_count": 0, "accepted_track_count": 0,
                    "error_slices": 0, "visualization_errors": 0, "rf_center_mhz": rf}
                rows = []
            for row in rows:
                if row.get("visualization"):
                    row["visualization"] = Path(row["visualization"]).relative_to(output).as_posix()
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            result["true_class_names"] = truth.get(str(path))
            result["class_set_correct"] = (result["status"] != "error" and set(result["predicted_classes"]) == set(result["true_class_names"])
                                            if result["true_class_names"] is not None else None)
            results.append(result)
            rows_by_file[str(path)] = rows
            logger.info("%s | %s | classes=%s | %s observations, %s tracks", path.name, result["status"],
                        result["predicted_classes"], result["observation_count"], result["track_count"])
            for track in result["signals"][:10]:
                logger.info("  %s | %s | %s | %.3f-%.3f ms | votes %s/%s", track["track_id"],
                    track["class_name"] or "unknown", frequency_text(track, rf), track["start_ms"],
                    track["end_ms"], track["valid_votes"], track["observation_count"])
    report = {"schema_version": 2, "signal_mode": "multi", "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(args.checkpoint_path.expanduser().resolve()),
        "yolo_weights": str(args.yolo_weights.expanduser().resolve()),
        "effective_config": {key: value for key, value in vars(cfg).items() if key != "device"},
        "arguments": vars(args), "device": str(cfg.device), "metrics": multi_evaluation_metrics(results, cfg.classes),
        "files": results}
    (output/"results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=False)+"\n", encoding="utf-8")

    def write_csv(name, records, columns):
        with (output/name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else value
                                 for key, value in record.items()})

    frequency_columns = ["frequency_center_offset_mhz", "frequency_low_offset_mhz", "frequency_high_offset_mhz",
        "frequency_span_mhz", "occupied_bandwidth_mhz", "frequency_bands_offset_mhz", "rf_center_mhz",
        "frequency_center_absolute_mhz", "frequency_low_absolute_mhz", "frequency_high_absolute_mhz", "frequency_bands_absolute_mhz"]
    observations = (group for rows in rows_by_file.values() for row in rows for group in row.get("groups", []))
    write_csv("signals.csv", observations, ["path", "slice_index", "group_id", "observation_id", "track_id", "class_name",
        "confidence", "start_ms", "end_ms", *frequency_columns, "vote_accepted", "vote_reason", "selected_boxes", "probabilities"])
    tracks = ({"path": result["path"], **track} for result in results for track in result["signals"])
    write_csv("tracks.csv", tracks, ["path", "track_id", "status", "class_name", "candidate_class_name", "start_ms", "end_ms",
        *frequency_columns, "valid_votes", "observation_count", "mean_probability", "vote_share", "vote_counts", "unknown_reason"])
    write_csv("results.csv", results, ["path", "input_type", "status", "predicted_classes", "total_slices", "observation_count",
        "track_count", "accepted_track_count", "rf_center_mhz", "error_slices", "visualization_errors", "true_class_names", "class_set_correct", "error"])
    write_html_report(output, report, rows_by_file)
    return report


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = run(args)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print("Reports: " + str(args.output_dir.resolve()))
    return 2 if (report["metrics"]["error_files"] or report["metrics"]["files_with_slice_errors"]
                 or report["metrics"].get("visualization_errors", 0)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
