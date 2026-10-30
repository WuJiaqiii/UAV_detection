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
    p.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--checkpoint_path", type=Path, required=True, help="Classifier .pth; effective settings and label IDs are read from config")
    p.add_argument("--yolo_weights", type=Path, required=True)
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
    p.add_argument("--label_from_parent", action=argparse.BooleanOptionalAction, default=False, help="Explicitly use parent folder names as truth; never changes model label IDs")
    p.add_argument("--label_from_filename", action=argparse.BooleanOptionalAction, default=False, help="Multi mode: use existing Protocol-[...] filename blocks as reporting/export truth only")
    p.add_argument("--fail_fast", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--export_features", action=argparse.BooleanOptionalAction, default=False, help="Multi mode: save one penultimate feature per accepted track to features.npz")
    p.add_argument("--recording_ids_json", type=Path, help="Input-path-to-original-recording-ID mapping; required for crops/slices sharing a recording")
    p.add_argument("--open_set_profile", type=Path, help="Frozen known-only profile; enables rejection and global unknown discovery in multi mode")
    p.add_argument("--novelty_policy", choices=["calibrated_track", "observation_guard"], default="calibrated_track", help="Default uses the calibrated track score; guard is an optional stricter ablation")
    p.add_argument("--min_novel_observation_share", type=float, default=.5, help="Only vetoes unknowns under novelty_policy observation_guard; always reported diagnostically")
    from util.unknown_discovery import add_discovery_arguments
    add_discovery_arguments(p)
    from util.feature_visualization import add_feature_plot_arguments
    add_feature_plot_arguments(p)
    return p


def run(args, predictor=None):
    if not getattr(args, "output_dir", None):
        from util.run_context import execute_run
        return execute_run("predict", args, lambda managed: run(managed, predictor))
    import torch
    from util.single_signal_eval import SingleSignalPredictor, evaluate_file, load_truth, evaluation_metrics
    from util.eval_common import collect_inputs, resolve_config, validate_prediction_options

    from util.run_context import check_output_directory
    output = check_output_directory(args)
    from util.feature_visualization import plot_options
    plot_options(args)
    inputs = collect_inputs(args.inputs, args.recursive)
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    from util.checkpoint import load_checkpoint_object
    checkpoint = load_checkpoint_object(checkpoint_path)
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
    options = validate_prediction_options(args, cfg.classes)
    feature_mode = args.export_features or args.open_set_profile is not None
    if feature_mode and args.signal_mode != "multi":
        raise ValueError("Open-set profiles and feature export require --signal_mode multi")
    if args.label_from_filename and args.signal_mode != "multi":
        raise ValueError("label_from_filename requires --signal_mode multi")
    if args.export_features and args.open_set_profile:
        raise ValueError("Export reference/calibration features without open-set rejection")
    signature, profile, recording_ids = None, None, {}
    if feature_mode:
        from util.open_set import inference_signature, OpenSetProfile
        from util.multi_signal_eval import read_path_mapping
        signature = inference_signature(args, cfg)
        if args.open_set_profile:
            profile = OpenSetProfile(args.open_set_profile, signature, cfg.classes)
            from util.unknown_discovery import validate_discovery_options, discovery_kwargs
            cluster_options = discovery_kwargs(args)
            validate_discovery_options(cluster_options["backend"], cluster_options["min_cluster_size"],
                cluster_options["min_samples"], cluster_options["min_recordings"], cluster_options["dcg_options"])
        if args.recording_ids_json:
            recording_ids = read_path_mapping(args.recording_ids_json)
            if any(not isinstance(value, str) or not value.strip() for value in recording_ids.values()):
                raise ValueError("Recording IDs must be nonempty strings")
    if args.signal_mode == "multi":
        from util.multi_signal_eval import load_multi_truth, resolve_rf_centers
        resolve_rf_centers(args.rf_centers_json, args.rf_center_mhz)
        truth = load_multi_truth(args.labels_json, inputs, cfg.classes, args.label_from_parent,
                                allow_unknown=args.open_set_profile is not None, label_from_filename=args.label_from_filename)
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
                SignalPreprocessor(cfg, logger), cfg, device, args.classifier_batch_size, return_features=feature_mode)
        else:
            predictor = SingleSignalPredictor(model, YoloV5Detector(cfg, yolo_device),
                SignalPreprocessor(cfg, logger), cfg, device, args.group_policy or "main")
    del checkpoint
    output.mkdir(parents=True, exist_ok=True)
    if args.signal_mode == "multi":
        return run_multi(args, predictor, cfg, inputs, output, options, truth, logger, signature, profile, recording_ids)
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


def run_multi(args, predictor, cfg, inputs, output, options, truth, logger, signature=None, profile=None, recording_ids=None):
    from util.multi_signal_eval import evaluate_multi_file, multi_evaluation_metrics, resolve_rf_centers
    from util.signal_eval_report import frequency_text, write_html_report

    centers = resolve_rf_centers(args.rf_centers_json, args.rf_center_mhz)
    track_options = dict(frequency_tolerance_mhz=args.track_freq_tolerance_mhz,
                         min_band_iou=args.track_min_band_iou, max_gap_ms=args.track_max_gap_ms)
    results, rows_by_file, candidates, plot_items = [], {}, [], []
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
        result["true_class_names"] = truth.get(str(path))
        result["class_set_correct"] = (result["status"] != "error" and set(result["predicted_classes"]) == set(result["true_class_names"])
                                        if result["true_class_names"] is not None else None)
        results.append(result)
        rows_by_file[str(path)] = rows
        if profile:
            if args.save_feature_plots:
                from util.open_set import iter_track_features
                for source, track, feature, accepted, _ in iter_track_features([result], {str(path): rows}):
                    if feature is not None:
                        plot_items.append({"path": str(path), "track": track, "feature": feature,
                                           "component_types": sorted({g.get("group_type", "unspecified") for g in accepted})})
            from util.open_set import apply_open_set
            candidates.extend(apply_open_set(profile, [result], {str(path): rows}, args.min_novel_observation_share, args.novelty_policy))
            for row in rows:
                for group in row.get("groups", []):
                    group.pop("embedding", None)
        elif args.export_features:
            from util.open_set import cache_track_features
            cache_track_features([result], {str(path): rows})
        logger.info("%s | %s | %s observations, %s tracks", path.name, result["status"],
                    result["observation_count"], result["track_count"])

    feature_archive, discovery = None, None
    if args.export_features:
        from util.open_set import export_feature_archive
        feature_archive = export_feature_archive(output/"features.npz", signature, cfg.classes,
                                                  results, rows_by_file, recording_ids)
    if profile:
        from util.open_set import FEATURE_VERSION, refresh_open_set_results, file_sha256, recording_identity, save_npz
        from util.unknown_discovery import discover_unknown, discovery_kwargs, save_unknown_prototypes
        identities = {}
        for candidate in candidates:
            path = candidate["path"]
            if path not in identities:
                identities[path] = recording_identity(path, file_sha256(path), recording_ids)
            candidate["recording_id"] = identities[path]
        discovery = discover_unknown(candidates, **discovery_kwargs(args))
        save_unknown_prototypes(output/"unknown_prototypes.npz", discovery, signature, profile.reference.shape[1])
        discovery["prototype_archive"] = "unknown_prototypes.npz"
        import numpy as np
        positions = ["start_ms", "end_ms", "known_candidate_class", "ood_score", "ood_threshold",
            "frequency_center_offset_mhz", "frequency_low_offset_mhz", "frequency_high_offset_mhz",
            "frequency_bands_offset_mhz", "rf_center_mhz", "frequency_center_absolute_mhz",
            "frequency_low_absolute_mhz", "frequency_high_absolute_mhz", "frequency_bands_absolute_mhz"]
        candidate_records = [{"path": candidate["path"], "recording_id": candidate["recording_id"],
                              "track_id": candidate["track"]["track_id"],
                              "unknown_cluster_id": candidate["track"]["unknown_cluster_id"],
                              **{key: candidate["track"].get(key) for key in positions}} for candidate in candidates]
        candidate_features = (np.stack([candidate["feature"] for candidate in candidates]) if candidates
                              else np.empty((0, profile.reference.shape[1]), dtype=np.float32))
        save_npz(output/"unknown_features.npz", {"kind": "unknown_candidates", "schema_version": 1,
                 "feature_version": FEATURE_VERSION, "signature": signature, "records": candidate_records},
                 features=candidate_features)
        refresh_open_set_results(results, rows_by_file)
        # Re-render only previously selected previews with FINAL track decisions.
        # The original slice predictions remain in class_name/known_candidate_class.
        from util.single_signal_eval import load_spectrum
        from util.signal_eval_report import save_slice_preview
        for result in results:
            for row in rows_by_file[result["path"]]:
                if not row.get("visualization"):
                    continue
                try:
                    spectrum, domain = load_spectrum(Path(result["path"]), cfg,
                        row if result["input_type"] == "dat" else None)
                    save_slice_preview(result["path"], row["slice_index"], spectrum, domain, cfg, row,
                                       output/"visualizations", result["rf_center_mhz"])
                except Exception as exc:
                    if args.fail_fast:
                        raise
                    # Never publish an old closed-set preview as an open-set result.
                    row.pop("visualization", None)
                    row["visualization_error"] = f"{type(exc).__name__}: {exc}"
            result["visualization_errors"] = sum("visualization_error" in row for row in rows_by_file[result["path"]])
        (output/"unknown_clusters.json").write_text(json.dumps(discovery, ensure_ascii=False, indent=2,
                                                             allow_nan=False)+"\n", encoding="utf-8")

    feature_visualization = None
    if profile and args.save_feature_plots:
        from util.feature_visualization import write_feature_plots, plot_options
        fields = ("track_id", "recognition_status", "display_label", "class_name", "unknown_cluster_id",
                  "start_ms", "end_ms", "frequency_bands_offset_mhz", "ood_score", "ood_threshold")
        plot_records = [{"path": item["path"], "recording_id": (recording_ids or {}).get(item["path"], item["path"]),
                         "true_class_names": truth.get(item["path"]), "component_types": item["component_types"],
                         **{key: item["track"].get(key) for key in fields}} for item in plot_items]
        all_features = np.stack([item["feature"] for item in plot_items]) if plot_items else np.empty((0, profile.reference.shape[1]), dtype=np.float32)
        # Keep all frozen track vectors for later plots, including missed unknowns accepted as known.
        save_npz(output/"inference_features.npz", {"kind": "inference_features", "schema_version": 1,
            "feature_version": FEATURE_VERSION, "signature": signature, "classes": cfg.classes, "records": plot_records},
            features=all_features)
        try:
            feature_visualization = write_feature_plots(output/"feature_plots", all_features, plot_records, **plot_options(args))
            feature_visualization["archive"] = "inference_features.npz"
            feature_visualization["tracks_without_features"] = sum(len(r["signals"]) for r in results) - len(plot_items)
        except Exception as error:
            logger.warning("Feature projection failed; recognition results retained: %s", error)
            feature_visualization = {"status": "error", "error": str(error), "archive": "inference_features.npz", "plots": []}

    with (output/"slices.jsonl").open("w", encoding="utf-8") as stream:
        for rows in rows_by_file.values():
            for row in rows:
                for group in row.get("groups", []):
                    group.pop("embedding", None)  # Keep large vectors in the feature archive only.
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    for result in results:
        for track in result["signals"]:
            track.pop("_feature", None)
    from util.class_evidence import summarize_class_evidence
    summarize_class_evidence(results, rows_by_file)
    metrics = multi_evaluation_metrics(results, cfg.classes)
    if profile:
        metrics["feature_plot_errors"] = int(feature_visualization is not None and feature_visualization["status"] == "error")
        from util.open_set_metrics import evaluate_open_set
        evaluation = evaluate_open_set(results, cfg.classes, discovery, recording_ids)
        metrics["open_set_evaluation"] = evaluation["metrics"]
        (output/"open_set_metrics.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2,
                                                             allow_nan=False)+"\n", encoding="utf-8")
        # Closed-set file-set accuracy would silently ignore anonymous novel IDs.
        metrics.pop("file_class_set_accuracy", None)
        metrics.pop("unknown_files", None)  # Legacy vote abstention is not a novel-class count.
        metrics["known_file_class_presence_metrics"] = metrics.pop("file_class_presence_metrics")
        labeled = [result for result in results if result.get("true_class_names") is not None]
        actual_novel = lambda row: bool(set(row["true_class_names"]) - set(cfg.classes))
        predicted_novel = lambda row: row.get("novel_track_count", 0) > 0
        tp = sum(actual_novel(row) and predicted_novel(row) for row in labeled)
        fp = sum(not actual_novel(row) and predicted_novel(row) for row in labeled)
        fn = sum(actual_novel(row) and not predicted_novel(row) for row in labeled)
        metrics["file_unknown_presence_metrics"] = {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None}
        metrics.update(known_tracks=sum(row.get("known_track_count", 0) for row in results),
                       files_with_novel_candidates=sum(row.get("novel_track_count", 0) > 0 for row in results),
                       uncertain_tracks=sum(row.get("uncertain_track_count", 0) for row in results),
                       no_signal_files=sum(row["status"] == "no_signal" for row in results),
                       no_detection_slices=sum(row.get("status") == "no_detection" for rows in rows_by_file.values() for row in rows),
                       no_group_slices=sum(row.get("status") == "no_group" for rows in rows_by_file.values() for row in rows),
                       uncertain_files=sum(row["status"] == "uncertain" for row in results),
                       **{key: discovery[key] for key in ("unknown_candidate_tracks", "estimated_unknown_class_count",
                                                           "clustered_unknown_tracks", "unassigned_unknown_tracks")})
    report = {"schema_version": 3 if profile or args.export_features else 2, "signal_mode": "multi", "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(args.checkpoint_path.expanduser().resolve()),
        "yolo_weights": str(args.yolo_weights.expanduser().resolve()),
        "effective_config": {key: value for key, value in vars(cfg).items() if key != "device"},
        "arguments": vars(args), "device": str(cfg.device), "metrics": metrics,
        "files": results}
    if feature_archive:
        report["feature_archive"] = feature_archive
    if feature_visualization is not None:
        report["feature_visualization"] = feature_visualization
    if profile:
        report.update(open_set={"profile": str(args.open_set_profile.resolve()),
                                "profile_sha256": file_sha256(args.open_set_profile),
                                "threshold": profile.threshold, "k": profile.k,
                                "calibration": {key: value for key, value in profile.metadata.items() if key != "signature"}},
                      discovery=discovery)
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
    open_columns = ["recognition_status", "display_label", "known_candidate_class", "ood_score", "ood_threshold",
        "track_ood_score", "novel_observation_share", "novelty_policy", "profile_class_name", "unknown_cluster_id", "cluster_status", "cluster_membership", "cluster_similarity"] if profile else []
    write_csv("signals.csv", observations, ["path", "slice_index", "group_id", "observation_id", "track_id", "class_name",
        "confidence", *open_columns, "start_ms", "end_ms", *frequency_columns, "vote_accepted", "vote_reason", "selected_boxes", "probabilities"])
    tracks = ({"path": result["path"], **track} for result in results for track in result["signals"])
    write_csv("tracks.csv", tracks, ["path", "track_id", "status", "class_name", "candidate_class_name", "start_ms", "end_ms",
        *frequency_columns, *open_columns, "component_types", "valid_votes", "observation_count", "mean_probability", "vote_share", "vote_counts", "unknown_reason"])
    evidence = ({"path": result["path"], **item} for result in results for item in result["class_evidence"])
    write_csv("class_evidence.csv", evidence, ["path", "display_label", "recognition_status", "class_name", "identity_status", "components"])
    write_csv("results.csv", results, ["path", "input_type", "status", "predicted_classes", "total_slices", "observation_count",
        "track_count", "accepted_track_count", "rf_center_mhz", "error_slices", "visualization_errors", "true_class_names", "class_set_correct",
        "predicted_unknown_classes", "known_track_count", "novel_track_count", "uncertain_track_count", "no_signal_reason", "error"])
    write_html_report(output, report, rows_by_file)
    return report


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    report = run(args)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print("Reports: " + str(args.output_dir.resolve()))
    return 2 if (report["metrics"]["error_files"] or report["metrics"]["files_with_slice_errors"]
                 or report["metrics"].get("visualization_errors", 0) or report["metrics"].get("feature_plot_errors", 0)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
