"""Known-only training -> independent calibration -> labeled discovery experiment."""
import argparse
import json
from pathlib import Path

import yaml


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--input_type", choices=["iq_dat", "mat", "png"], default="iq_dat")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--known_classes", nargs="+")
    parser.add_argument("--unknown_classes", nargs="+")
    parser.add_argument("--exclude_classes", nargs="*", default=[], help="Discovered classes omitted from the entire experiment, e.g. Background")
    parser.add_argument("--val_ratio", type=float, default=.15)
    parser.add_argument("--calibration_ratio", type=float, default=.20)
    parser.add_argument("--test_ratio", type=float, default=.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recording_ids_json", type=Path)
    parser.add_argument("--split_only", action=argparse.BooleanOptionalAction, default=False,
                        help="Write the split manifest and stage configs without training or running models")
    return parser


def stage_arguments(args, config_path):
    """Validate both stages before data indexing or costly training begins."""
    from util.cli import build_task_parser, load_settings, settings_tokens
    from util.train_entry import build_parser as train_parser, validate_training_options
    from util.run_context import planned_directory, run_type
    output = (getattr(args, "output_dir", None) or planned_directory(run_type("experiment", args))).resolve()
    train_settings, _ = load_settings(config_path, "train")
    managed_train = {"dataset_path", "train_dataset_path", "val_dataset_path", "class_names", "exclude_classes",
                     "incremental_new_classes", "incremental_split_mode", "checkpoint_path", "experiment_dir", "recording_ids_json"}
    if managed_train & train_settings.keys():
        raise ValueError("experiment manages training data/classes/checkpoint paths; remove: " + str(sorted(managed_train & train_settings.keys())))
    if train_settings.get("input_type", args.input_type) != args.input_type:
        raise ValueError("train.input_type must agree with experiment.input_type")
    if train_settings.get("run_mode", "train") != "train" or train_settings.get("trainer_type", "normal") != "normal":
        raise ValueError("experiment uses normal known-only training; LwF/resume remain separate train tasks")
    train_settings.update(input_type=args.input_type, run_mode="train", trainer_type="normal", seed=args.seed)
    train = train_parser().parse_args(settings_tokens(train_parser(), train_settings))
    validate_training_options(train)
    train._run_dir = output / "training"
    from util.spectrogram_frontend import apply_frontend_manifest
    apply_frontend_manifest(train)
    if train.epochs < 1 or train.train_signal_mode != "single" or train.sample_ratio != 1:
        raise ValueError("experiment requires epochs >= 1, single-signal training and sample_ratio=1")

    evaluation, _ = load_settings(config_path, "discover")
    managed_eval = {"inputs", "checkpoint_path", "classes_json", "reference_inputs", "calibration_inputs", "test_inputs",
                    "output_dir", "test_output_dir", "reference_labels_json", "calibration_labels_json", "test_labels_json",
                    "labels_json", "label_mode", "test_label_mode", "label_from_parent", "label_from_filename",
                    "open_set_profile", "export_features", "recording_ids_json"}
    if managed_eval & evaluation.keys():
        raise ValueError("experiment manages profile/test data and labels; remove: " + str(sorted(managed_eval & evaluation.keys())))
    if str(evaluation.get("yolo_weights", train.yolo_weights)) != str(train.yolo_weights):
        raise ValueError("Experiment training/profile must use the same yolo_weights")
    evaluation.update(checkpoint_path=str(output / "training/models/best.pth"), yolo_weights=train.yolo_weights,
                      reference_inputs=["generated-train-files"], calibration_inputs=["generated-calibration-files"],
                      test_inputs=["generated-test-files"], label_mode="json", test_label_mode="none",
                      reference_labels_json=str(output / "reference_labels.json"),
                      calibration_labels_json=str(output / "calibration_labels.json"),
                      test_labels_json=str(output / "test_labels.json"),
                      recording_ids_json=str(output / "recording_ids.json"))
    # Explicit training frontend overrides are saved in the checkpoint. Capture
    # rate is source-specific and inference defaults deliberately do not inherit it.
    if args.input_type == "iq_dat":
        evaluation.setdefault("iq_sample_rate", train.iq_sample_rate)
        evaluation.setdefault("iq_byte_order", train.iq_byte_order)
    parser = build_task_parser("discover")
    profile = parser.parse_args(settings_tokens(parser, evaluation))
    profile.output_dir, profile.test_output_dir = output / "open_set", output / "test"
    if profile.signal_mode != "multi" or profile.ignore_classes or profile.exclude_vote_classes:
        raise ValueError("experiment requires multi mode with no ignored/excluded head classes")
    from util.eval_common import validate_prediction_options
    validate_prediction_options(profile, {"known": 0})
    if args.input_type == "iq_dat" and (profile.iq_sample_rate != train.iq_sample_rate or profile.iq_byte_order != train.iq_byte_order):
        raise ValueError("Training and evaluation must use the same IQ capture rate/byte order")
    from util.unknown_discovery import discovery_kwargs, validate_discovery_options
    options = discovery_kwargs(profile)
    validate_discovery_options(options["backend"], options["min_cluster_size"], options["min_samples"],
                               options["min_recordings"], options["dcg_options"])
    if (profile.knn_k < 1 or profile.min_calibration_per_class < 1 or
            profile.max_reference_per_class < profile.knn_k or not 0 < profile.known_false_reject_rate < 1):
        raise ValueError("Invalid known-profile calibration settings")
    return train, profile


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def _write_stage_config(output, train, profile):
    from util.cli import build_task_parser
    _write_json(output / "effective_train.json", vars(train))
    _write_json(output / "effective_discover.json", vars(profile))
    # Copy exactly the effective feature-producing options; no test truth or
    # training data paths are exported into the deployment config.
    allowed = {action.dest for action in build_task_parser("infer")._actions}
    blocked = {"inputs", "output_dir", "labels_json", "label_from_parent", "label_from_filename",
               "open_set_profile", "export_features", "help"}
    settings = {key: value for key, value in vars(profile).items() if key in allowed - blocked and value is not None}
    settings = json.loads(json.dumps(settings, default=str))
    deployment = {"evaluation": settings, "infer": {
        "inputs": ["/path/to/unlabeled_test"],
        "open_set_profile": str(output / "open_set/known_profile.npz")}}
    (output / "inference.yaml").write_text(yaml.safe_dump(deployment, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_experiment(args, config_path=None, train_runner=None, predictor=None):
    if not getattr(args, "output_dir", None):
        from util.run_context import execute_run
        return execute_run("experiment", args, lambda managed: run_experiment(managed, config_path, train_runner, predictor))
    from util.experiment_split import split_dataset
    from util.cli import run_profile
    from util.run_context import check_output_directory
    output = check_output_directory(args)
    train, profile = stage_arguments(args, config_path)
    if not args.split_only and not Path(train.yolo_weights).is_file():
        raise FileNotFoundError(f"YOLO weights not found: {train.yolo_weights}")
    manifest = split_dataset(args, train)
    for name in manifest["known_classes"]:
        counts = manifest["summary"][name]
        if counts["train"]["recordings"] < profile.knn_k or counts["calibration"]["recordings"] < profile.min_calibration_per_class:
            raise ValueError(f"Insufficient independent reference/calibration recordings for {name!r}: {counts}; "
                             "adjust split ratios or profile counts before training")
    rows = manifest["records"]
    select = lambda *splits: [r["path"] for r in rows if r["split"] in splits]
    train.dataset_path = None
    train.train_dataset_path, train.val_dataset_path = select("train"), select("val")
    train.class_names = manifest["known_classes"]
    train.recording_ids_json = output / "recording_ids.json"
    profile.reference_inputs, profile.calibration_inputs = select("train"), select("calibration")
    profile.test_inputs = select("test_known", "test_unknown")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "split_manifest.json", manifest)
    _write_json(output / "recording_ids.json", {r["path"]: r["recording_id"] for r in rows})
    for file, splits in (("reference", ("train",)), ("calibration", ("calibration",)),
                         ("test", ("test_known", "test_unknown"))):
        _write_json(output / (file + "_labels.json"), {r["path"]: r["class_name"] for r in rows if r["split"] in splits})
    _write_stage_config(output, train, profile)
    state = {"status": "split_only" if args.split_only else "running", "stage": "split",
             "manifest": str(output / "split_manifest.json"), "checkpoint_path": str(profile.checkpoint_path),
             "profile_path": str(output / "open_set/known_profile.npz"),
             "inference_config": str(output / "inference.yaml")}
    _write_json(output / "experiment.json", state)
    if args.split_only:
        return state
    try:
        state["stage"] = "training"
        _write_json(output / "experiment.json", state)
        if train_runner is None:
            from util.train_entry import main as train_runner
        train_runner(train)
        if not profile.checkpoint_path.is_file():
            raise RuntimeError(f"Training did not produce a best checkpoint: {profile.checkpoint_path}")
        # Ensure a custom runner or stale file cannot change the known head space.
        import torch
        checkpoint = torch.load(profile.checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("config", {}).get("classes") != manifest["classes"]:
            raise ValueError("Best checkpoint class mapping does not match the known split")
        del checkpoint
        state["stage"] = "profile_and_test"
        _write_json(output / "experiment.json", state)
        report = run_profile(profile, discover=True, predictor=predictor)
        from util.run_context import metric_error_count
        status = "completed_with_errors" if metric_error_count(report["metrics"]) else "complete"
        state.update(status=status, stage="complete", metrics=report["metrics"],
                     test_report=str(output / "test/results.json"))
        _write_json(output / "experiment.json", state)
        return state
    except BaseException as exc:
        state.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        _write_json(output / "experiment.json", state)
        raise
