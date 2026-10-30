"""Task dispatch and safe YAML-to-argparse adaptation, with lazy model imports."""
import argparse
import copy
import json
from pathlib import Path
import sys

import yaml

TASKS = ("train", "predict", "profile", "cluster", "discover", "experiment", "infer", "visualize")
SECTIONS = {"train", "evaluation", "predict", "profile", "cluster", "discovery", "experiment", "infer", "visualize"}
PATH_OPTIONS = {
    "dataset_path", "train_dataset_path", "val_dataset_path", "yolo_weights", "checkpoint_path",
    "lwf_teacher_checkpoint", "bbox_cache_path", "bbox_cache_dataset_root", "frontend_config",
    "classes_json", "inputs", "features", "reference_inputs", "calibration_inputs",
    "test_inputs", "reference_labels_json", "calibration_labels_json",
    "test_labels_json", "labels_json", "recording_ids_json", "rf_centers_json", "open_set_profile", "results_json",
}


def load_settings(path, task):
    if path is None:
        return {}, None
    path = path.expanduser().resolve()
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document)-SECTIONS:
        raise ValueError(f"Config must contain only sections: {', '.join(sorted(SECTIONS))}")
    for key, value in document.items():
        if not isinstance(value, dict):
            raise ValueError(f"Config section {key} must be a mapping")
    if task in {"train", "cluster", "experiment", "visualize"}:
        settings = document.get(task, {}).copy()
    else:
        settings = document.get("evaluation", {}).copy()
        settings.update(document.get(task if task in {"predict", "infer"} else "profile", {}))
        if task == "discover":
            settings.update(document.get("discovery", {}))
    for key, value in list(settings.items()):
        if key in PATH_OPTIONS and value is not None:
            def resolve(item):
                if not isinstance(item, (str, Path)):
                    raise ValueError(f"{key} requires path strings")
                if not str(item).strip():
                    return str(item)
                target = Path(item).expanduser()
                return str(target if target.is_absolute() else (path.parent/target).resolve())
            settings[key] = [resolve(item) for item in value] if isinstance(value, list) else resolve(value)
    return settings, path


def settings_tokens(parser, settings):
    """Use each task's real argparse schema, including booleans and list options."""
    actions = {action.dest: action for action in parser._actions if action.option_strings}
    tokens = []
    for key, value in settings.items():
        if key not in actions or key == "help":
            if key in {"output_dir", "test_output_dir", "experiment_dir"}:
                raise ValueError(f"Remove {key}: output directories are automatic under the repository's experiment directory")
            raise ValueError(f"Unknown setting for this task: {key}")
        action = actions[key]
        flag = next(option for option in action.option_strings if option.startswith("--") and not option.startswith("--no-"))
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise ValueError(f"{key} requires a YAML boolean")
            tokens.append(flag if value else "--no-"+flag[2:])
        elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            if not isinstance(value, bool):
                raise ValueError(f"{key} requires a YAML boolean")
            if value != action.default:
                tokens.append(flag)
        elif value is None:
            continue
        elif isinstance(action, argparse._AppendAction):
            if key == "settings" and isinstance(value, dict):
                value = [name+"="+json.dumps(item, ensure_ascii=False, allow_nan=False) for name, item in value.items()]
            if not isinstance(value, list):
                raise ValueError(f"{key} requires a list, or a mapping for settings")
            for item in value:
                tokens.extend([flag, str(item)])
        elif action.nargs in ("*", "+"):
            if not isinstance(value, list):
                raise ValueError(f"{key} requires a YAML list")
            tokens.extend([flag, *map(str, value)])
        else:
            if isinstance(value, (list, dict, bool)):
                raise ValueError(f"{key} requires a scalar value")
            tokens.extend([flag, str(value)])
    return tokens


def profile_parser(discover=False):
    from scripts.evaluate_single_signal import build_parser
    parser = build_parser()
    parser.description = "Build a known-only profile from independent reference/calibration inputs"
    # Retain the exact inference option schema for all feature-producing stages.
    for action in parser._actions:
        if action.dest == "inputs":
            action.required = False
            action.help = argparse.SUPPRESS
    parser.set_defaults(save_visuals=discover)
    parser.add_argument("--reference_inputs", nargs="+", required=True)
    parser.add_argument("--calibration_inputs", nargs="+", required=True)
    parser.add_argument("--label_mode", choices=["filename", "parent", "json"], default="filename")
    parser.add_argument("--reference_labels_json", type=Path)
    parser.add_argument("--calibration_labels_json", type=Path)
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--known_false_reject_rate", type=float, default=.01)
    parser.add_argument("--min_calibration_per_class", type=int, default=20)
    parser.add_argument("--max_reference_per_class", type=int, default=2000)
    parser.add_argument("--profile_score_method", choices=["record_class_knn", "global_knn"], default="record_class_knn")
    parser.add_argument("--profile_scale_prior_records", type=float, default=5.)
    parser.add_argument("--ignore_classes", nargs="*", default=[], help="Explicit abstention head classes, e.g. Background")
    if discover:
        parser.description += "; then predict a mixed batch and discover anonymous classes"
        parser.add_argument("--test_inputs", nargs="+", required=True)
        parser.add_argument("--test_labels_json", type=Path)
        parser.add_argument("--test_label_mode", choices=["none", "filename", "parent"], default="none")
    return parser


def build_task_parser(task):
    if task == "visualize":
        from scripts.visualize_features import build_parser
        return build_parser()
    if task == "experiment":
        from util.open_set_experiment import build_parser
        return build_parser()
    if task == "train":
        from util.train_entry import build_parser
        return build_parser()
    if task in {"predict", "infer"}:
        from scripts.evaluate_single_signal import build_parser
        parser = build_parser()
        if task == "infer":
            parser.description = "Unlabeled open-set inference and batch unknown clustering; requires a frozen profile"
            for action in parser._actions:
                if action.dest == "open_set_profile":
                    action.required = True
        return parser
    if task == "cluster":
        from scripts.cluster_unknown_features import build_parser
        return build_parser()
    if task in {"profile", "discover"}:
        return profile_parser(task == "discover")
    raise ValueError(f"Unknown task parser: {task}")


def run_profile(args, discover=False, predictor=None):
    if not getattr(args, "output_dir", None):
        from util.run_context import execute_run
        return execute_run("discover" if discover else "profile", args,
                           lambda managed: run_profile(managed, discover, predictor))
    from scripts.evaluate_single_signal import run
    from util.open_set import build_profile

    from util.run_context import check_output_directory
    output = check_output_directory(args)
    if args.inputs or args.open_set_profile or args.signal_mode != "multi":
        raise ValueError("Profile uses reference_inputs/calibration_inputs and signal_mode multi; do not set inputs/open_set_profile")
    if args.labels_json or args.label_from_parent or args.label_from_filename:
        raise ValueError("Use profile label_mode and the separate reference/calibration label mappings")
    if args.label_mode == "json" and (not args.reference_labels_json or not args.calibration_labels_json):
        raise ValueError("JSON mode requires both reference_labels_json and calibration_labels_json")
    if discover and getattr(args, "test_output_dir", None):
        test_output = args.test_output_dir.expanduser().resolve()
        if test_output.exists() and (not test_output.is_dir() or any(test_output.iterdir())):
            raise ValueError("Test output_dir must be new or empty")
    for split in ("reference", "calibration"):
        extraction = copy.deepcopy(args)
        extraction.inputs = getattr(args, split+"_inputs")
        extraction.output_dir = output/split
        extraction.export_features, extraction.open_set_profile = True, None
        extraction.save_visuals = False
        extraction.label_from_filename = args.label_mode == "filename"
        extraction.label_from_parent = args.label_mode == "parent"
        extraction.labels_json = getattr(args, split+"_labels_json") if args.label_mode == "json" else None
        run(extraction, predictor)
    target = output/"known_profile.npz"
    metadata = build_profile(output/"reference/features.npz", output/"calibration/features.npz", target,
        args.knn_k, args.known_false_reject_rate, args.min_calibration_per_class,
        args.max_reference_per_class, args.ignore_classes,
        score_method=args.profile_score_method, scale_prior_records=args.profile_scale_prior_records)
    if not discover:
        return {"profile": str(target), "threshold": metadata["threshold"]}
    prediction = copy.deepcopy(args)
    prediction.inputs = args.test_inputs
    prediction.output_dir = getattr(args, "test_output_dir", None) or output/"discovery"
    prediction.open_set_profile, prediction.export_features = target, False
    prediction.labels_json = args.test_labels_json
    prediction.label_from_filename = args.test_label_mode == "filename"
    prediction.label_from_parent = args.test_label_mode == "parent"
    return run(prediction, predictor)


def validate_infer_args(args):
    if args.signal_mode != "multi" or args.export_features or args.labels_json or args.label_from_parent or args.label_from_filename:
        raise ValueError("infer requires multi mode and unlabeled inputs; use predict/discover for labeled evaluation")


def dispatch(task, args):
    from util.run_context import execute_run
    return execute_run(task, args, lambda managed: _dispatch(task, managed))


def _dispatch(task, args):
    if task == "visualize":
        from scripts.visualize_features import run
        return run(args)
    if task == "experiment":
        from util.open_set_experiment import run_experiment
        return run_experiment(args, args._config_path)
    if task == "infer":
        validate_infer_args(args)
        from scripts.evaluate_single_signal import run
        return run(args)
    if task == "train":
        from util.train_entry import main
        return main(args)
    if task == "predict":
        from scripts.evaluate_single_signal import run
        return run(args)
    if task == "cluster":
        from scripts.cluster_unknown_features import run
        return run(args)
    return run_profile(args, task == "discover")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: python main.py {" + ",".join(TASKS) + "} [--config FILE] [--dry_run] [options]\n"
              "Config sections follow task names; evaluation is shared. CLI values override YAML.\n"
              "Use TASK --help for the underlying options. Relative YAML paths are based on its directory.")
        return
    task = argv.pop(0)
    if task not in TASKS:
        # Preserve the original direct training/GT-evaluation CLI during phase 1.
        if task.startswith("-"):
            from util.train_entry import build_parser
            return dispatch("train", build_parser().parse_args([task, *argv]))
        raise ValueError(f"Unknown task {task}; choose {', '.join(TASKS)}")
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--config", type=Path)
    wrapper.add_argument("--dry_run", action="store_true")
    controls, explicit = wrapper.parse_known_args(argv)
    parser = build_task_parser(task)
    if "--help" in explicit or "-h" in explicit:
        parser.parse_args(["--help"])
    settings, path = load_settings(controls.config, task)
    args = parser.parse_args([*settings_tokens(parser, settings), *explicit])
    args._config_path = path
    if hasattr(args, "feature_plot_method"):
        from util.feature_visualization import plot_options
        plot_options(args)
    if task == "experiment":
        from util.open_set_experiment import stage_arguments
        if controls.dry_run:
            stage_arguments(args, path)
    if task == "infer" and controls.dry_run:
        validate_infer_args(args)
    if controls.dry_run:
        if task == "train":
            from util.train_entry import validate_training_options
            validate_training_options(args)
        from util.run_context import EXPERIMENT_ROOT, run_type
        print(json.dumps({"task": task, "config": str(path) if path else None,
                          "automatic_output_root": str(EXPERIMENT_ROOT), "experiment_type": run_type(task, args),
                          "effective": {k: v for k, v in vars(args).items() if not k.startswith("_")}}, default=str, ensure_ascii=False, indent=2, allow_nan=False))
        return args
    result = dispatch(task, args)
    if isinstance(result, dict):
        summary = result.get("metrics", {key: value for key, value in result.items() if key != "clusters"})
        print(json.dumps(summary, default=str, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
