"""One repository-local output policy for every public command."""
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import traceback
from time import monotonic

import pytz


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1] / "experiment"
TIMEZONE = pytz.timezone("Asia/Shanghai")


def now():
    return datetime.now(TIMEZONE)


def run_type(task, args):
    if task == "train":
        if getattr(args, "run_mode", "train") == "infer":
            return "validation"
        mode = getattr(args, "incremental_split_mode", "normal")
        if getattr(args, "trainer_type", "normal") == "lwf" or mode == "incremental":
            return "incremental_train"
        return "base_train" if mode == "base" else "train"
    if task == "experiment":
        return "open_set_split" if getattr(args, "split_only", False) else "open_set_experiment"
    if task == "predict":
        labeled = any(getattr(args, key, None) for key in ("labels_json", "label_from_parent", "label_from_filename"))
        return ("open_set_" if getattr(args, "open_set_profile", None) else "") + ("validation" if labeled else "inference")
    return {"infer": "open_set_inference", "profile": "known_profile", "discover": "open_set_discovery",
            "cluster": "unknown_clustering", "profile_archive": "known_profile"}[task]


def planned_directory(kind):
    """A display/config path only; never creates files during --dry_run."""
    return EXPERIMENT_ROOT / f"{kind}_{now():%Y%m%d_%H%M%S}"


def create_directory(kind):
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    candidate = planned_directory(kind)
    for suffix in range(10000):
        path = candidate if suffix == 0 else candidate.with_name(f"{candidate.name}_{suffix:03d}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("Too many runs with the same timestamp")


def execute_run(task, args, callback):
    """Allocate once; internal stages explicitly receive their parent's paths."""
    path = create_directory(run_type(task, args))
    args._run_dir = path
    args._managed_output = True
    if task != "train":
        args.output_dir = path
    started = now()
    start_clock = monotonic()
    record = {"schema_version": 1, "task": task, "experiment_type": run_type(task, args),
              "run_dir": str(path), "timezone": "Asia/Shanghai", "started_at": started.isoformat(),
              "status": "running", "config_file": str(getattr(args, "_config_path", "") or ""),
              "arguments": {key: value for key, value in vars(args).items() if not key.startswith("_")}}
    def save():
        (path / "run.json").write_text(json.dumps(_json_safe(record), ensure_ascii=False, indent=2,
                                                  default=str, allow_nan=False) + "\n", encoding="utf-8")
    save()
    print(f"Experiment directory: {path}", file=sys.stderr)
    try:
        result = callback(args)
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        record["status"] = "completed_with_errors" if metric_error_count(metrics) else "complete"
        if isinstance(result, dict):
            result["run_dir"] = str(path)
        return result if result is not None else {"run_dir": str(path), "status": "complete"}
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                      error=f"{type(exc).__name__}: {exc}")
        (path / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        finished = now()
        record.update(finished_at=finished.isoformat(), duration_seconds=monotonic() - start_clock)
        save()


def check_output_directory(args):
    """Permit only our own run metadata in a freshly allocated root."""
    path = args.output_dir.expanduser().resolve()
    owner = getattr(args, "_run_dir", None)
    allowed = {"run.json"} if getattr(args, "_managed_output", False) and owner is not None and path == Path(owner).resolve() else set()
    if path.exists() and (not path.is_dir() or any(p.name not in allowed for p in path.iterdir())):
        raise ValueError("Output directory must be new or empty")
    return path


def metric_error_count(metrics):
    return sum(int(metrics.get(key, 0) or 0) for key in ("error_files", "files_with_slice_errors", "visualization_errors"))


def command_exit_code(result):
    return 2 if isinstance(result, dict) and metric_error_count(result.get("metrics", {})) else 0


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value) if isinstance(value, float) and not math.isfinite(value) else value
