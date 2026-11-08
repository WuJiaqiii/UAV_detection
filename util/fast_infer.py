"""Compact inference output and progress; feature-producing code stays unchanged."""
from contextlib import AbstractContextManager
import csv
import json
from pathlib import Path
import sys

from tqdm import tqdm


def configure_fast_inference(args):
    if (args.signal_mode != "multi" or not args.open_set_profile or args.export_features
            or args.labels_json or args.label_from_parent or args.label_from_filename):
        raise ValueError("--fast requires unlabeled multi-mode inference with an open_set_profile; "
                         "use normal predict/profile for feature export or labeled evaluation")
    args.save_visuals = False
    args.save_feature_plots = False


class _ProgressPredictor:
    def __init__(self, predictor, bar):
        self.predictor, self.bar = predictor, bar

    def predict(self, spectrum, domain):
        try:
            return self.predictor.predict(spectrum, domain)
        finally:
            self.bar.update(1)

    def set_source(self,path):
        if hasattr(self.predictor,'set_source'):self.predictor.set_source(path)


class InferenceProgress(AbstractContextManager):
    """File and slice bars on stderr, with deterministic cleanup on failures."""
    def __init__(self, inputs, args, config):
        self.args, self.config, self.slice_bar, self.identity_bar = args, config, None, None
        self.files = tqdm(inputs, desc="Files", unit="file", file=sys.stderr,
                          disable=not args.progress, dynamic_ncols=True, position=0)

    def __iter__(self):
        return iter(self.files)

    def predictor(self, predictor, path):
        if not self.args.progress:
            return predictor
        try:
            total = 1
            if Path(path).suffix.lower() == ".dat":
                from data.iq_dat_loader import BYTES_PER_COMPLEX_SAMPLE
                from util.eval_common import plan_iq_slices
                windows, _ = plan_iq_slices(Path(path).stat().st_size // BYTES_PER_COMPLEX_SAMPLE,
                    self.config.iq_sample_rate, self.args.iq_stride_ms, self.args.tail_policy, self.args.max_slices)
                total = len(windows)
        except (OSError, ValueError):
            # The evaluator owns input errors; progress must not change them.
            return predictor
        self.slice_bar = tqdm(total=total, desc=Path(path).name, unit="slice", file=sys.stderr,
                              dynamic_ncols=True, position=1, leave=False)
        return _ProgressPredictor(predictor, self.slice_bar)

    def finish_file(self):
        if self.slice_bar is not None:
            # Slices failing before model invocation were also processed.
            self.slice_bar.update(max(0, self.slice_bar.total - self.slice_bar.n))
            self.slice_bar.close()
            self.slice_bar = None

    def clustering(self, count):
        if self.args.progress:
            tqdm.write(f"Clustering: {count} unknown track(s)...", file=sys.stderr)

    def identities(self, results):
        self.identity_bar = tqdm(results, desc="Recording IDs", unit="file", file=sys.stderr,
                                 disable=not self.args.progress, dynamic_ncols=True)
        return self.identity_bar

    def __exit__(self, *exc):
        self.finish_file()
        self.files.close()
        if self.identity_bar is not None:
            self.identity_bar.close()


def input_recording_identity(result, recording_ids):
    """Avoid hashing large captures when an explicit ID/Figure ID already wins."""
    from util.recordings import recording_identity, file_sha256
    path = result["path"]
    if not result["signals"]:
        return (recording_ids or {}).get(path, path)
    identity = recording_identity(path, "", recording_ids)
    return identity or recording_identity(path, file_sha256(path), recording_ids)


def retain_submission_bursts(result, rows):
    """Keep per-box offsets before releasing slice details; never merge repeated bursts."""
    from util.burst_normalization import collect_raw_bursts
    collect_raw_bursts(result,rows)


def write_predictions(output, results, input_mode, discovery, classes=None):
    """One row per file in single mode; one row per frequency track in multi."""
    from util.burst_normalization import finalize_bursts
    finalize_bursts(results)
    target = output / "predictions.csv"
    columns = ["path", "label", "status", "error"]
    if input_mode == "multi":
        columns = ["path", "track_id", "label", "frequency_low_offset_mhz", "frequency_high_offset_mhz",
                   "frequency_low_absolute_mhz", "frequency_high_absolute_mhz", "status", "error", "bursts_json"]
    count = 0
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            error = result.get("error", "")
            if result.get("error_slices"):
                error = error or f"{result['error_slices']} slice(s) failed"
            base = {"path": result["path"], "error": error}
            if result["status"] == "error":
                rows = [{**base, "label": "error", "status": "error"}]
            elif input_mode == "single":
                decision = result["single_signal_prediction"]
                rows = [{**base, "label": decision["display_label"],
                         "status": decision["recognition_status"]}]
            elif not result["signals"]:
                rows = [{**base, "label": "no_signal", "status": result["status"]}]
            else:
                rows = [{**base, "track_id": track["track_id"], "label": track["display_label"],
                         "status": track["recognition_status"],
                         "bursts_json": json.dumps(track.get("submission_bursts", []), separators=(",", ":"), allow_nan=False),
                         **{key: track.get(key) for key in columns if key.startswith("frequency_")}}
                        for track in result["signals"] if track.get('submission_bursts')]
            for row in rows:
                writer.writerow(row)
                count += 1
    metrics = {"total_files": len(results), "output_rows": count,
               'submission_unit': 'recording' if input_mode == 'single' else 'burst',
               "error_files": sum(row["status"] == "error" for row in results),
               "files_with_slice_errors": sum(row.get("error_slices", 0) > 0 for row in results),
               "estimated_unknown_class_count": discovery["estimated_unknown_class_count"],
               'discovery_constraint_failures':int(discovery.get('constraint_satisfied') is False),
               'burst_label_conflicts':sum(r.get('burst_diagnostics',{}).get('label_conflicts',0) for r in results)}
    metrics['event_export_valid'] = not bool(metrics['burst_label_conflicts'])
    from util.pipeline_diagnostics import write_diagnostics
    metrics['pipeline_diagnostics']=write_diagnostics(output,results,classes)
    return {"metrics": metrics, "predictions_csv": str(target), "files": results, "discovery": discovery}
