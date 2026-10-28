"""Legacy file-level policy on shared group prediction, slicing and voting."""
from collections import Counter
from pathlib import Path

from data.iq_dat_loader import BYTES_PER_COMPLEX_SAMPLE
from util.eval_common import (
    INFERENCE_DEFAULTS, SUPPORTED_SUFFIXES, collect_inputs, validate_classes,
    resolve_config, plan_iq_slices, load_spectrum, vote_predictions, read_path_mapping,
)
from util.group_predictor import GroupPredictor


class SingleSignalPredictor(GroupPredictor):
    def __init__(self, classifier, detector, preprocessor, config, device="cpu", group_policy="main"):
        if group_policy not in {"main", "union"}:
            raise ValueError("group_policy must be main or union")
        super().__init__(classifier, detector, preprocessor, config, device)
        self.group_policy = group_policy

    def predict(self, spectrum, domain):
        import numpy as np
        result, groups = self.detect_groups(spectrum, domain)
        if not groups:
            return result
        main = max(groups, key=lambda group: float(group.get("score", 0.)))
        selected = (main["boxes"] if self.group_policy == "main" else
                    np.unique(np.concatenate([g["boxes"] for g in groups]), axis=0))
        predicted = self.classify_groups(spectrum, [{**main, "boxes": selected}])[0]
        result.update({key: predicted[key] for key in (
            "class_id", "confidence", "probabilities", "selected_boxes", "selected_group_score")})
        return result


def evaluate_file(path, predictor, config, vote_options=None, stride_ms=50., tail_policy="drop",
                  max_slices=0, fail_fast=False):
    path = Path(path).resolve()
    if path.suffix.lower() == ".dat":
        size = path.stat().st_size
        if size % BYTES_PER_COMPLEX_SAMPLE:
            raise ValueError("DAT size is not a whole number of int16 I/Q pairs")
        windows, sampling = plan_iq_slices(size // BYTES_PER_COMPLEX_SAMPLE, config.iq_sample_rate,
                                          stride_ms, tail_policy, max_slices)
    else:
        windows, sampling = [None], None
    rows = []
    for index, window in enumerate(windows):
        row = {"path": str(path), "slice_index": index, **(window or {})}
        try:
            spectrum, domain = load_spectrum(path, config, window)
            row.update(predictor.predict(spectrum, domain))
        except Exception as exc:
            if fail_fast:
                raise
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    result = vote_predictions(rows, config.classes, **(vote_options or {}))
    result.update(path=str(path), input_type=path.suffix.lower()[1:], sampling=sampling,
                  error_slices=sum(row["status"] == "error" for row in rows))
    if result["error_slices"] == len(rows):
        result["status"] = "error"
    return result, rows


def load_truth(path, inputs, classes, label_from_parent=False):
    truth = {}
    if path:
        labels = read_path_mapping(path)
        for file, name in labels.items():
            if name not in classes:
                raise ValueError(f"Unknown truth class {name!r} for {file}")
            truth[file] = name
    if label_from_parent:
        for file in inputs:
            name = file.parent.name
            if name not in classes:
                raise ValueError(f"Parent folder is not a classifier class: {file}")
            if str(file) in truth and truth[str(file)] != name:
                raise ValueError(f"Conflicting truth labels for {file}")
            truth[str(file)] = name
    return truth


def evaluation_metrics(results, classes):
    labeled = [row for row in results if row.get("true_class_name") is not None]
    decided = [row for row in labeled if row["status"] == "ok"]
    correct = sum(row.get("class_name") == row["true_class_name"] for row in labeled)
    confusion = {name: {} for name in classes}
    for row in labeled:
        prediction = row.get("class_name") if row["status"] == "ok" else "<abstained>"
        counter = confusion[row["true_class_name"]]
        counter[prediction] = counter.get(prediction, 0) + 1
    return {"total_files": len(results), "classified_files": sum(r["status"] == "ok" for r in results),
            "unknown_files": sum(r["status"] == "unknown" for r in results),
            "error_files": sum(r["status"] == "error" for r in results),
            "files_with_slice_errors": sum(r.get("error_slices", 0) > 0 for r in results),
            "labeled_files": len(labeled), "correct_files": correct,
            "file_accuracy": correct / len(labeled) if labeled else None,
            "labeled_coverage": len(decided) / len(labeled) if labeled else None,
            "accuracy_on_classified": correct / len(decided) if decided else None,
            "confusion_counts": confusion}
