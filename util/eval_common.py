"""Shared input, checkpoint config, native slicing and voting for inference."""
from collections import Counter
from collections.abc import Mapping
import glob
import json
import math
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from data.iq_dat_loader import read_iq_dat, iq_to_spectrogram
from util.spectrogram_frontend import (
    FrontendConfig, SPEC_SHAPE, TARGET_SAMPLE_RATE, apply_frontend_manifest,
    read_mat_spectrum, require_shape, validate_iq_settings,
)


INFERENCE_DEFAULTS = {
    "backbone": "resnet18", "cnn_dropout": 0.,
    "mask_img_size": 224, "mask_in_chans": 1, "mask_fill_value": 255,
    "sampling_rate": float(TARGET_SAMPLE_RATE), "frequency_axis": "fft",
    "mat_key": "summed_submatrices", "mat_spec_domain": "amplitude",
    "iq_sample_rate": 100_000_000, "iq_segment_ms": 50.,
    "iq_n_fft": 512, "iq_hop_length": 512, "iq_byte_order": "<",
    "yolo_imgsz_h": 640, "yolo_imgsz_w": 640, "yolo_conf_thres": .05,
    "yolo_iou_thres": .30, "yolo_max_det": 1000, "yolo_classes": None,
    "grouping_mode": "frequency", "hop_association": "shape",
    "hop_shape_log_tol": .35, "hop_use_contrast_gate": False,
    "hop_contrast_delta": 3., "hop_period_rel_tol": .15,
    "hop_max_missing_pulses": 2, "hop_max_overlap_ratio": .2,
    "hop_max_conflict_fraction": .2, "hop_min_timing_consistency": .85,
    "hop_min_pulses": 3, "hop_min_time_extent_ratio": .15,
    "hop_min_time_coverage_ratio": .02,
    "min_area": 4, "min_ratio": 0., "min_width": 2, "min_height": 2,
    "max_width": 0, "max_height": 0, "ring_margin": 5, "min_contrast_z": -2.,
    "freq_eps": 5., "freq_min_samples": 1, "merge_freq_thresh": 10.,
    "merge_w_log_thresh": .35, "merge_h_log_thresh": .35, "merge_energy_thresh": 1.,
    "min_group_len": 2, "min_group_time_span_ratio": .15, "nms_thresh": .2,
    "score_n_boxes_weight": .5, "score_time_span_weight": 2.,
    "score_contrast_weight": 1., "score_w_std_weight": 1.,
    "score_h_std_weight": 10., "score_contrast_std_weight": 3.,
    "frontend_mode": "global", "frontend_z_min": -2., "frontend_z_max": 8.,
    "frontend_min_scale_db": 1., "frontend_frequency_quantile": .2,
    "frontend_frequency_smooth": 9, "frontend_background_band_mhz": 50.,
}


SUPPORTED_SUFFIXES = {".mat", ".png", ".dat"}


def collect_inputs(values, recursive=False):
    found = set()
    for value in values:
        path = Path(value).expanduser()
        matches = [path] if path.exists() else [Path(p) for p in glob.glob(str(path), recursive=recursive)]
        if not matches:
            raise FileNotFoundError(value)
        for match in matches:
            if match.is_dir():
                candidates = match.rglob("*") if recursive else match.iterdir()
                found.update(p.resolve() for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
            elif match.is_file() and match.suffix.lower() in SUPPORTED_SUFFIXES:
                found.add(match.resolve())
            else:
                raise ValueError(f"Expected MAT, PNG or DAT input: {match}")
    if not found:
        raise ValueError("No MAT, PNG or DAT inputs found")
    return sorted(found)


def validate_classes(classes):
    if not isinstance(classes, Mapping) or not classes:
        raise ValueError("Need a nonempty class-name-to-ID mapping from checkpoint config.classes or --classes_json")
    if any(not isinstance(name, str) or not name or isinstance(idx, bool) or not isinstance(idx, Integral)
           for name, idx in classes.items()):
        raise ValueError("Classes must map nonempty string names to integer IDs")
    classes = {name: int(idx) for name, idx in classes.items()}
    if sorted(classes.values()) != list(range(len(classes))):
        raise ValueError("Class IDs must be unique and contiguous, starting at 0")
    return classes


def resolve_config(checkpoint, classes=None, overrides=None, frontend_config=None, logger=None):
    from util.checkpoint import _validate_checkpoint_classes

    saved = checkpoint.get("config", {})
    if not isinstance(saved, Mapping):
        raise ValueError("checkpoint.config must be a dictionary")
    class_map = validate_classes(classes if classes is not None else saved.get("classes"))
    _validate_checkpoint_classes(checkpoint, class_map, logger)
    # Only effective inference settings: no stale training paths/devices/frozen flags.
    values = {key: saved.get(key, default) for key, default in INFERENCE_DEFAULTS.items()}
    cfg = SimpleNamespace(**values, classes=class_map, num_classes=len(class_map))
    cfg.frontend_config = str(frontend_config or "")
    # input_type is intentionally absent: a mixed run must not import source-specific
    # MAT domains or IQ capture rates from a manifest for another input type.
    apply_frontend_manifest(cfg)
    for key, value in (overrides or {}).items():
        if key not in INFERENCE_DEFAULTS:
            raise ValueError(f"Unknown inference setting: {key}")
        default = INFERENCE_DEFAULTS[key]
        if isinstance(default, bool) and not isinstance(value, bool):
            raise ValueError(f"{key} requires a boolean")
        if isinstance(default, int) and (isinstance(value, bool) or not isinstance(value, Integral)):
            raise ValueError(f"{key} requires an integer")
        setattr(cfg, key, value)
    FrontendConfig.from_config(cfg)
    if cfg.sampling_rate != TARGET_SAMPLE_RATE or cfg.frequency_axis != "fft":
        raise ValueError("Native output grid requires sampling_rate=122880000 and frequency_axis=fft")
    if cfg.mask_in_chans != 1 or int(cfg.mask_img_size) <= 0 or not 0 <= cfg.mask_fill_value <= 255:
        raise ValueError("Need one mask channel, positive mask size and fill value in [0,255]")
    if cfg.mat_spec_domain not in {"amplitude", "power", "db", "image"}:
        raise ValueError("Unsupported mat_spec_domain")
    if cfg.iq_byte_order not in {"<", ">"}:
        raise ValueError("iq_byte_order must be '<' or '>'")
    validate_iq_settings(cfg.iq_sample_rate, cfg.iq_segment_ms, cfg.iq_n_fft, cfg.iq_hop_length)
    return cfg


def plan_iq_slices(total_samples, sample_rate, stride_ms=50., tail_policy="drop", max_slices=0):
    """Plan deterministic 50 ms windows in source-complex-sample coordinates.

    No padding. 'cover' adds an end-aligned window and may overlap the last one.
    A limit samples evenly across the plan, rather than favoring the file start.
    """
    size = validate_iq_settings(sample_rate)
    if not math.isfinite(stride_ms) or stride_ms <= 0:
        raise ValueError("iq_stride_ms must be finite and > 0")
    stride = round(stride_ms * sample_rate / 1000.)
    if stride < 1 or max_slices < 0 or tail_policy not in {"drop", "cover"}:
        raise ValueError("Invalid slice stride, max_slices or tail policy")
    if total_samples < size:
        raise ValueError("DAT is shorter than one complete 50 ms slice; padding is not used")
    # Keep a range, so long recordings do not allocate all unused window starts.
    regular = range(0, total_samples - size + 1, stride)
    extra = tail_policy == "cover" and regular[-1] != total_samples - size
    planned = len(regular) + int(extra)
    indices = ([((planned - 1) // 2)] if max_slices == 1 else
               np.linspace(0, planned - 1, max_slices, dtype=np.int64).tolist()
               if max_slices and max_slices < planned else range(planned))
    starts = [regular[i] if i < len(regular) else total_samples - size for i in indices]
    windows = [{"start_sample": start, "end_sample": start + size,
                "start_ms": start / sample_rate * 1000., "end_ms": (start + size) / sample_rate * 1000.}
               for start in starts]
    # Report actual unobserved samples, including gaps or a slice count limit.
    covered, end = 0, 0
    for window in windows:
        covered += max(0, window["end_sample"] - max(end, window["start_sample"]))
        end = max(end, window["end_sample"])
    return windows, {"total_samples": total_samples, "sample_rate": sample_rate,
                     "planned_slices": planned, "evaluated_slices": len(windows),
                     "unobserved_samples": total_samples - covered,
                     "tail_policy": tail_policy, "stride_ms": stride_ms}


def load_spectrum(path, config, window=None):
    path = Path(path)
    if path.suffix.lower() == ".dat":
        if window is None:
            raise ValueError("DAT requires a slice window")
        iq = read_iq_dat(path, window["start_sample"], window["end_sample"] - window["start_sample"],
                         byte_order=config.iq_byte_order)
        return iq_to_spectrogram(iq, sample_rate=config.iq_sample_rate,
                                 n_fft=config.iq_n_fft, hop_length=config.iq_hop_length), "amplitude"
    if path.suffix.lower() == ".mat":
        return read_mat_spectrum(path, config.mat_key, config.mat_spec_domain), (
            "image" if config.mat_spec_domain == "image" else "amplitude")
    if path.suffix.lower() == ".png":
        # Match UAVDataset's grayscale loading, while handling Windows Unicode paths.
        encoded = np.fromfile(path, dtype=np.uint8)
        native = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if native is None or native.dtype != np.uint8:
            raise ValueError("Expected an 8-bit rendered PNG")
        gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Cannot decode PNG: {path}")
        return np.ascontiguousarray(require_shape(gray)), "image"
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def vote_predictions(rows, classes, method="majority", min_slice_confidence=0.,
                     min_votes=1, min_vote_share=0., exclude_classes=()):
    """One vote per valid slice; ties use average probabilities, then lowest ID."""
    classes = validate_classes(classes)
    if method not in {"majority", "soft"}:
        raise ValueError("vote_method must be majority or soft")
    if not 0 <= min_slice_confidence <= 1 or not 0 <= min_vote_share <= 1 or min_votes < 1:
        raise ValueError("Invalid voting thresholds")
    exclude_classes = set(exclude_classes)
    if exclude_classes - classes.keys():
        raise ValueError(f"Unknown excluded vote classes: {exclude_classes - classes.keys()}")
    names = {idx: name for name, idx in classes.items()}
    accepted, reasons = [], Counter()
    for row in rows:
        reason = row["status"]
        if reason == "ok":
            probabilities = np.asarray(row["probabilities"], dtype=float)
            if (probabilities.shape != (len(classes),) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1., atol=1e-5)):
                raise ValueError("Slice probabilities must be a finite normalized class vector")
            label = int(np.argmax(probabilities))
            row.update(class_id=label, class_name=names[label], confidence=float(probabilities[label]))
            if row["confidence"] < min_slice_confidence:
                reason = "low_confidence"
            elif names[label] in exclude_classes:
                reason = "excluded_class"
            else:
                accepted.append(probabilities)
        row["vote_accepted"] = reason == "ok"
        row["vote_reason"] = reason
        if reason != "ok":
            reasons[reason] += 1
    counts = np.zeros(len(classes), dtype=int)
    mean = None
    if accepted:
        matrix = np.asarray(accepted)
        counts = np.bincount(matrix.argmax(axis=1), minlength=len(classes))
        mean = matrix.mean(axis=0)
    result = {"status": "unknown", "class_id": None, "class_name": None,
              "candidate_class_name": None, "mean_probability": None, "vote_share": None,
              "valid_votes": len(accepted), "total_slices": len(rows),
              "vote_counts": {names[i]: int(counts[i]) for i in range(len(classes))},
              "mean_probabilities": None if mean is None else {names[i]: float(mean[i]) for i in range(len(classes))},
              "abstention_counts": dict(reasons), "vote_method": method}
    if not accepted:
        result["unknown_reason"] = "no_valid_votes"
        return result
    if method == "majority":
        candidates = np.flatnonzero(counts == counts.max())
        winner = int(candidates[np.argmax(mean[candidates])])
    else:
        # Excluded classes must not re-enter as a soft-vote runner-up.
        candidates = [i for i in range(len(classes)) if names[i] not in exclude_classes]
        winner = max(candidates, key=lambda i: (mean[i], -i))
    share = float(counts[winner] / len(accepted))
    result.update(candidate_class_name=names[winner], mean_probability=float(mean[winner]), vote_share=share)
    if len(accepted) < min_votes or share < min_vote_share:
        result["unknown_reason"] = "too_few_votes" if len(accepted) < min_votes else "low_vote_share"
    else:
        result.update(status="ok", class_id=winner, class_name=names[winner])
    return result


def read_path_mapping(path):
    path = Path(path).expanduser().resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Expected a JSON object keyed by input paths")
    result = {}
    for name, value in raw.items():
        file = Path(name).expanduser()
        key = str((file if file.is_absolute() else path.parent/file).resolve())
        if key in result:
            raise ValueError(f"Duplicate normalized input path: {name}")
        result[key] = value
    return result
