#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Background-only group-level .mat samples from original single-signal .mat files.

Main logic:
    original single-signal mat
        -> YOLO detect
        -> SignalPreprocessor.select_signal_groups()
        -> match groups to filename target by center_freq / bandwidth
        -> if target is missed: generate nothing
        -> if target is matched: save all unmatched groups as Background samples

Output mat content:
    summed_submatrices:
        full-size matrix where only one background group boxes are kept.
        pixels outside this background group are set to 0.

Filename format:
    Background-[0,center_freq,1000,bandwidth]-SNR-snr-Src-hash-Group-idx.mat

Notes:
    - The second bracket field is center frequency.
    - The fourth bracket field is bandwidth.
    - No positive samples are generated.
"""

from __future__ import annotations

import os
import re
import sys
import json
import glob
import argparse
import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from scipy.io import loadmat, savemat


# ----------------------------------------------------------------------
# Make project imports work
# ----------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()

# If this script is placed at project root: UAV_detection/generate_background_only.py
if (_THIS_FILE.parent / "util").exists():
    PROJECT_ROOT = _THIS_FILE.parent
# If this script is placed under scripts/: UAV_detection/scripts/generate_background_only.py
elif (_THIS_FILE.parents[1] / "util").exists():
    PROJECT_ROOT = _THIS_FILE.parents[1]
else:
    raise RuntimeError(
        "Cannot locate project root. Please place this script under project root "
        "or under project_root/scripts/."
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.detector import YoloV5Detector
from util.preprocess import SignalPreprocessor

try:
    from util.bboxcache import BBoxCache
except Exception:
    BBoxCache = None


# ----------------------------------------------------------------------
# Default class mapping
# Used only to validate source filename protocols.
# Background must exist because generated files are labeled as Background.
# Please keep this consistent with your training Config.
# ----------------------------------------------------------------------
DEFAULT_CLASSES = {
    "Background": 0,
    "FPV1": 1,
    "Lightbridge1": 2,
    "Ocusync_mini1": 3,
    "Ocusync21": 4,
    "Ocusync31": 5,
    "Ocusync41": 6,
    "Skylink11": 7,
    "Skylink21": 8,
}


SIGNAL_RE = re.compile(r"(?P<protocol>[A-Za-z0-9_]+)-\[(?P<bracket>[^\]]+)\]")
SNR_RE = re.compile(r"-SNR-(?P<snr>[-+]?\d+(?:\.\d+)?)")


# ----------------------------------------------------------------------
# Logger
# ----------------------------------------------------------------------
def create_logger(name: str = "generate_background_only") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ----------------------------------------------------------------------
# File collection / parsing
# ----------------------------------------------------------------------
def collect_mat_files(input_dir: str, recursive: bool = False) -> List[str]:
    input_dir = str(input_dir)
    pattern = "**/*.mat" if recursive else "*.mat"
    files = sorted(glob.glob(os.path.join(input_dir, pattern), recursive=recursive))
    return [f for f in files if os.path.isfile(f)]


def load_classes(classes_json: str | None) -> Dict[str, int]:
    if not classes_json:
        return dict(DEFAULT_CLASSES)

    s = str(classes_json).strip()

    if os.path.isfile(s):
        with open(s, "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        obj = json.loads(s)

    if not isinstance(obj, dict):
        raise ValueError("--classes_json must be a dict JSON string or a JSON file")

    return {str(k): int(v) for k, v in obj.items()}


def parse_single_signal_target(
    fp: str,
    classes: Dict[str, int],
) -> Tuple[Dict[str, Any] | None, float]:
    """
    Parse one source filename.

    Expected bracket format:
        Protocol-[field0, center_freq, field2, bandwidth]

    Returns:
        target dict or None
        snr
    """
    fname = os.path.basename(fp)
    stem = os.path.splitext(fname)[0]

    snr = float("nan")
    m_snr = SNR_RE.search(stem)
    if m_snr is not None:
        try:
            snr = float(m_snr.group("snr"))
        except ValueError:
            snr = float("nan")

    prefix = stem.split("-SNR-")[0]
    matches = list(SIGNAL_RE.finditer(prefix))

    # This background generator is designed for original single-signal files only.
    if len(matches) != 1:
        return None, snr

    m = matches[0]
    protocol = m.group("protocol").strip()

    if protocol not in classes:
        return None, snr

    parts = [p.strip() for p in m.group("bracket").split(",")]
    if len(parts) < 4:
        return None, snr

    try:
        center_freq = float(parts[1])
        bandwidth = float(parts[3])
    except ValueError:
        return None, snr

    if bandwidth <= 0:
        return None, snr

    target = {
        "label": int(classes[protocol]),
        "class_name": protocol,
        "center_freq": float(center_freq),
        "bandwidth": float(bandwidth),
    }
    return target, snr


# ----------------------------------------------------------------------
# Mat helpers
# ----------------------------------------------------------------------
def load_mat_array(fp: str, mat_key: str) -> np.ndarray:
    mat = loadmat(fp, variable_names=[mat_key])
    if mat_key not in mat:
        raise KeyError(f"key '{mat_key}' not found in {fp}")

    x = np.asarray(mat[mat_key])
    if x.ndim != 2:
        raise ValueError(f"data dim != 2, got shape={x.shape}")

    return np.asarray(x, dtype=np.float32)


def group_to_mat_fullsize(
    spec: np.ndarray,
    boxes,
    mode: str = "raw_in_boxes",
    mask_value: float = 1.0,
) -> np.ndarray:
    """
    Convert one group to a full-size matrix.

    mode:
        raw_in_boxes:
            keep original spectrogram values inside group boxes,
            set outside to 0.

        mask:
            group boxes are mask_value,
            outside is 0.
    """
    spec = np.asarray(spec, dtype=np.float32)
    h, w = spec.shape[:2]

    boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)

    out = np.zeros_like(spec, dtype=np.float32)

    if mode not in {"raw_in_boxes", "mask"}:
        raise ValueError(f"Unsupported --output_mat_mode={mode}")

    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        if mode == "raw_in_boxes":
            out[y1:y2, x1:x2] = spec[y1:y2, x1:x2]
        elif mode == "mask":
            out[y1:y2, x1:x2] = float(mask_value)

    return np.ascontiguousarray(out, dtype=np.float32)


def safe_float_str(x: float, ndigits: int = 4) -> str:
    x = float(x)
    s = f"{x:.{ndigits}f}"
    s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def snr_to_str(snr: float) -> str:
    if np.isnan(float(snr)):
        return "nan"
    return safe_float_str(float(snr), ndigits=2)


def source_hash(fp: str, root: str | None = None) -> str:
    if root:
        try:
            rel = os.path.relpath(fp, root)
        except Exception:
            rel = fp
    else:
        rel = fp

    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]


def make_background_filename(
    center_freq: float,
    bandwidth: float,
    snr: float,
    src_fp: str,
    group_idx: int,
    src_root: str | None = None,
) -> str:
    """
    New filename format:
        Background-[0,center_freq,1000,bandwidth]-SNR-snr-Src-hash-Group-group_idx.mat
    """
    c = safe_float_str(center_freq, ndigits=4)
    bw = safe_float_str(max(float(bandwidth), 1e-6), ndigits=4)
    snr_s = snr_to_str(snr)
    h = source_hash(src_fp, root=src_root)

    return f"Background-[0,{c},1000,{bw}]-SNR-{snr_s}-Src-{h}-Group-{int(group_idx)}.mat"


def save_background_mat(
    save_path: str,
    mat_key: str,
    group_mat: np.ndarray,
    source_fp: str,
    source_class_name: str,
    source_center_freq: float,
    source_bandwidth: float,
    bg_center_freq: float,
    bg_bandwidth: float,
    group_idx: int,
    boxes,
):
    boxes_arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)

    payload = {
        mat_key: np.asarray(group_mat, dtype=np.float32),

        # Label metadata for generated sample
        "class_name": np.array("Background"),
        "center_freq": np.array([[float(bg_center_freq)]], dtype=np.float32),
        "bandwidth": np.array([[float(bg_bandwidth)]], dtype=np.float32),
        "is_background": np.array([[1]], dtype=np.int32),

        # Source metadata
        "source_file": np.array(source_fp),
        "source_class_name": np.array(source_class_name),
        "source_center_freq": np.array([[float(source_center_freq)]], dtype=np.float32),
        "source_bandwidth": np.array([[float(source_bandwidth)]], dtype=np.float32),

        # Group metadata
        "group_idx": np.array([[int(group_idx)]], dtype=np.int32),
        "boxes": boxes_arr.astype(np.int32),
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    savemat(save_path, payload, do_compression=True)


# ----------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------
def match_groups_to_single_target(
    groups: List[Dict[str, Any]],
    target: Dict[str, Any],
    match_freq_thresh: float,
    match_bandwidth_weight: float,
    skip_unmatched: bool = True,
):
    """
    Match groups to the only true target.

    Returns:
        matched_group_indices: list[int], length 0 or 1
        unmatched_group_indices: list[int]

    If target is missed, return:
        matched_group_indices = []
        unmatched_group_indices = all groups
    """
    if groups is None:
        groups = []

    if len(groups) == 0:
        return [], []

    target_freq = float(target.get("center_freq", 0.0))
    target_bw = float(target.get("bandwidth", 0.0))

    costs = []
    for gi, g in enumerate(groups):
        gf = float(g.get("center_freq", 0.0))
        gbw = float(g.get("bandwidth", 0.0))

        freq_diff = abs(target_freq - gf)
        bw_diff = abs(target_bw - gbw)
        cost = float(freq_diff + float(match_bandwidth_weight) * bw_diff)
        costs.append((cost, freq_diff, gi))

    costs = sorted(costs, key=lambda x: x[0])
    _, best_freq_diff, best_gi = costs[0]

    if skip_unmatched and best_freq_diff > float(match_freq_thresh):
        return [], list(range(len(groups)))

    matched = [int(best_gi)]
    unmatched = [i for i in range(len(groups)) if i != int(best_gi)]

    return matched, unmatched


# ----------------------------------------------------------------------
# BBox cache wrapper
# ----------------------------------------------------------------------
def detect_with_optional_cache(
    detector,
    bbox_cache,
    spec,
    fp: str,
    strict_read: bool = False,
):
    if bbox_cache is not None:
        cached = bbox_cache.get(fp)
        if cached is not None:
            if torch.is_tensor(cached):
                cached = cached.detach().cpu().numpy()
            return np.asarray(cached, dtype=np.int32).reshape(-1, 4)

        if strict_read and getattr(bbox_cache, "mode", "") == "read":
            raise RuntimeError(f"[BBoxCache] cache miss in strict read mode: {fp}")

    boxes = detector.detect(spec)
    boxes = (
        np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        if len(boxes) > 0
        else np.zeros((0, 4), dtype=np.int32)
    )

    if bbox_cache is not None:
        bbox_cache.put(fp, torch.as_tensor(boxes, dtype=torch.int32))

    return boxes


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
def build_config(args) -> SimpleNamespace:
    classes = load_classes(args.classes_json)

    if "Background" not in classes:
        raise ValueError("classes must contain 'Background'.")

    cfg = SimpleNamespace(**vars(args))
    cfg.classes = classes
    cfg.yolo_classes = None

    if args.yolo_classes:
        cfg.yolo_classes = [
            int(x)
            for x in str(args.yolo_classes).split(",")
            if str(x).strip()
        ]

    return cfg


# ----------------------------------------------------------------------
# Main generation
# ----------------------------------------------------------------------
def process_input_dir(
    input_dir: str,
    output_dir: str,
    cfg,
    logger,
    detector,
    preprocessor,
    bbox_cache=None,
):
    files = collect_mat_files(input_dir, recursive=cfg.recursive)
    logger.info(f"[Input] found {len(files)} mat files from {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    n_total = 0
    n_bad_parse = 0
    n_no_group = 0
    n_missed = 0
    n_no_background_group = 0
    n_bg_saved = 0
    n_failed = 0

    for fp in files:
        n_total += 1

        try:
            target, snr = parse_single_signal_target(fp, cfg.classes)
            if target is None:
                n_bad_parse += 1
                logger.warning(f"[SkipParse] {os.path.basename(fp)}")
                continue

            spec = load_mat_array(fp, cfg.mat_key)

            yolo_boxes = detect_with_optional_cache(
                detector=detector,
                bbox_cache=bbox_cache,
                spec=spec,
                fp=fp,
                strict_read=cfg.bbox_cache_strict_read,
            )

            groups = preprocessor.select_signal_groups(
                yolo_boxes,
                spectrogram=spec,
            ) or []

            if len(groups) == 0:
                n_no_group += 1
                continue

            matched_idx, unmatched_idx = match_groups_to_single_target(
                groups=groups,
                target=target,
                match_freq_thresh=cfg.match_freq_thresh,
                match_bandwidth_weight=cfg.match_bandwidth_weight,
                skip_unmatched=cfg.skip_unmatched,
            )

            # Key rule:
            # If the real signal is missed, do not generate any samples.
            if len(matched_idx) == 0:
                n_missed += 1
                continue

            # If there is only the real signal group, no background can be generated.
            if len(unmatched_idx) == 0:
                n_no_background_group += 1
                continue

            for gi in unmatched_idx:
                g = groups[int(gi)]
                boxes = np.asarray(g.get("boxes", []), dtype=np.int32).reshape(-1, 4)

                if len(boxes) == 0:
                    continue

                bg_center_freq = float(g.get("center_freq", 0.0))
                bg_bandwidth = float(g.get("bandwidth", 0.0))
                bg_bandwidth = max(bg_bandwidth, 1e-6)

                group_mat = group_to_mat_fullsize(
                    spec=spec,
                    boxes=boxes,
                    mode=cfg.output_mat_mode,
                    mask_value=cfg.mask_value,
                )

                out_name = make_background_filename(
                    center_freq=bg_center_freq,
                    bandwidth=bg_bandwidth,
                    snr=snr,
                    src_fp=fp,
                    group_idx=int(gi),
                    src_root=input_dir,
                )

                out_path = os.path.join(output_dir, out_name)

                save_background_mat(
                    save_path=out_path,
                    mat_key=cfg.mat_key,
                    group_mat=group_mat,
                    source_fp=fp,
                    source_class_name=str(target["class_name"]),
                    source_center_freq=float(target["center_freq"]),
                    source_bandwidth=float(target["bandwidth"]),
                    bg_center_freq=bg_center_freq,
                    bg_bandwidth=bg_bandwidth,
                    group_idx=int(gi),
                    boxes=boxes,
                )

                n_bg_saved += 1

        except Exception as e:
            n_failed += 1
            logger.warning(f"[Failed] {os.path.basename(fp)}: {e}")

    summary = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "total_files": n_total,
        "bad_parse_or_invalid_target": n_bad_parse,
        "no_group": n_no_group,
        "missed_target_no_samples": n_missed,
        "no_background_group": n_no_background_group,
        "background_saved": n_bg_saved,
        "failed": n_failed,
        "output_mat_mode": cfg.output_mat_mode,
        "mat_key": cfg.mat_key,
        "match_freq_thresh": cfg.match_freq_thresh,
        "match_bandwidth_weight": cfg.match_bandwidth_weight,
        "skip_unmatched": cfg.skip_unmatched,
    }

    summary_path = os.path.join(output_dir, "background_generation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"[Summary] {json.dumps(summary, ensure_ascii=False)}")
    logger.info(f"[Summary] saved to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Background-only group-level mat samples."
    )

    # I/O
    parser.add_argument(
        "--input_dir",
        type=str,
        default='/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/dataSet/generated_dataset/new_dataset/new_dataset_awgn_space/train/1',
        help="Original single-signal mat directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/data/background',
        help="Output directory for generated Background mat files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search mat files.",
    )
    parser.add_argument(
        "--mat_key",
        type=str,
        default="summed_submatrices",
    )

    # Classes
    parser.add_argument(
        "--classes_json",
        type=str,
        default="",
        help=(
            "Optional classes mapping JSON string or JSON file. "
            "Must contain Background and source signal protocols."
        ),
    )

    # Output group matrix
    parser.add_argument(
        "--output_mat_mode",
        type=str,
        default="raw_in_boxes",
        choices=["raw_in_boxes", "mask"],
        help="How to save each background group as mat.",
    )
    parser.add_argument(
        "--mask_value",
        type=float,
        default=1.0,
    )

    # YOLO
    parser.add_argument("--yolo_weights", type=str, default='/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/best.pt')
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--yolo_conf_thres", type=float, default=0.25)
    parser.add_argument("--yolo_iou_thres", type=float, default=0.45)
    parser.add_argument("--yolo_max_det", type=int, default=1000)
    parser.add_argument("--yolo_imgsz_h", type=int, default=640)
    parser.add_argument("--yolo_imgsz_w", type=int, default=640)
    parser.add_argument(
        "--yolo_classes",
        type=str,
        default="",
        help="Optional comma-separated YOLO class ids. Usually empty.",
    )

    # Matching
    parser.add_argument("--match_freq_thresh", type=float, default=10.0)
    parser.add_argument("--match_bandwidth_weight", type=float, default=0.2)
    parser.add_argument(
        "--skip_unmatched",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, reject match when frequency diff > match_freq_thresh.",
    )

    # Optional BBox cache
    parser.add_argument(
        "--bbox_cache_mode",
        type=str,
        default="off",
        choices=["off", "read", "write", "refresh", "readwrite"],
    )
    parser.add_argument("--bbox_cache_path", type=str, default="")
    parser.add_argument("--bbox_cache_dataset_root", type=str, default="")
    parser.add_argument("--bbox_cache_mem_max", type=int, default=0)
    parser.add_argument(
        "--bbox_cache_strict_read",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true and bbox_cache_mode=read, cache miss raises an error.",
    )

    # Preprocessor geometry filters
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--min_width", type=int, default=2)
    parser.add_argument("--min_height", type=int, default=2)
    parser.add_argument("--min_ratio", type=float, default=0.0)
    parser.add_argument("--max_width", type=int, default=0)
    parser.add_argument("--max_height", type=int, default=0)

    # Energy filter
    parser.add_argument("--ring_margin", type=int, default=5)
    parser.add_argument("--min_contrast_z", type=float, default=-1e9)

    # Frequency / clustering
    parser.add_argument("--sampling_rate", type=float, default=122.88e6)
    parser.add_argument("--freq_eps", type=float, default=6.0)
    parser.add_argument("--freq_min_samples", type=int, default=1)
    parser.add_argument("--freq_bw_weight", type=float, default=1.0)

    # Merge / NMS
    parser.add_argument("--nms_thresh", type=float, default=0.5)
    parser.add_argument("--merge_freq_thresh", type=float, default=6.0)
    parser.add_argument("--merge_h_log_thresh", type=float, default=0.8)
    parser.add_argument("--merge_energy_thresh", type=float, default=999.0)

    # Group quality thresholds
    parser.add_argument("--min_group_len", type=int, default=1)
    parser.add_argument("--min_group_time_span_ratio", type=float, default=0.0)

    # Group score
    parser.add_argument("--score_n_boxes_weight", type=float, default=1.0)
    parser.add_argument("--score_time_span_weight", type=float, default=1.0)
    parser.add_argument("--score_contrast_weight", type=float, default=0.0)
    parser.add_argument("--score_contrast_std_weight", type=float, default=0.0)
    parser.add_argument("--score_w_std_weight", type=float, default=0.0)
    parser.add_argument("--score_h_std_weight", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()
    logger = create_logger()

    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"--input_dir is not a directory: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = build_config(args)

    logger.info("Initializing YOLO detector...")
    detector = YoloV5Detector(cfg, device=cfg.device)

    logger.info("Initializing preprocessor...")
    preprocessor = SignalPreprocessor(cfg, logger)

    bbox_cache = None
    if cfg.bbox_cache_mode != "off":
        if BBoxCache is None:
            raise ImportError("util.bboxcache.BBoxCache not available.")
        if not cfg.bbox_cache_path:
            raise ValueError("--bbox_cache_path must be provided when bbox_cache_mode != off")

        dataset_root = cfg.bbox_cache_dataset_root.strip() or None
        bbox_cache = BBoxCache(
            base_dir=cfg.bbox_cache_path,
            dataset_root=dataset_root,
            mode=cfg.bbox_cache_mode,
            mem_max=cfg.bbox_cache_mem_max,
            logger=logger,
        )
        logger.info(
            f"[BBoxCache] enabled: mode={cfg.bbox_cache_mode}, "
            f"path={cfg.bbox_cache_path}, dataset_root={dataset_root}"
        )
    else:
        logger.info("[BBoxCache] disabled")

    process_input_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cfg=cfg,
        logger=logger,
        detector=detector,
        preprocessor=preprocessor,
        bbox_cache=bbox_cache,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()