from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List

import numpy as np
from scipy.io import loadmat


# Recommended placement:
#   util/preprocess_multi_signal_infer.py
#   util/vis_multi_signal.py
#   scripts/infer_multi_signal_mat.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.detector import YoloV5Detector
from util.preprocess_multi_signal_infer import MultiSignalPreprocessor
from util.vis_multi_signal import draw_multi_signal_groups, save_groups_json


def build_detector_config(args) -> SimpleNamespace:
    # Match the current main-branch detector construction style:
    # detector = YoloV5Detector(config, yolo_device)
    cfg = SimpleNamespace(
        yolo_weights=args.weights,
        yolo_imgsz_h=args.imgsz,
        yolo_imgsz_w=args.imgsz,
        yolo_conf_thres=args.conf_thres,
        yolo_iou_thres=args.iou_thres,
        yolo_max_det=args.max_det,
        yolo_classes=None,
        yolo_half=args.half,
        yolo_warmup=args.warmup,
    )
    return cfg


def load_spectrogram_from_mat(mat_path: str | Path, mat_key: str = "summed_submatrices") -> np.ndarray:
    mat = loadmat(str(mat_path))
    if mat_key not in mat:
        raise KeyError(f"Key '{mat_key}' not found in {mat_path}. Available keys: {list(mat.keys())}")
    spec = mat[mat_key]
    spec = np.asarray(spec)
    if spec.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram from key '{mat_key}', got shape={spec.shape}")
    return spec.astype(np.float32, copy=False)


def iter_mat_files(mat_path: str | None, input_dir: str | None) -> Iterable[Path]:
    if mat_path:
        yield Path(mat_path)
        return
    if not input_dir:
        raise ValueError("Either --mat_path or --input_dir must be provided.")
    p = Path(input_dir)
    if not p.is_dir():
        raise NotADirectoryError(f"input_dir not found: {input_dir}")
    for fp in sorted(p.glob("*.mat")):
        yield fp


def build_preprocessor(args) -> MultiSignalPreprocessor:
    return MultiSignalPreprocessor(
        min_area=args.min_area,
        min_ratio=args.min_ratio,
        min_width=args.min_width,
        min_height=args.min_height,
        max_width=args.max_width,
        max_height=args.max_height,
        exclude_bottom_ratio=args.exclude_bottom_ratio,
        ring_margin=args.ring_margin,
        min_contrast_z=args.min_contrast_z,
        min_integrated_energy=args.min_integrated_energy,
        min_bright_ratio=args.min_bright_ratio,
        bright_z_thresh=args.bright_z_thresh,
        link_dt_max_ratio=args.link_dt_max_ratio,
        link_cost_thresh=args.link_cost_thresh,
        link_time_weight=args.link_time_weight,
        link_freq_weight=args.link_freq_weight,
        link_width_weight=args.link_width_weight,
        link_height_weight=args.link_height_weight,
        link_area_weight=args.link_area_weight,
        link_contrast_weight=args.link_contrast_weight,
        link_bright_weight=args.link_bright_weight,
        min_group_len=args.min_group_len,
        min_group_time_span_ratio=args.min_group_time_span_ratio,
        max_group_shape_std=args.max_group_shape_std,
        max_group_energy_std=args.max_group_energy_std,
        nms_iou_thresh=args.group_nms_iou_thresh,
        nms_cover_small_thresh=args.group_nms_cover_small_thresh,
        score_count_weight=args.score_count_weight,
        score_time_span_weight=args.score_time_span_weight,
        score_shape_consistency_weight=args.score_shape_consistency_weight,
        score_energy_consistency_weight=args.score_energy_consistency_weight,
        score_mean_contrast_weight=args.score_mean_contrast_weight,
        score_freq_span_penalty=args.score_freq_span_penalty,
    )


def summarize_groups(groups: List[dict]) -> str:
    if len(groups) == 0:
        return "No valid signal groups found."
    lines = [f"Detected {len(groups)} signal groups:"]
    for g in groups:
        lines.append(
            f"  - S{g['id']}: {g['group_type']}, n={g['n_boxes']}, "
            f"score={g['score']:.2f}, t_span={g['time_span_ratio']:.3f}, f_span={g['freq_span_ratio']:.3f}, "
            f"mean_wh=({g['mean_w']:.1f}, {g['mean_h']:.1f}), mean_contrast={g['mean_contrast_z']:.2f}"
        )
    return "\n".join(lines)


def process_one_file(mat_fp: Path, detector, preprocessor: MultiSignalPreprocessor, out_dir: Path, args) -> None:
    spec = load_spectrogram_from_mat(mat_fp, mat_key=args.mat_key)
    yolo_boxes = detector.detect(spec)
    groups = preprocessor.select_signal_groups(yolo_boxes, spec)

    stem = mat_fp.stem
    png_path = out_dir / f"{stem}_multi_signal.png"
    json_path = out_dir / f"{stem}_multi_signal.json"

    draw_multi_signal_groups(
        spec,
        groups,
        save_path=png_path,
        show_yolo=args.show_yolo,
        yolo_boxes=yolo_boxes,
        line_thickness=args.line_thickness,
        font_scale=args.font_scale,
        p_low=args.vis_p_low,
        p_high=args.vis_p_high,
        log_gain=args.vis_log_gain,
    )
    save_groups_json(
        groups,
        save_path=json_path,
        extra={
            "file": str(mat_fp),
            "mat_key": args.mat_key,
            "num_yolo_boxes": int(len(yolo_boxes)) if yolo_boxes is not None else 0,
        },
    )

    print(f"\n[{mat_fp.name}]")
    print(summarize_groups(groups))
    print(f"Saved image: {png_path}")
    print(f"Saved json : {json_path}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Inference-only multi-signal grouping for .mat spectrograms.",
    )

    # input/output
    parser.add_argument("--mat_path", type=str, default=None, help="Single .mat file to process.")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing .mat files.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for PNG and JSON results.")
    parser.add_argument("--mat_key", type=str, default="summed_submatrices", help="MAT key containing the 2D spectrogram.")

    # detector
    parser.add_argument("--weights", type=str, required=True, help="YOLOv5 weights path.")
    parser.add_argument("--device", type=str, default="", help="YOLO device string. Empty means auto/default.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf_thres", type=float, default=0.25)
    parser.add_argument("--iou_thres", type=float, default=0.45)
    parser.add_argument("--max_det", type=int, default=1000)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--warmup", action="store_true")

    # low-level box filtering
    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--min_ratio", type=float, default=0.0)
    parser.add_argument("--min_width", type=int, default=2)
    parser.add_argument("--min_height", type=int, default=2)
    parser.add_argument("--max_width", type=int, default=0)
    parser.add_argument("--max_height", type=int, default=0)
    parser.add_argument("--exclude_bottom_ratio", type=float, default=0.0)

    # energy filtering
    parser.add_argument("--ring_margin", type=int, default=5)
    parser.add_argument("--min_contrast_z", type=float, default=0.6)
    parser.add_argument("--min_integrated_energy", type=float, default=8.0)
    parser.add_argument("--min_bright_ratio", type=float, default=0.02)
    parser.add_argument("--bright_z_thresh", type=float, default=1.5)

    # temporal grouping / association
    parser.add_argument("--link_dt_max_ratio", type=float, default=0.22)
    parser.add_argument("--link_cost_thresh", type=float, default=2.3)
    parser.add_argument("--link_time_weight", type=float, default=1.0)
    parser.add_argument("--link_freq_weight", type=float, default=0.35)
    parser.add_argument("--link_width_weight", type=float, default=1.4)
    parser.add_argument("--link_height_weight", type=float, default=1.4)
    parser.add_argument("--link_area_weight", type=float, default=0.8)
    parser.add_argument("--link_contrast_weight", type=float, default=1.0)
    parser.add_argument("--link_bright_weight", type=float, default=0.8)

    # group filtering / scoring
    parser.add_argument("--min_group_len", type=int, default=2)
    parser.add_argument("--min_group_time_span_ratio", type=float, default=0.01)
    parser.add_argument("--max_group_shape_std", type=float, default=0.80)
    parser.add_argument("--max_group_energy_std", type=float, default=1.20)
    parser.add_argument("--group_nms_iou_thresh", type=float, default=0.75)
    parser.add_argument("--group_nms_cover_small_thresh", type=float, default=0.98)
    parser.add_argument("--score_count_weight", type=float, default=1.2)
    parser.add_argument("--score_time_span_weight", type=float, default=1.0)
    parser.add_argument("--score_shape_consistency_weight", type=float, default=1.3)
    parser.add_argument("--score_energy_consistency_weight", type=float, default=1.0)
    parser.add_argument("--score_mean_contrast_weight", type=float, default=0.6)
    parser.add_argument("--score_freq_span_penalty", type=float, default=0.15)

    # visualization
    parser.add_argument("--show_yolo", action="store_true", help="Overlay raw YOLO boxes in gray for reference.")
    parser.add_argument("--line_thickness", type=int, default=2)
    parser.add_argument("--font_scale", type=float, default=0.5)
    parser.add_argument("--vis_p_low", type=float, default=1.0)
    parser.add_argument("--vis_p_high", type=float, default=99.5)
    parser.add_argument("--vis_log_gain", type=float, default=9.0)

    return parser


def main():
    args = get_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    detector_cfg = build_detector_config(args)
    detector = YoloV5Detector(detector_cfg, args.device)
    preprocessor = build_preprocessor(args)

    files = list(iter_mat_files(args.mat_path, args.input_dir))
    if len(files) == 0:
        raise FileNotFoundError("No .mat files found.")

    for fp in files:
        process_one_file(fp, detector, preprocessor, out_dir, args)


if __name__ == "__main__":
    main()
