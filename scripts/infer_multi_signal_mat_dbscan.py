import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

# Make repo root importable when this script is placed under scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.detector import YoloV5Detector
from util.preprocess_multi_signal_dbscan import MultiSignalDBSCANPreprocessor
from util.vis_multi_signal_dbscan import draw_multi_signal_groups, save_groups_json


def build_detector_args(args):
    # detector in current repo expects a config-like object with yolo_* attributes
    return SimpleNamespace(
        yolo_weights=args.weights,
        yolo_imgsz_h=args.imgsz_h,
        yolo_imgsz_w=args.imgsz_w,
        yolo_conf_thres=args.conf_thres,
        yolo_iou_thres=args.iou_thres,
        yolo_max_det=args.max_det,
        yolo_classes=args.yolo_classes,
        yolo_half=args.half,
        yolo_warmup=args.warmup,
    )


def load_spec_from_mat(mat_path: Path, mat_key: str) -> np.ndarray:
    obj = loadmat(str(mat_path))
    if mat_key not in obj:
        keys = ", ".join(sorted(k for k in obj.keys() if not k.startswith("__")))
        raise KeyError(f"Key '{mat_key}' not found in {mat_path}. Available keys: {keys}")
    spec = obj[mat_key]
    spec = np.asarray(spec)
    if spec.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram in '{mat_key}', got shape={spec.shape}")
    return spec.astype(np.float32, copy=False)


def infer_one(mat_path: Path, out_dir: Path, detector, preprocessor, mat_key: str, show_yolo: bool):
    spec = load_spec_from_mat(mat_path, mat_key=mat_key)
    yolo_boxes = detector.detect(spec)
    groups = preprocessor.select_signal_groups(yolo_boxes, spectrogram=spec)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = mat_path.stem
    png_path = out_dir / f"{stem}_multi_signal.png"
    json_path = out_dir / f"{stem}_multi_signal.json"

    draw_multi_signal_groups(
        spec=spec,
        groups=groups,
        save_path=str(png_path),
        yolo_boxes=yolo_boxes,
        show_yolo=show_yolo,
    )
    save_groups_json(groups=groups, save_path=str(json_path), source_file=str(mat_path))
    return groups, png_path, json_path


def collect_mat_paths(args) -> list[Path]:
    if args.mat_path:
        return [Path(args.mat_path)]
    if not args.input_dir:
        raise ValueError("Provide either --mat_path or --input_dir")
    root = Path(args.input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"input_dir not found: {root}")
    if args.recursive:
        return sorted(root.rglob("*.mat"))
    return sorted(root.glob("*.mat"))


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Inference-only multi-signal .mat pipeline using DBSCAN frequency clustering + top-N groups.",
    )

    src = parser.add_argument_group("Input")
    src.add_argument("--mat_path", type=str, default=None, help="Single .mat file to process.")
    src.add_argument("--input_dir", type=str, default=None, help="Directory containing .mat files.")
    src.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Recursively search input_dir for .mat files.")
    src.add_argument("--out_dir", type=str, required=True, help="Output directory for PNG and JSON results.")
    src.add_argument("--mat_key", type=str, default="summed_submatrices", help="Key inside .mat file containing the 2D spectrogram.")

    yolo = parser.add_argument_group("YOLO detector")
    yolo.add_argument("--weights", type=str, required=True, help="Path to YOLO weights.")
    yolo.add_argument("--device", type=str, default="", help='YOLO device. "" for auto, "cpu", or GPU id like "0".')
    yolo.add_argument("--imgsz_h", type=int, default=640)
    yolo.add_argument("--imgsz_w", type=int, default=640)
    yolo.add_argument("--conf_thres", type=float, default=0.25)
    yolo.add_argument("--iou_thres", type=float, default=0.45)
    yolo.add_argument("--max_det", type=int, default=1000)
    yolo.add_argument("--yolo_classes", type=int, nargs="*", default=None)
    yolo.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    yolo.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)

    phys = parser.add_argument_group("Physical mapping")
    phys.add_argument("--sampling_rate", type=float, default=122.88e6)
    phys.add_argument("--n_fft", type=int, default=512)
    phys.add_argument("--hop_length", type=int, default=int(122.88e6 * 0.05 / 750))

    pre = parser.add_argument_group("Multi-signal preprocess")
    pre.add_argument("--min_area", type=int, default=20)
    pre.add_argument("--min_ratio", type=float, default=0.0)
    pre.add_argument("--min_width", type=int, default=2)
    pre.add_argument("--min_height", type=int, default=2)
    pre.add_argument("--max_width", type=int, default=0)
    pre.add_argument("--max_height", type=int, default=0)
    pre.add_argument("--exclude_bottom_ratio", type=float, default=0.0)

    pre.add_argument("--ring_margin", type=int, default=5)
    pre.add_argument("--min_contrast_z", type=float, default=0.6)
    pre.add_argument("--min_integrated_energy", type=float, default=8.0)
    pre.add_argument("--min_bright_ratio", type=float, default=0.02)
    pre.add_argument("--bright_z_thresh", type=float, default=1.5)

    pre.add_argument("--freq_eps", type=float, default=12.0)
    pre.add_argument("--freq_min_samples", type=int, default=1)

    pre.add_argument("--merge_freq_thresh", type=float, default=10.0)
    pre.add_argument("--merge_w_log_thresh", type=float, default=0.35)
    pre.add_argument("--merge_h_log_thresh", type=float, default=0.35)
    pre.add_argument("--merge_energy_thresh", type=float, default=1.0)
    pre.add_argument("--merge_bright_thresh", type=float, default=0.12)

    pre.add_argument("--min_group_len", type=int, default=2)
    pre.add_argument("--min_group_time_span_ratio", type=float, default=0.01)
    pre.add_argument("--nms_iou_thresh", type=float, default=0.7)
    pre.add_argument("--use_cover_small_rule", action=argparse.BooleanOptionalAction, default=False)
    pre.add_argument("--max_groups", type=int, default=3)

    pre.add_argument("--score_boxes_weight", type=float, default=1.0)
    pre.add_argument("--score_time_span_weight", type=float, default=1.2)
    pre.add_argument("--score_area_weight", type=float, default=20.0)
    pre.add_argument("--score_mean_contrast_weight", type=float, default=0.8)
    pre.add_argument("--score_shape_std_weight", type=float, default=1.0)
    pre.add_argument("--score_energy_std_weight", type=float, default=0.8)

    vis = parser.add_argument_group("Visualization")
    vis.add_argument("--show_yolo", action=argparse.BooleanOptionalAction, default=False, help="Overlay original YOLO boxes in gray for reference.")

    return parser


def main():
    args = get_parser().parse_args()

    mat_paths = collect_mat_paths(args)
    if len(mat_paths) == 0:
        raise FileNotFoundError("No .mat files found.")

    detector_cfg = build_detector_args(args)
    detector = YoloV5Detector(detector_cfg, args.device)

    preprocessor = MultiSignalDBSCANPreprocessor(
        sampling_rate=args.sampling_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
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
        freq_eps=args.freq_eps,
        freq_min_samples=args.freq_min_samples,
        merge_freq_thresh=args.merge_freq_thresh,
        merge_w_log_thresh=args.merge_w_log_thresh,
        merge_h_log_thresh=args.merge_h_log_thresh,
        merge_energy_thresh=args.merge_energy_thresh,
        merge_bright_thresh=args.merge_bright_thresh,
        min_group_len=args.min_group_len,
        min_group_time_span_ratio=args.min_group_time_span_ratio,
        nms_iou_thresh=args.nms_iou_thresh,
        use_cover_small_rule=args.use_cover_small_rule,
        max_groups=args.max_groups,
        score_boxes_weight=args.score_boxes_weight,
        score_time_span_weight=args.score_time_span_weight,
        score_area_weight=args.score_area_weight,
        score_mean_contrast_weight=args.score_mean_contrast_weight,
        score_shape_std_weight=args.score_shape_std_weight,
        score_energy_std_weight=args.score_energy_std_weight,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(mat_paths)} .mat files")
    for idx, mat_path in enumerate(mat_paths, start=1):
        try:
            groups, png_path, json_path = infer_one(
                mat_path=mat_path,
                out_dir=out_dir,
                detector=detector,
                preprocessor=preprocessor,
                mat_key=args.mat_key,
                show_yolo=args.show_yolo,
            )
            print(f"[{idx}/{len(mat_paths)}] {mat_path.name}: groups={len(groups)} -> {png_path.name}, {json_path.name}")
        except Exception as e:
            print(f"[{idx}/{len(mat_paths)}] {mat_path.name}: ERROR: {e}")


if __name__ == "__main__":
    main()
