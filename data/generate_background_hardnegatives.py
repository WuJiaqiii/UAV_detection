#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate background hard-negative .mat samples using the current inference chain:
background image -> YOLO -> preprocess.select_signal_groups(...)

This version adds:
    --erase_class_ids
Only YOLO label boxes whose class id is in erase_class_ids will be removed from
the source image when constructing the background image.

Example:
    --erase_class_ids 0

Warning:
If labels 1/2 are also real signals and you do NOT erase them, they will remain
in the image and may be mined as "background" by mistake. Use this mode mainly
as a debugging / exploratory attempt, and inspect the saved PNGs carefully.
"""

import os
import re
import csv
import json
import random
import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.io import savemat
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.detector import YoloV5Detector
from util.preprocess import SignalPreprocessor


class SimpleLogger:
    def info(self, msg):
        print(msg)

    def warning(self, msg):
        print(f"[WARN] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}")


def read_gray_u8(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def parse_yolo_label_line(line: str):
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO label line: {line!r}")
    class_id = int(float(parts[0]))
    xc, yc, w, h = map(float, parts[1:])
    return class_id, xc, yc, w, h


def yolo_to_xyxy(label_line: str, img_shape):
    class_id, xc, yc, bw, bh = parse_yolo_label_line(label_line)
    H, W = img_shape[:2]
    x1 = int(round((xc - bw / 2.0) * W))
    y1 = int(round((yc - bh / 2.0) * H))
    x2 = int(round((xc + bw / 2.0) * W))
    y2 = int(round((yc + bh / 2.0) * H))

    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(1, min(W, x2))
    y2 = max(1, min(H, y2))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return class_id, x1, y1, x2, y2


def parse_source_meta_from_name(stem: str):
    snr = 0.0
    snrspace = 0.0

    m = re.search(r"-SNR-([+-]?\d+(?:\.\d+)?)", stem)
    if m:
        snr = float(m.group(1))

    m = re.search(r"-SNRSPACE([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", stem, flags=re.IGNORECASE)
    if m:
        snrspace = float(m.group(1))

    return snr, snrspace


def make_background_from_image(img_u8: np.ndarray, valid_boxes, mode="median", noise_std=3.0) -> np.ndarray:
    H, W = img_u8.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    for x1, y1, x2, y2 in valid_boxes:
        mask[y1:y2, x1:x2] = 255

    pixels = img_u8[mask == 0]
    if pixels.size == 0:
        base_value = 0.0
    else:
        if mode == "mean":
            base_value = float(np.mean(pixels))
        elif mode == "black":
            base_value = 0.0
        else:
            base_value = float(np.median(pixels))

    bg = np.full((H, W), base_value, dtype=np.float32)
    if noise_std is not None and noise_std > 0:
        bg += np.random.normal(0, noise_std, bg.shape).astype(np.float32)
    return np.clip(bg, 0, 255).astype(np.uint8)


def spec_to_uint8_vis_log(spec: np.ndarray, p_low=1.0, p_high=99.5, log_gain=9.0) -> np.ndarray:
    x = np.asarray(spec, dtype=np.float32)
    finite_mask = np.isfinite(x)
    if not finite_mask.any():
        return np.zeros_like(x, dtype=np.uint8)

    valid = x[finite_mask]
    lo = float(np.percentile(valid, p_low))
    hi = float(np.percentile(valid, p_high))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    x = np.log1p(log_gain * x) / np.log1p(log_gain)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def draw_boxes_rgb(gray_u8: np.ndarray, boxes, color=(255, 0, 0), thickness=2):
    from PIL import Image, ImageDraw
    img = Image.fromarray(gray_u8).convert("RGB")
    draw = ImageDraw.Draw(img)
    arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4) if boxes is not None else np.zeros((0, 4), dtype=np.int32)
    for b in arr:
        x1, y1, x2, y2 = [int(v) for v in b]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=int(thickness))
    return img


def draw_groups_rgb(gray_u8: np.ndarray, groups, thickness=2):
    from PIL import Image, ImageDraw
    img = Image.fromarray(gray_u8).convert("RGB")
    draw = ImageDraw.Draw(img)

    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 255, 0),
    ]

    for gi, g in enumerate(groups):
        boxes = np.asarray(g.get("boxes", []), dtype=np.int32).reshape(-1, 4)
        if len(boxes) == 0:
            continue
        color = colors[gi % len(colors)]
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=int(thickness))

        x1, y1, x2, y2 = [int(v) for v in boxes[0]]
        label = f"G{gi}"
        if "score" in g:
            label += f" s={float(g['score']):.2f}"
        if "center_freq" in g:
            label += f" f={float(g['center_freq']):.1f}"
        if "bandwidth" in g:
            label += f" bw={float(g['bandwidth']):.1f}"
        draw.text((x1, max(0, y1 - 12)), label, fill=color)

    return img


class BackgroundHardNegativeGenerator:
    def __init__(self, args):
        self.args = args
        self.logger = SimpleLogger()

        self.H = int(args.out_h)
        self.W = int(args.out_w)
        self.erase_class_ids = set(args.erase_class_ids or [])

        self.out_dir = Path(args.out_dir)
        self.mat_dir = self.out_dir / "mats"
        self.png_dir = self.out_dir / "vis"
        self.meta_dir = self.out_dir / "meta"
        self.mat_dir.mkdir(parents=True, exist_ok=True)
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = SimpleNamespace(
            run_mode="infer",
            train_signal_mode="single",
            sampling_rate=args.sampling_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            mat_key=args.mat_key,

            yolo_weights=args.yolo_weights,
            yolo_device=args.yolo_device,
            yolo_imgsz_h=args.yolo_imgsz_h,
            yolo_imgsz_w=args.yolo_imgsz_w,
            yolo_conf_thres=args.yolo_conf_thres,
            yolo_iou_thres=args.yolo_iou_thres,
            yolo_max_det=args.yolo_max_det,
            yolo_classes=args.yolo_classes,
            yolo_half=args.yolo_half,
            yolo_warmup=args.yolo_warmup,

            min_area=args.min_area,
            min_ratio=args.min_ratio,
            min_width=args.min_width,
            min_height=args.min_height,
            max_width=args.max_width,
            max_height=args.max_height,
            ring_margin=args.ring_margin,
            min_contrast_z=args.min_contrast_z,
            freq_eps=args.freq_eps,
            freq_min_samples=args.freq_min_samples,
            merge_freq_thresh=args.merge_freq_thresh,
            merge_w_log_thresh=args.merge_w_log_thresh,
            merge_h_log_thresh=args.merge_h_log_thresh,
            merge_energy_thresh=args.merge_energy_thresh,
            min_group_len=args.min_group_len,
            min_group_time_span_ratio=args.min_group_time_span_ratio,
            score_n_boxes_weight=args.score_n_boxes_weight,
            score_time_span_weight=args.score_time_span_weight,
            score_contrast_weight=args.score_contrast_weight,
            score_w_std_weight=args.score_w_std_weight,
            score_h_std_weight=args.score_h_std_weight,
            score_contrast_std_weight=args.score_contrast_std_weight,
            nms_thresh=args.nms_thresh,

            match_freq_thresh=args.match_freq_thresh,
            match_bw_thresh=args.match_bw_thresh,
            match_bw_weight=args.match_bw_weight,
            match_size_penalty=args.match_size_penalty,
            skip_unmatched=args.skip_unmatched,
            match_use_bandwidth=args.match_use_bandwidth,

            cnn_input_mode=args.cnn_input_mode,
            box_draw_thickness=args.box_draw_thickness,
            box_draw_value=args.box_draw_value,
            mask_img_size=args.mask_img_size,
        )

        self.detector = YoloV5Detector(self.cfg, args.yolo_device)
        self.preprocessor = SignalPreprocessor(self.cfg, self.logger)
        self.rows = []

    def build_background_sources(self):
        img_dir = Path(self.args.img_dir)
        label_dir = Path(self.args.label_dir)

        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image dir not found: {img_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Label dir not found: {label_dir}")

        img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}])

        samples = []
        skipped_no_target_class = 0

        for img_path in tqdm(img_paths, desc="Building background sources"):
            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            img = read_gray_u8(str(img_path))
            with open(label_path, "r", encoding="utf-8") as f:
                raw_lines = [line.strip() for line in f if line.strip()]
            if not raw_lines:
                continue

            valid_boxes = []
            for line in raw_lines:
                try:
                    class_id, x1, y1, x2, y2 = yolo_to_xyxy(line, img.shape)
                except Exception:
                    continue

                if self.erase_class_ids and class_id not in self.erase_class_ids:
                    continue

                valid_boxes.append((x1, y1, x2, y2))

            if not valid_boxes:
                skipped_no_target_class += 1
                continue

            bg = make_background_from_image(
                img_u8=img,
                valid_boxes=valid_boxes,
                mode=self.args.background_fill_mode,
                noise_std=self.args.background_noise_std,
            )

            if bg.shape[:2] != (self.H, self.W):
                bg = cv2.resize(bg, (self.W, self.H), interpolation=cv2.INTER_AREA)

            snr, snrspace = parse_source_meta_from_name(img_path.stem)
            samples.append({
                "source_img": str(img_path),
                "source_label": str(label_path),
                "stem": img_path.stem,
                "background": bg,
                "snr": snr,
                "snrspace": snrspace,
            })

        if len(samples) == 0:
            raise RuntimeError("No usable background sources were built.")

        self.logger.info(f"Built {len(samples)} background sources.")
        self.logger.info(f"Skipped {skipped_no_target_class} images because they contained no erase_class_ids targets.")
        return samples

    def _freq_to_name_fields(self, group):
        center = float(group.get("center_freq", 0.0))
        bandwidth = max(float(group.get("bandwidth", 0.0)), 0.0)
        upper = center + bandwidth / 2.0
        return center, upper

    def _save_one_sample(self, bg_u8, src_info, group, yolo_boxes, groups, sample_idx):
        center, upper = self._freq_to_name_fields(group)
        snr = src_info["snr"]
        snrspace = src_info["snrspace"]

        fname = (
            f"Background-[0,{center:.1f},1000,{upper:.1f}]"
            f"-SNR-{snr:g}-SNRSPACE{snrspace:.6e}-Figure-{sample_idx}.mat"
        )
        mat_path = self.mat_dir / fname

        savemat(str(mat_path), {
            "summed_submatrices": np.asarray(bg_u8, dtype=np.uint8)
        })

        vis_base = self.png_dir / mat_path.stem
        gray_vis = spec_to_uint8_vis_log(bg_u8)

        cv2.imwrite(str(vis_base.with_name(vis_base.name + "_input.png")), gray_vis)

        img = draw_boxes_rgb(gray_vis, yolo_boxes, color=(255, 0, 0), thickness=2)
        img.save(str(vis_base.with_name(vis_base.name + "_yolo.png")))

        img = draw_groups_rgb(gray_vis, groups, thickness=2)
        img.save(str(vis_base.with_name(vis_base.name + "_groups.png")))

        selected_boxes = np.asarray(group.get("boxes", []), dtype=np.int32).reshape(-1, 4)
        img = draw_boxes_rgb(gray_vis, selected_boxes, color=(0, 255, 0), thickness=2)
        img.save(str(vis_base.with_name(vis_base.name + "_selected.png")))

        self.rows.append({
            "mat_file": fname,
            "source_img": src_info["source_img"],
            "source_label": src_info["source_label"],
            "source_stem": src_info["stem"],
            "snr": snr,
            "snrspace": snrspace,
            "center_freq_mhz": float(group.get("center_freq", 0.0)),
            "bandwidth_mhz": float(group.get("bandwidth", 0.0)),
            "upper_freq_mhz": upper,
            "score": float(group.get("score", 0.0)),
            "n_boxes": int(group.get("n_boxes", len(selected_boxes))),
            "num_yolo_boxes": int(len(np.asarray(yolo_boxes).reshape(-1, 4))) if len(yolo_boxes) > 0 else 0,
            "num_final_groups": int(len(groups)),
            "keep_mode": self.args.keep_mode,
            "erase_class_ids": ",".join(map(str, sorted(self.erase_class_ids))),
        })

    def run(self):
        sources = self.build_background_sources()
        saved = 0
        skipped_no_yolo = 0
        skipped_no_group = 0
        skipped_multi_group = 0

        for src in tqdm(sources, desc="Mining background hard negatives"):
            bg_u8 = np.asarray(src["background"], dtype=np.uint8)
            spec = bg_u8.astype(np.float32)

            yolo_boxes = self.detector.detect(spec)
            yolo_boxes = np.asarray(yolo_boxes, dtype=np.int32).reshape(-1, 4) if len(yolo_boxes) > 0 else np.zeros((0, 4), dtype=np.int32)

            if len(yolo_boxes) == 0:
                skipped_no_yolo += 1
                continue

            groups = self.preprocessor.select_signal_groups(yolo_boxes, spectrogram=spec)
            if groups is None or len(groups) == 0:
                skipped_no_group += 1
                continue

            if self.args.keep_mode == "exact_one":
                if len(groups) != 1:
                    skipped_multi_group += 1
                    continue
                selected = groups[0]
            elif self.args.keep_mode == "top1":
                groups_sorted = sorted(groups, key=lambda g: float(g.get("score", 0.0)), reverse=True)
                selected = groups_sorted[0]
            else:
                raise ValueError(f"Unsupported keep_mode={self.args.keep_mode}")

            saved += 1
            self._save_one_sample(bg_u8, src, selected, yolo_boxes, groups, saved)

            if self.args.max_samples > 0 and saved >= self.args.max_samples:
                break

        csv_path = self.meta_dir / "background_samples.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if len(self.rows) > 0:
                writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.rows)

        summary = {
            "num_saved": saved,
            "skipped_no_yolo": skipped_no_yolo,
            "skipped_no_group": skipped_no_group,
            "skipped_multi_group": skipped_multi_group,
            "erase_class_ids": sorted(list(self.erase_class_ids)),
            "keep_mode": self.args.keep_mode,
            "img_dir": self.args.img_dir,
            "label_dir": self.args.label_dir,
            "out_dir": self.args.out_dir,
        }
        with open(self.meta_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Saved {saved} background .mat samples.")
        self.logger.info(f"Summary written to: {self.meta_dir / 'summary.json'}")
        self.logger.info(f"CSV written to: {csv_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate background hard-negative .mat samples aligned with current main.py arguments."
    )

    parser.add_argument("--img_dir", type=str, default='/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/dataset/images', help="Annotated image directory (PNG/JPG).")
    parser.add_argument("--label_dir", type=str, default="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/dataset/labels", help="YOLO label directory corresponding to img_dir.")
    parser.add_argument("--out_dir", type=str, default="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/data/background", help="Output directory for mined background samples.")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--keep_mode", type=str, default="top1", choices=["exact_one", "top1"])
    parser.add_argument("--out_h", type=int, default=512)
    parser.add_argument("--out_w", type=int, default=750)

    parser.add_argument("--erase_class_ids", type=int, nargs="*", default=[0],
                        help="Only remove these YOLO label classes when building background images. Example: --erase_class_ids 0")

    parser.add_argument("--background_fill_mode", type=str, default="median", choices=["median", "mean", "black"])
    parser.add_argument("--background_noise_std", type=float, default=3.0)

    parser.add_argument("--sampling_rate", type=float, default=122.88e6)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=int(122.88e6 * 0.05 / 750))
    parser.add_argument("--mat_key", type=str, default="summed_submatrices")

    parser.add_argument("--yolo_weights", type=str, default="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/best.pt")
    parser.add_argument("--yolo_device", type=str, default="")
    parser.add_argument("--yolo_imgsz_h", type=int, default=640)
    parser.add_argument("--yolo_imgsz_w", type=int, default=640)
    parser.add_argument("--yolo_conf_thres", type=float, default=0.40)
    parser.add_argument("--yolo_iou_thres", type=float, default=0.10)
    parser.add_argument("--yolo_max_det", type=int, default=1000)
    parser.add_argument("--yolo_classes", type=int, nargs="*", default=None)
    parser.add_argument("--yolo_half", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--yolo_warmup", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--min_area", type=int, default=20)
    parser.add_argument("--min_ratio", type=float, default=0.0)
    parser.add_argument("--min_width", type=int, default=2)
    parser.add_argument("--min_height", type=int, default=2)
    parser.add_argument("--max_width", type=int, default=0)
    parser.add_argument("--max_height", type=int, default=0)
    parser.add_argument("--ring_margin", type=int, default=5)
    parser.add_argument("--min_contrast_z", type=float, default=0.5)
    parser.add_argument("--freq_eps", type=float, default=5.0)
    parser.add_argument("--freq_min_samples", type=int, default=1)
    parser.add_argument("--merge_freq_thresh", type=float, default=10.0)
    parser.add_argument("--merge_w_log_thresh", type=float, default=0.35)
    parser.add_argument("--merge_h_log_thresh", type=float, default=0.35)
    parser.add_argument("--merge_energy_thresh", type=float, default=1.0)
    parser.add_argument("--min_group_len", type=int, default=2)
    parser.add_argument("--min_group_time_span_ratio", type=float, default=0.50)
    parser.add_argument("--score_n_boxes_weight", type=float, default=0.50)
    parser.add_argument("--score_time_span_weight", type=float, default=2.00)
    parser.add_argument("--score_contrast_weight", type=float, default=1.0)
    parser.add_argument("--score_w_std_weight", type=float, default=1.0)
    parser.add_argument("--score_h_std_weight", type=float, default=10.0)
    parser.add_argument("--score_contrast_std_weight", type=float, default=3.0)
    parser.add_argument("--nms_thresh", type=float, default=0.2)

    parser.add_argument("--match_freq_thresh", type=float, default=30.0)
    parser.add_argument("--match_bw_thresh", type=float, default=20.0)
    parser.add_argument("--match_bw_weight", type=float, default=1.0)
    parser.add_argument("--match_size_penalty", type=float, default=1.0)
    parser.add_argument("--skip_unmatched", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--match_use_bandwidth", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--cnn_input_mode", type=str, default="mask", choices=["mask", "raw", "raw_with_boxes", "raw_in_boxes"])
    parser.add_argument("--box_draw_thickness", type=int, default=2)
    parser.add_argument("--box_draw_value", type=int, default=255)
    parser.add_argument("--mask_img_size", type=int, default=224)

    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    gen = BackgroundHardNegativeGenerator(args)
    gen.run()


if __name__ == "__main__":
    main()
