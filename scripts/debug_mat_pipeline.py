
import os
import sys
import json
import math
import argparse
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from scipy.io import loadmat

# Make project root importable
THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.detector import YoloV5Detector
from util.preprocess import SignalPreprocessor
from util.boxmask import boxes_to_white_mask
from model.resnet import MaskImageClassifier
from util.checkpoint import load_checkpoint


# ----------------------------- visualization helpers -----------------------------
def spec_to_uint8_vis_log(spec: np.ndarray, p_low: float = 1.0, p_high: float = 99.5, log_gain: float = 9.0) -> np.ndarray:
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


def draw_boxes_on_gray(spec_u8: np.ndarray, boxes, color=(0, 255, 0), thickness=2, labels=None) -> np.ndarray:
    if spec_u8.ndim == 2:
        canvas = cv2.cvtColor(spec_u8, cv2.COLOR_GRAY2BGR)
    else:
        canvas = spec_u8.copy()
    if boxes is None:
        return canvas
    arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4) if len(boxes) > 0 else np.zeros((0, 4), dtype=np.int32)
    H, W = canvas.shape[:2]
    for i, b in enumerate(arr.tolist()):
        x1, y1, x2, y2 = [int(v) for v in b]
        x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color=color, thickness=thickness)
        if labels is not None and i < len(labels):
            cv2.putText(canvas, str(labels[i]), (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return canvas


def draw_clustered_boxes(spec_u8: np.ndarray, clusters, title_prefix='C') -> np.ndarray:
    if spec_u8.ndim == 2:
        canvas = cv2.cvtColor(spec_u8, cv2.COLOR_GRAY2BGR)
    else:
        canvas = spec_u8.copy()
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (0, 128, 255), (255, 128, 0),
        (180, 80, 255), (0, 180, 120)
    ]
    for gi, g in enumerate(clusters):
        color = palette[gi % len(palette)]
        boxes = np.asarray(g.get('boxes', []), dtype=np.int32).reshape(-1, 4) if len(g.get('boxes', [])) > 0 else np.zeros((0,4), dtype=np.int32)
        for bi, b in enumerate(boxes.tolist()):
            x1, y1, x2, y2 = b
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if boxes.shape[0] > 0:
            x1 = int(np.min(boxes[:,0])); y1 = int(np.min(boxes[:,1]))
            text = f"{title_prefix}{gi+1} n={g.get('n_boxes', boxes.shape[0])} s={g.get('score', 0):.2f}"
            cv2.putText(canvas, text, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


# ----------------------------- model helpers -----------------------------
def make_detector(args):
    cfg = SimpleNamespace(
        yolo_weights=args.yolo_weights,
        yolo_conf_thres=args.yolo_conf_thres,
        yolo_iou_thres=args.yolo_iou_thres,
        yolo_classes=None,
        yolo_max_det=args.yolo_max_det,
        yolo_imgsz_h=args.yolo_imgsz,
        yolo_imgsz_w=args.yolo_imgsz,
    )
    return YoloV5Detector(cfg, args.device)


def make_preprocessor(args):
    return SignalPreprocessor(
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
        nms_iou_thresh=args.nms_iou_thresh,
        merge_freq_thresh=args.merge_freq_thresh,
        merge_w_log_thresh=args.merge_w_log_thresh,
        merge_h_log_thresh=args.merge_h_log_thresh,
        merge_energy_thresh=args.merge_energy_thresh,
        merge_bright_thresh=args.merge_bright_thresh,
        min_group_len=args.min_group_len,
        min_group_time_span_ratio=args.min_group_time_span_ratio,
        min_group_contrast=args.min_group_contrast,
        min_group_bright=args.min_group_bright,
        max_group_w_std=args.max_group_w_std,
        max_group_h_std=args.max_group_h_std,
        max_group_contrast_std=args.max_group_contrast_std,
        score_abs_thresh=args.score_abs_thresh,
        score_rel_thresh=args.score_rel_thresh,
        score_n_boxes_weight=args.score_n_boxes_weight,
        score_time_span_weight=args.score_time_span_weight,
        score_contrast_weight=args.score_contrast_weight,
        score_bright_weight=args.score_bright_weight,
        score_w_std_weight=args.score_w_std_weight,
        score_h_std_weight=args.score_h_std_weight,
        score_contrast_std_weight=args.score_contrast_std_weight,
    )


def load_classifier_if_needed(args):
    if not args.classifier_checkpoint:
        return None, None
    model = MaskImageClassifier(
        backbone=args.backbone,
        num_classes=args.num_classes,
        in_chans=1,
        pretrained=False,
        dropout=args.cnn_dropout,
        freeze_backbone=False,
    )
    load_checkpoint({"model": model}, path=args.classifier_checkpoint, device="cpu", logger=None)
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() and torch.cuda.is_available() else (args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")))
    model.to(device)
    model.eval()

    if args.class_names and os.path.isfile(args.class_names):
        with open(args.class_names, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = [str(i) for i in range(args.num_classes)]
    return model, class_names


def make_input_tensor(spec, group_boxes, mode: str, img_size: int, draw_value: int = 255, draw_thickness: int = 2):
    mode = str(mode).lower()
    raw = spec_to_uint8_vis_log(spec)
    if mode == 'mask':
        img = boxes_to_white_mask(spec.shape, group_boxes, fill_value=255, mode='fill')
    elif mode == 'raw':
        img = raw
    elif mode == 'raw_in_boxes':
        if group_boxes is None or len(group_boxes) == 0:
            img = np.zeros_like(raw)
        else:
            mask = boxes_to_white_mask(spec.shape, group_boxes, fill_value=255, mode='fill')
            img = np.zeros_like(raw)
            keep = mask > 0
            img[keep] = raw[keep]
    elif mode == 'raw_with_boxes':
        img = draw_boxes_on_gray(raw, group_boxes, color=(draw_value, draw_value, draw_value), thickness=draw_thickness)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f'Unsupported cnn_input_mode={mode}')

    x = img.astype(np.float32) / 255.0
    x = cv2.resize(x, (img_size, img_size), interpolation=cv2.INTER_NEAREST if mode == 'mask' else cv2.INTER_LINEAR)
    x = np.expand_dims(x, axis=0)
    return torch.from_numpy(x).float().unsqueeze(0)


# ----------------------------- step-by-step debug -----------------------------
def run_preprocess_debug(spec, yolo_boxes, pre: SignalPreprocessor):
    H, W = spec.shape[:2]
    out = {}

    # step 1: basic filter
    basic = pre._basic_filter(yolo_boxes, img_h=H, img_w=W)
    out['basic_boxes'] = basic.astype(np.int32).tolist() if basic.size > 0 else []

    # step 2: energy filter (with fallback semantics recorded)
    used_energy_fallback = False
    if basic.size > 0:
        e_boxes, e_stats = pre._filter_boxes_by_energy(basic, spec)
        if e_boxes.size > 0:
            boxes_after_energy, stats_after_energy = e_boxes, e_stats
        else:
            boxes_after_energy = basic.astype(np.int32, copy=False)
            stats_after_energy = [pre._box_energy_stats(b, spec) for b in boxes_after_energy]
            used_energy_fallback = True
    else:
        boxes_after_energy = np.zeros((0, 4), dtype=np.int32)
        stats_after_energy = []
    out['energy_boxes'] = boxes_after_energy.astype(np.int32).tolist() if boxes_after_energy.size > 0 else []
    out['energy_fallback'] = bool(used_energy_fallback)

    # step 3: raw clusters by frequency
    raw_groups = []
    if boxes_after_energy.size == 1:
        g = pre._build_group_stats(boxes_after_energy, stats_after_energy, spec.shape)
        if g is not None:
            raw_groups = [g]
    elif boxes_after_energy.size > 1:
        clusters = pre._cluster_boxes_by_frequency(boxes_after_energy)
        out['cluster_indices'] = [idx.tolist() for idx in clusters]
        for idx in clusters:
            c_boxes = boxes_after_energy[idx]
            c_stats = [stats_after_energy[int(i)] for i in idx.tolist()] if stats_after_energy is not None else None
            time_centers = (c_boxes[:, 0] + c_boxes[:, 2]) / 2.0
            order = np.argsort(time_centers)
            c_boxes = c_boxes[order]
            if c_stats is not None:
                c_stats = [c_stats[int(i)] for i in order.tolist()]
            g = pre._build_group_stats(c_boxes, c_stats, spec.shape)
            if g is not None:
                raw_groups.append(g)
    else:
        out['cluster_indices'] = []

    out['raw_groups'] = [serialize_group(g) for g in raw_groups]

    # step 4: merge groups
    merged_groups = pre._merge_groups(raw_groups, spec.shape) if len(raw_groups) > 0 else []
    out['merged_groups'] = [serialize_group(g) for g in merged_groups]

    # step 5: nms inside each group + recompute stats
    cleaned = []
    for g in merged_groups:
        boxes_list = [list(map(int, b)) for b in g['boxes'].tolist()]
        boxes_list = pre._nms(boxes_list)
        if len(boxes_list) == 0:
            continue
        boxes_arr = np.asarray(boxes_list, dtype=np.int32).reshape(-1, 4)
        time_centers = (boxes_arr[:, 0] + boxes_arr[:, 2]) / 2.0
        order = np.argsort(time_centers)
        boxes_arr = boxes_arr[order]
        stat_map = {}
        if g.get('stats') is not None:
            for b, st in zip(g['boxes'].tolist(), g['stats']):
                stat_map[tuple(map(int, b))] = st
            stats_kept = [stat_map.get(tuple(map(int, b)), None) for b in boxes_arr.tolist()]
        else:
            stats_kept = None
        gg = pre._build_group_stats(boxes_arr, stats_kept, spec.shape)
        if gg is not None:
            cleaned.append(gg)
    out['cleaned_groups'] = [serialize_group(g) for g in cleaned]

    # step 6: thresholding
    final_groups = []
    if cleaned:
        best_score = max(g['score'] for g in cleaned)
        for g in cleaned:
            passes = pre._group_passes_thresholds(g, best_score)
            g_copy = dict(g)
            g_copy['passes_thresholds'] = bool(passes)
            g_copy['group_type'] = pre._infer_group_type(g)
            if passes:
                final_groups.append(g_copy)
        final_groups = sorted(final_groups, key=lambda x: x['score'], reverse=True)
        out['best_score'] = float(best_score)
    else:
        out['best_score'] = None
    out['final_groups'] = [serialize_group(g) for g in final_groups]
    return out


def serialize_group(g):
    if g is None:
        return None
    out = {}
    for k, v in g.items():
        if k == 'stats':
            if v is None:
                out[k] = None
            else:
                out[k] = [serialize_stat(x) for x in v]
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def serialize_stat(st):
    if st is None:
        return None
    out = {}
    for k, v in st.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def classify_groups(spec, final_groups, clf, class_names, args):
    if clf is None or not final_groups:
        return []
    device = next(clf.parameters()).device
    results = []
    for gi, g in enumerate(final_groups):
        boxes = g['boxes']
        x = make_input_tensor(spec, boxes, args.cnn_input_mode, args.mask_img_size, args.box_draw_value, args.box_draw_thickness)
        x = x.to(device)
        with torch.no_grad():
            logits = clf(x)
            prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred = int(np.argmax(prob))
        results.append({
            'group_idx': gi,
            'pred_label': pred,
            'pred_class_name': class_names[pred] if pred < len(class_names) else str(pred),
            'confidence': float(prob[pred]),
            'probs': prob.tolist(),
        })
    return results


def annotate_final_predictions(spec_u8, final_groups, cls_results):
    if spec_u8.ndim == 2:
        canvas = cv2.cvtColor(spec_u8, cv2.COLOR_GRAY2BGR)
    else:
        canvas = spec_u8.copy()
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (0, 128, 255), (255, 128, 0),
    ]
    result_map = {r['group_idx']: r for r in cls_results}
    for gi, g in enumerate(final_groups):
        color = palette[gi % len(palette)]
        boxes = np.asarray(g.get('boxes', []), dtype=np.int32).reshape(-1, 4)
        for b in boxes.tolist():
            cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), color, 2)
        if boxes.shape[0] > 0:
            x1 = int(np.min(boxes[:, 0])); y1 = int(np.min(boxes[:, 1]))
            cls = result_map.get(gi)
            if cls is None:
                text = f"G{gi+1} {g.get('group_type','?')} score={g.get('score',0):.2f}"
            else:
                text = f"G{gi+1} {cls['pred_class_name']} conf={cls['confidence']:.2f} | {g.get('group_type','?')}"
            cv2.putText(canvas, text, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def process_one_file(mat_path: Path, args, detector, preprocessor, classifier, class_names):
    mat = loadmat(str(mat_path), variable_names=[args.mat_key])
    if args.mat_key not in mat:
        raise KeyError(f"key '{args.mat_key}' not found in {mat_path}")
    spec = np.asarray(mat[args.mat_key])
    if spec.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got {spec.shape} in {mat_path}")
    spec = np.asarray(spec, dtype=np.float32)

    stem = mat_path.stem
    out_dir = Path(args.out_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    vis = spec_to_uint8_vis_log(spec, p_low=args.vis_p_low, p_high=args.vis_p_high, log_gain=args.vis_log_gain)
    save_image(out_dir / f"{stem}_00_input.png", vis)

    # detector
    yolo_boxes = detector.detect(spec)
    save_image(out_dir / f"{stem}_01_yolo.png", draw_boxes_on_gray(vis, yolo_boxes, color=(0, 255, 0), thickness=2))

    # preprocess step-by-step
    dbg = run_preprocess_debug(spec, yolo_boxes, preprocessor)

    save_image(out_dir / f"{stem}_02_basic_filter.png", draw_boxes_on_gray(vis, dbg['basic_boxes'], color=(255, 255, 0), thickness=2))
    save_image(out_dir / f"{stem}_03_energy_filter.png", draw_boxes_on_gray(vis, dbg['energy_boxes'], color=(255, 0, 255), thickness=2))

    raw_groups_img = draw_clustered_boxes(vis, dbg['raw_groups'], title_prefix='C')
    save_image(out_dir / f"{stem}_04_raw_clusters.png", raw_groups_img)

    merged_groups_img = draw_clustered_boxes(vis, dbg['merged_groups'], title_prefix='M')
    save_image(out_dir / f"{stem}_05_merged_groups.png", merged_groups_img)

    cleaned_groups_img = draw_clustered_boxes(vis, dbg['cleaned_groups'], title_prefix='N')
    save_image(out_dir / f"{stem}_06_cleaned_groups.png", cleaned_groups_img)

    final_groups = dbg['final_groups']
    final_groups_img = draw_clustered_boxes(vis, final_groups, title_prefix='G')
    save_image(out_dir / f"{stem}_07_final_groups.png", final_groups_img)

    cls_results = classify_groups(spec, final_groups, classifier, class_names, args)
    pred_img = annotate_final_predictions(vis, final_groups, cls_results)
    save_image(out_dir / f"{stem}_08_final_predictions.png", pred_img)

    # save per-group cropped/constructed classifier input images
    if classifier is not None:
        for gi, g in enumerate(final_groups):
            x = make_input_tensor(spec, g['boxes'], args.cnn_input_mode, args.mask_img_size, args.box_draw_value, args.box_draw_thickness)
            img = (x[0,0].numpy() * 255.0).astype(np.uint8)
            save_image(out_dir / f"{stem}_group_{gi+1:02d}_{args.cnn_input_mode}.png", img)

    payload = {
        'file': str(mat_path),
        'yolo_boxes': yolo_boxes,
        'debug': dbg,
        'classification': cls_results,
    }
    with open(out_dir / f"{stem}_debug.json", 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def collect_mat_files(args):
    if args.mat_path:
        return [Path(args.mat_path)]
    root = Path(args.input_dir)
    if args.recursive:
        return sorted(root.rglob('*.mat'))
    return sorted(root.glob('*.mat'))


def build_parser():
    p = argparse.ArgumentParser(description='Standalone debug script for current detector + preprocess + classifier on raw .mat files.')
    p.add_argument('--mat_path', type=str, default=None)
    p.add_argument('--input_dir', type=str, default=None)
    p.add_argument('--recursive', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--out_dir', type=str, required=True)
    p.add_argument('--mat_key', type=str, default='summed_submatrices')

    # detector
    p.add_argument('--yolo_weights', type=str, required=True)
    p.add_argument('--device', type=str, default='')
    p.add_argument('--yolo_conf_thres', type=float, default=0.85)
    p.add_argument('--yolo_iou_thres', type=float, default=0.05)
    p.add_argument('--yolo_max_det', type=int, default=1000)
    p.add_argument('--yolo_imgsz', type=int, default=640)

    # preprocess params (keep aligned with current main branch / uploaded preprocess)
    p.add_argument('--sampling_rate', type=float, default=122.88e6)
    p.add_argument('--n_fft', type=int, default=512)
    p.add_argument('--hop_length', type=int, default=int(122.88e6 * 0.05 / 750))

    p.add_argument('--min_area', type=int, default=20)
    p.add_argument('--min_ratio', type=float, default=0.0)
    p.add_argument('--min_width', type=int, default=2)
    p.add_argument('--min_height', type=int, default=2)
    p.add_argument('--max_width', type=int, default=0)
    p.add_argument('--max_height', type=int, default=0)
    p.add_argument('--exclude_bottom_ratio', type=float, default=0.0)

    p.add_argument('--ring_margin', type=int, default=5)
    p.add_argument('--min_contrast_z', type=float, default=0.6)
    p.add_argument('--min_integrated_energy', type=float, default=8.0)
    p.add_argument('--min_bright_ratio', type=float, default=0.02)
    p.add_argument('--bright_z_thresh', type=float, default=1.5)

    p.add_argument('--freq_eps', type=float, default=12.0)
    p.add_argument('--freq_min_samples', type=int, default=1)
    p.add_argument('--nms_iou_thresh', type=float, default=0.6)

    p.add_argument('--merge_freq_thresh', type=float, default=10.0)
    p.add_argument('--merge_w_log_thresh', type=float, default=0.35)
    p.add_argument('--merge_h_log_thresh', type=float, default=0.35)
    p.add_argument('--merge_energy_thresh', type=float, default=1.0)
    p.add_argument('--merge_bright_thresh', type=float, default=0.12)

    p.add_argument('--min_group_len', type=int, default=2)
    p.add_argument('--min_group_time_span_ratio', type=float, default=0.01)
    p.add_argument('--min_group_contrast', type=float, default=0.0)
    p.add_argument('--min_group_bright', type=float, default=0.0)
    p.add_argument('--max_group_w_std', type=float, default=0.60)
    p.add_argument('--max_group_h_std', type=float, default=0.60)
    p.add_argument('--max_group_contrast_std', type=float, default=2.50)
    p.add_argument('--score_abs_thresh', type=float, default=0.0)
    p.add_argument('--score_rel_thresh', type=float, default=0.25)

    p.add_argument('--score_n_boxes_weight', type=float, default=0.80)
    p.add_argument('--score_time_span_weight', type=float, default=2.00)
    p.add_argument('--score_contrast_weight', type=float, default=0.60)
    p.add_argument('--score_bright_weight', type=float, default=1.20)
    p.add_argument('--score_w_std_weight', type=float, default=0.50)
    p.add_argument('--score_h_std_weight', type=float, default=0.50)
    p.add_argument('--score_contrast_std_weight', type=float, default=0.25)

    # classifier (optional)
    p.add_argument('--classifier_checkpoint', type=str, default=None)
    p.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'mobilenet_v3_small'])
    p.add_argument('--num_classes', type=int, default=8)
    p.add_argument('--cnn_dropout', type=float, default=0.0)
    p.add_argument('--cnn_input_mode', type=str, default='mask', choices=['mask', 'raw', 'raw_with_boxes', 'raw_in_boxes'])
    p.add_argument('--mask_img_size', type=int, default=224)
    p.add_argument('--box_draw_thickness', type=int, default=2)
    p.add_argument('--box_draw_value', type=int, default=255)
    p.add_argument('--class_names', type=str, default=None, help='Optional text file, one class name per line.')

    # visualization
    p.add_argument('--vis_p_low', type=float, default=1.0)
    p.add_argument('--vis_p_high', type=float, default=99.5)
    p.add_argument('--vis_log_gain', type=float, default=9.0)
    return p


def main():
    args = build_parser().parse_args()
    if not args.mat_path and not args.input_dir:
        raise ValueError('Please provide --mat_path or --input_dir')
    files = collect_mat_files(args)
    if not files:
        raise FileNotFoundError('No .mat files found')

    detector = make_detector(args)
    preprocessor = make_preprocessor(args)
    classifier, class_names = load_classifier_if_needed(args)

    for fp in files:
        print(f'[INFO] Processing {fp}')
        process_one_file(fp, args, detector, preprocessor, classifier, class_names)


if __name__ == '__main__':
    main()
