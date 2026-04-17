#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.detector import YoloV5Detector
from util.preprocess import SignalPreprocessor
from model.resnet import MaskImageClassifier
from util.config import Config

try:
    from util.utils import create_logger
except Exception:
    create_logger = None


# 这里是原始全类别映射。
# 如果你推理时传入了 exclude_classes，脚本会自动重新映射。
BASE_CLASS_MAP: Dict[str, int] = {
    "FPV1": 0,
    "Lightbridge1": 1,
    "Ocusync_mini1": 2,
    "Ocusync21": 3,
    "Ocusync31": 4,
    "Ocusync41": 5,
    "Skylink11": 6,
    "Skylink21": 7,
}


def build_class_map(exclude_classes: List[str]) -> Dict[str, int]:
    exclude_set = set(exclude_classes or [])
    kept = [k for k in BASE_CLASS_MAP.keys() if k not in exclude_set]
    return {name: i for i, name in enumerate(kept)}


def parse_class_from_filename(fp: str, class_map: Dict[str, int]) -> Tuple[str, int]:
    """
    单信号命名示例:
      Ocusync21-[0,-6.7,1000,18]-SNR-17-SNRSPACE...-Figure-10.mat
    """
    name = Path(fp).stem
    m = re.match(r"(?P<cls>[A-Za-z0-9_]+)-\[[^\]]+\]", name)
    if m is None:
        raise ValueError(f"Cannot parse class name from filename: {fp}")
    cls_name = m.group("cls")
    if cls_name not in class_map:
        raise KeyError(f"Class '{cls_name}' is not in active class_map")
    return cls_name, class_map[cls_name]


def spec_to_uint8(spec: np.ndarray) -> np.ndarray:
    x = np.asarray(spec)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape={x.shape}")
    if x.dtype == np.uint8:
        return x.copy()

    x = x.astype(np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi > lo:
        x = (x - lo) / (hi - lo)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def spec_to_uint8_vis_log(
    spec: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.5,
    log_gain: float = 9.0,
) -> np.ndarray:
    x = np.asarray(spec, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.uint8)

    valid = x[finite]
    lo = float(np.percentile(valid, p_low))
    hi = float(np.percentile(valid, p_high))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    x = np.log1p(log_gain * x) / np.log1p(log_gain)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def boxes_to_white_mask(
    image_shape: Tuple[int, int],
    boxes,
    fill_value: int = 255,
) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    if boxes is None:
        return mask

    arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
    for b in arr:
        x1, y1, x2, y2 = [int(v) for v in b]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = int(fill_value)
    return mask


def draw_boxes_on_gray(
    gray_u8: np.ndarray,
    boxes,
    value: int = 255,
    thickness: int = 2,
) -> np.ndarray:
    img = gray_u8.copy()
    if boxes is None:
        return img

    h, w = img.shape[:2]
    arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
    for b in arr:
        x1, y1, x2, y2 = [int(v) for v in b]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), color=int(value), thickness=int(thickness))
    return img


def gray_to_tensor(img_u8: np.ndarray, out_size: int) -> torch.Tensor:
    x = img_u8.astype(np.float32) / 255.0
    x = cv2.resize(x, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    x = np.expand_dims(x, axis=0)  # [1, H, W]
    return torch.from_numpy(x).float()


def build_cnn_input(
    spec: np.ndarray,
    final_boxes,
    cnn_input_mode: str,
    mask_img_size: int,
    box_draw_thickness: int = 2,
    box_draw_value: int = 255,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Returns:
      tensor: [1, H, W] float32
      vis_u8: uint8 image actually fed to classifier before resize
    """
    mode = str(cnn_input_mode).lower()
    raw_u8 = spec_to_uint8(spec)

    if mode == "mask":
        vis_u8 = boxes_to_white_mask(spec.shape, final_boxes, fill_value=255)

    elif mode == "raw":
        vis_u8 = raw_u8

    elif mode == "raw_with_boxes":
        vis_u8 = draw_boxes_on_gray(
            raw_u8,
            final_boxes,
            value=box_draw_value,
            thickness=box_draw_thickness,
        )

    elif mode == "raw_in_boxes":
        mask = boxes_to_white_mask(spec.shape, final_boxes, fill_value=255)
        vis_u8 = np.zeros_like(raw_u8, dtype=np.uint8)
        keep = mask > 0
        vis_u8[keep] = raw_u8[keep]

    else:
        raise ValueError(f"Unsupported cnn_input_mode={cnn_input_mode}")

    tensor = gray_to_tensor(vis_u8, mask_img_size)
    return tensor, vis_u8


def save_boxes_image(
    spec: np.ndarray,
    boxes,
    save_path: str,
    color=(255, 0, 0),
    thickness: int = 2,
):
    from PIL import Image, ImageDraw

    bg = spec_to_uint8_vis_log(spec)
    img = Image.fromarray(bg).convert("RGB")
    draw = ImageDraw.Draw(img)

    if boxes is not None:
        arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        for b in arr:
            x1, y1, x2, y2 = [int(v) for v in b]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

    img.save(save_path)


def load_classifier_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        # 兼容 PyTorch 2.6+，并允许加载包含 Config 等对象的可信 checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load classifier checkpoint: {ckpt_path}\n"
            f"Original error: {e}"
        ) from e

    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "classifier", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                model.load_state_dict(ckpt[key], strict=False)
                return
        try:
            model.load_state_dict(ckpt, strict=False)
            return
        except Exception:
            pass

    raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    out_dir: Path,
    normalize: bool = False,
):
    if len(y_true) == 0:
        return

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.save(out_dir / "confusion_matrix.npy", cm)

    cm_plot = cm.astype(np.float64)
    if normalize:
        row_sum = cm_plot.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm_plot = cm_plot / row_sum

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm_plot.max() / 2.0 if cm_plot.size > 0 else 0.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            txt = f"{cm_plot[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            plt.text(
                j,
                i,
                txt,
                horizontalalignment="center",
                color="white" if cm_plot[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    out_name = "confusion_matrix_norm.png" if normalize else "confusion_matrix.png"
    plt.savefig(out_dir / out_name, dpi=200)
    plt.close()


def make_minimal_config(args) -> Config:
    ns = SimpleNamespace(
        dataset_path=str(args.input_dir or args.mat_path or ""),
        input_type="mat",
        val_ratio=0.2,
        num_workers=0,
        batch_size=1,
        sample_ratio=1.0,
        exclude_classes=args.exclude_classes,
        sampling_rate=args.sampling_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        pre_min_area=args.pre_min_area,
        pre_min_ratio=args.pre_min_ratio,
        pre_freq_eps=args.pre_freq_eps,
        pre_freq_min_samples=args.pre_freq_min_samples,
        pre_nms_iou_thresh=args.pre_nms_iou_thresh,
        yolo_weights=args.yolo_weights,
        yolo_device=args.device,
        yolo_imgsz_h=args.yolo_imgsz_h,
        yolo_imgsz_w=args.yolo_imgsz_w,
        yolo_conf_thres=args.yolo_conf_thres,
        yolo_iou_thres=args.yolo_iou_thres,
        yolo_max_det=args.yolo_max_det,
        yolo_classes=args.yolo_classes,
        yolo_half=args.yolo_half,
        yolo_warmup=args.yolo_warmup,
        backbone=args.backbone,
        mask_img_size=args.mask_img_size,
        mask_source="final",
        mask_fill_value=255,
        mask_in_chans=1,
        mask_pretrained=False,
        freeze_backbone=False,
        cnn_dropout=0.0,
        cnn_input_mode=args.cnn_input_mode,
        box_draw_thickness=args.box_draw_thickness,
        box_draw_value=args.box_draw_value,
        checkpoint_path=None,
        bbox_cache_mode="off",
        bbox_cache_path=None,
        precompute_boxes=False,
        lr=1e-4,
        weight_decay=1e-2,
        early_stop_patience=10,
        cosine_annealing_T0=10,
        cosine_annealing_mult=2,
        use_data_parallel=False,
        use_amp_autocast=False,
        save_detect_vis_once=False,
        detect_vis_num_samples=0,
        epochs=1,
        save_interval=1,
    )
    cfg = Config(ns)
    cfg.make_dir()
    return cfg


def build_logger(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "infer_log.log"
    if create_logger is not None:
        return create_logger(str(log_file))

    import logging
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(str(log_file))
        ch = logging.StreamHandler()
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


def collect_mat_files(
    mat_path: Optional[str],
    input_dir: Optional[str],
    recursive: bool,
) -> List[Path]:
    if mat_path:
        return [Path(mat_path)]
    if not input_dir:
        raise ValueError("Either --mat_path or --input_dir must be provided.")

    root = Path(input_dir)
    if recursive:
        return sorted(root.rglob("*.mat"))
    return sorted(root.glob("*.mat"))


@torch.no_grad()
def run_one_file(
    fp: Path,
    detector: YoloV5Detector,
    preprocessor: SignalPreprocessor,
    classifier: MaskImageClassifier,
    device: torch.device,
    args,
    out_dir: Path,
    class_map: Dict[str, int],
    inv_class_map: Dict[int, str],
) -> Optional[Dict]:
    # 解析 GT；若该类别被 exclude，则直接跳过
    try:
        cls_name, gt_label = parse_class_from_filename(str(fp), class_map)
    except KeyError:
        return None

    mat = loadmat(str(fp), variable_names=["summed_submatrices"])
    if "summed_submatrices" not in mat:
        raise KeyError(f"'summed_submatrices' not found in {fp}")

    spec = np.asarray(mat["summed_submatrices"], dtype=np.float32)
    if spec.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram in {fp}, got shape={spec.shape}")

    sample_dir = out_dir / fp.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    # input vis
    cv2.imwrite(str(sample_dir / "00_input.png"), spec_to_uint8_vis_log(spec))

    # YOLO
    yolo_boxes = detector.detect(spec)
    yolo_boxes = (
        np.asarray(yolo_boxes, dtype=np.int32).reshape(-1, 4)
        if len(yolo_boxes) > 0 else np.zeros((0, 4), dtype=np.int32)
    )
    save_boxes_image(spec, yolo_boxes, str(sample_dir / "01_yolo.png"), color=(255, 0, 0))

    # preprocess
    final_boxes = preprocessor.select_main_boxes(yolo_boxes, spectrogram=spec)
    final_boxes = (
        np.asarray(final_boxes, dtype=np.int32).reshape(-1, 4)
        if len(final_boxes) > 0 else np.zeros((0, 4), dtype=np.int32)
    )
    save_boxes_image(spec, final_boxes, str(sample_dir / "02_final.png"), color=(0, 255, 0))

    # classifier input
    x, vis_u8 = build_cnn_input(
        spec=spec,
        final_boxes=final_boxes,
        cnn_input_mode=args.cnn_input_mode,
        mask_img_size=args.mask_img_size,
        box_draw_thickness=args.box_draw_thickness,
        box_draw_value=args.box_draw_value,
    )
    cv2.imwrite(str(sample_dir / "03_classifier_input.png"), vis_u8)

    # classify
    x = x.unsqueeze(0).to(device)  # [1,1,H,W]
    logits = classifier(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_label = int(np.argmax(probs))
    pred_name = inv_class_map.get(pred_label, str(pred_label))
    ok = int(pred_label == gt_label)

    result = {
        "file": str(fp),
        "gt_label": gt_label,
        "gt_name": cls_name,
        "pred_label": pred_label,
        "pred_name": pred_name,
        "correct": ok,
        "num_yolo_boxes": int(len(yolo_boxes)),
        "num_final_boxes": int(len(final_boxes)),
        "probs": probs.tolist(),
    }

    with open(sample_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Inference / evaluation script for one-signal-resnet branch."
    )

    src = parser.add_argument_group("Input Source")
    src.add_argument("--mat_path", type=str, default=None, help="Single .mat file")
    src.add_argument("--input_dir", type=str, default=None, help="Directory containing .mat files")
    src.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    src.add_argument("--out_dir", type=str, required=True, help="Directory to save outputs")

    data = parser.add_argument_group("Data")
    data.add_argument("--exclude_classes", type=str, nargs="*", default=[])

    stft = parser.add_argument_group("STFT / Physical Mapping")
    stft.add_argument("--sampling_rate", type=float, default=122.88e6)
    stft.add_argument("--n_fft", type=int, default=512)
    stft.add_argument("--hop_length", type=int, default=int(122.88e6 * 0.05 / 750))

    pre = parser.add_argument_group("Preprocessor")
    pre.add_argument("--pre_min_area", type=int, default=20)
    pre.add_argument("--pre_min_ratio", type=float, default=0.0)
    pre.add_argument("--pre_freq_eps", type=int, default=5)
    pre.add_argument("--pre_freq_min_samples", type=int, default=2)
    pre.add_argument("--pre_nms_iou_thresh", type=float, default=0.5)

    yolo = parser.add_argument_group("YOLO Detector")
    yolo.add_argument("--yolo_weights", type=str, required=True)
    yolo.add_argument("--device", type=str, default="")
    yolo.add_argument("--yolo_imgsz_h", type=int, default=640)
    yolo.add_argument("--yolo_imgsz_w", type=int, default=640)
    yolo.add_argument("--yolo_conf_thres", type=float, default=0.85)
    yolo.add_argument("--yolo_iou_thres", type=float, default=0.05)
    yolo.add_argument("--yolo_max_det", type=int, default=1000)
    yolo.add_argument("--yolo_classes", type=int, nargs="*", default=None)
    yolo.add_argument("--yolo_half", action=argparse.BooleanOptionalAction, default=False)
    yolo.add_argument("--yolo_warmup", action=argparse.BooleanOptionalAction, default=True)

    cnn = parser.add_argument_group("CNN Classifier")
    cnn.add_argument("--classifier_checkpoint", type=str, required=True)
    cnn.add_argument("--backbone", type=str, default="resnet18",
                     choices=["resnet18", "resnet34", "mobilenet_v3_small"])
    cnn.add_argument("--num_classes", type=int, default=None,
                     help="If omitted, will be inferred from active class_map")
    cnn.add_argument("--mask_img_size", type=int, default=224)
    cnn.add_argument("--cnn_input_mode", type=str, default="mask",
                     choices=["mask", "raw", "raw_with_boxes", "raw_in_boxes"])
    cnn.add_argument("--box_draw_thickness", type=int, default=2)
    cnn.add_argument("--box_draw_value", type=int, default=255)

    return parser


def main():
    args = get_parser().parse_args()

    class_map = build_class_map(args.exclude_classes)
    inv_class_map = {v: k for k, v in class_map.items()}
    class_names = [inv_class_map[i] for i in range(len(inv_class_map))]

    if args.num_classes is None:
        args.num_classes = len(class_map)

    mat_files = collect_mat_files(args.mat_path, args.input_dir, args.recursive)
    if len(mat_files) == 0:
        raise FileNotFoundError("No .mat files found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(out_dir)

    cfg = make_minimal_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    detector = YoloV5Detector(cfg, args.device)
    preprocessor = SignalPreprocessor(
        sampling_rate=cfg.sampling_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        min_area=cfg.pre_min_area,
        min_ratio=cfg.pre_min_ratio,
        freq_eps=cfg.pre_freq_eps,
        freq_min_samples=cfg.pre_freq_min_samples,
        nms_iou_thresh=cfg.pre_nms_iou_thresh,
    )
    classifier = MaskImageClassifier(
        backbone=args.backbone,
        num_classes=args.num_classes,
        in_chans=1,
        pretrained=False,
        dropout=0.0,
        freeze_backbone=False,
    ).to(device)
    classifier.eval()
    load_classifier_checkpoint(classifier, args.classifier_checkpoint, device)

    results = []
    logger.info(f"Found {len(mat_files)} mat files")
    logger.info(f"Active classes: {class_names}")
    logger.info(f"Excluded classes: {args.exclude_classes}")

    for i, fp in enumerate(mat_files, 1):
        logger.info(f"[{i}/{len(mat_files)}] Processing: {fp.name}")
        try:
            res = run_one_file(
                fp=fp,
                detector=detector,
                preprocessor=preprocessor,
                classifier=classifier,
                device=device,
                args=args,
                out_dir=out_dir,
                class_map=class_map,
                inv_class_map=inv_class_map,
            )
            if res is not None:
                results.append(res)
        except Exception as e:
            logger.exception(f"Failed on {fp}: {e}")
            results.append({
                "file": str(fp),
                "error": str(e),
                "correct": 0,
            })

    valid = [r for r in results if "gt_label" in r and "pred_label" in r]
    accuracy = (sum(int(r["correct"]) for r in valid) / len(valid)) if valid else 0.0

    y_true = [int(r["gt_label"]) for r in valid]
    y_pred = [int(r["pred_label"]) for r in valid]

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_dir=out_dir,
        normalize=False,
    )
    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        out_dir=out_dir,
        normalize=True,
    )

    summary = {
        "num_files_total": len(mat_files),
        "num_valid": len(valid),
        "accuracy": accuracy,
        "cnn_input_mode": args.cnn_input_mode,
        "exclude_classes": args.exclude_classes,
        "active_classes": class_names,
        "results": results,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary.csv", "w", encoding="utf-8") as f:
        f.write("file,gt_name,pred_name,correct,num_yolo_boxes,num_final_boxes,error\n")
        for r in results:
            f.write(
                f'{r.get("file","")},'
                f'{r.get("gt_name","")},'
                f'{r.get("pred_name","")},'
                f'{r.get("correct","")},'
                f'{r.get("num_yolo_boxes","")},'
                f'{r.get("num_final_boxes","")},'
                f'{r.get("error","")}\n'
            )

    print(f"Done. Accuracy = {accuracy:.4f} ({sum(int(r['correct']) for r in valid)}/{len(valid) if valid else 0})")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()