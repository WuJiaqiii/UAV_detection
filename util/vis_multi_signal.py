from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


PALETTE: Sequence[Tuple[int, int, int]] = [
    (255, 80, 80),
    (80, 220, 120),
    (80, 180, 255),
    (255, 200, 80),
    (220, 80, 255),
    (80, 255, 240),
    (255, 120, 180),
    (180, 255, 80),
]


def spec_to_uint8_vis_log(
    spec: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.5,
    log_gain: float = 9.0,
) -> np.ndarray:
    x = np.asarray(spec, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape={x.shape}")
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


def _group_label_text(group: Dict[str, Any]) -> str:
    return (
        f"S{group['id']} {group['group_type']} | "
        f"n={group['n_boxes']} score={group['score']:.2f} | "
        f"t={group['time_span_ratio']:.3f} f={group['freq_span_ratio']:.3f}"
    )


def draw_multi_signal_groups(
    spec: np.ndarray,
    groups: List[Dict[str, Any]],
    save_path: str | Path,
    show_yolo: bool = False,
    yolo_boxes: np.ndarray | List[List[int]] | None = None,
    line_thickness: int = 2,
    font_scale: float = 0.5,
    p_low: float = 1.0,
    p_high: float = 99.5,
    log_gain: float = 9.0,
) -> Path:
    gray = spec_to_uint8_vis_log(spec, p_low=p_low, p_high=p_high, log_gain=log_gain)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    H, W = img.shape[:2]

    if show_yolo and yolo_boxes is not None and len(yolo_boxes) > 0:
        arr = np.asarray(yolo_boxes, dtype=np.int32).reshape(-1, 4)
        for b in arr:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), (120, 120, 120), 1)

    for idx, group in enumerate(groups):
        color = PALETTE[idx % len(PALETTE)]
        boxes = np.asarray(group["boxes"], dtype=np.int32).reshape(-1, 4)
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

        # label near leftmost/topmost box in this group
        x_left = int(np.min(boxes[:, 0]))
        y_top = int(np.min(boxes[:, 1]))
        label = _group_label_text(group)
        text_pos = (max(2, x_left), max(18, y_top - 6))
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        rx1, ry1 = text_pos[0], max(0, text_pos[1] - th - baseline - 4)
        rx2, ry2 = min(W - 1, text_pos[0] + tw + 6), min(H - 1, text_pos[1] + baseline + 2)
        cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, -1)
        cv2.putText(img, label, (text_pos[0] + 3, text_pos[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), img)
    return save_path


def groups_to_jsonable(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    out_groups = []
    for g in groups:
        out_groups.append({
            "id": int(g["id"]),
            "group_type": str(g["group_type"]),
            "score": float(g["score"]),
            "n_boxes": int(g["n_boxes"]),
            "time_span_px": float(g["time_span_px"]),
            "freq_span_px": float(g["freq_span_px"]),
            "time_span_ratio": float(g["time_span_ratio"]),
            "freq_span_ratio": float(g["freq_span_ratio"]),
            "mean_w": float(g["mean_w"]),
            "mean_h": float(g["mean_h"]),
            "std_log_w": float(g["std_log_w"]),
            "std_log_h": float(g["std_log_h"]),
            "mean_contrast_z": float(g["mean_contrast_z"]),
            "std_contrast_z": float(g["std_contrast_z"]),
            "mean_bright_ratio": float(g["mean_bright_ratio"]),
            "std_bright_ratio": float(g["std_bright_ratio"]),
            "boxes": np.asarray(g["boxes"], dtype=np.int32).reshape(-1, 4).tolist(),
        })
    return {"num_groups": len(out_groups), "groups": out_groups}


def save_groups_json(groups: List[Dict[str, Any]], save_path: str | Path, extra: Dict[str, Any] | None = None) -> Path:
    data = groups_to_jsonable(groups)
    if extra:
        data.update(extra)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return save_path
