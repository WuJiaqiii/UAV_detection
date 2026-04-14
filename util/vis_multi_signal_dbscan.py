import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 128, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 255, 0),
]


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


def draw_multi_signal_groups(
    spec: np.ndarray,
    groups: List[Dict],
    save_path: str,
    yolo_boxes: Optional[np.ndarray] = None,
    show_yolo: bool = False,
    draw_thickness: int = 2,
    font_scale: float = 0.45,
) -> None:
    img_u8 = spec_to_uint8_vis_log(spec)
    img = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    H, W = img.shape[:2]

    if show_yolo and yolo_boxes is not None and len(yolo_boxes) > 0:
        yolo_boxes = np.asarray(yolo_boxes, dtype=np.int32).reshape(-1, 4)
        for b in yolo_boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(100, 100, 100), thickness=1)

    for idx, g in enumerate(groups):
        color = COLORS[idx % len(COLORS)]
        boxes = np.asarray(g["boxes"], dtype=np.int32).reshape(-1, 4)
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=draw_thickness)

        x_min = int(boxes[:, 0].min())
        y_min = int(boxes[:, 1].min())
        label = (
            f"S{idx+1} {g.get('group_type', 'unknown')} | "
            f"n={g.get('n_boxes', 0)} score={g.get('score', 0.0):.2f} | "
            f"t={g.get('time_span_ratio', 0.0):.2f} f={g.get('freq_span_ratio', 0.0):.2f}"
        )
        text_org = (max(2, x_min), max(14, y_min - 6))
        cv2.putText(img, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, img)


def save_groups_json(
    groups: List[Dict],
    save_path: str,
    source_file: Optional[str] = None,
) -> None:
    payload = {
        "source_file": source_file,
        "num_groups": len(groups),
        "groups": [],
    }
    for idx, g in enumerate(groups, start=1):
        item = {k: v for k, v in g.items() if k != "boxes"}
        item["id"] = idx
        item["boxes"] = np.asarray(g["boxes"], dtype=np.int32).reshape(-1, 4).tolist()
        payload["groups"].append(item)

    save_path = str(save_path)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
