from __future__ import annotations

from typing import List, Union
from pathlib import Path
import sys

import numpy as np
import torch
import cv2

# def spec_float32_to_uint8_gray(spec: np.ndarray) -> np.ndarray:
#     """
#     spec: float32/float64 2D matrix, shape (H, W)
#     return: uint8 grayscale image, shape (H, W)

#     Match your old .mat -> .png logic:
#         uint8(matrix / max(matrix) * 255)
#     """
#     if not isinstance(spec, np.ndarray):
#         spec = np.asarray(spec)

#     if spec.ndim != 2:
#         raise ValueError(f"Expected 2D matrix, got {spec.shape}")

#     x = spec.astype(np.float32, copy=False)

#     maxv = float(np.max(x))
#     if (not np.isfinite(maxv)) or maxv <= 0:
#         return np.zeros_like(x, dtype=np.uint8)

#     x = x / maxv * 255.0
#     x = np.clip(x, 0.0, 255.0)
#     return x.astype(np.uint8)

# def spec_float32_to_uint8_gray(
#     spec: np.ndarray,
#     norm_mode: str = "max",
#     p_low: float = 1.0,
#     p_high: float = 99.5,
# ) -> np.ndarray:
#     """
#     Convert 2D spectrogram matrix to uint8 grayscale image for YOLO.

#     norm_mode:
#         max:
#             Original behavior:
#                 x / max(x) * 255

#         percentile:
#             Percentile clipping:
#                 lo = percentile(x, p_low)
#                 hi = percentile(x, p_high)
#                 x = clip(x, lo, hi)
#                 x = (x - lo) / (hi - lo) * 255
#     """
#     if not isinstance(spec, np.ndarray):
#         spec = np.asarray(spec)

#     if spec.ndim != 2:
#         raise ValueError(f"Expected 2D matrix, got {spec.shape}")

#     x = spec.astype(np.float32, copy=False)

#     finite_mask = np.isfinite(x)
#     if not finite_mask.any():
#         return np.zeros_like(x, dtype=np.uint8)

#     valid = x[finite_mask]
#     norm_mode = str(norm_mode).lower()

#     if norm_mode == "max":
#         maxv = float(np.max(valid))
#         if maxv <= 0:
#             return np.zeros_like(x, dtype=np.uint8)

#         x = x / maxv * 255.0

#     elif norm_mode == "percentile":
#         p_low = float(p_low)
#         p_high = float(p_high)

#         if not (0.0 <= p_low < p_high <= 100.0):
#             raise ValueError(
#                 f"Invalid percentile range: p_low={p_low}, p_high={p_high}. "
#                 "Expected 0 <= p_low < p_high <= 100."
#             )

#         lo = float(np.percentile(valid, p_low))
#         hi = float(np.percentile(valid, p_high))

#         if hi <= lo:
#             return np.zeros_like(x, dtype=np.uint8)

#         x = np.clip(x, lo, hi)
#         x = (x - lo) / (hi - lo) * 255.0

#     else:
#         raise ValueError(
#             f"Unsupported yolo_input_norm={norm_mode}, expected 'max' or 'percentile'."
#         )

#     x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)
#     x = np.clip(x, 0.0, 255.0)

#     return x.astype(np.uint8)

def spec_float32_to_uint8_gray(spec: np.ndarray) -> np.ndarray:
    """
    Temporary YOLO input visualization logic:
        percentile clipping + log enhancement + uint8.

    This is for testing whether YOLO miss-detection is caused by weak signals
    being too dark after the original x / max(x) normalization.
    """
    if not isinstance(spec, np.ndarray):
        spec = np.asarray(spec)

    if spec.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {spec.shape}")

    x = spec.astype(np.float32, copy=False)

    finite_mask = np.isfinite(x)
    if not finite_mask.any():
        return np.zeros_like(x, dtype=np.uint8)

    valid = x[finite_mask]

    # 你可以先用这一组参数测试
    p_low = 1.0
    p_high = 99.5
    log_gain = 9.0

    lo = float(np.percentile(valid, p_low))
    hi = float(np.percentile(valid, p_high))

    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)

    # 1) 分位数裁剪，避免极端亮点压暗弱信号
    x = np.clip(x, lo, hi)

    # 2) 归一化到 0~1
    x = (x - lo) / (hi - lo)

    # 3) log 增强弱信号
    x = np.log1p(log_gain * x) / np.log1p(log_gain)

    # 4) 转 uint8
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x * 255.0, 0.0, 255.0)

    return x.astype(np.uint8)

class YoloV5Detector:
    """
    High-level YOLOv5 detector using the same inference path as plot.py:
        model = torch.hub.load(..., "custom", source="local")
        results = model(image_rgb, size=imgsz)
        det = results.xyxy[0]

    detect(spec) input:
        - spec: np.ndarray or torch.Tensor, shape (H, W)

    returns:
        - List[[x1, y1, x2, y2], ...]
    """

    def __init__(self, config, device):
        self.config = config
        self.device_str = device if device is not None else ""

        # Resolve yolov5 local repo path
        repo_root = Path(__file__).resolve().parents[1]
        yolov5_dir = repo_root / "yolov5"
        if not yolov5_dir.exists():
            raise FileNotFoundError(f"Local yolov5 directory not found: {yolov5_dir}")
        
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        import yolov5.hubconf as hubconf

        self.model = hubconf.custom(
            path=str(config.yolo_weights),
            autoshape=True,
            _verbose=False,
            device=self.device_str if self.device_str != "" else None,
        )

        # device selection, same idea as plot.py
        if self.device_str:
            if str(self.device_str).lower() == "cpu":
                self.device = torch.device("cpu")
            elif str(self.device_str).isdigit():
                self.device = torch.device(f"cuda:{self.device_str}")
            else:
                self.device = torch.device(self.device_str)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        # Same runtime parameters as plot.py
        self.model.conf = float(config.yolo_conf_thres)
        self.model.iou = float(config.yolo_iou_thres)
        self.model.classes = getattr(config, "yolo_classes", None)
        self.model.max_det = int(config.yolo_max_det)

        # plot.py uses one imgsz value; here keep compatibility with current config
        self.imgsz = int(max(config.yolo_imgsz_h, config.yolo_imgsz_w))

        self.input_norm = str(config.yolo_input_norm).lower()
        self.input_p_low = float(config.yolo_input_p_low)
        self.input_p_high = float(config.yolo_input_p_high)

    @torch.inference_mode()
    def detect(self, spec: Union[np.ndarray, torch.Tensor]) -> List[List[int]]:

        if isinstance(spec, torch.Tensor):
            spec_np = spec.detach().cpu().numpy()
        else:
            spec_np = spec

        if not isinstance(spec_np, np.ndarray):
            raise TypeError(f"spec must be np.ndarray or torch.Tensor, got {type(spec)}")

        if spec_np.ndim != 2:
            raise ValueError(f"Expected spec shape (H, W), got {spec_np.shape}")

        H0, W0 = spec_np.shape

        gray_u8 = spec_float32_to_uint8_gray(spec_np)
        # gray_u8 = spec_float32_to_uint8_gray(
        #     spec_np,
        #     norm_mode=self.input_norm,
        #     p_low=self.input_p_low,
        #     p_high=self.input_p_high,
        # )
        image_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)

        results = self.model(image_rgb, size=self.imgsz)

        det = results.xyxy[0].cpu()
        if det is None or len(det) == 0:
            return []

        boxes = det[:, :4].numpy().astype(np.int32)

        boxes[:, 0] = np.clip(boxes[:, 0], 0, W0 - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, W0 - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, H0 - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, H0 - 1)

        # ensure x1 <= x2, y1 <= y2
        x1 = np.minimum(boxes[:, 0], boxes[:, 2])
        x2 = np.maximum(boxes[:, 0], boxes[:, 2])
        y1 = np.minimum(boxes[:, 1], boxes[:, 3])
        y2 = np.maximum(boxes[:, 1], boxes[:, 3])
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        return boxes.tolist()