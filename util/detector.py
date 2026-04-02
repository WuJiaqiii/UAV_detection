# util/yolov5_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# --- YOLOv5 official imports ---
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device

import sys

def register_yolov5_legacy_module_aliases():
    """
    Make old YOLOv5 checkpoint module names (models.*, utils.*)
    resolvable after yolov5 is integrated as a subpackage.
    """
    import yolov5.models as y5_models
    import yolov5.models.yolo as y5_models_yolo
    import yolov5.models.common as y5_models_common
    import yolov5.models.experimental as y5_models_experimental

    import yolov5.utils as y5_utils
    import yolov5.utils.general as y5_utils_general
    import yolov5.utils.torch_utils as y5_utils_torch_utils
    import yolov5.utils.autoanchor as y5_utils_autoanchor
    import yolov5.utils.dataloaders as y5_utils_dataloaders
    import yolov5.utils.plots as y5_utils_plots
    import yolov5.utils.loss as y5_utils_loss

    sys.modules.setdefault("models", y5_models)
    sys.modules.setdefault("models.yolo", y5_models_yolo)
    sys.modules.setdefault("models.common", y5_models_common)
    sys.modules.setdefault("models.experimental", y5_models_experimental)

    sys.modules.setdefault("utils", y5_utils)
    sys.modules.setdefault("utils.general", y5_utils_general)
    sys.modules.setdefault("utils.torch_utils", y5_utils_torch_utils)
    sys.modules.setdefault("utils.autoanchor", y5_utils_autoanchor)
    sys.modules.setdefault("utils.dataloaders", y5_utils_dataloaders)
    sys.modules.setdefault("utils.plots", y5_utils_plots)
    sys.modules.setdefault("utils.loss", y5_utils_loss)

# def _to_uint8_gray(spec: np.ndarray) -> np.ndarray:
#     """
#     Robustly convert spectrogram array to uint8 grayscale image.
#     spec: (H,W) float/uint16/uint8
#     """
#     if spec.ndim != 2:
#         raise ValueError(f"Expected 2D grayscale spec (H,W), got shape={spec.shape}")

#     if spec.dtype == np.uint8:
#         return spec

#     x = spec.astype(np.float32, copy=False)

#     # robust percentile scaling to avoid outliers
#     lo = np.percentile(x, 1.0)
#     hi = np.percentile(x, 99.0)
#     if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
#         # fallback: min-max
#         lo = float(np.min(x))
#         hi = float(np.max(x))
#         if hi <= lo:
#             return np.zeros_like(x, dtype=np.uint8)

#     x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)
#     return (x * 255.0).astype(np.uint8)

def _to_uint8_gray(spec: np.ndarray) -> np.ndarray:
    """
    Match the old .mat -> .png generation logic:
        uint8(matrix / max(matrix) * 255)

    Assumption:
        spectrogram values are non-negative.
    """
    if spec.ndim != 2:
        raise ValueError(f"Expected 2D grayscale spec (H,W), got shape={spec.shape}")

    if spec.dtype == np.uint8:
        return spec

    x = spec.astype(np.float32, copy=False)

    maxv = float(np.max(x))
    if (not np.isfinite(maxv)) or maxv <= 0:
        return np.zeros_like(x, dtype=np.uint8)

    x = x / maxv * 255.0
    x = np.clip(x, 0.0, 255.0)
    return x.astype(np.uint8)


def _letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    stride: int = 32,
    auto: bool = True,
    scaleup: bool = True,
) -> np.ndarray:
    """
    Letterbox resize (HWC) -> (HWC) with padding, like YOLOv5.
    new_shape: (h, w)
    """
    # Lazy import cv2 (YOLOv5 uses cv2 inside utils.general)
    import cv2

    assert im.ndim == 3 and im.shape[2] == 3, f"Expected HWC 3ch image, got {im.shape}"
    h0, w0 = im.shape[:2]
    new_h, new_w = int(new_shape[0]), int(new_shape[1])

    # Scale ratio (new / old)
    r = min(new_h / h0, new_w / w0)
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    unpad_w, unpad_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = new_w - unpad_w, new_h - unpad_h

    if auto:
        # make padding a multiple of stride
        dw %= stride
        dh %= stride

    dw /= 2
    dh /= 2

    # resize
    if (w0, h0) != (unpad_w, unpad_h):
        im = cv2.resize(im, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)

    # pad
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

class YoloV5Detector:
    """
    detect(spec) -> boxes_xyxy on original image coordinates.
    - spec: np.ndarray or torch.Tensor, shape (H,W)
    - returns: List[[x1,y1,x2,y2], ...] (ints), clipped to image bounds.
    """
    
    def __init__(self, config, device):
        
        register_yolov5_legacy_module_aliases()
        
        self.config = config
        self.device = device
        self.model = DetectMultiBackend(config.yolo_weights, device=select_device(self.device), dnn=False, data=None, fp16=config.yolo_half)
        self.stride = int(self.model.stride)
        self.pt = bool(self.model.pt)

        # Check image size to be divisible by stride (same as detect.py)
        self.imgsz = check_img_size((config.yolo_imgsz_h, config.yolo_imgsz_w), s=self.stride)  # returns (h,w)

        if config.yolo_warmup:
            bs = 1
            self.model.warmup(imgsz=(bs, 3, int(self.imgsz[0]), int(self.imgsz[1])))

    @torch.inference_mode()
    def detect(self, spec: Union[np.ndarray, torch.Tensor]) -> List[List[int]]:
        # --- 1) to numpy (H,W) ---
        if isinstance(spec, torch.Tensor):
            spec_np = spec.detach().to("cpu").numpy()
        else:
            spec_np = spec
        if not isinstance(spec_np, np.ndarray):
            raise TypeError(f"spec must be np.ndarray or torch.Tensor, got {type(spec)}")

        if spec_np.ndim != 2:
            raise ValueError(f"Expected spec shape (H,W), got {spec_np.shape}")

        H0, W0 = int(spec_np.shape[0]), int(spec_np.shape[1])

        # --- 2) grayscale uint8 -> 3ch ---
        gray_u8 = _to_uint8_gray(spec_np)
        im0 = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)  # HWC, 3ch

        # --- 3) letterbox to imgsz ---
        im = _letterbox(
            im0,
            new_shape=(int(self.imgsz[0]), int(self.imgsz[1])),
            stride=self.stride,
            auto=False,
            scaleup=True,
        )

        # --- 4) to torch (1,3,H,W), normalize to 0-1 ---
        # YOLOv5 expects RGB? For grayscale replicated channels, channel order doesn't matter.
        im_t = torch.from_numpy(im).to(self.model.device)
        im_t = im_t.permute(2, 0, 1).contiguous()  # HWC -> CHW
        im_t = im_t.half() if self.model.fp16 else im_t.float()
        im_t /= 255.0
        if im_t.ndim == 3:
            im_t = im_t.unsqueeze(0)  # add batch

        # --- 5) inference ---
        pred = self.model(im_t, augment=False, visualize=False)

        # --- 6) NMS ---
        pred = non_max_suppression(
            pred,
            self.config.yolo_conf_thres,
            self.config.yolo_iou_thres,
            classes=self.config.yolo_classes,
            agnostic=False,
            max_det=self.config.yolo_max_det,
        )

        det = pred[0]  # batch=1

        if det is None or len(det) == 0:
            return []

        # --- 7) scale boxes back to original image coords ---
        # det[:, :4] are xyxy on letterboxed image; scale_boxes will invert letterbox based on shapes.
        det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], im0.shape).round()

        # --- 8) return xyxy int list, clipped ---
        boxes = det[:, :4].detach().to("cpu").numpy().astype(np.int32)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, W0 - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, W0 - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, H0 - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, H0 - 1)

        # ensure x1<=x2, y1<=y2
        x1 = np.minimum(boxes[:, 0], boxes[:, 2])
        x2 = np.maximum(boxes[:, 0], boxes[:, 2])
        y1 = np.minimum(boxes[:, 1], boxes[:, 3])
        y2 = np.maximum(boxes[:, 1], boxes[:, 3])
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        return boxes.tolist()