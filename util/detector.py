from __future__ import annotations

from typing import List, Union
from pathlib import Path
import sys

import numpy as np
import torch
import numpy as np
import cv2


# def _to_uint8_gray_png_style(spec: np.ndarray) -> np.ndarray:
#     """
#     Match the old .mat -> .png generation logic exactly:
#         uint8(matrix / max(matrix) * 255)
#     """
#     if spec.ndim != 2:
#         raise ValueError(f"Expected 2D grayscale spec (H, W), got shape={spec.shape}")

#     if spec.dtype == np.uint8:
#         return spec

#     x = spec.astype(np.float32, copy=False)

#     maxv = float(np.max(x))
#     if (not np.isfinite(maxv)) or maxv <= 0:
#         return np.zeros_like(x, dtype=np.uint8)

#     x = x / maxv * 255.0
#     x = np.clip(x, 0.0, 255.0)
#     return x.astype(np.uint8)



def spec_float32_to_uint8_gray(spec: np.ndarray) -> np.ndarray:
    """
    spec: float32/float64 2D matrix, shape (H, W)
    return: uint8 grayscale image, shape (H, W)

    Match your old .mat -> .png logic:
        uint8(matrix / max(matrix) * 255)
    """
    if not isinstance(spec, np.ndarray):
        spec = np.asarray(spec)

    if spec.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {spec.shape}")

    x = spec.astype(np.float32, copy=False)

    maxv = float(np.max(x))
    if (not np.isfinite(maxv)) or maxv <= 0:
        return np.zeros_like(x, dtype=np.uint8)

    x = x / maxv * 255.0
    x = np.clip(x, 0.0, 255.0)
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
        
        # 关键：把项目根目录放进 sys.path，这样可以按包方式 import yolov5.hubconf
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        # 不再用 torch.hub.load，直接导入 yolov5.hubconf
        import yolov5.hubconf as hubconf

        # 仍然走 high-level custom 接口，和 plot.py 同类
        self.model = hubconf.custom(
            path=str(config.yolo_weights),
            autoshape=True,
            _verbose=False,
            device=self.device_str if self.device_str != "" else None,
        )

        # # Keep same behavior as plot.py
        # sys.path.insert(0, str(yolov5_dir))

        # self.model = torch.hub.load(
        #     str(yolov5_dir),
        #     "custom",
        #     path=str(config.yolo_weights),
        #     source="local",
        # )

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

    @torch.inference_mode()
    def detect(self, spec: Union[np.ndarray, torch.Tensor]) -> List[List[int]]:
        # 1) to numpy (H, W)
        if isinstance(spec, torch.Tensor):
            spec_np = spec.detach().cpu().numpy()
        else:
            spec_np = spec

        if not isinstance(spec_np, np.ndarray):
            raise TypeError(f"spec must be np.ndarray or torch.Tensor, got {type(spec)}")

        if spec_np.ndim != 2:
            raise ValueError(f"Expected spec shape (H, W), got {spec_np.shape}")

        H0, W0 = spec_np.shape

        # 2) Convert mat matrix to the exact same uint8 grayscale style as old PNG generation
        gray_u8 = _to_uint8_gray_png_style(spec_np)

        # 3) Make it RGB HWC uint8, same kind of input as plot.py
        # image_rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
        # image_rgb = np.ascontiguousarray(image_rgb)
        
        gray_u8 = spec_float32_to_uint8_gray(spec_np)
        image_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)

        # 4) High-level inference path, same as plot.py
        results = self.model(image_rgb, size=self.imgsz)

        # 5) Read raw detections
        det = results.xyxy[0].cpu()
        if det is None or len(det) == 0:
            return []

        boxes = det[:, :4].numpy().astype(np.int32)

        # 6) Clip to original image bounds
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