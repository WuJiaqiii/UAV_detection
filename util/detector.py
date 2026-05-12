from __future__ import annotations

from typing import List, Union
from pathlib import Path
import sys

import numpy as np
import torch
import cv2

from util.utils import spectrogram_to_yolo_uint8

def spec_float32_to_uint8_gray(spec: np.ndarray, config) -> np.ndarray:
    return spectrogram_to_yolo_uint8(
        data=spec,
        db_min=float(config.yolo_db_min),
        db_max=float(config.yolo_db_max),
        eps=float(config.yolo_db_eps),
    )

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

        gray_u8 = spectrogram_to_yolo_uint8(data=spec_np, db_min=float(self.config.yolo_db_min), db_max=float(self.config.yolo_db_max), eps=float(self.config.yolo_db_eps))

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