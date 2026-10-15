"""YOLO inference on exactly the same RGB rasters exported for annotation."""
from pathlib import Path
import numpy as np
import torch

from util.spectrogram_frontend import (
    FrontendConfig, render_tiles, nms_indices,
)


class YoloV5Detector:
    def __init__(self, config, device):
        self.config = config
        self.frontend = FrontendConfig.from_config(config)
        # Dataset adapters explicitly declare the representation reaching this API.
        # Standalone MAT utilities can pass power/amplitude/db/image themselves.
        self.input_domain = getattr(config, "detector_spec_domain", "power")
        repo = Path(__file__).resolve().parents[1] / "yolov5"
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError("Initialize the local yolov5 submodule first")
        import yolov5.hubconf as hubconf
        self.model = hubconf.custom(
            path=str(config.yolo_weights), autoshape=True,
            _verbose=False, device=device or None,
        )
        self.model.eval()
        self.model.conf = float(config.yolo_conf_thres)
        self.model.iou = float(config.yolo_iou_thres)
        self.model.classes = getattr(config, "yolo_classes", None)
        self.model.max_det = int(config.yolo_max_det)
        self.imgsz = self.frontend.image_size

    def prepare_tiles(self, spec):
        if torch.is_tensor(spec):
            spec = spec.detach().cpu().numpy()
        return render_tiles(spec, self.frontend, self.input_domain)

    @torch.inference_mode()
    def detect(self, spec):
        if torch.is_tensor(spec):
            spec = spec.detach().cpu().numpy()
        if np.asarray(spec).ndim != 2:
            raise ValueError("Expected [frequency, time] spectrum")
        boxes, scores, classes = [], [], []
        for tile in self.prepare_tiles(spec):
            # Already square, stride-aligned uint8 RGB. AutoShape adds no resize
            # or padding; it only performs its usual tensor conversion and NMS.
            result = self.model(tile.image, size=self.imgsz)
            det = result.xyxy[0]
            if det is None or len(det) == 0:
                continue
            det = det.detach().cpu().numpy()
            mapped = tile.to_source(det[:, :4])
            valid = (mapped[:, 2] > mapped[:, 0]) & (mapped[:, 3] > mapped[:, 1])
            boxes.extend(mapped[valid])
            scores.extend(det[valid, 4])
            classes.extend(det[valid, 5].astype(int))
        if not boxes:
            return []
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        classes = np.asarray(classes)
        keep = []
        for cls in np.unique(classes):
            indices = np.flatnonzero(classes == cls)
            keep.extend(indices[nms_indices(boxes[indices], scores[indices],
                                            self.frontend.nms_iou)].tolist())
        keep.sort(key=lambda i: -float(scores[i]))
        keep = keep[:int(self.config.yolo_max_det)]
        selected = boxes[keep]
        # xyxy uses exclusive right/bottom edges, consistent with numpy slicing.
        selected[:, :2] = np.floor(selected[:, :2])
        selected[:, 2:] = np.ceil(selected[:, 2:])
        return selected.astype(np.int32).tolist()
