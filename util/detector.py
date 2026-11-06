"""Native 512x750 rasters share preprocessing with exported annotation PNGs."""
from pathlib import Path
import sys
import numpy as np
import torch
from util.spectrogram_frontend import FrontendConfig, render_rgb, prepare_yolo_image


def load_yolo_model(weights, device=""):
    repo = Path(__file__).resolve().parents[1] / "yolov5"
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError("Initialize the actual local yolov5 submodule first")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import yolov5.hubconf as hubconf
    from models import experimental
    from unittest.mock import patch
    from util.checkpoint import load_checkpoint_object
    # YOLO checkpoints can also contain pathlib objects from the training OS.
    # Scope the trusted reader to model construction; never replace pathlib globally.
    with patch.object(experimental, "torch_load", load_checkpoint_object):
        return hubconf.custom(path=str(weights), autoshape=True, _verbose=False, device=device or None)


class YoloV5Detector:
    def __init__(self, config, device):
        self.config = config
        self.frontend = FrontendConfig.from_config(config)
        self.input_domain = getattr(config, "detector_spec_domain", "amplitude")
        self.model = load_yolo_model(config.yolo_weights, device)
        self.model.eval()
        self.model.conf = float(config.yolo_conf_thres)
        self.model.iou = float(config.yolo_iou_thres)
        self.model.classes = getattr(config, "yolo_classes", None)
        self.model.max_det = int(config.yolo_max_det)
        self.imgsz = self.frontend.image_size
        self.last_detections = []

    def prepare_rgb(self, spec):
        if torch.is_tensor(spec):
            spec = spec.detach().cpu().numpy()
        return render_rgb(spec, self.frontend, self.input_domain)

    @torch.inference_mode()
    def detect_rgb(self, rgb):
        prepared = prepare_yolo_image(rgb, self.frontend)
        # AutoShape sees an already prepared square, so cannot introduce another
        # resize. Annotation PNGs themselves remain native 512x750.
        result = self.model(prepared.image, size=self.imgsz)
        det = result.xyxy[0]
        self.last_detections = []
        if det is None or len(det) == 0:
            return []
        native=det.detach().cpu().numpy()
        boxes = prepared.to_source(native[:, :4])
        keep=(boxes[:,2]>boxes[:,0]) & (boxes[:,3]>boxes[:,1])
        boxes=boxes[keep]
        confidences=native[keep,4]
        # Stabilize exact integer round trips against float32 interpolation error.
        boxes = np.round(boxes, decimals=4)
        boxes[:, :2] = np.floor(boxes[:, :2])
        boxes[:, 2:] = np.ceil(boxes[:, 2:])
        output=boxes.astype(np.int32).tolist()
        self.last_detections=[{'box':b,'confidence':float(p)} for b,p in zip(output,confidences)]
        return output

    def detect(self, spec):
        return self.detect_rgb(self.prepare_rgb(spec))
