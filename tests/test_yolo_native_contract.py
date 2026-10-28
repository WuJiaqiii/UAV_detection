"""Optional actual YOLO inference input check; no trained weights or training tools."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("UAV_TEST_REAL_YOLO") == "1",
                     "Set UAV_TEST_REAL_YOLO=1 with an actual YOLO checkout/dependencies")
class ActualYoloInputTests(unittest.TestCase):
    def test_native_amplitude_and_rendered_png_reach_the_same_actual_backend_tensor(self):
        import torch
        torch.set_num_threads(2)
        repo = ROOT/"yolov5"
        sys.path.insert(0, str(repo))
        from models.yolo import Model
        from models.common import AutoShape
        from util.detector import YoloV5Detector
        from util.spectrogram_frontend import (
            FrontendConfig, canonical_amplitude, render_rgb, prepare_yolo_image, rgb_to_tensor_array,
        )
        cfg = FrontendConfig()
        raw = np.random.default_rng(9).random((512, 750), dtype=np.float32)
        raw[200:220, 100:300] += 3
        amplitude = canonical_amplitude(raw)
        rgb = render_rgb(amplitude, cfg)
        model_config = repo/"models/yolov5n.yaml"
        if not model_config.is_file():
            model_config = repo/"models/yolov5s.yaml"
        model = AutoShape(Model(str(model_config), ch=3, nc=1).float().eval(), verbose=False)
        model.conf = 1.0
        detector = YoloV5Detector.__new__(YoloV5Detector)
        detector.frontend, detector.input_domain = cfg, "amplitude"
        detector.imgsz, detector.model = 640, model
        captured = []
        hook = model.model.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().cpu().numpy().copy()))
        try:
            first = detector.detect(amplitude)
            second = detector.detect_rgb(rgb)
        finally:
            hook.remove()
        self.assertEqual(first, second)
        self.assertEqual(len(captured), 2)
        expected = rgb_to_tensor_array(prepare_yolo_image(rgb, cfg).image)[None]
        np.testing.assert_array_equal(captured[0], expected)
        np.testing.assert_array_equal(captured[0], captured[1])
