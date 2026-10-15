"""Optional integration with a real YOLO source checkout; no trained weights required."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("UAV_TEST_REAL_YOLO") == "1",
                     "Set UAV_TEST_REAL_YOLO=1 with an actual YOLO checkout/dependencies")
class ActualYoloInputTests(unittest.TestCase):
    def test_real_training_batch_equals_actual_autoshape_backend_input(self):
        import torch
        torch.set_num_threads(2)
        repo = ROOT/"yolov5"
        sys.path.insert(0, str(repo))
        from util.spectrogram_frontend import FrontendConfig, canonical_amplitude, render_rgb
        from util.yolo_dataset_contract import install_yolo_dataset_contract
        from util.detector import YoloV5Detector
        module = install_yolo_dataset_contract()
        from models.yolo import Model
        from models.common import AutoShape
        cfg = FrontendConfig()
        raw = np.random.default_rng(9).random((512,750), dtype=np.float32)
        raw[200:220,100:300] += 3
        amplitude = canonical_amplitude(raw)
        rgb = render_rgb(amplitude, cfg)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_dir, label_dir = root/"images"/"train", root/"labels"/"train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            path = image_dir/"native.png"
            cv2.imwrite(str(path), rgb[:,:,::-1])
            (label_dir/"native.txt").write_text("0 0.5 0.5 0.25 0.25\n")
            loader, dataset = module.create_dataloader(
                str(image_dir), 640, 1, 32, single_cls=True, hyp={},
                augment=True, rect=True, workers=0, pad=0.5, cache=False,
            )
            self.assertFalse(dataset.augment)
            self.assertFalse(dataset.rect)
            training = next(iter(loader))[0].float().numpy()/255.0
            yaml = repo/"models"/"yolov5n.yaml"
            if not yaml.is_file():
                yaml = repo/"models"/"yolov5s.yaml"
            model = AutoShape(Model(str(yaml), ch=3, nc=1).float().eval(), verbose=False)
            model.conf = 1.0  # This checks tensors; random network detections have no accuracy meaning.
            detector = YoloV5Detector.__new__(YoloV5Detector)
            detector.frontend, detector.input_domain = cfg, "amplitude"
            detector.imgsz, detector.model = 640, model
            captured = []
            hook = model.model.register_forward_pre_hook(
                lambda module, args: captured.append(args[0].detach().cpu().numpy().copy()))
            try:
                a = detector.detect(amplitude)
                loaded = cv2.imread(str(path))[:,:,::-1].copy()
                b = detector.detect_rgb(loaded)
            finally:
                hook.remove()
            self.assertEqual(a, b)
            self.assertEqual(len(captured), 2)
            np.testing.assert_array_equal(training, captured[0])
            np.testing.assert_array_equal(training, captured[1])


if __name__ == "__main__":
    unittest.main()
