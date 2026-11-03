import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from util.checkpoint import load_checkpoint_object
from util.detector import load_yolo_model


class DetectorPortabilityTests(unittest.TestCase):
    def run_fake(self, fail=False):
        package = ModuleType("yolov5")
        hub = ModuleType("yolov5.hubconf")
        models = ModuleType("models")
        experimental = ModuleType("models.experimental")
        original = lambda *args, **kwargs: "original-reader"
        experimental.torch_load = original
        package.hubconf = hub
        models.experimental = experimental
        expected = object()
        def custom(**kwargs):
            self.assertIs(experimental.torch_load, load_checkpoint_object)
            if fail:
                raise RuntimeError("deliberate load failure")
            return expected
        hub.custom = custom
        modules = {"yolov5": package, "yolov5.hubconf": hub, "models": models, "models.experimental": experimental}
        with patch.dict(sys.modules, modules), patch("pathlib.Path.is_file", return_value=True):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "load failure"):
                    load_yolo_model("trusted.pt", "cpu")
            else:
                self.assertIs(load_yolo_model("trusted.pt", "cpu"), expected)
        self.assertIs(experimental.torch_load, original)

    def test_reader_is_scoped_to_construction(self):
        self.run_fake()

    def test_reader_is_restored_after_failure(self):
        self.run_fake(fail=True)


if __name__ == "__main__":
    unittest.main()
