import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from scipy.io import savemat
import torch
from torch import nn

from scripts.evaluate_single_signal import build_parser, run
from util.mask_input import boxes_to_centered_mask, build_mask_input
from util.preprocess import SignalPreprocessor
from util.single_signal_eval import (
    SingleSignalPredictor, collect_inputs, evaluate_file, evaluation_metrics,
    load_spectrum, load_truth, plan_iq_slices, resolve_config, validate_classes, vote_predictions,
)
from util.spectrogram_frontend import SPEC_SHAPE, render_gray, FrontendConfig


CLASSES = {"alpha": 0, "beta": 1, "Background": 2}


def prediction(probabilities):
    return {"status": "ok", "probabilities": probabilities}


class SliceVotingTests(unittest.TestCase):
    def test_majority_and_soft_are_distinct(self):
        rows = [prediction([.51, .49, 0]), prediction([.51, .49, 0]), prediction([.01, .99, 0])]
        hard = vote_predictions(copy.deepcopy(rows), CLASSES)
        soft = vote_predictions(copy.deepcopy(rows), CLASSES, method="soft")
        self.assertEqual(hard["class_name"], "alpha")
        self.assertEqual(hard["vote_counts"], {"alpha": 2, "beta": 1, "Background": 0})
        self.assertAlmostEqual(hard["vote_share"], 2/3)
        self.assertEqual(soft["class_name"], "beta")

    def test_majority_tie_uses_average_probabilities_then_id(self):
        result = vote_predictions([prediction([.6, .4, 0]), prediction([.1, .9, 0])], CLASSES)
        self.assertEqual(result["class_name"], "beta")
        result = vote_predictions([prediction([.6, .4, 0]), prediction([.4, .6, 0])], CLASSES)
        self.assertEqual(result["class_name"], "alpha")

    def test_abstentions_confidence_and_background(self):
        rows = [{"status": "no_detection"}, {"status": "no_group"},
                prediction([.1, .1, .8]), prediction([.51, .49, 0]), prediction([.9, .1, 0])]
        result = vote_predictions(rows, CLASSES, min_slice_confidence=.7, exclude_classes=["Background"])
        self.assertEqual(result["class_name"], "alpha")
        self.assertEqual(result["valid_votes"], 1)
        self.assertEqual(result["abstention_counts"], {"no_detection": 1, "no_group": 1, "excluded_class": 1, "low_confidence": 1})
        self.assertEqual([r["vote_accepted"] for r in rows], [False, False, False, False, True])
        self.assertEqual(vote_predictions([prediction([0, 0, 1])], CLASSES)["class_name"], "Background")

    def test_soft_vote_does_not_revive_excluded_class(self):
        rows = [prediction([.51, 0, .49]), prediction([0, .51, .49])]
        result = vote_predictions(rows, CLASSES, method="soft", exclude_classes=["Background"])
        self.assertEqual(result["class_name"], "alpha")

    def test_unknown_when_no_votes_or_threshold_not_met(self):
        self.assertEqual(vote_predictions([], CLASSES)["unknown_reason"], "no_valid_votes")
        self.assertIsNone(vote_predictions([{"status": "no_group"}], CLASSES)["class_name"])
        result = vote_predictions([prediction([1, 0, 0])], CLASSES, min_votes=2)
        self.assertEqual(result["unknown_reason"], "too_few_votes")
        result = vote_predictions([prediction([1, 0, 0]), prediction([0, 1, 0])], CLASSES, min_vote_share=.6)
        self.assertEqual(result["unknown_reason"], "low_vote_share")
        self.assertIsNone(result["class_name"])

    def test_invalid_probabilities_thresholds_and_classes(self):
        for probabilities in ([.1, .2], [0, 0, 0], [float("nan"), 0, 1], [-1, 1, 1]):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                vote_predictions([prediction(probabilities)], CLASSES)
        for classes in ({"a": 1}, {"a": 0, "b": 0}, {"a": True}, {"a": .0}, {}):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                validate_classes(classes)
        with self.assertRaises(ValueError):
            vote_predictions([], CLASSES, exclude_classes=["not_a_class"])


class IQWindowTests(unittest.TestCase):
    def test_drop_and_cover_tail_without_padding(self):
        rate = 100_000_000
        windows, info = plan_iq_slices(12_000_000, rate)
        self.assertEqual([w["start_sample"] for w in windows], [0, 5_000_000])
        self.assertEqual(info["unobserved_samples"], 2_000_000)
        windows, info = plan_iq_slices(12_000_000, rate, tail_policy="cover")
        self.assertEqual([w["start_sample"] for w in windows], [0, 5_000_000, 7_000_000])
        self.assertEqual(windows[-1]["end_ms"], 120.)
        self.assertEqual(info["unobserved_samples"], 0)

    def test_overlap_and_even_limit(self):
        windows, info = plan_iq_slices(30_000_000, 100_000_000, stride_ms=25., max_slices=3)
        self.assertEqual([w["start_sample"] for w in windows], [0, 12_500_000, 25_000_000])
        self.assertEqual(info["planned_slices"], 11)
        self.assertEqual(info["unobserved_samples"], 15_000_000)
        windows, _ = plan_iq_slices(30_000_000, 100_000_000, max_slices=1)
        self.assertEqual(windows[0]["start_sample"], 10_000_000)

    def test_native_rate_sample_coordinates(self):
        windows, _ = plan_iq_slices(12_288_000, 122_880_000)
        self.assertEqual(windows[1]["start_sample"], 6_144_000)
        self.assertEqual(windows[1]["start_ms"], 50.)
        self.assertEqual(windows[1]["end_ms"], 100.)

    def test_short_dat_and_invalid_rate_stride(self):
        for total, rate, stride in [(1, 100_000_000, 50), (5_000_000, 80_000_000, 50),
                                    (5_000_000, 100_000_000, 0), (5_000_000, 100_000_000, float("nan"))]:
            with self.subTest(total=total, rate=rate, stride=stride), self.assertRaises(ValueError):
                plan_iq_slices(total, rate, stride)


class InputAndPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uav-single-eval-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ckpt = {"config": {"classes": CLASSES, "grouping_mode": "hopping", "mask_img_size": 32},
                     "model_state_dict": {}}
        self.cfg = resolve_config(self.ckpt)

    def _png(self, path, gray):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", gray)[1].tofile(path)

    def test_configuration_uses_saved_label_ids_and_effective_parameters(self):
        self.assertEqual(self.cfg.classes, CLASSES)
        self.assertEqual(self.cfg.mask_img_size, 32)
        self.assertEqual(self.cfg.grouping_mode, "hopping")
        changed = {"alpha": 1, "beta": 0, "Background": 2}
        with self.assertRaisesRegex(ValueError, "class mapping mismatch"):
            resolve_config(self.ckpt, classes=changed)
        self.assertEqual(resolve_config(self.ckpt, overrides={"grouping_mode": "frequency"}).grouping_mode, "frequency")
        with self.assertRaisesRegex(ValueError, "Unknown inference setting"):
            resolve_config(self.ckpt, overrides={"typo": 2})

    def test_legacy_checkpoint_requires_explicit_classes(self):
        with self.assertRaises(ValueError):
            resolve_config({"model_state_dict": {}})
        with self.assertWarns(UserWarning):
            cfg = resolve_config({"model_state_dict": {}}, classes=CLASSES)
        self.assertEqual(cfg.classes, CLASSES)

    def test_mat_png_reading_and_mixed_input_discovery(self):
        matrix = np.ones(SPEC_SHAPE, dtype=np.float32)
        matrix[100:110, 20:60] = 9
        mat = self.root / "arbitrary_name.MAT"
        savemat(mat, {"summed_submatrices": matrix})
        amplitude, domain = load_spectrum(mat, self.cfg)
        self.assertEqual(domain, "amplitude")
        gray = render_gray(amplitude, FrontendConfig.from_config(self.cfg))
        png = self.root / "中文" / "same.PNG"
        self._png(png, gray)
        raster, domain = load_spectrum(png, self.cfg)
        self.assertEqual(domain, "image")
        np.testing.assert_array_equal(raster, gray)
        self.assertEqual(collect_inputs([self.root], recursive=True), sorted([mat.resolve(), png.resolve()]))
        self.assertEqual(collect_inputs([str(mat), str(mat)]), [mat.resolve()])
        with self.assertRaises(FileNotFoundError):
            collect_inputs([self.root / "missing.dat"])

    def test_png_wrong_shape_and_16bit_are_rejected(self):
        for image in (np.zeros((10, 10), dtype=np.uint8), np.zeros(SPEC_SHAPE, dtype=np.uint16)):
            path = self.root / "wrong.png"
            self._png(path, image)
            with self.assertRaises(ValueError):
                load_spectrum(path, self.cfg)

    def test_shared_mask_centering_and_resize(self):
        boxes = [[1, 1, 4, 3], [7, 3, 10, 5]]
        expected = np.zeros((10, 12), dtype=np.uint8)
        expected[3:5, 1:4] = 255
        expected[5:7, 7:10] = 255
        np.testing.assert_array_equal(boxes_to_centered_mask((10, 12), boxes, self.cfg), expected)
        tensor, mask = build_mask_input(np.ones((10, 12)), boxes, self.cfg)
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(tuple(tensor.shape), (1, 32, 32))
        self.assertEqual(set(tensor.unique().tolist()), {0., 1.})

    def test_actual_grouping_mask_and_torch_classifier(self):
        cfg = resolve_config(self.ckpt, overrides={"grouping_mode": "frequency"})
        spectrum = np.ones(SPEC_SHAPE, dtype=np.float32)
        boxes = [[10, 100, 70, 110], [150, 100, 210, 110], [300, 100, 360, 110]]
        for x1, y1, x2, y2 in boxes:
            spectrum[y1:y2, x1:x2] = 100
        spectrum /= spectrum.max()
        classifier = nn.Sequential(nn.Flatten(), nn.Linear(32*32, 3))
        with torch.no_grad():
            classifier[1].weight.zero_()
            classifier[1].bias.copy_(torch.tensor([0., 5., 0.]))
        detector = Mock()
        detector.detect_rgb.return_value = boxes
        engine = SingleSignalPredictor(classifier, detector, SignalPreprocessor(cfg, None), cfg)
        result = engine.predict(spectrum, "amplitude")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["class_id"], 1)
        self.assertEqual(len(result["selected_boxes"]), 3)
        self.assertEqual(detector.detect_rgb.call_args.args[0].shape, (512, 750, 3))
        detector.detect_rgb.return_value = []
        self.assertEqual(engine.predict(spectrum, "amplitude")["status"], "no_detection")

    def test_union_and_no_group_are_explicit(self):
        detector, preprocessor, classifier = Mock(), Mock(), Mock()
        detector.detect_rgb.return_value = [[10, 100, 70, 110]]
        preprocessor.select_signal_groups.return_value = []
        # A real nn.Module avoids Mock.to().eval() replacing the prediction object.
        model = nn.Sequential(nn.Flatten(), nn.Linear(32*32, 3))
        engine = SingleSignalPredictor(model, detector, preprocessor, self.cfg)
        self.assertEqual(engine.predict(np.zeros(SPEC_SHAPE), "image")["status"], "no_group")
        preprocessor.select_signal_groups.return_value = [
            {"boxes": np.array([[10, 100, 70, 110]]), "score": 2},
            {"boxes": np.array([[200, 200, 260, 210]]), "score": 1},
        ]
        self.assertEqual(len(engine.predict(np.zeros(SPEC_SHAPE), "image")["selected_boxes"]), 1)
        engine.group_policy = "union"
        self.assertEqual(len(engine.predict(np.zeros(SPEC_SHAPE), "image")["selected_boxes"]), 2)

    def test_real_iq_slice_loading_and_file_vote(self):
        # Sparse zero recording: real int16 reads, resampling, FFT and canonicalization.
        path = self.root / "recording.dat"
        with path.open("wb") as stream:
            stream.truncate(10_000_000 * 4)
        predictor = Mock()
        predictor.predict.side_effect = [prediction([.8, .1, .1]), prediction([.9, .05, .05])]
        result, rows = evaluate_file(path, predictor, self.cfg)
        self.assertEqual(result["class_name"], "alpha")
        self.assertEqual(result["valid_votes"], 2)
        self.assertEqual([row["start_ms"] for row in rows], [0., 50.])
        spectrum, domain = predictor.predict.call_args.args
        self.assertEqual(domain, "amplitude")
        self.assertEqual(spectrum.shape, SPEC_SHAPE)
        self.assertTrue(np.isfinite(spectrum).all())

    def test_bad_dat_size_and_short_dat(self):
        for data in (b"123", b"1234"):
            path = self.root / "bad.dat"
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                evaluate_file(path, Mock(), self.cfg)

    def test_reports_are_per_file_and_labels_do_not_affect_prediction(self):
        ckpt_path = self.root / "model.pth"
        torch.save(self.ckpt, ckpt_path)
        first, second = self.root / "first.png", self.root / "second.png"
        self._png(first, np.zeros(SPEC_SHAPE, dtype=np.uint8))
        self._png(second, np.zeros(SPEC_SHAPE, dtype=np.uint8))
        labels = self.root / "labels.json"
        labels.write_text(json.dumps({"first.png": "alpha", "second.png": "beta"}), encoding="utf-8")
        args = build_parser().parse_args(["--signal_mode", "single", "--inputs", str(first), str(second), "--checkpoint_path", str(ckpt_path),
                                          "--yolo_weights", str(self.root / "fake.pt"), "--output_dir", str(self.root / "reports"),
                                          "--labels_json", str(labels), "--device", "cpu"])
        predictor = Mock()
        predictor.predict.side_effect = [prediction([.8, .1, .1]), {"status": "no_detection"}]
        report = run(args, predictor=predictor)
        self.assertEqual(report["metrics"]["file_accuracy"], .5)
        self.assertEqual(report["metrics"]["labeled_coverage"], .5)
        self.assertEqual(report["metrics"]["accuracy_on_classified"], 1.)
        self.assertEqual(report["files"][1]["status"], "unknown")
        self.assertIsNone(report["files"][1]["class_name"])
        self.assertTrue((args.output_dir / "results.csv").is_file())
        saved = json.loads((args.output_dir / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["effective_config"]["classes"], CLASSES)
        self.assertEqual(len((args.output_dir / "slices.jsonl").read_text().splitlines()), 2)
        with self.assertRaisesRegex(ValueError, "empty output_dir"):
            run(args, predictor=predictor)

    def test_slice_errors_are_not_votes(self):
        png = self.root / "broken.png"
        png.write_bytes(b"broken")
        result, rows = evaluate_file(png, Mock(), self.cfg)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["valid_votes"], 0)
        self.assertEqual(rows[0]["vote_reason"], "error")
        with self.assertRaises(ValueError):
            evaluate_file(png, Mock(), self.cfg, fail_fast=True)

    def test_truth_parent_and_unclassified_count_in_accuracy(self):
        path = self.root / "alpha" / "arbitrary.dat"
        self.assertEqual(load_truth(None, [path], CLASSES, True)[str(path)], "alpha")
        metrics = evaluation_metrics([
            {"status": "ok", "class_name": "alpha", "true_class_name": "alpha"},
            {"status": "error", "class_name": None, "true_class_name": "beta"},
        ], CLASSES)
        self.assertEqual(metrics["file_accuracy"], .5)
        self.assertEqual(metrics["confusion_counts"]["beta"]["<abstained>"], 1)

    def test_real_resnet_checkpoint_is_loaded_strictly(self):
        try:
            from model.resnet import MaskImageClassifier
        except ImportError as exc:
            self.skipTest(str(exc))
        model = MaskImageClassifier(num_classes=3, in_chans=1, pretrained=False)
        checkpoint_path = self.root / "resnet.pth"
        torch.save({"config": self.ckpt["config"], "model_state_dict": model.state_dict()}, checkpoint_path)
        image = self.root / "raw.png"
        self._png(image, np.zeros(SPEC_SHAPE, dtype=np.uint8))
        weights = self.root / "detector.pt"
        weights.touch()
        args = build_parser().parse_args(["--signal_mode", "single", "--inputs", str(image), "--checkpoint_path", str(checkpoint_path),
                                          "--yolo_weights", str(weights), "--output_dir", str(self.root/"resnet_result"), "--device", "cpu"])
        boxes = [[10, 100, 70, 110], [150, 100, 210, 110], [300, 100, 360, 110]]
        with patch("util.detector.YoloV5Detector") as detector:
            detector.return_value.detect_rgb.return_value = boxes
            result = run(args)
        self.assertEqual(result["files"][0]["status"], "ok")
        self.assertEqual(result["files"][0]["valid_votes"], 1)
        state = model.state_dict()
        state.pop(next(iter(state)))
        torch.save({"config": self.ckpt["config"], "model_state_dict": state}, checkpoint_path)
        args.output_dir = self.root / "incompatible_result"
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            run(args)


if __name__ == "__main__":
    unittest.main()
