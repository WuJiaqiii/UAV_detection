from tests.fixtures import parse_test_args
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
import torch
from torch import nn

from scripts.evaluate_single_signal import build_parser, run
from util.multi_signal_eval import (
    FrequencyTracker, MultiSignalPredictor, band_iou, evaluate_multi_file,
    group_geometry, load_multi_truth, merge_bands, multi_evaluation_metrics, resolve_rf_centers,
)
from util.single_signal_eval import resolve_config
from util.signal_eval_report import write_html_report


CLASSES = {"alpha": 0, "beta": 1, "Background": 2}


def group(group_id, row, probabilities, width=60):
    label = int(np.argmax(probabilities))
    return {"group_id": group_id, "status": "ok", "class_id": label,
            "class_name": list(CLASSES)[label], "confidence": float(max(probabilities)),
            "probabilities": probabilities, "selected_group_score": 2.,
            "selected_boxes": [[10, row, 10+width, row+10], [150, row, 150+width, row+10], [300, row, 300+width, row+10]]}


def observation(index, row, probabilities, group_id=0):
    result = group(group_id, row, probabilities)
    result.update(slice_index=index, observation_id=f"S{index}_G{group_id}")
    result.update(group_geometry(result["selected_boxes"], (512, 750), 122_880_000, index*50.))
    return result


class FrequencyGeometryTests(unittest.TestCase):
    def test_fft_grid_and_absolute_center_are_not_source_capture_rate(self):
        result = group_geometry([[0, 0, 750, 512]], (512, 750), 122_880_000, 100., rf_center_mhz=2400.)
        self.assertAlmostEqual(result["frequency_low_offset_mhz"], -61.44)
        self.assertAlmostEqual(result["frequency_high_offset_mhz"], 61.44)
        self.assertAlmostEqual(result["frequency_center_absolute_mhz"], 2400.)
        self.assertEqual(result["start_ms"], 100.)
        self.assertEqual(result["end_ms"], 150.)
        self.assertAlmostEqual(result["pulses"][0]["frequency_low_absolute_mhz"], 2338.56)

    def test_hopping_bands_keep_gaps_and_original_coordinates(self):
        result = group_geometry([[0, 100, 75, 110], [150, 300, 225, 310]], (512, 750), 122_880_000, 50.)
        self.assertEqual(len(result["frequency_bands_offset_mhz"]), 2)
        self.assertAlmostEqual(result["occupied_bandwidth_mhz"], 4.8)
        self.assertAlmostEqual(result["frequency_span_mhz"], 50.4)
        self.assertAlmostEqual(result["pulses"][0]["frequency_low_offset_mhz"], -37.44)
        self.assertEqual(result["pulses"][1]["start_ms"], 60.)
        self.assertEqual(result["pulses"][1]["end_ms"], 65.)
        self.assertIsNone(result["frequency_center_absolute_mhz"])

    def test_bands_merge_and_overlap_use_intervals_not_envelopes(self):
        self.assertEqual(merge_bands([[0, 2], [1, 3], [5, 6]]), [[0., 3.], [5., 6.]])
        self.assertEqual(band_iou([[0, 1], [9, 10]], [[4, 6]]), 0.)
        self.assertAlmostEqual(band_iou([[0, 2]], [[1, 3]]), 1/3)
        for boxes in ([], [[0, 2, 0, 3]], [[0, float("nan"), 2, 3]]):
            with self.assertRaises(ValueError):
                group_geometry(boxes, (512, 750), 122_880_000)


class TrackingTests(unittest.TestCase):
    def test_same_class_at_different_frequencies_stays_two_tracks(self):
        tracker = FrequencyTracker()
        first = [observation(0, 100, [.9, .1, 0]), observation(0, 350, [.9, .1, 0], 1)]
        second = [observation(1, 351, [.8, .2, 0]), observation(1, 101, [.8, .2, 0], 1)]
        tracker.update(first)
        tracker.update(second)
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(first[0]["track_id"], second[1]["track_id"])
        self.assertEqual(first[1]["track_id"], second[0]["track_id"])
        self.assertNotEqual(first[0]["track_id"], first[1]["track_id"])

    def test_class_flip_is_associated_before_per_track_vote(self):
        tracker = FrequencyTracker()
        for index, probabilities in enumerate(([.9, .1, 0], [.3, .7, 0], [.8, .2, 0])):
            tracker.update([observation(index, 100, probabilities)])
        result = tracker.summarize(CLASSES)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class_name"], "alpha")
        self.assertEqual(result[0]["valid_votes"], 3)
        self.assertEqual(result[0]["vote_counts"]["alpha"], 2)

    def test_one_track_cannot_consume_two_groups_in_same_slice(self):
        tracker = FrequencyTracker()
        tracker.update([observation(0, 100, [.9, .1, 0])])
        current = [observation(1, 100, [.9, .1, 0]), observation(1, 101, [.9, .1, 0], 1)]
        tracker.update(current)
        self.assertEqual(len({row["track_id"] for row in current}), 2)
        self.assertEqual(sorted(len(track["observations"]) for track in tracker.tracks), [1, 2])

    def test_disjoint_carriers_and_long_time_gaps_start_new_tracks(self):
        tracker = FrequencyTracker(frequency_tolerance_mhz=100., min_band_iou=0., max_gap_ms=100.)
        tracker.update([observation(0, 100, [.9, .1, 0])])
        tracker.update([observation(1, 150, [.9, .1, 0])])
        tracker.update([observation(20, 100, [.9, .1, 0])])
        self.assertEqual(len(tracker.tracks), 3)

    def test_vote_thresholds_apply_to_each_track_independently(self):
        tracker = FrequencyTracker()
        first = [observation(0, 100, [.8, .2, 0]), observation(0, 350, [.2, .8, 0], 1)]
        tracker.update(first)
        tracker.update([observation(1, 100, [.9, .1, 0])])
        result = tracker.summarize(CLASSES, {"min_votes": 2})
        self.assertEqual([row["status"] for row in result], ["ok", "unknown"])
        self.assertEqual(result[1]["unknown_reason"], "too_few_votes")
        self.assertEqual(len(result[1]["observation_ids"]), 1)


class MultiSignalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uav-multi-eval-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = {"config": {"classes": CLASSES, "mask_img_size": 32}, "model_state_dict": {}}
        self.cfg = resolve_config(self.checkpoint)

    def _png(self, name="image.png"):
        path = self.root/name
        cv2.imencode(".png", np.zeros((512, 750), dtype=np.uint8))[1].tofile(path)
        return path

    def test_predictor_classifies_every_group_in_batches(self):
        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []
            def forward(self, tensor):
                self.batch_sizes.append(len(tensor))
                return torch.tensor([[4., 1., 0.]], device=tensor.device).repeat(len(tensor), 1)
        classifier, detector, preprocessor = Classifier(), Mock(), Mock()
        detector.detect_rgb.return_value = [[10, 100, 70, 110]]
        preprocessor.select_signal_groups.return_value = [
            {"boxes": np.array(group(i, 100+i*50, [.8, .2, 0])["selected_boxes"]), "score": 5-i}
            for i in range(3)]
        engine = MultiSignalPredictor(classifier, detector, preprocessor, self.cfg, batch_size=2)
        result = engine.predict(np.zeros((512, 750)), "image")
        self.assertEqual(len(result["groups"]), 3)
        self.assertEqual(classifier.batch_sizes, [2, 1])
        self.assertEqual([row["class_name"] for row in result["groups"]], ["alpha"]*3)

    def test_real_dat_slices_keep_two_signals_and_file_times(self):
        path = self.root/"record.dat"
        with path.open("wb") as stream:
            stream.truncate(10_500_000*4)
        predictor = Mock()
        predictor.predict.side_effect = [
            {"status": "ok", "groups": [group(0, 100, [.9, .1, 0]), group(1, 350, [.1, .9, 0])]},
            {"status": "ok", "groups": [group(0, 101, [.8, .2, 0]), group(1, 351, [.2, .8, 0])]},
        ]
        result, rows = evaluate_multi_file(path, predictor, self.cfg, {"min_votes": 2}, rf_center_mhz=2400.)
        self.assertEqual(result["predicted_classes"], ["alpha", "beta"])
        self.assertEqual(result["track_count"], 2)
        self.assertEqual(result["observation_count"], 4)
        self.assertGreater(rows[1]["groups"][0]["start_ms"], 50.)
        self.assertEqual(rows[1]["slice_start_ms"], 50.)
        self.assertAlmostEqual(rows[0]["groups"][0]["frequency_center_absolute_mhz"], 2363.76)
        self.assertEqual(result["sampling"]["unobserved_samples"], 500_000)

    def test_cli_reports_images_html_and_multilabel_accuracy(self):
        image = self._png("多信号.png")
        model = self.root/"model.pth"
        torch.save(self.checkpoint, model)
        truth = self.root/"labels.json"
        truth.write_text(json.dumps({image.name: ["alpha", "beta"]}, ensure_ascii=False), encoding="utf-8")
        args = parse_test_args(build_parser(), ["--inputs", str(image), "--checkpoint_path", str(model),
            "--yolo_weights", str(self.root/"mock.pt"), "--output_dir", str(self.root/"reports"),
            "--rf_center_mhz", "2400", "--labels_json", str(truth), "--device", "cpu"])
        predictor = Mock()
        predictor.predict.return_value = {"status": "ok", "groups": [group(0, 100, [.9, .1, 0]), group(1, 350, [.1, .9, 0])]}
        report = run(args, predictor=predictor)
        self.assertEqual(report["signal_mode"], "multi")
        self.assertEqual(report["metrics"]["file_class_set_accuracy"], 1.)
        for name in ["results.csv", "results.json", "signals.csv", "tracks.csv", "slices.jsonl", "report.html"]:
            self.assertTrue((args.output_dir/name).is_file(), name)
        row = json.loads((args.output_dir/"slices.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(len(row["groups"]), 2)
        self.assertTrue((args.output_dir/row["visualization"]).is_file())
        html = (args.output_dir/"report.html").read_text(encoding="utf-8")
        self.assertIn("2400", html)
        self.assertIn("T0001", html)
        self.assertIn("Frequency range", html)
        self.assertEqual(len(report["files"][0]["signals"]), 2)

    def test_no_detection_is_recorded_and_visualized(self):
        predictor = Mock()
        predictor.predict.return_value = {"status": "no_detection", "groups": []}
        result, rows = evaluate_multi_file(self._png(), predictor, self.cfg, visualization_dir=self.root/"previews")
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["track_count"], 0)
        self.assertTrue(Path(rows[0]["visualization"]).is_file())

    def test_preview_failure_does_not_discard_classification(self):
        from unittest.mock import patch
        predictor = Mock()
        predictor.predict.return_value = {"status": "ok", "groups": [group(0, 100, [.9, .1, 0])]}
        with patch("util.signal_eval_report.save_slice_preview", side_effect=OSError("cannot write image")):
            result, rows = evaluate_multi_file(self._png(), predictor, self.cfg, visualization_dir=self.root/"previews")
        self.assertEqual(result["predicted_classes"], ["alpha"])
        self.assertEqual(result["visualization_errors"], 1)
        self.assertEqual(rows[0]["status"], "ok")

    def test_bad_slice_and_incorrect_predictor_are_explicit_errors(self):
        predictor = Mock()
        predictor.predict.return_value = {"status": "ok", "probabilities": [.9, .1, 0]}
        result, rows = evaluate_multi_file(self._png(), predictor, self.cfg)
        self.assertEqual(result["status"], "error")
        self.assertIn("returning groups", rows[0]["error"])

    def test_per_file_rf_center_and_multiclass_truth(self):
        path = self._png()
        centers = self.root/"centers.json"
        centers.write_text(json.dumps({path.name: 5800.}), encoding="utf-8")
        self.assertEqual(resolve_rf_centers(centers, 2400.)[str(path.resolve())], 5800.)
        centers.write_text(json.dumps({path.name: float("nan")}), encoding="utf-8")
        with self.assertRaises(ValueError):
            resolve_rf_centers(centers)
        labels = self.root/"truth.json"
        labels.write_text(json.dumps({path.name: ["alpha", "beta", "alpha"]}), encoding="utf-8")
        self.assertEqual(load_multi_truth(labels, [path], CLASSES)[str(path)], ["alpha", "beta"])

    def test_same_file_class_set_is_not_instance_accuracy(self):
        result = {"status": "ok", "predicted_classes": ["alpha"], "true_class_names": ["alpha", "beta"],
                  "observation_count": 2, "track_count": 2, "accepted_track_count": 2}
        metrics = multi_evaluation_metrics([result], CLASSES)
        self.assertEqual(metrics["file_class_set_accuracy"], 0.)
        self.assertEqual(metrics["file_class_presence_metrics"]["beta"]["fn"], 1)

    def test_multi_mode_rejects_union_instead_of_merging_signals(self):
        path = self._png()
        checkpoint = self.root/"model.pth"
        torch.save(self.checkpoint, checkpoint)
        args = parse_test_args(build_parser(), ["--inputs", str(path), "--checkpoint_path", str(checkpoint),
            "--yolo_weights", "unused.pt", "--output_dir", str(self.root/"out"), "--group_policy", "union"])
        with self.assertRaisesRegex(ValueError, "every group"):
            run(args, predictor=Mock())


if __name__ == "__main__":
    unittest.main()
