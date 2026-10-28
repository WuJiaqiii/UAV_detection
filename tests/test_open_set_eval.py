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

from tests.fixtures import archive, track_fixture
from model.resnet import MaskImageClassifier
from scripts.evaluate_single_signal import build_parser, run
from util.open_set import (
    FEATURE_VERSION, OpenSetProfile, apply_open_set, build_profile, inference_signature,
    knn_scores, load_npz, normalize_features, refresh_open_set_results, save_npz,
)
from util.single_signal_eval import resolve_config
from util.unknown_discovery import discover_unknown


class FeatureCompatibilityTests(unittest.TestCase):
    def test_real_group_classifier_emits_normalized_penultimate_features(self):
        from util.multi_signal_eval import MultiSignalPredictor
        config = resolve_config({"config": {"classes": {"known": 0}, "mask_img_size": 32}})
        classifier = MaskImageClassifier("resnet18", 1, 1, pretrained=False)
        detector, preprocessor = Mock(), Mock()
        detector.detect_rgb.return_value = [[10, 100, 50, 110]]
        preprocessor.select_signal_groups.return_value = [
            {"boxes": np.array([[10, 100, 50, 110], [100, 100, 140, 110]])},
            {"boxes": np.array([[200, 300, 210, 310], [250, 300, 260, 310]])}]
        predictor = MultiSignalPredictor(classifier, detector, preprocessor, config, batch_size=1, return_features=True)
        groups = predictor.predict(np.zeros((512, 750)), "image")["groups"]
        self.assertEqual(len(groups), 2)
        for group in groups:
            self.assertEqual(len(group["embedding"]), 512)
            self.assertAlmostEqual(float(np.linalg.norm(group["embedding"])), 1., places=6)
            self.assertEqual(group["class_name"], "known")

    def test_all_backbones_preserve_logits_and_checkpoint_keys(self):
        torch.set_num_threads(2)
        for name, dimension in (("resnet18", 512), ("resnet34", 512), ("mobilenet_v3_small", 1024)):
            with self.subTest(backbone=name):
                model = MaskImageClassifier(name, 3, 1, pretrained=False).eval()
                saved = copy.deepcopy(model.state_dict())
                tensor = torch.randn(2, 1, 64, 64)
                with torch.no_grad():
                    before = model(tensor)
                    logits, features = model(tensor, return_features=True)
                torch.testing.assert_close(before, logits, rtol=1e-5, atol=1e-6)
                self.assertEqual(features.shape, (2, dimension))
                self.assertEqual(set(saved), set(model.state_dict()))
                clone = MaskImageClassifier(name, 3, 1, pretrained=False)
                clone.load_state_dict(saved, strict=True)

    def test_zero_nan_and_wrong_shape_features_fail_instead_of_becoming_novel(self):
        for values in ([[0., 0.]], [[float("nan"), 1]], [1., 2.]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalize_features(values)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uav-open-profile-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root/"reference.npz"
        self.calibration = self.root/"calibration.npz"
        self.profile = self.root/"profile.npz"
        archive(self.reference, [[1., .00, 0], [1., .01, 0], [1., -.01, 0]], ["known"]*3, "train")
        archive(self.calibration, [[1., .02, 0], [1., -.02, 0], [1., .03, 0]], ["known"]*3, "val")

    def build(self, **kwargs):
        build_profile(self.reference, self.calibration, self.profile, k=1, min_calibration_per_class=1, **kwargs)
        return OpenSetProfile(self.profile, {"sha256": "fixture"}, {"known": 0})

    def test_calibration_is_external_and_known_novel_distances_separate(self):
        profile = self.build(known_false_reject_rate=.1)
        self.assertGreater(profile.threshold, 0.)
        self.assertLess(profile.score([[1., 0, 0]])[0], profile.threshold)
        self.assertGreater(profile.score([[0, 1., 0]])[0], profile.threshold)
        self.assertEqual(profile.metadata["empirical_calibration_false_reject_rate"], 0.)
        self.assertTrue(profile.metadata["warnings"])

    def test_overlap_by_recording_or_identical_content_is_rejected(self):
        metadata, arrays = load_npz(self.calibration, "track_features")
        for key, value in (("recording_id", "train_0"), ("source_sha256", "train_hash_0")):
            altered = copy.deepcopy(metadata)
            altered["records"][0][key] = value
            filename = self.root/f"overlap_{key}.npz"
            save_npz(filename, altered, **arrays)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "overlap"):
                build_profile(self.reference, filename, self.profile, k=1, min_calibration_per_class=1)

    def test_profile_requires_truth_and_all_known_classes_in_calibration(self):
        metadata, arrays = load_npz(self.calibration, "track_features")
        metadata["records"][0]["true_class_name"] = None
        invalid = self.root/"unlabeled.npz"
        save_npz(invalid, metadata, **arrays)
        with self.assertRaisesRegex(ValueError, "truth"):
            build_profile(self.reference, invalid, self.profile, k=1, min_calibration_per_class=1)
        with self.assertRaisesRegex(ValueError, "calibration tracks"):
            build_profile(self.reference, self.calibration, self.profile, k=1, min_calibration_per_class=20)

    def test_profile_rejects_different_checkpoint_or_preprocessing_signature(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "does not match"):
            OpenSetProfile(self.profile, {"sha256": "different"}, {"known": 0})
        with self.assertRaisesRegex(ValueError, "does not match"):
            OpenSetProfile(self.profile, {"sha256": "fixture"}, {"other": 0})

    def test_existing_archives_are_not_overwritten(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            save_npz(self.reference, {})

    def test_persistent_novelty_routes_one_track_not_each_slice(self):
        profile = self.build()
        results, rows = track_fixture([[0, 1, 0]]*4)
        candidates = apply_open_set(profile, results, rows)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["track"]["recognition_status"], "novel_candidate")
        self.assertIsNone(candidates[0]["track"]["class_name"])
        self.assertEqual(candidates[0]["track"]["known_candidate_class"], "known")
        discovery = discover_unknown(candidates)
        refresh_open_set_results(results, rows)
        self.assertEqual(discovery["estimated_unknown_class_count"], 0)
        self.assertEqual(results[0]["novel_track_count"], 1)
        self.assertEqual(rows["test.png"][0]["groups"][0]["display_label"], "Unknown_unassigned")

    def test_insufficient_votes_are_uncertain_and_not_clustered(self):
        results, rows = track_fixture([[0, 1, 0]], min_votes=2)
        candidates = apply_open_set(self.build(), results, rows)
        self.assertFalse(candidates)
        self.assertEqual(results[0]["signals"][0]["recognition_status"], "uncertain")

    def test_inconsistent_novelty_is_uncertain(self):
        results, rows = track_fixture([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
        candidates = apply_open_set(self.build(), results, rows, min_novel_observation_share=.6)
        self.assertFalse(candidates)
        self.assertEqual(results[0]["signals"][0]["unknown_reason"], "inconsistent_novelty_across_observations")

    def test_no_unknown_quota_when_all_tracks_are_known(self):
        results, rows = track_fixture([[1, 0, 0]]*10)
        self.assertFalse(apply_open_set(self.build(), results, rows))
        self.assertEqual(results[0]["signals"][0]["recognition_status"], "known")


class DiscoveryTests(unittest.TestCase):
    @staticmethod
    def candidates(features, same_recording=False):
        return [{"path": f"file_{i}.dat", "recording_id": "same" if same_recording else f"record_{i}",
                 "feature": feature, "track": {"track_id": "T0001", "unknown_cluster_id": None}}
                for i, feature in enumerate(features)]

    def test_zero_or_too_few_candidates_estimates_zero_without_fabricated_classes(self):
        self.assertEqual(discover_unknown([])["estimated_unknown_class_count"], 0)
        result = discover_unknown(self.candidates([[0, 1, 0]]*2))
        self.assertEqual(result["estimated_unknown_class_count"], 0)
        self.assertEqual(result["unassigned_unknown_tracks"], 2)

    def test_two_feature_populations_discover_two_anonymous_classes(self):
        rng = np.random.default_rng(42)
        features = np.r_[rng.normal([0, 1., 0], .005, (12, 3)), rng.normal([0, 0, 1.], .005, (12, 3))]
        candidates = self.candidates(features)
        result = discover_unknown(candidates)
        self.assertEqual(result["estimated_unknown_class_count"], 2)
        self.assertEqual(result["clustered_unknown_tracks"], 24)
        self.assertNotEqual(candidates[0]["track"]["unknown_cluster_id"], candidates[-1]["track"]["unknown_cluster_id"])

    def test_single_class_is_allowed_and_recording_fragments_are_not_independent_support(self):
        rng = np.random.default_rng(13)
        features = rng.normal([0, 1., 0], .01, (40, 3))
        result = discover_unknown(self.candidates(features), min_recordings=2)
        self.assertEqual(result["estimated_unknown_class_count"], 1)
        result = discover_unknown(self.candidates(features, same_recording=True), min_recordings=2)
        self.assertEqual(result["estimated_unknown_class_count"], 0)
        self.assertEqual(result["unassigned_unknown_tracks"], 40)


class OpenSetCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uav-open-cli-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root/"model.pth"
        self.weights = self.root/"detector.pt"
        self.weights.write_bytes(b"detector fixture; never loaded")
        torch.save({"model_state_dict": {}, "config": {"classes": {"known": 0}, "mask_img_size": 32}}, self.checkpoint)

    def png(self, parent, name, value):
        path = self.root/parent/name
        path.parent.mkdir(exist_ok=True, parents=True)
        image = np.zeros((512, 750), dtype=np.uint8)
        image[0, 0] = value
        cv2.imencode(".png", image)[1].tofile(path)
        return path

    def args(self, inputs, output, *extra):
        return parse_test_args(build_parser(), ["--inputs", *map(str, inputs), "--checkpoint_path", str(self.checkpoint),
            "--yolo_weights", str(self.weights), "--output_dir", str(self.root/output), "--device", "cpu", *extra])

    @staticmethod
    def prediction(feature):
        return {"status": "ok", "groups": [{"group_id": 0, "status": "ok", "class_id": 0,
                "class_name": "known", "confidence": 1., "probabilities": [1.], "selected_group_score": 2.,
                "selected_boxes": [[10, 100, 40, 110], [100, 100, 130, 110]], "embedding": feature}]}

    def test_export_calibrate_and_discover_end_to_end_with_final_html_png_csv(self):
        # Deterministic feature fixtures exercise orchestration, not real accuracy.
        train = self.png("train/known", "train.png", 1)
        calibration = self.png("val/known", "calibration.png", 2)
        for path, output, feature in ((train, "ref", [1., 0, 0]), (calibration, "cal", [1., .01, 0])):
            predictor = Mock()
            predictor.predict.return_value = self.prediction(feature)
            run(self.args([path], output, "--export_features", "--label_from_parent", "--no-save_visuals"), predictor)
        profile = self.root/"profile.npz"
        build_profile(self.root/"ref/features.npz", self.root/"cal/features.npz", profile, k=1, min_calibration_per_class=1)
        paths = [self.png("test", f"{i}.png", i+3) for i in range(7)]
        predictor = Mock()
        features = [[1., 0, 0], *[[0, 1., i*.001] for i in range(6)]]
        predictor.predict.side_effect = [self.prediction(feature) for feature in features]
        labels = self.root/"truth.json"
        labels.write_text(json.dumps({str(path): "known" if i == 0 else "UnseenProtocol" for i, path in enumerate(paths)}), encoding="utf-8")
        args = self.args(paths, "reports", "--open_set_profile", str(profile),
                         "--unknown_min_cluster_size", "3", "--unknown_min_samples", "2", "--labels_json", str(labels))
        report = run(args, predictor)
        self.assertEqual(report["metrics"]["known_tracks"], 1)
        self.assertEqual(report["metrics"]["unknown_candidate_tracks"], 6)
        self.assertEqual(report["metrics"]["estimated_unknown_class_count"], 1)
        self.assertNotIn("file_class_set_accuracy", report["metrics"])
        self.assertEqual(report["metrics"]["file_unknown_presence_metrics"]["recall"], 1.)
        self.assertTrue((args.output_dir/"unknown_clusters.json").is_file())
        self.assertIn("Unknown_001", (args.output_dir/"tracks.csv").read_text(encoding="utf-8-sig"))
        self.assertIn("Unknown_001", (args.output_dir/"report.html").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (args.output_dir/"slices.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertNotIn("embedding", rows[0]["groups"][0])
        self.assertTrue((args.output_dir/rows[-1]["visualization"]).is_file())
        labels.write_text(json.dumps({str(path): "known" for path in paths}), encoding="utf-8")
        args.output_dir = self.root/"misleading_truth"
        args.save_visuals = False
        predictor.predict.side_effect = [self.prediction(feature) for feature in features]
        second = run(args, predictor)
        self.assertEqual(second["discovery"], report["discovery"])
        self.assertEqual(second["metrics"]["file_unknown_presence_metrics"]["fp"], 6)

    def test_profile_mismatch_fails_before_creating_output_or_running_predictor(self):
        path = self.png("test", "one.png", 3)
        args = self.args([path], "bad")
        reference, calibration, profile = [self.root/name for name in ("r.npz", "c.npz", "p.npz")]
        archive(reference, [[1., 0, 0]], ["known"], "r")
        archive(calibration, [[1., .1, 0]], ["known"], "c")
        build_profile(reference, calibration, profile, k=1, min_calibration_per_class=1)
        args.open_set_profile = profile
        predictor = Mock()
        with self.assertRaisesRegex(ValueError, "does not match"):
            run(args, predictor)
        predictor.predict.assert_not_called()
        self.assertFalse(args.output_dir.exists())

    def test_feature_modes_cannot_silently_use_single_signal_voting(self):
        path = self.png("test", "one.png", 3)
        with self.assertRaisesRegex(ValueError, "require --signal_mode multi"):
            run(self.args([path], "single", "--signal_mode", "single", "--export_features"), Mock())


if __name__ == "__main__":
    unittest.main()
