import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from scripts.cluster_unknown_features import build_parser as cluster_parser, run as cluster_run
from scripts.evaluate_single_signal import build_parser, run
from util.dcg_discovery import cluster_dcg
from util.multi_signal_eval import load_multi_truth
from util.open_set import (
    FEATURE_VERSION, OpenSetProfile, apply_open_set, build_profile, inference_signature,
    load_npz, normalize_features, recording_identity, save_npz,
)
from util.single_signal_eval import resolve_config
from util.unknown_discovery import discover_unknown


def candidates(features, same_recording=False):
    return [{"path": f"file_{i}.dat", "recording_id": "same_capture" if same_recording else f"capture_{i}",
             "feature": feature, "track": {"track_id": "T0001"}} for i, feature in enumerate(features)]


def known_archive(path, features, split, signature=None, classes=None, recording=None):
    labels = classes or {"known": 0}
    records = [{"path": f"{split}/{i}.png", "recording_id": recording or f"{split}_{i}",
                "source_sha256": f"hash_{split}_{i}", "true_class_name": "known", "track_id": "T0001"}
               for i in range(len(features))]
    save_npz(path, {"kind": "track_features", "schema_version": 1, "feature_version": FEATURE_VERSION,
                   "signature": signature or {"sha256": "fixture"}, "classes": labels, "records": records},
             features=normalize_features(features))


class DcgDiscoveryTests(unittest.TestCase):
    @staticmethod
    def populations():
        rng = np.random.default_rng(42)
        return np.r_[rng.normal([0, 1., 0], .005, (12, 3)), rng.normal([0, 0, 1.], .005, (12, 3))]

    def test_two_populations_and_an_isolated_outlier(self):
        items = candidates(np.r_[self.populations(), [[-1., 0, 0]]])
        result = discover_unknown(items, backend="dcg")
        self.assertEqual(result["estimated_unknown_class_count"], 2)
        self.assertEqual(result["clustered_unknown_tracks"], 24)
        self.assertEqual(result["unassigned_unknown_tracks"], 1)
        self.assertEqual(items[-1]["track"]["display_label"], "Unknown_unassigned")
        self.assertIsNone(items[0]["track"]["cluster_membership"])
        self.assertGreater(items[0]["track"]["cluster_similarity"], .99)

    def test_no_edges_do_not_turn_singletons_into_new_classes(self):
        items = candidates(np.eye(6))
        result = discover_unknown(items, backend="dcg", min_cluster_size=2)
        self.assertEqual(result["estimated_unknown_class_count"], 0)
        self.assertEqual(result["diagnostics"]["isolated_tracks"], 6)
        self.assertEqual(result["unassigned_unknown_tracks"], 6)

    def test_identical_embeddings_keep_all_support_despite_knn_ties(self):
        items = candidates([[0., 1., 0.]]*30)
        result = discover_unknown(items, backend="dcg", dcg_options={"topk": 3})
        self.assertEqual(result["estimated_unknown_class_count"], 1)
        self.assertEqual(result["clustered_unknown_tracks"], 30)
        self.assertEqual(result["diagnostics"]["duplicate_tracks_collapsed"], 29)

    def test_independent_recording_support_and_small_communities_are_required(self):
        result = discover_unknown(candidates(self.populations(), same_recording=True), backend="dcg")
        self.assertEqual(result["estimated_unknown_class_count"], 0)
        small = discover_unknown(candidates([[0., 1., 0.]]*3), backend="dcg", min_cluster_size=5)
        self.assertEqual(small["estimated_unknown_class_count"], 0)

    def test_zero_and_one_candidate_are_valid(self):
        for values in ([], [[1., 0.]]):
            with self.subTest(count=len(values)):
                result = discover_unknown(candidates(values), backend="dcg")
                self.assertEqual(result["estimated_unknown_class_count"], 0)

    def test_backend_failure_propagates_instead_of_fabricating_one_class(self):
        with patch("networkx.algorithms.community.louvain_communities", side_effect=RuntimeError("community backend failed")):
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                cluster_dcg(np.array([[1., 0.], [1., .01], [1., .02]]))

    def test_seeded_results_and_prototypes_are_reproducible(self):
        first = discover_unknown(candidates(self.populations()), backend="dcg")
        second = discover_unknown(candidates(self.populations()), backend="dcg")
        self.assertEqual(first, second)
        for cluster in first["clusters"]:
            self.assertAlmostEqual(np.linalg.norm(cluster["prototype"]), 1., places=6)

    def test_invalid_graph_options_fail_explicitly(self):
        for options in ({"topk": 0}, {"cosine_threshold": 1.1}, {"resolution": 0}, {"seed": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                discover_unknown([], backend="dcg", dcg_options=options)


class ProfileAndArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uav-openworld-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_figure_exports_have_a_shared_recording_identity_and_explicit_override(self):
        first, second = self.root/"capture-Figure-1.png", self.root/"capture-Figure-2.mat"
        identity = recording_identity(first, "hash_a")
        self.assertEqual(identity, recording_identity(second, "hash_b"))
        self.assertEqual(recording_identity(first, "hash_a", {str(first): "session_override"}), "session_override")
        ref, cal, profile = [self.root/name for name in ("ref.npz", "cal.npz", "profile.npz")]
        known_archive(ref, [[1., 0, 0]], "train", recording=identity)
        known_archive(cal, [[1., .01, 0]], "val", recording=identity)
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_profile(ref, cal, profile, k=1, min_calibration_per_class=1)

    def test_existing_filename_labels_are_only_reporting_metadata(self):
        file = self.root/"known-[0,1,1000,2]-SNR-0-Figure-1.png"
        truth = load_multi_truth(None, [file], {"known": 0}, label_from_filename=True)
        self.assertEqual(truth[str(file)], ["known"])
        unknown = self.root/"Unseen-[0,1,1000,2]-SNR-0-Figure-1.png"
        with self.assertRaises(ValueError):
            load_multi_truth(None, [unknown], {"known": 0}, label_from_filename=True)
        self.assertEqual(load_multi_truth(None, [unknown], {"known": 0}, allow_unknown=True,
                                         label_from_filename=True)[str(unknown)], ["Unseen"])

    def test_ignored_background_needs_no_reference_and_is_uncertain(self):
        from tests.test_open_set_eval import track_fixture
        classes = {"known": 0, "Background": 1}
        ref, cal, profile = [self.root/name for name in ("ref.npz", "cal.npz", "profile.npz")]
        known_archive(ref, [[1., 0, 0]], "train", classes=classes)
        known_archive(cal, [[1., .01, 0]], "val", classes=classes)
        build_profile(ref, cal, profile, k=1, min_calibration_per_class=1, ignore_classes=["Background"])
        scorer = OpenSetProfile(profile, {"sha256": "fixture"}, classes)
        results, rows = track_fixture([[0., 1., 0.]])
        results[0]["signals"][0]["class_name"] = "Background"
        pool = apply_open_set(scorer, results, rows)
        self.assertFalse(pool)
        self.assertEqual(results[0]["signals"][0]["recognition_status"], "uncertain")
        self.assertEqual(results[0]["signals"][0]["unknown_reason"], "ignored_profile_class")

    def test_offline_backends_share_exact_candidates_and_keep_frequency_metadata(self):
        features = DcgDiscoveryTests.populations()
        records = [{"path": f"file_{i}.dat", "recording_id": f"r{i}", "track_id": "T0001",
                    "start_ms": 50., "end_ms": 90., "frequency_center_offset_mhz": float(i), "rf_center_mhz": 2400.}
                   for i in range(len(features))]
        archive = self.root/"candidates.npz"
        save_npz(archive, {"kind": "unknown_candidates", "schema_version": 1, "feature_version": FEATURE_VERSION,
                           "signature": {"sha256": "fixture"}, "records": records}, features=features)
        reports = []
        for backend in ("hdbscan", "dcg"):
            output = self.root/backend
            reports.append(cluster_run(cluster_parser().parse_args(["--features", str(archive), "--output_dir", str(output),
                                                                    "--unknown_cluster_backend", backend])))
            metadata, arrays = load_npz(output/"unknown_features.npz", "unknown_candidates")
            np.testing.assert_allclose(arrays["features"], normalize_features(features), atol=1e-7)
            self.assertEqual(metadata["records"][5]["frequency_center_offset_mhz"], 5.)
            prototypes, values = load_npz(output/"unknown_prototypes.npz", "unknown_prototypes")
            self.assertEqual(values["features"].shape, (2, 3))
            self.assertIn("frequency_center_offset_mhz", (output/"unknown_assignments.csv").read_text(encoding="utf-8-sig"))
        self.assertEqual(reports[0]["candidate_archive_sha256"], reports[1]["candidate_archive_sha256"])
        self.assertEqual([report["estimated_unknown_class_count"] for report in reports], [2, 2])

    def test_empty_archive_and_duplicate_identity_handling(self):
        archive = self.root/"empty.npz"
        metadata = {"kind": "unknown_candidates", "schema_version": 1, "feature_version": FEATURE_VERSION,
                    "signature": {"sha256": "fixture"}, "records": []}
        save_npz(archive, metadata, features=np.empty((0, 3), dtype=np.float32))
        args = cluster_parser().parse_args(["--features", str(archive), "--output_dir", str(self.root/"empty"),
                                            "--unknown_cluster_backend", "dcg"])
        self.assertEqual(cluster_run(args)["estimated_unknown_class_count"], 0)
        metadata["records"] = [{"path": "a.dat", "track_id": "T0001", "recording_id": "a"}]*2
        duplicate = self.root/"duplicate.npz"
        save_npz(duplicate, metadata, features=np.array([[1., 0, 0], [1., 0, 0]]))
        args.features, args.output_dir = duplicate, self.root/"bad"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            cluster_run(args)
        self.assertFalse(args.output_dir.exists())

    def test_dat_slices_track_known_and_unknown_signals_then_dcg_reclusters_pool(self):
        # Artificial logits/features exercise real DAT loading/tracking/reporting.
        dat = self.root/"capture.dat"
        with dat.open("wb") as stream:
            stream.truncate(10_000_000*4)
        checkpoint, weights, profile = [self.root/name for name in ("classifier.pth", "detector.pt", "profile.npz")]
        saved = {"config": {"classes": {"known": 0}, "mask_img_size": 32}, "model_state_dict": {}}
        torch.save(saved, checkpoint)
        weights.write_bytes(b"fixture")
        args = build_parser().parse_args(["--inputs", str(dat), "--checkpoint_path", str(checkpoint), "--yolo_weights", str(weights),
            "--output_dir", str(self.root/"report"), "--device", "cpu", "--min_votes", "2", "--rf_center_mhz", "2400",
            "--unknown_cluster_backend", "dcg", "--unknown_min_cluster_size", "2", "--unknown_min_recordings", "1"])
        signature = inference_signature(args, resolve_config(saved))
        ref, cal = self.root/"ref.npz", self.root/"cal.npz"
        known_archive(ref, [[1., 0, 0]], "train", signature)
        known_archive(cal, [[1., .02, 0]], "val", signature)
        build_profile(ref, cal, profile, k=1, min_calibration_per_class=1)
        args.open_set_profile = profile
        groups = [{"group_id": i, "status": "ok", "class_name": "known", "class_id": 0, "confidence": 1.,
                   "probabilities": [1.], "selected_group_score": 2., "selected_boxes": [[10, y, 70, y+10], [150, y, 210, y+10]],
                   "embedding": [1., 0., 0.] if i == 0 else [0., 1., 0.]}
                  for i, y in enumerate((100, 270, 350))]
        predictor = Mock()
        predictor.predict.side_effect = [{"status": "ok", "detected_boxes": 6, "groups": copy.deepcopy(groups)} for _ in range(2)]
        report = run(args, predictor)
        self.assertEqual(report["metrics"]["known_tracks"], 1)
        self.assertEqual(report["metrics"]["unknown_candidate_tracks"], 2)
        self.assertEqual(report["metrics"]["estimated_unknown_class_count"], 1)
        self.assertEqual(report["files"][0]["total_slices"], 2)
        self.assertTrue(all(track["observation_count"] == 2 for track in report["files"][0]["signals"]))
        metadata, arrays = load_npz(args.output_dir/"unknown_features.npz", "unknown_candidates")
        self.assertEqual(arrays["features"].shape, (2, 3))
        self.assertIsNotNone(metadata["records"][0]["frequency_center_absolute_mhz"])
        self.assertGreater(metadata["records"][0]["end_ms"], 50.)
        self.assertIn("DCG-Louvain", (args.output_dir/"report.html").read_text(encoding="utf-8"))
        self.assertIn("Unknown_001", (args.output_dir/"tracks.csv").read_text(encoding="utf-8-sig"))
        clustered = cluster_run(cluster_parser().parse_args(["--features", str(args.output_dir/"unknown_features.npz"),
            "--output_dir", str(self.root/"reclustered"), "--unknown_cluster_backend", "hdbscan",
            "--unknown_min_cluster_size", "2", "--unknown_min_samples", "1", "--unknown_min_recordings", "1"]))
        self.assertEqual(clustered["unknown_candidate_tracks"], 2)
