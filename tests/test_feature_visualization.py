import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from util.feature_visualization import project_features, write_feature_plots
from util.open_set import FEATURE_VERSION, save_npz


class FeatureVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def records(n):
        return [{"path": f"capture-{i}.dat", "track_id": "T1", "recognition_status": "novel_candidate",
                 "unknown_cluster_id": "Unknown_001" if i % 2 else None, "true_class_names": ["Protocol-A"]}
                for i in range(n)]

    def test_truth_changes_colors_only_not_projection_or_sampling(self):
        x = np.random.default_rng(1).normal(size=(12, 4))
        records = self.records(12)
        first = write_feature_plots(self.root/"first", x, records, max_points=7)
        changed = copy.deepcopy(records)
        for row in changed:
            row["true_class_names"] = ["Another-protocol"]
        second = write_feature_plots(self.root/"second", x, changed, max_points=7)
        with np.load(self.root/"first/pca_coordinates.npz") as a, np.load(self.root/"second/pca_coordinates.npz") as b:
            np.testing.assert_array_equal(a["coordinates"], b["coordinates"])
            np.testing.assert_array_equal(a["source_indices"], b["source_indices"])
        self.assertEqual(first["plotted_tracks"], 7)
        self.assertEqual(second["omitted_tracks"], 5)

    def test_unlabeled_or_multilabel_files_do_not_get_instance_truth(self):
        rows = self.records(4)
        for row in rows:
            row["true_class_names"] = ["A", "B"]
        summary = write_feature_plots(self.root/"out", np.eye(4), rows)
        self.assertFalse(any(plot["file"].endswith("_truth.png") for plot in summary["plots"]))

    def test_empty_singleton_identical_and_one_dimensional_features(self):
        for i, x in enumerate((np.empty((0, 3)), np.ones((1, 3)), np.ones((3, 3)), np.array([[1.], [2.], [3.]]))):
            summary = write_feature_plots(self.root/str(i), x, self.records(len(x)))
            if len(x):
                with np.load(self.root/str(i)/"pca_coordinates.npz") as archive:
                    self.assertEqual(archive["coordinates"].shape, (len(x), 2))
                    self.assertTrue(np.isfinite(archive["coordinates"]).all())
            else:
                self.assertEqual(summary["status"], "skipped")

    def test_small_tsne_fallback_and_real_tsne_are_finite(self):
        xy, info = project_features(np.eye(3), "tsne", 42)
        self.assertEqual(info["method"], "pca")
        self.assertEqual(xy.shape, (3, 2))
        x = np.random.default_rng(5).normal(size=(8, 4))
        xy, info = project_features(x, "tsne", 42)
        self.assertEqual(info["method"], "tsne")
        self.assertTrue(np.isfinite(xy).all())

    def test_invalid_or_misaligned_features_fail(self):
        for x, records in ((np.ones((2, 3)), self.records(1)), (np.array([[np.nan, 0]]), self.records(1))):
            with self.assertRaisesRegex(ValueError, "aligned"):
                write_feature_plots(self.root/"bad", x, records)

    def test_offline_command_handles_old_candidates_and_exact_path_truth(self):
        from scripts.visualize_features import build_parser
        from util.cli import main
        archive = self.root/"candidates.npz"
        rows = [{"path": f"/original/{i}.dat", "track_id": "T1", "unknown_cluster_id": None} for i in range(4)]
        save_npz(archive, {"kind": "unknown_candidates", "schema_version": 1, "feature_version": FEATURE_VERSION,
                          "records": rows}, features=np.eye(4))
        truth = self.root/"results.json"
        truth.write_text(json.dumps({"files": [{"path": row["path"], "true_class_names": ["UAV-A"]} for row in rows]}))
        with patch("util.run_context.EXPERIMENT_ROOT", self.root/"runs"):
            report = main(["visualize", "--features", str(archive), "--results_json", str(truth)])
        output = Path(report["run_dir"])
        self.assertTrue((output/"feature_report.html").exists())
        self.assertTrue((output/"feature_plots/pca_unknown_clusters.png").exists())
        points = json.loads((output/"feature_plots/pca_points.json").read_text())
        self.assertTrue(all(row["recognition_status"] == "novel_candidate" for row in points))
        self.assertTrue(all(row["true_class_names"] == ["UAV-A"] for row in points))
        self.assertNotIn("--output_dir", build_parser()._option_string_actions)


if __name__ == "__main__":
    unittest.main()
