import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.compare_open_set import build_parser
from tests.test_ood_ablation import split
from util.open_set import FEATURE_VERSION, save_npz
from util.open_set_ablation import run


class AblationPipelineTests(unittest.TestCase):
    def test_managed_frozen_matrix_counts_missing_records_and_holds_pool_fixed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/"source"
            ref = split([[1, .1, 0], [1, -.1, 0], [.1, 1, 0], [-.1, 1, 0]], ["a", "a", "b", "b"], "ref")
            cal = split([[1, .15, 0], [1, -.15, 0], [.15, 1, 0], [-.15, 1, 0]], ["a", "a", "b", "b"], "cal")
            test = split([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.01, 0, 1], [-.01, 0, 1]], ["a", "b", "u", "u", "u"], "test")
            for data in (ref, cal, test):
                for row, record in zip(data.report["files"], data.records):
                    name = record["true_class_name"] if record["true_class_name"] in data.classes else "a"
                    record["head_class"] = name
                    row["signals"][0].update(status="ok", recognition_status="known", class_name=name,
                                             known_candidate_class=name, candidate_class_name=name, ood_score=.1)
            test.report["files"].append({"path": "/test/missing.dat", "true_class_names": ["a"], "signals": []})
            test.report["discovery"] = {"estimated_unknown_class_count": 0, "backend": "hdbscan"}
            for relative, data, archive in (("open_set/reference", ref, "features.npz"),
                                             ("open_set/calibration", cal, "features.npz"),
                                             ("test", test, "inference_features.npz")):
                directory = source/relative
                directory.mkdir(parents=True)
                metadata = {**data.metadata, "records": data.records, "schema_version": 1,
                            "kind": "track_features", "feature_version": FEATURE_VERSION}
                save_npz(directory/archive, metadata, features=data.features)
                (directory/"results.json").write_text(json.dumps(data.report), encoding="utf-8")
            args = build_parser().parse_args(["--source_experiment", str(source), "--methods", "record_knn",
                "--calibration_units", "primary", "record_max", "--known_false_reject_rates", ".01", ".05",
                "--knn_k", "1", "--unknown_min_cluster_size", "2", "--unknown_min_samples", "1",
                "--unknown_min_recordings", "1"])
            output = root/"managed"
            with patch("util.run_context.create_directory", return_value=output):
                output.mkdir()
                result = run(args)
            summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["metrics"]["completed_cases"], 9)
            self.assertEqual(summary["cases"][0]["known_recordings"], 3)
            self.assertEqual(summary["cases"][0]["unknown_recordings"], 3)
            cases = [r for r in summary["cases"] if r["stage"] == "R4"]
            self.assertEqual(len(cases), 3)
            self.assertEqual(len({r["unknown_detection_recall"] for r in cases}), 1)
            self.assertTrue(all(r["unit"] == "primary" for r in cases))
            hashes = [json.loads((output/"cases"/r["name"]/"clusters.json").read_text())["candidate_pool_sha256"] for r in cases]
            self.assertEqual(len(set(hashes)), 1)
            self.assertEqual(json.loads((output/"run.json").read_text())["status"], "complete")
            for name in ("summary.csv", "per_class.csv", "audit.json", "report.html", "roc.png", "performance.png"):
                self.assertTrue((output/name).is_file())


if __name__ == "__main__":
    unittest.main()
