import contextlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from util.cli import build_task_parser, main
from util.run_context import check_output_directory, command_exit_code, create_directory, execute_run, run_type


class RunContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uav-run-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "experiment"
        self.patcher = patch("util.run_context.EXPERIMENT_ROOT", self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_same_second_parallel_runs_are_unique(self):
        fixed = datetime.fromisoformat("2026-10-28T12:34:56+08:00")
        with patch("util.run_context.now", return_value=fixed), ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda _: create_directory("train"), range(4)))
        self.assertEqual(len(set(paths)), 4)
        self.assertTrue(all(p.parent == self.root for p in paths))
        self.assertIn("train_20261028_123456", {p.name for p in paths})

    def test_types_include_training_incremental_validation_and_discovery(self):
        for task, fields, expected in (
            ("train", {}, "train"), ("train", {"incremental_split_mode": "base"}, "base_train"),
            ("train", {"trainer_type": "lwf"}, "incremental_train"),
            ("train", {"run_mode": "infer"}, "validation"),
            ("predict", {"label_from_parent": True}, "validation"),
            ("predict", {"open_set_profile": "profile.npz"}, "open_set_inference"),
            ("experiment", {}, "open_set_experiment"), ("experiment", {"split_only": True}, "open_set_split"),
            ("cluster", {}, "unknown_clustering"), ("infer", {}, "open_set_inference")):
            with self.subTest(task=task, fields=fields):
                self.assertEqual(run_type(task, SimpleNamespace(**fields)), expected)

    def test_metadata_and_paths_are_automatic(self):
        args = SimpleNamespace(inputs=["capture.dat"])
        def callback(managed):
            self.assertEqual(check_output_directory(managed), managed.output_dir)
            (managed.output_dir / "results.json").write_text("{}")
            return {"metrics": {"error_files": 0}}
        report = execute_run("infer", args, callback)
        self.assertEqual(Path(report["run_dir"]).parent, self.root)
        record = json.loads((args.output_dir / "run.json").read_text())
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["timezone"], "Asia/Shanghai")
        self.assertTrue(record["started_at"].endswith("+08:00"))
        self.assertNotIn("_run_dir", record["arguments"])
        with self.assertRaisesRegex(ValueError, "new or empty"):
            check_output_directory(args)

    def test_failures_interruptions_and_nonfinite_arguments_are_recorded(self):
        for error, expected in ((RuntimeError("broken"), "failed"), (KeyboardInterrupt(), "interrupted")):
            args = SimpleNamespace(value=float("nan"))
            def fail(_):
                raise error
            with self.subTest(expected=expected), self.assertRaises(type(error)):
                execute_run("profile", args, fail)
            record = json.loads((args.output_dir / "run.json").read_text())
            self.assertEqual(record["status"], expected)
            self.assertEqual(record["arguments"]["value"], "nan")
            self.assertTrue((args.output_dir / "error.log").is_file())

    def test_partial_errors_have_an_honest_status_and_exit_code(self):
        args = SimpleNamespace()
        report = execute_run("infer", args, lambda _: {"metrics": {"error_files": 1}})
        self.assertEqual(command_exit_code(report), 2)
        self.assertEqual(json.loads((args.output_dir / "run.json").read_text())["status"], "completed_with_errors")

    def test_all_public_parsers_remove_manual_output_options(self):
        for task in ("train", "experiment", "predict", "infer", "profile", "discover", "cluster"):
            parser = build_task_parser(task)
            options = {flag for action in parser._actions for flag in action.option_strings}
            self.assertFalse(options & {"--output_dir", "--experiment_dir", "--test_output_dir"}, task)
        with self.assertRaisesRegex(ValueError, "Unknown task"):
            build_task_parser("typo")

    def test_dry_run_has_no_filesystem_side_effects(self):
        with contextlib.redirect_stdout(io.StringIO()):
            main(["predict", "--inputs", "missing", "--checkpoint_path", "model.pth", "--yolo_weights", "detector.pt", "--dry_run"])
        self.assertFalse(self.root.exists())

    def test_cli_and_legacy_training_share_the_output_manager(self):
        with patch("util.train_entry.main", return_value=None) as training, contextlib.redirect_stdout(io.StringIO()):
            report = main(["--yolo_weights", "detector.pt"])
        self.assertEqual(Path(report["run_dir"]).parent, self.root)
        self.assertEqual(training.call_args.args[0]._run_dir, Path(report["run_dir"]))

    def test_old_output_yaml_is_rejected(self):
        config = self.root.parent / "old.yaml"
        config.write_text("predict:\n  output_dir: custom\n  inputs: [missing]\n  checkpoint_path: model.pth\n  yolo_weights: detector.pt\n")
        with self.assertRaisesRegex(ValueError, "output_dir"):
            main(["predict", "--config", str(config), "--dry_run"])
        self.assertFalse(self.root.exists())
