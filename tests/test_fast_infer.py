import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from scripts.evaluate_single_signal import build_parser, run
from tests.fixtures import archive, parse_test_args
from util.eval_common import resolve_config
from util.fast_infer import configure_fast_inference, input_recording_identity
from util.open_set import build_profile, inference_signature
from util.run_context import command_exit_code, execute_run


class Predictor:
    def predict(self, spectrum, domain):
        value = int(spectrum[0, 0])
        if value == 30:
            return {"status": "no_detection", "detected_boxes": 0, "groups": []}
        # Two separate frequency tracks, with independent known/unknown decisions.
        features = [[1., 0., 0.], [0., 1., (value-22)*.001]]
        if value == 10:
            features = [[1., 0., 0.]]
        elif value >= 24:
            features.reverse()
        groups = [{"group_id": i, "status": "ok", "class_name": "known", "class_id": 0,
                   "confidence": 1., "probabilities": [1.], "embedding": feature,
                   "selected_boxes": [[10, 100+i*200, 60, 110+i*200]], "group_type": "frequency"}
                  for i, feature in enumerate(features)]
        return {"status": "ok", "detected_boxes": len(groups), "groups": groups}


class FastInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uav-fast-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root/"inputs"
        self.inputs.mkdir()
        for i in (10, 20, 21, 22, 23, 24, 25, 30):
            cv2.imencode(".png", np.full((512, 750), i, dtype=np.uint8))[1].tofile(self.inputs/f"signal-{i}.png")
        self.checkpoint = self.root/"model.pth"
        torch.save({"config": {"classes": {"known": 0}, "mask_img_size": 32}, "model_state_dict": {}}, self.checkpoint)
        self.yolo = self.root/"detector.pt"
        self.yolo.write_bytes(b"fixture")
        self.profile = self.root/"profile.npz"
        self.cfg = resolve_config({"config": {"classes": {"known": 0}, "mask_img_size": 32}})
        args = self.args("signature")
        self.signature = inference_signature(args, self.cfg)
        ref, cal = self.root/"ref.npz", self.root/"cal.npz"
        archive(ref, [[1., 0., 0.], [1., .03, 0.], [1., -.03, 0.]], ["known"]*3, "reference")
        archive(cal, [[1., i, 0.] for i in (0., .01, -.01, .02, -.02)], ["known"]*5, "calibration")
        # Align fixtures with the actual feature signature, without a trained YOLO.
        from util.open_set import load_npz, save_npz
        for source in (ref, cal):
            metadata, arrays = load_npz(source, "track_features")
            metadata["signature"] = self.signature
            target = source.with_stem(source.stem+"_signed")
            save_npz(target, metadata, **arrays)
        build_profile(ref.with_stem("ref_signed"), cal.with_stem("cal_signed"), self.profile,
                      min_calibration_per_class=5)

    def args(self, output, mode="multi", fast=False, extra=()):
        return parse_test_args(build_parser(), ["--inputs", str(self.inputs), "--checkpoint_path", str(self.checkpoint),
            "--yolo_weights", str(self.yolo), "--open_set_profile", str(self.profile), "--device", "cpu",
            "--unknown_input_mode", mode, "--unknown_min_cluster_size", "2", "--unknown_min_samples", "1",
            "--rf_center_mhz", "2400", "--output_dir", str(self.root/output),
            *(["--fast", "--no-progress"] if fast else ["--no-save_visuals", "--no-save_feature_plots"]), *extra])

    def read_table(self, directory):
        with (directory/"predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_multi_matches_full_labels_clusters_and_frequency_bounds(self):
        full_args, fast_args = self.args("full"), self.args("fast", fast=True)
        full = run(full_args, Predictor())
        with patch("util.signal_eval_report.save_slice_preview", side_effect=AssertionError("preview called")), \
                patch("util.signal_eval_report.write_html_report", side_effect=AssertionError("html called")), \
                patch("util.feature_visualization.write_feature_plots", side_effect=AssertionError("plot called")):
            fast = run(fast_args, Predictor())
        self.assertEqual({p.name for p in fast_args.output_dir.iterdir()}, {"predictions.csv", "unknown_clusters.json", "failure_diagnostics.json"})
        self.assertEqual(full['metrics']['pipeline_diagnostics']['stage_counts'],
                         fast['metrics']['pipeline_diagnostics']['stage_counts'])
        self.assertEqual(json.loads((fast_args.output_dir/'unknown_clusters.json').read_text())['estimated_unknown_class_count'],
                         fast['discovery']['estimated_unknown_class_count'])
        self.assertGreater(fast["discovery"]["estimated_unknown_class_count"], 0)
        self.assertEqual(full["discovery"]["estimated_unknown_class_count"], fast["discovery"]["estimated_unknown_class_count"])
        expected = {(row["path"], track["track_id"]): track for row in full["files"] for track in row["signals"]}
        table = self.read_table(fast_args.output_dir)
        self.assertEqual(len(table), len(expected)+1)  # one no-signal file is retained
        for row in table:
            if not row["track_id"]:
                self.assertEqual(row["label"], "no_signal")
                continue
            track = expected[(row["path"], row["track_id"])]
            self.assertEqual(row["label"], track["display_label"])
            bursts = json.loads(row["bursts_json"])
            self.assertEqual(len(bursts), 1)
            self.assertAlmostEqual(bursts[0]["frequency_low_offset_mhz"], track["frequency_low_offset_mhz"])
            self.assertEqual(row["status"], track["recognition_status"])
            for key in ("frequency_low_offset_mhz", "frequency_high_offset_mhz",
                        "frequency_low_absolute_mhz", "frequency_high_absolute_mhz"):
                self.assertAlmostEqual(float(row[key]), track[key])
        self.assertTrue(any(row["label"].startswith("Unknown_") for row in table))

    def test_spectral_fast_and_full_keep_the_same_assignments_and_bursts(self):
        extra=['--unknown_cluster_backend','spectral','--unknown_bounded_strategy','spectral','--unknown_min_silhouette','0']
        full=run(self.args('spectral_full',extra=extra),Predictor())
        fast_args=self.args('spectral_fast',fast=True,extra=extra+['--save_discovery_cache'])
        fast=run(fast_args,Predictor())
        expected={(r['path'],t['track_id']):t['display_label'] for r in full['files'] for t in r['signals']}
        for row in self.read_table(fast_args.output_dir):
            if row['track_id']:
                self.assertEqual(row['label'],expected[(row['path'],row['track_id'])])
                self.assertTrue(json.loads(row['bursts_json']))
        self.assertEqual(full['discovery']['estimated_unknown_class_count'],fast['discovery']['estimated_unknown_class_count'])
        self.assertTrue((fast_args.output_dir/'unknown_features.npz').is_file())

    def test_single_uses_recording_primary_and_retains_one_row_per_file(self):
        full_args, fast_args = self.args("full-single", "single"), self.args("fast-single", "single", True)
        full, fast = run(full_args, Predictor()), run(fast_args, Predictor())
        self.assertEqual(fast["metrics"]["output_rows"], len(full["files"]))
        table = self.read_table(fast_args.output_dir)
        for row, source in zip(table, full["files"]):
            self.assertEqual(row["label"], source["single_signal_prediction"]["display_label"])
        # A secondary novel track never replaces a known primary in single mode.
        self.assertEqual({row["label"] for row in table}, {"known", "no_signal", "Unknown_001"})
        self.assertEqual(next(row["label"] for row in table if row["path"].endswith("signal-20.png")), "known")

    def test_managed_output_errors_and_progress(self):
        (self.inputs/"broken.png").write_bytes(b"invalid image")
        args = self.args("unused", fast=True, extra=["--progress"])
        del args.output_dir
        stderr = io.StringIO()
        with patch("util.run_context.EXPERIMENT_ROOT", self.root/"runs"), contextlib.redirect_stderr(stderr):
            result = execute_run("infer", args, lambda managed: run(managed, Predictor()))
        output = Path(result["run_dir"])
        self.assertEqual({p.name for p in output.iterdir()}, {"predictions.csv", "run.json", "unknown_clusters.json", "failure_diagnostics.json"})
        self.assertEqual(json.loads((output/"run.json").read_text())["status"], "completed_with_errors")
        self.assertEqual(command_exit_code(result), 2)
        error = next(row for row in self.read_table(output) if row["path"].endswith("broken.png"))
        self.assertEqual(error["label"], "error")
        self.assertIn("Files", stderr.getvalue())
        self.assertIn("slice", stderr.getvalue())
        self.assertIn("Clustering", stderr.getvalue())
        self.assertFalse(args.save_visuals)
        self.assertFalse(args.save_feature_plots)

    def test_fast_does_not_change_profile_signature(self):
        args = self.args("sig", fast=True)
        configure_fast_inference(args)
        self.assertEqual(inference_signature(args, self.cfg), self.signature)

    def test_dat_progress_and_partial_slice_errors_keep_valid_signals(self):
        path = self.root/"capture.dat"
        with path.open("wb") as stream:
            stream.truncate(10_000_000*4)  # two 50-ms windows at 100 MHz
        args = self.args("partial", fast=True, extra=["--progress"])
        args.inputs = [str(path)]
        captured = io.StringIO()
        with patch("util.multi_signal_eval.load_spectrum", side_effect=[
                (np.full((512, 750), 20, dtype=np.uint8), "image"), OSError("slice read failed")]), \
                contextlib.redirect_stderr(captured):
            result = run(args, Predictor())
        table = self.read_table(args.output_dir)
        self.assertEqual(len(table), 2)
        self.assertEqual(result["metrics"]["files_with_slice_errors"], 1)
        self.assertEqual(command_exit_code(result), 2)
        self.assertTrue(all("1 slice(s) failed" in row["error"] for row in table))
        self.assertIn("known", {row["label"] for row in table})
        self.assertIn("slice", captured.getvalue())

    def test_invalid_fast_combinations_fail_before_model_loading(self):
        for options in (["--export_features"], ["--label_from_parent"], ["--signal_mode", "single"]):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "--fast requires"):
                run(self.args("invalid", fast=True, extra=options), Predictor())
        self.assertFalse((self.root/"invalid").exists())

    def test_identity_hash_is_skipped_only_when_unneeded(self):
        row = {"path": str(self.inputs/"signal-10.png"), "signals": [{}]}
        with patch("util.recordings.file_sha256", side_effect=AssertionError("unnecessary hash")):
            self.assertEqual(input_recording_identity(row, {row["path"]: "capture-A"}), "capture-A")
            figure = {"path": str(self.root/"capture-Figure-1.png"), "signals": [{}]}
            self.assertEqual(input_recording_identity(figure, {}), "figure-source:capture")
        with patch("util.recordings.file_sha256", return_value="content-id") as digest:
            self.assertEqual(input_recording_identity(row, {}), "content-id")
            digest.assert_called_once_with(row["path"])

    def test_uncertain_and_unassigned_labels_are_not_invented_classes(self):
        args = self.args("unassigned", fast=True, extra=["--unknown_min_cluster_size", "100"])
        run(args, Predictor())
        self.assertTrue(any(row["label"] == "Unknown_unassigned" for row in self.read_table(args.output_dir)))
        args = self.args("uncertain", fast=True, extra=["--min_votes", "2"])
        # Changing voting settings normally requires a matching regenerated profile.
        with patch("util.open_set.inference_signature", return_value=self.signature):
            run(args, Predictor())
        self.assertTrue(any(row["label"] == "uncertain" for row in self.read_table(args.output_dir)))


if __name__ == "__main__":
    unittest.main()
