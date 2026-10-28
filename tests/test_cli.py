from tests.fixtures import parse_test_args
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

from util.cli import build_task_parser, load_settings, main, run_profile, settings_tokens

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uav-cli-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_root = patch("util.run_context.EXPERIMENT_ROOT", self.root/"runs")
        self.run_root.start()
        self.addCleanup(self.run_root.stop)

    def config(self, value):
        path = self.root/"experiment.yaml"
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        return path

    def dry(self, argv):
        with contextlib.redirect_stdout(io.StringIO()):
            return main([*argv, "--dry_run"])

    def test_yaml_paths_lists_boolean_and_cli_precedence(self):
        path = self.config({"evaluation": {"checkpoint_path": "weights/model.pth", "yolo_weights": "weights/yolo.pt",
            "recursive": True, "grouping_mode": "hopping"}, "predict": {"inputs": ["captures/*.dat"],
            "save_visuals": False, "min_votes": 3,
            "settings": {"min_group_len": 1}}})
        args = self.dry(["predict", "--config", str(path), "--min_votes", "2", "--save_visuals", "--no-recursive"])
        self.assertEqual(args.min_votes, 2)
        self.assertTrue(args.save_visuals)
        self.assertFalse(args.recursive)
        self.assertEqual(args.checkpoint_path, self.root/"weights/model.pth")
        self.assertEqual(args.inputs, [str(self.root/"captures/*.dat")])
        self.assertEqual(args.settings, ["min_group_len=1"])

    def test_unknown_keys_and_wrong_types_are_not_silently_ignored(self):
        for document in ({"typo_section": {}}, {"predict": {"inputz": ["x"]}},
                         {"predict": {"inputs": "x"}}, {"predict": {"recursive": "false"}}):
            with self.subTest(document=document):
                path = self.config(document)
                with self.assertRaises(ValueError):
                    settings, _ = load_settings(path, "predict")
                    settings_tokens(build_task_parser("predict"), settings)

    def test_discovery_and_profile_share_feature_options_without_test_label_leakage(self):
        path = self.config({"evaluation": {"checkpoint_path": "model.pth", "yolo_weights": "yolo.pt",
            "iq_sample_rate": 122880000, "min_votes": 2}, "profile": {"reference_inputs": ["train"],
            "calibration_inputs": ["cal"], "label_mode": "parent"},
            "discovery": {"test_inputs": ["test"], "test_labels_json": "truth.json", "unknown_cluster_backend": "dcg"}})
        profile = self.dry(["profile", "--config", str(path)])
        discovery = self.dry(["discover", "--config", str(path)])
        self.assertEqual(profile.iq_sample_rate, discovery.iq_sample_rate)
        self.assertEqual(profile.min_votes, discovery.min_votes)
        self.assertFalse(hasattr(profile, "test_inputs"))
        self.assertIsNone(discovery.labels_json)
        self.assertEqual(discovery.test_labels_json, self.root/"truth.json")

    def test_training_class_order_and_legacy_arguments_are_preserved(self):
        path = self.config({"train": {"yolo_weights": "yolo.pt", "class_names": ["B", "A", "Unknown"],
            "exclude_classes": ["Unknown"], "supcon_weight": .1, "seed": 7}})
        args = self.dry(["train", "--config", str(path)])
        from util.train_entry import setup_incremental_class_split
        config = SimpleNamespace(**vars(args))
        config.classes = {name: i for i, name in enumerate(args.class_names)}
        setup_incremental_class_split(config)
        self.assertEqual(config.classes, {"B": 0, "A": 1})
        self.assertEqual(args.seed, 7)
        with patch("util.train_entry.main") as train:
            main(["--yolo_weights", "legacy.pt", "--run_mode", "train"])
        self.assertEqual(train.call_args.args[0].yolo_weights, "legacy.pt")

    def test_example_configs_parse_for_every_documented_task(self):
        for task, name in (("train", "train.yaml"), ("train", "train_iq.yaml"), ("train", "train_incremental.yaml"), ("predict", "predict.yaml"),
                           ("profile", "open_set.yaml"), ("discover", "open_set.yaml"),
                           ("predict", "open_set.yaml"), ("cluster", "cluster.yaml")):
            with self.subTest(task=task, config=name):
                self.dry([task, "--config", str(ROOT/"config"/name)])

    def test_iq_example_does_not_override_folder_discovery_with_protocol_names(self):
        args = self.dry(["train", "--config", str(ROOT/"config/train_iq.yaml")])
        self.assertEqual(args.input_type, "iq_dat")
        self.assertIsNone(args.class_names)

    def test_offline_cluster_dispatch_does_not_import_torch_or_yolo(self):
        code = ("import sys, contextlib, io; from util.cli import main; "
                "stream=io.StringIO(); "
                "\nwith contextlib.redirect_stdout(stream): main(['cluster','--features','x.npz','--dry_run'])\n"
                "assert 'torch' not in sys.modules; assert 'util.detector' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_profile_rejects_existing_output_and_ambiguous_truth_before_prediction(self):
        parser = build_task_parser("profile")
        base = ["--reference_inputs", "train", "--calibration_inputs", "cal", "--checkpoint_path", "x.pth",
                "--yolo_weights", "y.pt", "--output_dir", str(self.root/"output")]
        args = parse_test_args(parser, [*base, "--label_from_filename"])
        with self.assertRaisesRegex(ValueError, "label_mode"):
            run_profile(args)
        self.assertFalse(args.output_dir.exists())
        args = parse_test_args(parser, base)
        args.output_dir.mkdir()
        (args.output_dir/"existing.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "new or empty"):
            run_profile(args)


class DiscoveryPipelineTests(unittest.TestCase):
    def test_profile_and_discover_reuse_options_and_preserve_novel_truth_and_frequency(self):
        with tempfile.TemporaryDirectory(prefix="uav-pipeline-") as temp:
            root = Path(temp)
            for split, values in (("ref", (10, 11)), ("cal", (12, 13)), ("test", (30, 31))):
                directory = root/split/"known"
                directory.mkdir(parents=True)
                for value in values:
                    cv2.imencode(".png", np.full((512, 750), value, dtype=np.uint8))[1].tofile(directory/f"{value}.png")
            checkpoint, yolo = root/"model.pth", root/"yolo.pt"
            torch.save({"model_state_dict": {}, "config": {"classes": {"known": 0}, "mask_img_size": 32}}, checkpoint)
            yolo.write_bytes(b"synthetic fixture, not trained weights")
            truth = root/"truth.json"
            truth.write_text(json.dumps({str(p): "unseen" for p in (root/"test/known").glob("*.png")}))
            parser = build_task_parser("discover")
            args = parse_test_args(parser, ["--reference_inputs", str(root/"ref"), "--calibration_inputs", str(root/"cal"),
                "--test_inputs", str(root/"test"), "--recursive", "--label_mode", "parent", "--checkpoint_path", str(checkpoint),
                "--yolo_weights", str(yolo), "--output_dir", str(root/"report"), "--knn_k", "1",
                "--min_calibration_per_class", "2", "--unknown_cluster_backend", "dcg", "--unknown_min_cluster_size", "2",
                "--unknown_min_samples", "1", "--unknown_min_recordings", "2", "--test_labels_json", str(truth), "--rf_center_mhz", "2400"])

            class Predictor:
                def predict(self, spectrum, domain):
                    value = float(spectrum[0, 0])
                    embedding = [0., 1., 0.] if value >= 30 else [1., value*1e-4, 0.]
                    return {"status": "ok", "detected_boxes": 2, "groups": [{"group_id": 0, "status": "ok",
                        "class_name": "known", "class_id": 0, "confidence": 1., "probabilities": [1.],
                        "selected_boxes": [[10, 100, 50, 110], [100, 100, 140, 110]],
                        "selected_group_score": 1., "embedding": embedding}]}

            report = run_profile(args, discover=True, predictor=Predictor())
            self.assertTrue((args.output_dir/"known_profile.npz").is_file())
            self.assertTrue((args.output_dir/"reference/features.npz").is_file())
            self.assertTrue((args.output_dir/"calibration/features.npz").is_file())
            self.assertEqual(report["metrics"]["estimated_unknown_class_count"], 1)
            self.assertEqual(report["metrics"]["unknown_candidate_tracks"], 2)
            self.assertEqual(report["metrics"]["file_unknown_presence_metrics"]["recall"], 1.)
            self.assertIn("Unknown_001", (args.output_dir/"discovery/tracks.csv").read_text(encoding="utf-8-sig"))
            self.assertTrue(all(row["signals"][0]["frequency_center_absolute_mhz"] is not None for row in report["files"]))
