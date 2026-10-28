"""Folder-derived DAT labels must survive splits, checkpoints and inference."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from data.iq_dat_loader import (
    RawIQDataset, create_iq_train_val_loaders, discover_iq_classes,
)
from util.checkpoint import load_training_checkpoint, save_training_checkpoint
from util.config import Config
from util.eval_common import resolve_config
from util.train_entry import build_parser, main, setup_incremental_class_split


class IQClassTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uav-iq-labels-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch("util.run_context.EXPERIMENT_ROOT", self.root/"runs")
        patcher.start()
        self.addCleanup(patcher.stop)

    def recording(self, split, name, index=0):
        path = self.root / split / name / f"capture_{index}.dat"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Index real-length 50 ms files without constructing spectrograms here.
        # Numerical conversion and crop behavior are covered by frontend tests.
        with path.open("wb") as stream:
            stream.write(f"{split}/{name}/{index}".encode("utf-8"))
            stream.truncate(100_000_000 // 20 * 4)
        return path

    def args(self, *extra):
        return build_parser().parse_args([
            "--yolo_weights", str(self.root / "detector.pt"),
            "--input_type", "iq_dat", *extra,
        ])

    def config(self, *extra):
        with patch("util.config.os.makedirs"):
            return Config(self.args(*extra))

    def test_config_discovers_real_names_and_shares_ids_across_explicit_splits(self):
        self.recording("train", "Device-Z")
        self.recording("train", "设备甲")
        self.recording("val", "设备甲")
        cfg = self.config("--train_dataset_path", str(self.root / "train"),
                          "--val_dataset_path", str(self.root / "val"),
                          "--dataset_path", str(self.root / "unused-path"))
        self.assertEqual(cfg.classes, {"Device-Z": 0, "设备甲": 1})
        setup_incremental_class_split(cfg)
        self.assertEqual(cfg.num_classes, 2)
        train = RawIQDataset(cfg, dataset_path=cfg.train_dataset_path)
        val = RawIQDataset(cfg, dataset_path=cfg.val_dataset_path, split_name="val")
        self.assertEqual({s["class_name"]: s["label"] for s in train.samples}, cfg.classes)
        self.assertEqual([(s["class_name"], s["label"]) for s in val.samples], [("设备甲", 1)])
        self.assertNotIn("Background", cfg.classes)

    def test_multiple_roots_class_directories_and_files_have_stable_mapping(self):
        z = self.recording("first", "Device-Z")
        a = self.recording("second", "Device-A")
        paths = [self.root / "first", a.parent, z]
        self.assertEqual(discover_iq_classes(paths), {"Device-A": 0, "Device-Z": 1})
        self.assertEqual(discover_iq_classes(paths[::-1]), discover_iq_classes(paths))
        cfg = self.config("--dataset_path", *(str(p) for p in paths))
        self.assertEqual(cfg.classes, {"Device-A": 0, "Device-Z": 1})

    def test_auto_split_retains_shared_ids_and_separates_recordings(self):
        for name in ("Device-Z", "Device-A"):
            for i in range(4):
                self.recording("all", name, i)
        cfg = self.config("--dataset_path", str(self.root / "all"),
                          "--val_ratio", "0.5", "--num_workers", "0")
        setup_incremental_class_split(cfg)
        train, val = create_iq_train_val_loaders(cfg)
        train_samples = [train.dataset.dataset.samples[i] for i in train.dataset.indices]
        val_samples = [val.dataset.dataset.samples[i] for i in val.dataset.indices]
        self.assertFalse({s["fp"] for s in train_samples} & {s["fp"] for s in val_samples})
        for samples in (train_samples, val_samples):
            self.assertEqual({s["label"] for s in samples}, {0, 1})
            self.assertTrue(all(s["label"] == cfg.classes[s["class_name"]] for s in samples))

    def test_exclusion_and_incremental_splits_use_actual_folder_names(self):
        for split in ("train", "val"):
            for name in ("Device-A", "Device-Z", "Held-out"):
                self.recording(split, name)
        for mode, classes, train_names in (
            ("normal", {"Device-A": 0, "Device-Z": 1}, {"Device-A", "Device-Z"}),
            ("base", {"Device-A": 0}, {"Device-A"}),
            ("incremental", {"Device-A": 0, "Device-Z": 1}, {"Device-Z"}),
        ):
            with self.subTest(mode=mode):
                cfg = self.config("--train_dataset_path", str(self.root / "train"),
                                  "--val_dataset_path", str(self.root / "val"),
                                  "--exclude_classes", "Held-out", "--incremental_split_mode", mode,
                                  "--incremental_new_classes", "Device-Z")
                setup_incremental_class_split(cfg)
                self.assertEqual(cfg.classes, classes)
                self.assertEqual(cfg.num_classes, len(classes))
                train = RawIQDataset(cfg, dataset_path=cfg.train_dataset_path)
                val = RawIQDataset(cfg, dataset_path=cfg.val_dataset_path, split_name="val")
                self.assertEqual({s["class_name"] for s in train.samples}, train_names)
                self.assertEqual({s["class_name"] for s in val.samples}, set(classes))

    def test_manual_class_order_cannot_silently_invent_or_drop_dat_classes(self):
        self.recording("all", "Device-A")
        self.recording("all", "Device-Z")
        base = ("--dataset_path", str(self.root / "all"), "--class_names")
        for names in (("Lightbridge", "Ocusync2"), ("Device-A",),
                      ("Device-A", "Device-Z", "Imaginary"), ("Device-A", "Device-A")):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "class_names"):
                self.config(*base, *names)
        cfg = self.config(*base, "Device-Z", "Device-A")
        self.assertEqual(cfg.classes, {"Device-Z": 0, "Device-A": 1})

    def test_incomplete_explicit_split_fails_before_falling_back_to_dataset_path(self):
        self.recording("all", "Device-A")
        for option in ("--train_dataset_path", "--val_dataset_path"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "Provide both"):
                self.config("--dataset_path", str(self.root / "all"), option, str(self.root / "all"))

    def test_loader_reports_unmapped_classes_but_respects_intentional_filters(self):
        self.recording("all", "Device-A")
        self.recording("all", "Device-Z")
        cfg = SimpleNamespace(classes={"Device-A": 0}, dataset_path=[str(self.root / "all")])
        with self.assertRaisesRegex(ValueError, "Device-Z.*missing from config.classes"):
            RawIQDataset(cfg)
        cfg.train_data_exclude_classes = ["Device-Z"]
        self.assertEqual(len(RawIQDataset(cfg)), 1)
        cfg.train_data_exclude_classes = []
        cfg.train_data_include_classes = ["Device-A"]
        self.assertEqual(len(RawIQDataset(cfg)), 1)

    def test_folder_mapping_is_saved_and_reused_by_resume_and_inference(self):
        self.recording("all", "Device-A")
        self.recording("all", "Device-Z")
        cfg = self.config("--dataset_path", str(self.root / "all"),
                          "--class_names", "Device-Z", "Device-A")
        setup_incremental_class_split(cfg)
        model = torch.nn.Linear(3, cfg.num_classes)
        checkpoint = save_training_checkpoint(model, None, None, 0, .5, self.root / "models", config=cfg)
        restored = torch.nn.Linear(3, cfg.num_classes)
        saved = load_training_checkpoint(restored, checkpoint, "cpu", expected_classes=cfg.classes)
        self.assertEqual(saved["config"]["classes"], cfg.classes)
        inferred = resolve_config(saved)
        self.assertEqual(inferred.classes, cfg.classes)
        self.assertEqual(inferred.num_classes, 2)
        with self.assertRaisesRegex(ValueError, "class mapping"):
            load_training_checkpoint(restored, checkpoint, "cpu",
                                     expected_classes={"Device-A": 0, "Device-Z": 1})

    def test_mat_defaults_and_explicit_order_remain_available(self):
        for names in (None, ["Custom-B", "Custom-A"]):
            args = SimpleNamespace(input_type="mat", class_names=names)
            with self.subTest(names=names), patch("util.config.os.makedirs"):
                cfg = Config(args)
            if names is None:
                self.assertEqual(cfg.classes["Lightbridge"], 1)
            else:
                self.assertEqual(cfg.classes, {"Custom-B": 0, "Custom-A": 1})

    def test_training_checks_yolo_submodule_at_project_root_after_entry_move(self):
        (self.root / "yolov5").mkdir()
        (self.root / "yolov5" / "hubconf.py").write_text("# submodule fixture\n")
        (self.root / "detector.pt").write_bytes(b"path validation fixture")
        args = self.args()
        class ReachedConfig(Exception):
            pass
        with patch("util.train_entry.__file__", str(self.root / "util" / "train_entry.py")), \
             patch("util.train_entry.Config", side_effect=ReachedConfig), \
             self.assertRaises(ReachedConfig):
            main(args)
