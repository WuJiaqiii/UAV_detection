from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from model.resnet import MaskImageClassifier
from util.checkpoint import load_training_checkpoint
from util.lwf_trainer import LwFTrainer
from util.recordings import train_val_indices


class AuditRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="uav-audit-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(2)

    def samples(self):
        samples = []
        for label in (0, 1):
            for index in range(6):
                fp = self.root / f"Class{label}-[{index},1,0,2]-SNR-1-Figure-1.mat"
                fp.write_bytes(f"{label}:{index}".encode())
                samples.append({"fp": str(fp), "label": label})
        # A differently named duplicate must not cross the train/val boundary.
        copy = self.root / "alias.mat"
        copy.write_bytes(Path(samples[0]["fp"]).read_bytes())
        samples.append({"fp": str(copy), "label": 0})
        return samples

    def test_auto_split_is_stratified_seeded_and_alias_safe(self):
        samples = self.samples()
        train, val = train_val_indices(samples, .5, 7)
        repeated = train_val_indices(samples, .5, 7)
        self.assertEqual(train.tolist(), repeated[0].tolist())
        self.assertNotEqual(train.tolist(), train_val_indices(samples, .5, 8)[0].tolist())
        self.assertEqual((0 in train), (12 in train))
        self.assertEqual({samples[i]["label"] for i in train}, {0, 1})
        self.assertEqual({samples[i]["label"] for i in val}, {0, 1})

    def test_one_original_recording_cannot_be_split_by_its_exports(self):
        samples = []
        for index in (1, 2):
            fp = self.root / f"A-[0,1,0,2]-SNR-1-Figure-{index}.mat"
            fp.write_bytes(str(index).encode())
            samples.append({"fp": str(fp), "label": 0})
        with self.assertRaisesRegex(ValueError, "two independent"):
            train_val_indices(samples, .5)

    def test_explicit_recording_ids_keep_differently_named_exports_together(self):
        import os
        samples = self.samples()
        mapping = {samples[0]["fp"]: "capture-A", samples[1]["fp"]: "capture-A"}
        train, _ = train_val_indices(samples, .5, recording_ids=mapping)
        self.assertEqual((0 in train), (1 in train))
        relative = [{**sample, "fp": os.path.relpath(sample["fp"])} for sample in samples]
        self.assertEqual(train.tolist(), train_val_indices(relative, .5, recording_ids=mapping)[0].tolist())

    def test_regular_checkpoint_cannot_silently_leave_random_parameters(self):
        model = torch.nn.Linear(3, 2)
        checkpoint = self.root / "incomplete.pth"
        torch.save({"model_state_dict": {"weight": model.weight.detach()}}, checkpoint)
        with self.assertRaises(RuntimeError):
            load_training_checkpoint(model, checkpoint, "cpu")

    def test_lwf_teacher_class_order_is_checked_before_model_mutation(self):
        checkpoint = self.root / "wrong_teacher.pth"
        torch.save({"model_state_dict": {}, "config": {"classes": {"old_b": 0, "old_a": 1}}}, checkpoint)
        config = SimpleNamespace(lwf_old_num_classes=2, num_classes=3, lwf_teacher_checkpoint=str(checkpoint),
                                 checkpoint_path=None, classes={"old_a": 0, "old_b": 1, "new": 2})
        classifier = Mock()
        with self.assertRaisesRegex(ValueError, "class mapping"):
            LwFTrainer(config, None, Mock(), Mock(), Mock(), classifier)
        classifier.load_state_dict.assert_not_called()

    def test_corrupt_optimizer_state_cannot_be_silently_ignored_on_resume(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters())
        checkpoint = self.root / "bad_optimizer.pth"
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": {"state": {}, "param_groups": []}}, checkpoint)
        with self.assertRaisesRegex(RuntimeError, "optimizer state"):
            load_training_checkpoint(model, checkpoint, "cpu", optimizer=optimizer, load_optimizer=True)

    def test_frozen_backbone_does_not_update_batchnorm_statistics(self):
        model = MaskImageClassifier("resnet18", 2, 1, pretrained=False, freeze_backbone=True).train()
        before = model.net.bn1.running_mean.clone()
        model(torch.randn(2, 1, 32, 32)).sum().backward()
        torch.testing.assert_close(model.net.bn1.running_mean, before)
        self.assertFalse(model.net.bn1.training)
        self.assertTrue(model.net.fc.training)
        self.assertIsNotNone(model.net.fc[-1].weight.grad)

    def test_bad_training_options_fail_before_dataset_or_model_work(self):
        from util.train_entry import build_parser, validate_training_options
        for option, value in (("save_interval", "0"), ("lr", "nan"), ("seed", "-1")):
            args = build_parser().parse_args(["--yolo_weights", "unused.pt", "--" + option, value])
            with self.subTest(option=option), self.assertRaises(ValueError):
                validate_training_options(args)
