import copy
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from torch import nn

from model.representation import (
    SupervisedContrastiveLoss, augment_group_boxes, contrastive_mask_batch, validate_representation_options,
)
from model.resnet import MaskImageClassifier
from util.single_signal_eval import resolve_config
from util.trainer import Trainer
from util.lwf_trainer import LwFTrainer


class ContrastiveLossTests(unittest.TestCase):
    def test_same_class_alignment_is_preferred_and_gradients_are_finite(self):
        aligned = torch.tensor([[[1., 0.], [1., .02]], [[1., -.02], [1., .01]],
                                [[0., 1.], [.01, 1.]], [[-.01, 1.], [.02, 1.]]], requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        criterion = SupervisedContrastiveLoss(.2)
        good = criterion(aligned, labels)
        bad = criterion(aligned, torch.tensor([0, 1, 0, 1]))
        self.assertLess(float(good.detach()), float(bad.detach()))
        good.backward()
        self.assertTrue(torch.isfinite(aligned.grad).all())
        self.assertGreater(float(aligned.grad.abs().sum()), 0.)

    def test_no_positive_pairs_and_empty_batch_give_connected_zero(self):
        for features, labels in ((torch.randn(3, 1, 8, requires_grad=True), torch.arange(3)),
                                 (torch.empty(0, 2, 8, requires_grad=True), torch.empty(0, dtype=torch.long))):
            with self.subTest(shape=features.shape):
                loss = SupervisedContrastiveLoss()(features, labels)
                self.assertEqual(float(loss.detach()), 0.)
                loss.backward()
                self.assertIsNotNone(features.grad)
                self.assertTrue(torch.isfinite(features.grad).all())

    def test_two_views_supply_positives_for_unique_sample_classes(self):
        features = torch.randn(4, 2, 16, requires_grad=True)
        loss = SupervisedContrastiveLoss()(features, torch.arange(4))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(float(loss.detach()), 0.)

    def test_low_temperature_does_not_overflow(self):
        features = torch.randn(4, 2, 32, requires_grad=True)
        loss = SupervisedContrastiveLoss(1e-4)(features, torch.tensor([0, 0, 1, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_invalid_options_and_frozen_encoder_are_rejected(self):
        for options in ({"supcon_weight": -1}, {"supcon_temperature": float("nan")},
                        {"supcon_drop_prob": 2}, {"supcon_jitter_px": -1},
                        {"supcon_weight": .1, "freeze_backbone": True},
                        {"supcon_weight": .1, "lwf_freeze_backbone": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_representation_options(SimpleNamespace(**options))


class MaskAugmentationTests(unittest.TestCase):
    def test_drop_is_capped_and_thin_rectangles_keep_their_shape(self):
        boxes = np.array([[i*20, 100+i, i*20+2+i, 102+i] for i in range(8)])
        config = SimpleNamespace(supcon_drop_prob=1., supcon_max_drop_fraction=.25, supcon_jitter_px=2)
        augmented = augment_group_boxes(boxes, (512, 750), config, np.random.default_rng(4))
        self.assertEqual(len(augmented), 6)
        original_shapes = {tuple(row) for row in boxes[:, 2:] - boxes[:, :2]}
        self.assertTrue(all(tuple(row) in original_shapes for row in augmented[:, 2:] - augmented[:, :2]))
        self.assertTrue((augmented[:, 0] >= 0).all())
        self.assertTrue((augmented[:, 2] <= 750).all())

    def test_singleton_and_pair_are_not_dropped_and_boundaries_are_safe(self):
        config = SimpleNamespace(supcon_drop_prob=1., supcon_max_drop_fraction=.5, supcon_jitter_px=8)
        boxes = np.array([[0, 0, 2, 2], [748, 510, 750, 512]])
        for size in (1, 2):
            augmented = augment_group_boxes(boxes[:size], (512, 750), config, np.random.default_rng(6))
            self.assertEqual(len(augmented), size)
            np.testing.assert_array_equal(augmented[:, 2:]-augmented[:, :2], np.full((size, 2), 2))

    def test_augmented_classifier_inputs_remain_binary_and_native_shape_is_used(self):
        config = resolve_config({"config": {"classes": {"a": 0}, "mask_img_size": 32}})
        config.supcon_jitter_px, config.supcon_drop_prob = 1, .1
        metas = [{"boxes": [[10, 100, 70, 110], [200, 100, 260, 110]], "spec_shape": (512, 750)}]
        batch = contrastive_mask_batch(metas, config)
        self.assertEqual(tuple(batch.shape), (1, 1, 32, 32))
        self.assertTrue(set(batch.unique().tolist()) <= {0., 1.})
        self.assertGreater(float(batch.sum()), 0.)


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        np.random.seed(42)
        self.temp = tempfile.TemporaryDirectory(prefix="uav-representation-", dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self, weight=.1, classes=None):
        config = resolve_config({"config": {"classes": classes or {"a": 0, "b": 1}, "mask_img_size": 32}})
        config.device = torch.device("cpu")
        for key, value in dict(lr=.001, weight_decay=0., log_dir=str(self.root/"log"), result_dir=str(self.root/"result"),
                               eval_exclude_classes=[], val_detect_vis_ratio=0., infer_detect_vis_ratio=0.,
                               checkpoint_path=None, run_mode="train", supcon_weight=weight, supcon_temperature=.2,
                               supcon_drop_prob=.1, supcon_jitter_px=1, supcon_max_drop_fraction=.25).items():
            setattr(config, key, value)
        return config

    def make_trainer(self, config, trainer_type=Trainer):
        count = 4
        targets = [[{"label": i % config.num_classes, "has_frequency_label": False}] for i in range(count)]
        batch = (torch.zeros(count, 512, 750), targets, torch.zeros(count), [f"capture-{i}.mat" for i in range(count)])
        detector, preprocessor = Mock(), Mock()
        boxes = np.array([[10, 100, 70, 110], [150, 100, 210, 110], [300, 100, 360, 110]])
        detector.detect.return_value = boxes
        preprocessor.select_signal_groups.return_value = [{"boxes": boxes, "score": 2.}]
        classifier = MaskImageClassifier("resnet18", config.num_classes, 1, pretrained=False)
        trainer = trainer_type(config, ([batch], [batch]), logging.getLogger("repr-test"), detector, preprocessor, classifier)
        self.addCleanup(trainer.writer.close)
        return trainer

    def test_real_detection_group_mask_training_step_and_checkpoint_round_trip(self):
        config = self.config()
        trainer = self.make_trainer(config)
        before = trainer.classifier.net.layer4[0].conv1.weight.detach().clone()
        loss, accuracy, matched = trainer.train_one_epoch(0)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(matched, 1.)
        self.assertFalse(torch.equal(before, trainer.classifier.net.layer4[0].conv1.weight))
        checkpoint = self.root/"trained.pth"
        torch.save({"config": vars(config), "model_state_dict": trainer.classifier.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict()}, checkpoint)
        clone = MaskImageClassifier("resnet18", 2, 1, pretrained=False)
        clone.load_state_dict(torch.load(checkpoint, weights_only=False)["model_state_dict"], strict=True)
        torch.testing.assert_close(clone.net.layer4[0].conv1.weight, trainer.classifier.net.layer4[0].conv1.weight)

    def test_zero_weight_preserves_original_ce_forward(self):
        trainer = self.make_trainer(self.config(weight=0.))
        trainer.classifier.eval()
        images, labels = torch.randn(3, 1, 32, 32), torch.tensor([0, 1, 0])
        expected_logits = trainer.classifier(images)
        expected_loss = nn.functional.cross_entropy(expected_logits, labels)
        logits, loss, ce, contrast = trainer._training_objective(images, labels, metas=None)
        torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        self.assertEqual(float(contrast), 0.)

    def test_all_components_expand_training_only_with_equal_record_weight(self):
        config = self.config(weight=0.)
        config.train_group_policy = "all"
        trainer = self.make_trainer(config)
        boxes = np.array([[10, 100, 70, 110], [150, 100, 210, 110]])
        trainer.preprocessor.select_signal_groups.return_value = [
            {"boxes": boxes, "score": 2.}, {"boxes": boxes+[0, 100, 0, 100], "score": 1.}]
        inputs, targets, _, fps = next(iter(trainer.train_loader))
        train = trainer._build_single_instances(inputs, targets, fps, split_name="train")
        val = trainer._build_single_instances(inputs, targets, fps, split_name="val")
        self.assertEqual(len(train[1]), 8)
        self.assertEqual(len(val[1]), 4)
        self.assertEqual(train[3:5], (4, 4))
        for fp in fps:
            self.assertEqual(sum(m["loss_weight"] for m in train[2] if m["fp"] == fp), 1.)

    def test_all_component_ce_is_invariant_to_duplicated_identical_component(self):
        config = self.config(weight=0.)
        config.train_group_policy = "all"
        trainer = self.make_trainer(config)
        trainer.classifier.eval()
        images = torch.rand(2, 1, 32, 32)
        labels = torch.tensor([0, 1])
        before = trainer._training_objective(images, labels, [{"loss_weight": 1.}]*2)[1]
        duplicate = images[[0, 0, 1]]
        after = trainer._training_objective(duplicate, labels[[0, 0, 1]],
            [{"loss_weight": .5}, {"loss_weight": .5}, {"loss_weight": 1.}])[1]
        torch.testing.assert_close(before, after)

    def test_all_component_supcon_training_step_has_finite_backbone_gradients(self):
        config = self.config(weight=.1)
        config.train_group_policy = "all"
        trainer = self.make_trainer(config)
        before = trainer.classifier.net.layer4[0].conv1.weight.detach().clone()
        loss, _, _ = trainer.train_one_epoch(0)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, trainer.classifier.net.layer4[0].conv1.weight))

    def test_supcon_reaches_backbone_gradients_and_one_sample_is_safe(self):
        trainer = self.make_trainer(self.config())
        images = torch.rand(4, 1, 32, 32)
        metas = [{"boxes": [[10, 100, 70, 110], [150, 100, 210, 110]], "spec_shape": (512, 750)}]*4
        _, loss, ce, contrast = trainer._training_objective(images, torch.tensor([0, 0, 1, 1]), metas)
        gradient = torch.autograd.grad(contrast, trainer.classifier.net.conv1.weight, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)
        _, single_loss, _, _ = trainer._training_objective(images[:1], torch.tensor([0]), metas[:1])
        single_loss.backward()
        self.assertTrue(torch.isfinite(single_loss))

    def test_lwf_uses_the_shared_representation_path(self):
        config = self.config(classes={"old_a": 0, "old_b": 1, "new": 2})
        teacher = MaskImageClassifier("resnet18", 2, 1, pretrained=False)
        checkpoint = self.root/"teacher.pth"
        torch.save({"model_state_dict": teacher.state_dict(), "config": {"classes": {"old_a": 0, "old_b": 1}}}, checkpoint)
        config.lwf_teacher_checkpoint = str(checkpoint)
        config.lwf_old_num_classes, config.lwf_temperature, config.lwf_lambda_kd = 2, 2., 1.
        trainer = self.make_trainer(config, LwFTrainer)
        before = trainer.classifier.net.conv1.weight.detach().clone()
        result = trainer.train_one_epoch(0)
        self.assertTrue(np.isfinite(result[0]))
        self.assertFalse(torch.equal(before, trainer.classifier.net.conv1.weight))
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.teacher.parameters()))

    def test_resume_selects_a_best_checkpoint_within_the_new_run(self):
        config = self.config(weight=0.)
        first = self.make_trainer(config)
        self.addCleanup(first.writer.close)
        checkpoint = self.root / "resume.pth"
        torch.save({"model_state_dict": first.classifier.state_dict(), "config": {"classes": config.classes},
                    "epoch": 0, "best_val_acc": 99.}, checkpoint)
        config.checkpoint_path = str(checkpoint)
        resumed = self.make_trainer(config)
        self.addCleanup(resumed.writer.close)
        self.assertEqual(resumed.start_epoch, 1)
        self.assertEqual(resumed.resume_best_val_acc, 99.)
        self.assertEqual(resumed.best_val_acc, -1.)
        config.epochs, config.save_interval, config.early_stop_patience = 2, 5, 10
        config.model_dir = str(self.root / "continuation_models")
        resumed.train_one_epoch = Mock(return_value=(1., 42., 1.))
        resumed.validate = Mock(return_value=(1., 42., 1.))
        resumed.train()
        self.assertTrue((Path(config.model_dir) / "best.pth").is_file())

    def test_resume_cannot_silently_run_zero_epochs(self):
        trainer = Trainer.__new__(Trainer)
        trainer.train_loader, trainer.val_loader = object(), object()
        trainer.start_epoch = 2
        trainer.config = type("Config", (), {"epochs": 2})()
        with self.assertRaisesRegex(ValueError, "greater than checkpoint"):
            trainer.train()
