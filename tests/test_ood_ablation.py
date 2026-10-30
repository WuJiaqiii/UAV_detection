import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from util.ablation_data import FrozenSplit, validate_splits
from util.ood_scores import FittedScore, calibrate, record_weights
from util.open_set import normalize_features


def split(values, labels, prefix="r", classes=None, logits=None, raw=None, identities=None, votes=None):
    classes = classes or {"a": 0, "b": 1}
    records, files = [], []
    for i, name in enumerate(labels):
        path = f"/{prefix}/{i}.dat"
        identity = identities[i] if identities is not None else f"{prefix}-{i}"
        vote = votes[i] if votes is not None else 1
        records.append({"path": path, "track_id": "T1", "recording_id": identity,
                        "source_sha256": f"sha-{prefix}-{i}", "true_class_name": name,
                        "head_class": "a", "valid_votes": vote, "observation_count": vote})
        files.append({"path": path, "true_class_names": [name], "signals": [
            {"track_id": "T1", "valid_votes": vote, "observation_count": vote}]})
    metadata = {"classes": classes, "signature": {"sha256": "same-frozen-model"}}
    return FrozenSplit(Path(prefix), metadata, normalize_features(values), records, {"files": files}, logits, raw)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.data = split([[1, 0]]*4, ["a"]*4, identities=["one", "one", "two", "two"], votes=[2, 1, 2, 1])

    def test_primary_and_max_have_different_objectives(self):
        values = [.1, .9, .2, .8]
        primary = calibrate(values, self.data, "primary", .1)
        maximum = calibrate(values, self.data, "record_max", .1)
        self.assertEqual(primary["threshold"], .2)
        self.assertEqual(maximum["threshold"], .9)
        self.assertEqual(primary["calibration_values"], 2)

    def test_each_record_has_equal_total_weight(self):
        weights = record_weights(self.data.records)
        self.assertAlmostEqual(weights[:2].sum(), .5)
        altered = self.data.records[:2] + [self.data.records[2]]*8
        weights = record_weights(altered)
        self.assertAlmostEqual(weights[:2].sum(), .5)

    def test_balanced_quantile_resists_repeated_entire_record(self):
        data = split([[1, 0]]*6, ["a"]*6, identities=["one"]*4+["two"]*2)
        fitted = calibrate([.1]*4+[.8]*2, data, "record_balanced", .4)
        self.assertEqual(fitted["threshold"], .8)

    def test_missing_primary_feature_does_not_choose_secondary(self):
        self.data.report["files"][0]["signals"].append({"track_id": "missing", "valid_votes": 3, "observation_count": 3})
        self.assertEqual(self.data.primary_indices().tolist(), [2])

    def test_nan_or_invalid_calibration_is_rejected(self):
        for values, rate in (([1, 2, np.nan, 4], .01), ([1, 2, 3, 4], 0), ([1, 2, 3, 4], 1)):
            with self.assertRaises(ValueError):
                calibrate(values, self.data, "primary", rate)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.ref = split([[1, .1, 0], [1, -.1, 0], [.1, 1, 0], [-.1, 1, 0]], ["a", "a", "b", "b"])
        self.query = split([[1, 0, 0], [0, 0, 1]], ["a", "unknown"], "test")

    def test_distinct_record_knn_cannot_gain_support_from_duplicates(self):
        before = FittedScore("record_knn", self.ref, k=2).score(self.query)
        altered = copy.deepcopy(self.ref)
        altered.features = np.vstack([altered.features, altered.features[0]])
        altered.records.append(dict(altered.records[0]))
        after = FittedScore("record_knn", altered, k=2).score(self.query)
        np.testing.assert_allclose(before, after)

    def test_class_knn_requires_distinct_reference_support(self):
        with self.assertRaisesRegex(ValueError, "independent"):
            FittedScore("record_class_knn", self.ref, k=3)

    def test_mahalanobis_is_finite_and_rotation_invariant(self):
        original = FittedScore("maha_pp_shrink", self.ref, k=1).score(self.query)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
        ref, query = copy.deepcopy(self.ref), copy.deepcopy(self.query)
        ref.features, query.features = ref.features @ rotation, query.features @ rotation
        rotated = FittedScore("maha_pp_shrink", ref, k=1).score(query)
        np.testing.assert_allclose(original, rotated, rtol=1e-7, atol=1e-7)
        self.assertGreater(original[1], original[0])

    def test_test_truth_does_not_change_scores(self):
        for method in ("record_knn", "record_class_knn", "record_class_knn_unscaled", "maha_pp_shrink"):
            scorer = FittedScore(method, self.ref, k=1)
            before = scorer.score(self.query)
            query = copy.deepcopy(self.query)
            for row in query.records:
                row["true_class_name"] = "arbitrary-hidden-label"
            np.testing.assert_array_equal(before, scorer.score(query))

    def test_scores_use_original_logits_not_softmax(self):
        self.ref.logits = np.array([[2., 0.]]*4)
        self.query.logits = np.array([[2., 0.], [2., 0.]])
        shifted = copy.deepcopy(self.query)
        shifted.logits += 10
        for method in ("maxlogit", "energy"):
            scorer = FittedScore(method, self.ref, k=1)
            np.testing.assert_allclose(scorer.score(shifted), scorer.score(self.query)-10)

    def test_nnguide_uses_distinct_record_confidence_weighted_cosines(self):
        ref = split([[1, 0], [0, 1]], ["a", "b"], logits=np.array([[2., 0.], [0., 1.]]))
        query = split([[1, 0]], ["unknown"], "test", logits=np.array([[1., 0.]]))
        score = FittedScore("nnguide_record", ref, k=1).score(query)
        expected = -np.log(np.exp(1)+1)*np.log(np.exp(2)+1)
        self.assertAlmostEqual(score[0], expected)

    def test_vim_detects_residual_direction_in_classifier_nullspace(self):
        rng = np.random.default_rng(12)
        raw = np.column_stack([rng.normal(2, 1, (30, 2)), rng.normal(0, .01, (30, 2))])
        w = np.array([[1., 0, 0, 0], [0, 1., 0, 0]])
        logits = raw @ w.T
        ref = split(raw, ["a"]*15+["b"]*15, raw=raw, logits=logits)
        query_raw = np.array([raw[0], raw[0]+[0, 0, 0, 50.]])
        query = split(query_raw, ["a", "unknown"], "q", raw=query_raw, logits=query_raw @ w.T)
        score = FittedScore("vim", ref, vim_dim=2, head={"weight": w, "bias": np.zeros(2)}).score(query)
        self.assertGreater(score[1], score[0]+10)

    def test_raw_methods_fail_without_raw_outputs(self):
        with self.assertRaisesRegex(ValueError, "logits"):
            FittedScore("energy", self.ref)

    def test_reference_and_calibration_identity_overlap_is_rejected(self):
        cal = copy.deepcopy(self.ref)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_splits(self.ref, cal, self.query)

    def test_unknown_reference_truth_is_rejected(self):
        ref = copy.deepcopy(self.ref)
        ref.records[0]["true_class_name"] = "unknown"
        with self.assertRaisesRegex(ValueError, "single-known"):
            validate_splits(ref, self.ref, self.query)


class CheckpointPortabilityTests(unittest.TestCase):
    def test_foreign_path_metadata_and_tensors_roundtrip(self):
        import os
        import pathlib
        import torch
        from util.checkpoint import load_checkpoint_object
        target = pathlib.PosixPath if os.name == "nt" else pathlib.WindowsPath
        class ForeignPath:
            def __reduce__(self):
                return target, ("/original/experiment",)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"model.pth"
            torch.save({"config": {"path": ForeignPath()}, "tensor": torch.tensor([2.])}, path)
            loaded = load_checkpoint_object(path)
            self.assertIsInstance(loaded["config"]["path"], pathlib.PurePath)
            torch.testing.assert_close(loaded["tensor"], torch.tensor([2.]))


class ComponentContrastTests(unittest.TestCase):
    def test_uniform_weights_preserve_original_supcon(self):
        import torch
        from model.representation import SupervisedContrastiveLoss
        features = torch.randn(4, 2, 6, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1])
        loss = SupervisedContrastiveLoss()
        torch.testing.assert_close(loss(features, labels), loss(features, labels, sample_weights=torch.ones(4)))
        weighted = loss(features, labels, sample_weights=torch.tensor([.5, .5, 1., 1.]))
        weighted.backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_invalid_component_weights_fail(self):
        import torch
        from model.representation import SupervisedContrastiveLoss
        with self.assertRaisesRegex(ValueError, "weights"):
            SupervisedContrastiveLoss()(torch.randn(2, 2, 4), torch.tensor([0, 1]), sample_weights=[0, 1])


if __name__ == "__main__":
    unittest.main()
