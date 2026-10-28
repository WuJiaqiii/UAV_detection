import copy
import unittest

from util.open_set_metrics import evaluate_open_set


def result(path, truth, status, prediction=None, cluster=None, votes=2, score=1.):
    return {"path": path, "true_class_names": [truth] if truth is not None else None,
            "signals": [{"track_id": "T1", "recognition_status": status, "class_name": prediction,
                         "unknown_cluster_id": cluster, "valid_votes": votes,
                         "observation_count": votes, "ood_score": score}]}


class OpenSetMetricsTests(unittest.TestCase):
    def test_permuted_cluster_ids_match_truth_and_known_labels_stay_fixed(self):
        rows = [result("k", "A", "known", "A", score=.1),
                result("u1", "U", "novel_candidate", cluster="Unknown_002", score=2.),
                result("u2", "V", "novel_candidate", cluster="Unknown_001", score=3.)]
        before = copy.deepcopy(rows)
        metrics = evaluate_open_set(rows, {"A": 0}, {"estimated_unknown_class_count": 2})["metrics"]
        self.assertEqual(metrics["all_accuracy"], 1.)
        self.assertEqual(metrics["unknown_accuracy"], 1.)
        self.assertEqual(metrics["unknown_cluster_to_truth_for_scoring_only"], {"Unknown_002": "U", "Unknown_001": "V"})
        self.assertEqual(metrics["ood_auroc"], 1.)
        self.assertEqual(rows, before)

    def test_missed_unassigned_and_misrouted_samples_stay_in_denominator(self):
        missed = result("missed", "U", "no_signal")
        missed["signals"] = []
        rows = [result("k", "A", "novel_candidate", cluster="X"),
                result("u1", "U", "novel_candidate", cluster="Y"),
                result("u2", "U", "novel_candidate"), result("u3", "U", "known", "A"), missed]
        metrics = evaluate_open_set(rows, {"A": 0}, {"estimated_unknown_class_count": 2})["metrics"]
        self.assertEqual(metrics["known_accuracy"], 0.)
        self.assertEqual(metrics["known_false_reject_rate"], 1.)
        self.assertEqual(metrics["unknown_accuracy"], .25)
        self.assertEqual(metrics["all_accuracy"], .2)
        self.assertEqual(metrics["unknown_cluster_coverage"], .25)
        self.assertEqual(metrics["unknown_detection_recall"], .5)
        self.assertEqual(metrics["unknown_class_count_absolute_error"], 1)

    def test_recording_units_are_not_inflated_by_exports_and_primary_is_truth_independent(self):
        rows = [result("slice1", "U", "known", "A", votes=1),
                result("slice2", "U", "novel_candidate", cluster="X", votes=3)]
        report = evaluate_open_set(rows, {"A": 0}, {"estimated_unknown_class_count": 1},
                                   {"slice1": "original", "slice2": "original"})
        self.assertEqual(report["metrics"]["evaluated_recordings"], 1)
        self.assertEqual(report["samples"][0]["primary_path"], "slice2")
        self.assertEqual(report["metrics"]["unknown_accuracy"], 1.)

    def test_unlabeled_multilabel_and_conflicting_units_are_explicitly_skipped(self):
        rows = [result("a", None, "known", "A"), result("b", "A", "known", "A")]
        rows[1]["true_class_names"] = ["A", "U"]
        report = evaluate_open_set(rows, {"A": 0}, {"estimated_unknown_class_count": 0})
        self.assertEqual(report["metrics"]["evaluated_recordings"], 0)
        self.assertEqual(report["metrics"]["skipped_recordings"], 2)
        self.assertIsNone(report["metrics"]["all_accuracy"])
        self.assertIsNone(report["metrics"]["true_unknown_class_count"])

    def test_oversegmentation_cannot_match_two_clusters_to_one_truth_class(self):
        rows = [result(str(i), "U", "novel_candidate", cluster="X" if i < 3 else "Y") for i in range(4)]
        metrics = evaluate_open_set(rows, {"A": 0}, {"estimated_unknown_class_count": 2})["metrics"]
        self.assertEqual(metrics["unknown_accuracy"], .75)
        self.assertEqual(metrics["unknown_class_count_error"], 1)

    def test_partial_truth_cannot_score_the_global_class_count(self):
        rows = [result("a", "U", "novel_candidate", cluster="X"), result("b", None, "novel_candidate", cluster="Y")]
        metrics = evaluate_open_set(rows, {"A": 0}, {"estimated_unknown_class_count": 2})["metrics"]
        self.assertEqual(metrics["observed_labeled_unknown_class_count"], 1)
        self.assertIsNone(metrics["unknown_class_count_error"])

    def test_known_labels_cannot_be_permuted_by_the_unknown_matching(self):
        rows = [result("a", "A", "known", "B"), result("b", "B", "known", "A"),
                result("u", "U", "novel_candidate", cluster="X")]
        metrics = evaluate_open_set(rows, {"A": 0, "B": 1}, {"estimated_unknown_class_count": 1})["metrics"]
        self.assertEqual(metrics["known_accuracy"], 0.)
        self.assertAlmostEqual(metrics["all_accuracy"], 1 / 3)
