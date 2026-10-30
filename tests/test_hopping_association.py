"""Cross-frequency seeds, final support and shape/conflict safeguards."""
import logging
from types import SimpleNamespace
import unittest

import numpy as np

from util.eval_common import INFERENCE_DEFAULTS
from util.preprocess import SignalPreprocessor


# Exact detector coordinates from sample_020__39226bb00cbe_grouping.json.
X20_BOXES = np.array([
    [533, 396, 579, 400], [217, 321, 264, 326],
    [7, 274, 55, 278], [324, 345, 369, 351],
    [637, 127, 684, 132], [429, 371, 475, 375],
    [112, 296, 158, 300], [744, 152, 750, 157],
], dtype=np.int32)


def process(boxes, **overrides):
    settings = dict(INFERENCE_DEFAULTS)
    settings.update(grouping_mode="hopping", hop_association="shape")
    settings.update(overrides)
    preprocessor = SignalPreprocessor(SimpleNamespace(**settings), logging.getLogger("hop-test"))
    # Controlled spectrum: each supplied detection has energy above background.
    spectrum = np.ones((512, 750), dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        spectrum[y1:y2, x1:x2] = 100.
    groups = preprocessor.select_signal_groups(boxes, spectrum)
    return groups, preprocessor.last_diagnostics


class HoppingAssociationTests(unittest.TestCase):
    def test_three_single_carrier_seeds_grow_into_one_hopping_group(self):
        boxes = np.array([[10, 100, 50, 104], [120, 200, 160, 204],
                          [230, 300, 270, 304]], dtype=np.int32)
        groups, _ = process(boxes)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["n_boxes"], 3)
        self.assertEqual(groups[0]["frequency_cluster_count"], 3)

    def test_two_pulses_can_seed_but_cannot_be_returned_as_valid_hopping_group(self):
        boxes = np.array([[10, 100, 50, 104], [230, 300, 270, 304]], dtype=np.int32)
        groups, diagnostics = process(boxes)
        self.assertEqual(groups, [])
        self.assertEqual(len(diagnostics["final_candidates"]), 1)
        candidate = diagnostics["final_candidates"][0]
        self.assertEqual(candidate["group_type"], "hopping")
        self.assertIn("n_boxes=2 < hop_min_pulses=3", candidate["rejection_reasons"])

    def test_x20_thin_pulses_group_and_clipped_edge_does_not_loosen_global_shape(self):
        groups, diagnostics = process(X20_BOXES, hop_shape_log_tol=.50)
        self.assertEqual(diagnostics["geometry_boxes"], 8)
        self.assertEqual(diagnostics["energy_boxes"], 8)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["n_boxes"], 7)
        expected = {tuple(b) for b in X20_BOXES if b[2] != 750}
        self.assertEqual({tuple(b) for b in groups[0]["boxes"]}, expected)

    def test_input_order_does_not_change_x20_membership(self):
        for boxes in (X20_BOXES, X20_BOXES[::-1], X20_BOXES[[7, 4, 0, 6, 3, 2, 5, 1]]):
            groups, _ = process(boxes, hop_shape_log_tol=.50)
            self.assertEqual(len(groups), 1)
            self.assertEqual({tuple(b) for b in groups[0]["boxes"]},
                             {tuple(b) for b in X20_BOXES if b[2] != 750})

    def test_shape_and_time_conflict_safeguards_still_reject(self):
        for boxes in (
            np.array([[10, 100, 50, 104], [120, 200, 160, 204],
                      [230, 300, 340, 304]], dtype=np.int32),
            np.array([[10, 100, 50, 104], [10, 200, 50, 204],
                      [230, 300, 270, 304]], dtype=np.int32),
        ):
            groups, _ = process(boxes)
            self.assertEqual(groups, [])

    def test_same_carrier_two_pulses_and_frequency_mode_keep_previous_behavior(self):
        boxes = np.array([[10, 100, 50, 104], [230, 100, 270, 104]], dtype=np.int32)
        groups, _ = process(boxes)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["group_type"], "frequency")
        self.assertEqual(groups[0]["n_boxes"], 2)
        for frequency in (True, False):
            groups, _ = process(X20_BOXES, grouping_mode="frequency" if frequency else "hopping",
                                hop_association="periodic", hop_shape_log_tol=.50)
            self.assertEqual(groups, [])


if __name__ == "__main__":
    unittest.main()
