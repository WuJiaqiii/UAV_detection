import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from util.bounded_density import bounded_density_partition
from util.open_set import normalize_features, save_npz, FEATURE_VERSION
from util.unknown_discovery import discover_unknown, validate_bounded_strategy


def fixture(noise=0):
    rng = np.random.default_rng(41)
    centers = [[1, 0, 0, 0], [1, .15, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    x, ids = [], []
    for c, center in enumerate(centers):
        for i in range(12):
            x.append(np.array(center) + rng.normal(0, .002, 4));ids.append(f'c{c}/{i:02}')
    for i in range(noise):
        x.append(rng.normal(size=4));ids.append(f'n/{i:02}')
    return normalize_features(x), ids


def seeded_density(features, ids, *args, **kwargs):
    # Deterministic density-core substitute isolates the merge stage from a
    # library's cluster extraction choices; actual density is tested below too.
    return (np.array([-1 if r.startswith('n/') else int(r[1]) for r in ids]),
            np.ones(len(ids)), dict(algorithm_version='test-density-seeds'))


class DensityBoundTests(unittest.TestCase):
    def test_supported_core_merge_recovers_four_geometric_populations(self):
        x, ids = fixture()
        with patch('util.unknown_density.guarded_hdbscan', side_effect=seeded_density):
            y, d = bounded_density_partition(x, ids, 3, 4, bootstrap_runs=2)
        self.assertTrue(d['constraint_satisfied']);self.assertEqual(d['selected_K'], 4)
        self.assertEqual(len(set(y)), 4)
        self.assertEqual(set(y[:12]), set(y[12:24]))
        self.assertEqual(sorted(np.bincount(y).tolist()), [12, 12, 12, 24])
        self.assertEqual(d['initial_supported_cores'], 5)
        self.assertTrue(all(c['bootstrap_evaluated'] == 2 for c in d['candidate_K_scores'] if c['valid']))

    def test_order_and_same_record_duplicate_components_do_not_change_partition(self):
        x, ids = fixture()
        with patch('util.unknown_density.guarded_hdbscan', side_effect=seeded_density):
            y, d = bounded_density_partition(x, ids, 3, 4, bootstrap_runs=1)
            reverse, _ = bounded_density_partition(x[::-1], ids[::-1], 3, 4, bootstrap_runs=1)
            dup, dd = bounded_density_partition(np.vstack([x, np.tile(x[0], (20, 1))]),
                ids+[ids[0]]*20, 3, 4, bootstrap_runs=1)
        np.testing.assert_array_equal(y, reverse[::-1])
        np.testing.assert_array_equal(y, dup[:len(x)])
        self.assertEqual(d['unique_fit_components'], dd['unique_fit_components'])

    def test_noise_stays_unassigned_and_does_not_become_a_class(self):
        x, ids = fixture(noise=4)
        with patch('util.unknown_density.guarded_hdbscan', side_effect=seeded_density):
            y, d = bounded_density_partition(x, ids, 3, 4, bootstrap_runs=0)
        self.assertTrue(d['constraint_satisfied']);self.assertTrue(np.all(y[-4:] == -1))
        self.assertEqual(d['noise_components'], 4)
        self.assertLess(d['record_balanced_coverage'], 1.)

    def test_missing_cores_are_not_fabricated_and_empty_task_two_is_valid(self):
        x, ids = fixture();x, ids = x[:12], ids[:12]
        with patch('util.unknown_density.guarded_hdbscan', side_effect=seeded_density):
            y, d = bounded_density_partition(x, ids, 3, 4)
        self.assertFalse(d['constraint_satisfied']);self.assertTrue(np.all(y == -1))
        self.assertEqual(d['reason'], 'insufficient_density_cores')
        y, d = bounded_density_partition(np.empty((0, 4)), [], 0, 5)
        self.assertTrue(d['constraint_satisfied']);self.assertEqual(len(y), 0)

    def test_identical_features_cannot_fake_distinct_classes(self):
        x = np.tile([1., 0, 0, 0], (60, 1));_, ids = fixture()
        with patch('util.unknown_density.guarded_hdbscan', side_effect=seeded_density):
            y, d = bounded_density_partition(x, ids, 3, 4, bootstrap_runs=0)
        self.assertFalse(d['constraint_satisfied']);self.assertTrue(np.all(y == -1))

    def test_fixed_k_stays_within_protocol(self):
        x, ids = fixture()
        with patch('util.unknown_density.guarded_hdbscan', side_effect=seeded_density):
            _, d = bounded_density_partition(x, ids, 3, 4, fixed_k=3, bootstrap_runs=0)
        self.assertEqual(d['selected_K'], 3)
        for invalid in (2, 5, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bounded_density_partition(x, ids, 3, 4, fixed_k=invalid)

    def test_invalid_feature_norms_and_strategy_combinations_fail(self):
        x, ids = fixture()
        with self.assertRaises(ValueError):
            bounded_density_partition(x*2, ids, 3, 4)
        for args in [('density_merge', 'unconstrained'), ('density_merge', 'task1_n2', 'dcg'),
                     ('density_merge', 'task1_n2', 'hdbscan', 'legacy')]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_bounded_strategy(*args)

    def test_actual_density_integration_and_joint_csv_pipeline(self):
        from scripts.recluster_competition_batch import main
        x, ids = fixture()
        candidates = [dict(path=f'1-2_{i:04}.dat', recording_id=r, feature=f,
            track=dict(track_id='T1', discovery_is_primary=True)) for i, (f, r) in enumerate(zip(x, ids))]
        d = discover_unknown(copy.deepcopy(candidates), input_mode='single', competition_profile='task1_n2',
                             bounded_strategy='density_merge')
        self.assertTrue(d['constraint_satisfied']);self.assertIn(d['estimated_unknown_class_count'], (3, 4))
        self.assertEqual(d['parameters']['bounded_strategy'], 'density_merge')
        with tempfile.TemporaryDirectory() as t:
            p = Path(t);archive = p/'features.npz';csv_path = p/'predictions.csv'
            records = [dict(path=c['path'], recording_id=c['recording_id'], track_id='T1',
                discovery_is_primary=True, recognition_status='novel_candidate',
                discovery_primary_path=c['path'], discovery_primary_track_id='T1') for c in candidates]
            save_npz(archive, dict(kind='unknown_candidates', schema_version=1, feature_version=FEATURE_VERSION,
                signature=dict(sha256='fixture'), records=records), features=x)
            with csv_path.open('w', newline='') as stream:
                w = csv.writer(stream);w.writerow(['path', 'label', 'status', 'error'])
                w.writerows([[c['path'], 'Unknown_unassigned', 'novel_candidate', ''] for c in candidates])
            import sys
            with patch.object(sys, 'argv', ['recluster_competition_batch.py', '--item', '1-2', str(archive),
                str(csv_path), '--bounded-strategy', 'density_merge', '--output', str(p/'joint')]):
                main()
            result = json.loads((p/'joint/joint_selection.json').read_text())
            self.assertIn(result['selected_counts']['1-2'], (3, 4))
            with Path(result['predictions']['1-2']).open(encoding='utf-8-sig') as stream:
                table = list(csv.DictReader(stream))
            self.assertEqual(len(table), len(candidates))
            self.assertNotIn('Unknown_unassigned', {r['label'] for r in table})


if __name__ == '__main__':
    unittest.main()
