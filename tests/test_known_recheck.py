import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from util.known_recheck import KnownRechecker, RecheckConfig, fit_known_rechecker
from util.open_set import (normalize_features, load_npz, save_npz, build_profile,
                          OpenSetProfile, apply_open_set, file_sha256)
from tests.fixtures import archive, track_fixture


def inputs():
    rng = np.random.default_rng(19)
    def split(prefix, n):
        features, records = [], []
        for c, name in enumerate(['known', 'other']):
            for i in range(n):
                f = np.eye(3)[c] + rng.normal(0, .015, 3)
                features.append(f)
                records.append(dict(path=f'{prefix}/{name}/{i}.dat', track_id='T1',
                    recording_id=f'{prefix}-{name}-{i}', source_sha256=f'hash-{prefix}-{name}-{i}',
                    true_class_name=name, known_candidate_class=name, is_primary=True))
        return normalize_features(features), records
    return split('ref', 8), split('cal', 6)


class RecheckTests(unittest.TestCase):
    def setUp(self):
        (self.x, self.rr), (self.z, self.cr) = inputs()

    def fit(self, **kwargs):
        return fit_known_rechecker(self.x, self.rr, self.z, self.cr,
            ['known', 'other'], minimum_calibration=5, **kwargs)

    def test_local_acceptance_requires_distance_and_head_and_margin(self):
        m, a = self.fit()
        gate = KnownRechecker(m, a)
        self.assertTrue(gate.decide([[1, 0, 0]], ['known'])[0]['accepted'])
        self.assertEqual(gate.decide([[1, 0, 0]], ['other'])[0]['reason'], 'class_disagreement')
        self.assertFalse(gate.decide([[0, 0, 1]], ['known'])[0]['accepted'])
        self.assertFalse(gate.decide([[1, 1, 0]], ['known'])[0]['accepted'])

    def test_reference_scales_do_not_fit_calibration_or_test(self):
        m, a = self.fit()
        z = normalize_features(self.z + [0, 0, .1])
        n, b = fit_known_rechecker(self.x, self.rr, z, self.cr, ['known', 'other'])
        np.testing.assert_array_equal(a['recheck_reference'], b['recheck_reference'])
        np.testing.assert_array_equal(a['recheck_scales'], b['recheck_scales'])
        self.assertNotEqual(m['radius'], n['radius'])
        gate = KnownRechecker(m, a)
        before = copy.deepcopy(gate.metadata)
        gate.decide([[0, 0, 1]]*50, ['known']*50)
        self.assertEqual(before, gate.metadata)

    def test_secondary_copies_never_inflate_bank_or_calibration(self):
        m, a = self.fit()
        rr, cr = copy.deepcopy(self.rr), copy.deepcopy(self.cr)
        rr.append({**rr[0], 'is_primary': False, 'track_id': 'secondary'})
        cr.append({**cr[0], 'is_primary': False, 'track_id': 'secondary'})
        n, b = fit_known_rechecker(np.vstack([self.x, [0, 0, 1]]), rr,
            np.vstack([self.z, [0, 0, 1]]), cr, ['known', 'other'])
        self.assertEqual(m, n)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_ineligible_calibration_head_cannot_inflate_gate_radius(self):
        m, a = self.fit()
        cr = copy.deepcopy(self.cr)
        cr.append({**cr[0], 'recording_id': 'extra', 'source_sha256': 'extra-hash',
                   'known_candidate_class': 'other'})
        n, b = fit_known_rechecker(self.x, self.rr, np.vstack([self.z, [0, 0, 1]]),
                                   cr, ['known', 'other'])
        self.assertEqual(m['radius'], n['radius'])
        self.assertEqual(m['margin'], n['margin'])

    def test_overlap_duplicate_record_and_unknown_calibration_fail(self):
        cr = copy.deepcopy(self.cr);cr[0]['recording_id'] = self.rr[0]['recording_id']
        with self.assertRaisesRegex(ValueError, 'overlap'):
            fit_known_rechecker(self.x, self.rr, self.z, cr, ['known', 'other'])
        rr = copy.deepcopy(self.rr);rr[1]['recording_id'] = rr[0]['recording_id']
        with self.assertRaisesRegex(ValueError, 'primary'):
            fit_known_rechecker(self.x, rr, self.z, self.cr, ['known', 'other'])
        cr = copy.deepcopy(self.cr);cr[0]['true_class_name'] = 'held_out'
        with self.assertRaisesRegex(ValueError, 'known-only'):
            fit_known_rechecker(self.x, self.rr, self.z, cr, ['known', 'other'])

    def test_persistence_and_baseline_profile_are_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            for name, x, records in [('ref', self.x, self.rr), ('cal', self.z, self.cr)]:
                save_npz(p/(name+'.npz'), dict(kind='track_features', schema_version=1,
                    feature_version='mask-penultimate-l2-accepted-track-mean-l2-v1',
                    signature={'sha256': 'fixture'}, classes={'known': 0, 'other': 1}, records=records), features=x)
            a = build_profile(p/'ref.npz', p/'cal.npz', p/'a.npz')
            b = build_profile(p/'ref.npz', p/'cal.npz', p/'b.npz', recheck_config=RecheckConfig())
            self.assertEqual(a['threshold'], b['threshold'])
            _, aa = load_npz(p/'a.npz', 'open_set_profile')
            _, bb = load_npz(p/'b.npz', 'open_set_profile')
            for key in aa:
                np.testing.assert_array_equal(aa[key], bb[key])
            gate = OpenSetProfile(p/'b.npz', {'sha256': 'fixture'}, {'known': 0, 'other': 1}).rechecker
            self.assertTrue(gate.decide([[1, 0, 0]], ['known'])[0]['accepted'])

    def profile(self):
        m, a = self.fit()
        return SimpleNamespace(threshold=.5, classes={'known': 0, 'other': 1},
            ignore_classes=set(), score_details=lambda x: (np.ones(len(x)), ['known']*len(x)),
            score=lambda x: np.ones(len(x)), rechecker=KnownRechecker(m, a))

    def test_live_gate_is_optional_and_recovered_track_leaves_unknown_pool(self):
        results, rows = track_fixture([[1, 0, 0]])
        a = apply_open_set(self.profile(), copy.deepcopy(results), copy.deepcopy(rows))
        self.assertEqual(len(a), 1)
        b = apply_open_set(self.profile(), results, rows, known_recheck=True)
        self.assertFalse(b)
        t = results[0]['signals'][0]
        self.assertEqual(t['recognition_status'], 'known')
        self.assertEqual(t['class_name'], 'known')
        self.assertTrue(t['known_recheck']['accepted'])

    def test_gate_does_not_promote_secondary_or_missing_features(self):
        results, rows = track_fixture([[1, 0, 0]])
        secondary = copy.deepcopy(results[0]['signals'][0]);secondary['track_id'] = 'secondary'
        primary = results[0]['signals'][0];primary['valid_votes'] = 10
        group = rows['test.png'][0]['groups'][0]
        other = copy.deepcopy(group);other['track_id'] = 'secondary'
        group['embedding'] = [0, 0, 1]
        rows['test.png'][0]['groups'].append(other)
        results[0]['signals'].append(secondary)
        candidates = apply_open_set(self.profile(), results, rows, known_recheck=True)
        self.assertEqual(len(candidates), 2)
        self.assertNotIn('known_recheck', secondary)

    def test_invalid_bank_config_and_single_class_fail(self):
        for config in (RecheckConfig(k=0), RecheckConfig(k=True), RecheckConfig(radius_frr=1),
                       RecheckConfig(margin_tail=-1), RecheckConfig(scale_prior_records=-1)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()
        with self.assertRaises(ValueError):
            fit_known_rechecker(self.x, self.rr, self.z, self.cr, ['known'])

    def test_class_without_agreed_calibration_cannot_be_recovered(self):
        m, a = self.fit()
        m['per_class_eligible_calibration']['known'] = 0
        d = KnownRechecker(m, a).decide([[1, 0, 0]], ['known'])[0]
        self.assertFalse(d['accepted'])
        self.assertEqual(d['reason'], 'no_agreed_class_calibration')

    def test_full_and_fast_live_recovery_match_and_write_diagnostics(self):
        import cv2
        import torch
        from scripts.evaluate_single_signal import build_parser, run
        from util.open_set import FEATURE_VERSION, inference_signature
        from util.eval_common import resolve_config
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            image = p/'sample.png'
            cv2.imencode('.png', np.zeros((512, 750), dtype=np.uint8))[1].tofile(image)
            checkpoint = p/'model.pth';yolo = p/'detector.pt'
            cfg = dict(classes={'known': 0, 'other': 1}, mask_img_size=32)
            torch.save(dict(config=cfg, model_state_dict={}), checkpoint);yolo.write_bytes(b'fixture')
            def args(fast):
                a = build_parser().parse_args(['--inputs', str(image), '--checkpoint_path', str(checkpoint),
                    '--yolo_weights', str(yolo), '--open_set_profile', str(p/'profile.npz'),
                    '--unknown_input_mode', 'single', '--known_recheck', '--no-save_visuals',
                    '--no-save_feature_plots', '--device', 'cpu', *(['--fast', '--no-progress'] if fast else [])])
                a.output_dir = p/('fast' if fast else 'full')
                return a
            signature = inference_signature(args(False), resolve_config(dict(config=cfg)))
            for name, x, records in [('ref', self.x, self.rr), ('cal', self.z, self.cr)]:
                save_npz(p/(name+'.npz'), dict(kind='track_features', schema_version=1,
                    feature_version=FEATURE_VERSION, signature=signature, classes=cfg['classes'], records=records), features=x)
            build_profile(p/'ref.npz', p/'cal.npz', p/'original.npz', recheck_config=RecheckConfig())
            m, a = load_npz(p/'original.npz', 'open_set_profile')
            m['threshold'] = 0.  # Explicit synthetic first-stage rejection, not a learned model.
            save_npz(p/'profile.npz', m, **a)
            class Predictor:
                def predict(self, spectrum, domain):
                    return dict(status='ok', detected_boxes=1, groups=[dict(group_id=0, status='ok',
                        class_name='known', class_id=0, confidence=1., probabilities=[1., 0.],
                        selected_boxes=[[10, 100, 50, 110]], embedding=[1., 0., 0.], group_type='frequency')])
            full, fast = run(args(False), Predictor()), run(args(True), Predictor())
            for r in (full, fast):
                self.assertEqual(r['metrics']['known_recheck_recovered'], 1)
            self.assertEqual((p/'full/predictions.csv').read_text(), (p/'fast/predictions.csv').read_text())
            self.assertTrue(json.loads((p/'fast/known_recheck_decisions.json').read_text())[0]['accepted'])


class PairedComparisonTests(unittest.TestCase):
    def test_event_conflict_scope_exception_is_explicit_and_does_not_hide_other_errors(self):
        from scripts.compare_known_recheck import comparison_exit_code
        result = dict(comparison_has_upstream_errors=True, source_non_event_error_count=0,
                      source_error_breakdown=dict(burst_label_conflicts=12))
        self.assertEqual(comparison_exit_code(result), 2)
        self.assertEqual(comparison_exit_code(result, True), 0)
        result['source_non_event_error_count'] = 1
        self.assertEqual(comparison_exit_code(result, True), 2)

    def fixture(self, root):
        from util.unknown_discovery import discover_unknown
        from util.discovery_context import annotate_discovery_candidates, candidate_record
        from util.open_set import FEATURE_VERSION
        (x, rr), (z, cr) = inputs()
        classes = {'known': 0, 'other': 1}
        repo = Path(__file__).resolve().parents[1]
        signature = {'sha256': 'paired-fixture', 'details': {'source_sha256': {
            name: file_sha256(repo/name) for name in ('util/open_set.py', 'util/known_recheck.py')}}}
        (root/'open_set').mkdir();(root/'test').mkdir()
        for name, values, records in [('reference', x, rr), ('calibration', z, cr)]:
            (root/'open_set'/name).mkdir()
            save_npz(root/'open_set'/name/'features.npz', dict(kind='track_features', schema_version=1,
                feature_version=FEATURE_VERSION, signature=signature, classes=classes, records=records), features=values)
        build_profile(root/'open_set/reference/features.npz', root/'open_set/calibration/features.npz',
                      root/'open_set/known_profile.npz')
        files, candidates = [], []
        # One baseline-rejected known, one truly novel, one missing detection.
        for i, (feature, truth) in enumerate([([1, 0, 0], 'known'), ([0, 0, 1], 'held_out')]):
            path = f'test-{i}.dat'
            t = dict(track_id='T1', valid_votes=1, observation_count=1, status='unknown',
                recognition_status='novel_candidate', class_name=None, known_candidate_class='known',
                ood_score=100., ood_threshold=1., display_label='Unknown_unassigned', closed_set_status='ok')
            files.append(dict(path=path, signals=[t], status='ok', true_class_names=[truth]))
            candidates.append(dict(path=path, recording_id=path, feature=np.array(feature, dtype=float), track=t))
        files.append(dict(path='empty.dat', signals=[], status='no_signal', true_class_names=['known']))
        annotate_discovery_candidates(candidates, files)
        discovery = discover_unknown(candidates, input_mode='single')
        source = dict(files=files, effective_config=dict(classes=classes), discovery=discovery,
                      arguments=dict(known_recheck=False), metrics={}, recording_ids={})
        settings = dict(known_recheck=False, unknown_input_mode='single', knn_k=5,
            known_false_reject_rate=.01, min_calibration_per_class=5, max_reference_per_class=2000,
            profile_score_method='maha_pp_shrink', profile_scale_prior_records=5.,
            profile_covariance_shrinkage=.1, profile_calibration_unit='primary',
            unknown_cluster_backend='hdbscan', unknown_min_cluster_size=5,
            unknown_min_samples=3, unknown_min_recordings=2, unknown_allow_single_cluster=True,
            unknown_cluster_policy='auto')
        manifest = dict(protocol='open_set', classes=classes, unknown_classes=['held_out'],
                        records=[dict(class_name='held_out', split='test_unknown')])
        for name, value in [('test/results.json', source), ('effective_discover.json', settings),
                            ('split_manifest.json', manifest)]:
            (root/name).write_text(json.dumps(value))
        save_npz(root/'test/unknown_features.npz', dict(kind='unknown_candidates', schema_version=1,
            feature_version=FEATURE_VERSION, signature=signature,
            records=[candidate_record(c) for c in candidates]), features=np.array([c['feature'] for c in candidates]))

    def test_paired_replay_preserves_samples_and_keeps_unknown_and_missing_detection(self):
        from scripts.compare_known_recheck import run_comparison
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);source = root/'source';source.mkdir();self.fixture(source)
            r = run_comparison(source, root/'B', {})
            self.assertEqual(r['paired_recordings'], 3)
            self.assertEqual(r['recovered_known_correct'], 1)
            self.assertEqual(r['newly_false_accepted_unknown'], 0)
            self.assertEqual(r['B']['known_abstention_rate'], .5)
            self.assertEqual(r['B']['unknown_false_accept_rate'], 0.)
            self.assertTrue((root/'B/predictions.csv').exists())

    def test_source_label_leakage_and_candidate_mismatch_fail(self):
        from scripts.compare_known_recheck import run_comparison
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);source = root/'source';source.mkdir();self.fixture(source)
            p = source/'split_manifest.json';m = json.loads(p.read_text())
            m['records'][0]['split'] = 'train';p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'leaked'):
                run_comparison(source, root/'B', {})

    def test_test_truth_changes_metrics_but_never_decisions(self):
        from scripts.compare_known_recheck import run_comparison
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);source = root/'source';source.mkdir();self.fixture(source)
            run_comparison(source, root/'B1', {})
            path = source/'test/results.json';r = json.loads(path.read_text())
            r['files'][0]['true_class_names'] = ['held_out']
            r['files'][1]['true_class_names'] = ['known']
            path.write_text(json.dumps(r))
            altered = run_comparison(source, root/'B2', {})
            self.assertEqual((root/'B1/known_recheck_decisions.json').read_text(),
                             (root/'B2/known_recheck_decisions.json').read_text())
            self.assertEqual(altered['newly_false_accepted_unknown'], 1)


if __name__ == '__main__':
    unittest.main()
