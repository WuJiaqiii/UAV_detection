import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import yaml
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from prepare_submission import KNOWN, read_xlsx_rows
from scripts.legacy_submission_common import (
    ITEMS, add_no_signal_class, checked_rows, export_results, read, read_csv,
    write_csv, verify_sources)
from scripts.supplement_full_duration import select_pending, merge_rows, check_frozen_inputs
from scripts import supplement_full_duration as supplement
from util.open_set import file_sha256


def row(sample, label='no_signal', status='no_signal', track=None):
    result = dict(path='/raw/'+sample+'.dat', label=label, status=status, error='')
    if track is not None:
        result['track_id'] = track
    return result


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.predictions = {}
        for item in ITEMS:
            if item.startswith('1-'):
                rows = [row(item+'_0000', next(iter(KNOWN)), 'known'),
                        row(item+'_0001', 'Unknown_001', 'novel_candidate'),
                        row(item+'_0002', 'Unknown_002', 'novel_candidate'), row(item+'_0003')]
            else:
                rows = [row(item+'_0000')]
            p = self.root/'source'/item/'predictions.csv'
            write_csv(p, list(rows[0]), rows)
            self.predictions[item] = p

    def test_no_signal_gets_separate_class_and_does_not_change_task_two_or_sources(self):
        original = {i: p.read_bytes() for i, p in self.predictions.items()}
        transformed, mapping = add_no_signal_class(self.predictions, self.root/'policy')
        for item in ITEMS:
            self.assertEqual(original[item], self.predictions[item].read_bytes())
            _, rows = read_csv(transformed[item])
            if item.startswith('1-'):
                self.assertEqual(mapping[item]['rows'], 1)
                self.assertEqual(rows[-1]['label'], 'Unknown_003')
                self.assertEqual(rows[-1]['status'], 'novel_candidate')
                self.assertEqual(rows[1]['label'], 'Unknown_001')
            else:
                self.assertEqual(rows[0]['label'], 'no_signal')

    def test_nonstandard_export_is_complete_numeric_and_explicitly_not_ready(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code, summary = export_results(self.predictions, self.root/'export', extra_no_signal=True)
        self.assertEqual(code, 0)
        self.assertEqual(summary['workbooks'], 6)
        self.assertFalse(summary['ready'])
        report = read(summary['report'])
        self.assertTrue(report['has_fallbacks'])
        self.assertTrue(report['preview'])
        for item in ITEMS[:4]:
            self.assertEqual(summary['no_signal_extra_classes'][item]['submission_code'], 22)
            table = read_xlsx_rows(Path(summary['output'])/(item+'.xlsx'))
            self.assertEqual(table[-1], [item+'_0003', '22'])
        with ZipFile(Path(summary['output'])/'2-1/2-1_0000.xlsx') as archive:
            xml = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        self.assertEqual([v.text for v in xml.findall('.//m:row[@r="2"]/m:c/m:v', ns)], ['-1', '0', '0'])

    def test_strict_export_still_blocks_no_signal_and_never_uses_automatic_fallback(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code, summary = export_results(self.predictions, self.root/'strict')
        self.assertEqual(code, 2)
        self.assertEqual(summary['workbooks'], 0)
        self.assertFalse(Path(summary['output']).exists())
        self.assertEqual(summary['no_signal_extra_classes'], {})

    def test_pending_refreshes_assigned_and_unassigned_unknowns_and_entire_overfull_files(self):
        rows = [row('2-1_0000', 'Unknown_001', 'novel_candidate', 'T1'),
                row('2-1_0001', 'Unknown_unassigned', 'novel_candidate', 'T1'),
                row('2-1_0002'), row('2-1_0003', 'error', 'error')]
        rows += [row('2-1_0004', name, 'known', 'T'+str(i)) for i, name in enumerate(list(KNOWN)[:4])]
        rows.append(row('2-1_0005', next(iter(KNOWN)), 'known', 'T1'))
        selected = select_pending(rows)
        self.assertEqual(set(selected), {f'2-1_{i:04d}' for i in range(5)})
        self.assertIn('refresh_entire_unknown_feature_pool', selected['2-1_0000'])
        self.assertIn('more_than_three_classes', selected['2-1_0004'])
        self.assertEqual(len(select_pending(rows, 'all')), 6)

    def test_merge_replaces_whole_file_and_preserves_other_decisions_and_burst_fields(self):
        kept = row('2-1_0000', next(iter(KNOWN)), 'known', 'T1')
        old = [kept, row('2-1_0001', 'Unknown_unassigned', 'novel_candidate', 'T1'),
               row('2-1_0001', 'another known', 'known', 'T2')]
        fresh = [row('2-1_0001', 'Unknown_001', 'novel_candidate', 'T7')]
        fresh[0]['bursts_json'] = '[{"start_ms":12,"end_ms":14}]'
        before = copy.deepcopy(old)
        merged = merge_rows(old, fresh, ['2-1_0001'])
        self.assertEqual(merged, [kept, fresh[0]])
        self.assertEqual(old, before)
        self.assertEqual(merged[-1]['bursts_json'], fresh[0]['bursts_json'])

    def test_merge_refuses_missing_extra_and_unrefreshed_novel_file(self):
        old = [row('1-1_0000', 'Unknown_001', 'novel_candidate'), row('1-1_0001')]
        with self.assertRaisesRegex(ValueError, 'Old novel'):
            merge_rows(old, [row('1-1_0001')], ['1-1_0001'])
        for fresh in [[], [row('1-1_0002')]]:
            with self.assertRaises(ValueError):
                merge_rows(old, fresh, ['1-1_0000'])

    def test_duplicate_tracks_missing_coverage_and_upstream_errors_are_not_ignored(self):
        p = self.root/'bad.csv'
        r = row('1-1_0000')
        write_csv(p, list(r), [r, r])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            checked_rows(p, '1-1')
        write_csv(p, list(r), [r])
        with self.assertRaisesRegex(ValueError, 'noncontiguous'):
            checked_rows(p, '1-1', 2)
        r['error'] = 'slice failed'
        write_csv(p, list(r), [r])
        with self.assertRaisesRegex(ValueError, 'upstream'):
            checked_rows(p, '1-1')
        self.assertEqual(len(checked_rows(p, '1-1', allow_errors=True)[1]), 1)

    def test_source_mutation_is_detected_before_resume(self):
        cache = self.root/'cache';cache.write_bytes(b'cache')
        csv_path = self.predictions['1-1']
        source = dict(predictions=str(csv_path), features=str(cache),
                      csv_sha256=file_sha256(csv_path), cache_sha256=file_sha256(cache))
        verify_sources({'1-1': source})
        cache.write_bytes(b'new cache')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            verify_sources({'1-1': source})

    def prepare_fixture(self, *, tail='cover', scope='pending', changed_inventory=False):
        source = self.root/'refresh';source.mkdir()
        training = self.root/'training'
        config = source/'inference.yaml'
        profile = source/'profile.npz';profile.write_bytes(b'cover profile')
        checkpoint = self.root/'model.pth';checkpoint.write_bytes(b'CNN')
        detector = self.root/'yolo.pt';detector.write_bytes(b'YOLO')
        config.write_text(yaml.safe_dump({'infer':dict(inputs=['/placeholder/1-1'],
            checkpoint_path=str(checkpoint), yolo_weights=str(detector), open_set_profile=str(profile),
            signal_mode='multi', tail_policy=tail, max_slices=0, iq_sample_rate=100000000)}))
        from scripts.legacy_submission_common import save
        save(source/'refresh_manifest.json', dict(status='complete', tail_policy='cover',
             training_run=str(training), profile=str(profile), profile_sha256=file_sha256(profile),
             inference_config=str(config)))
        artifacts = dict(checkpoint=str(checkpoint), yolo=str(detector), profile=str(profile),
                         checkpoint_sha256=file_sha256(checkpoint), detector_sha256=file_sha256(detector))
        sources, counts = {}, {}
        data = self.root/'raw';data.mkdir()
        for item, path in self.predictions.items():
            _, rows = read_csv(path);counts[item] = len(rows)
            sources[item] = dict(predictions=str(path), input_count=len(rows))
            folder = data/item;folder.mkdir()
            for r in rows:
                (folder/Path(r['path']).name).write_bytes(b'IQIQ')
        batch = dict(fingerprint=dict(training_run=str(training), data_root=str(data), **artifacts),
                     runs={i:dict(input_signature='original') for i in ITEMS})
        signatures = {i:'changed' if changed_inventory else 'original' for i in ITEMS}
        args = SimpleNamespace(batch=self.root/'batch.json', profile_refresh=source,
                               data_root=data, scope=scope, output=self.root/'plan')
        return args, batch, sources, artifacts, (data, counts, signatures)

    def call_prepare(self, fixture):
        args, batch, sources, artifacts, inspection = fixture
        with patch('scripts.supplement_full_duration.sources_from_batch', return_value=(batch, sources)), \
             patch('scripts.competition_pipeline.training_artifacts', return_value=artifacts), \
             patch('scripts.competition_pipeline.inspect_inputs', return_value=inspection), \
             patch('scripts.competition_pipeline.input_sampling_diagnostics', return_value={i:dict(uncovered_file_count=0) for i in ITEMS}), \
             patch('subprocess.run') as process, contextlib.redirect_stdout(io.StringIO()):
            result = supplement.prepare(args)
        process.assert_not_called()
        return result, args

    def test_prepare_writes_cover_configs_and_all_unknowns_without_launching_gpu(self):
        result, args = self.call_prepare(self.prepare_fixture())
        self.assertEqual(result, 0)
        plan = read(args.output/'supplement_plan.json')
        self.assertEqual(plan['status'], 'prepared')
        self.assertEqual(plan['items']['1-1']['selected_files'], 3)
        config = yaml.safe_load(Path(plan['items']['1-1']['config']).read_text())['infer']
        self.assertEqual(config['tail_policy'], 'cover')
        self.assertEqual(config['max_slices'], 0)
        self.assertTrue(config['fast'])
        self.assertFalse(config['known_recheck'])
        self.assertEqual(len(config['inputs']), 3)

    def test_prepare_rejects_drop_config_before_starting_any_work(self):
        with self.assertRaisesRegex(ValueError, 'Full-duration'):
            self.call_prepare(self.prepare_fixture(tail='drop'))
        self.assertFalse((self.root/'plan').exists())

    def test_pending_rejects_changed_original_dataset(self):
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            self.call_prepare(self.prepare_fixture(changed_inventory=True))

    def test_all_scope_reinfers_all_files_when_original_inventory_changed(self):
        result, args = self.call_prepare(self.prepare_fixture(scope='all', changed_inventory=True))
        self.assertEqual(result, 0)
        plan = read(args.output/'supplement_plan.json')
        self.assertEqual(plan['items']['1-1']['selected_files'], 4)

    def test_worker_rejects_extraction_errors_instead_of_treating_them_as_cluster_failures(self):
        import signal
        handler = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, handler)
        config = self.root/'config.yaml'
        config.write_text('infer:\n  unknown_input_mode: multi\n')
        args = SimpleNamespace(config=config, result=self.root/'result.json')
        with patch('util.cli.main', return_value=dict(run_dir='/a/run', metrics=dict(error_files=1))):
            with self.assertRaisesRegex(ValueError, 'extraction errors'):
                supplement.worker(args)
        self.assertFalse(args.result.exists())

    def test_worker_can_save_a_subset_cluster_failure_for_final_pool_reclustering(self):
        import signal
        handler = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, handler)
        config = self.root/'config.yaml'
        config.write_text('infer:\n  unknown_input_mode: single\n')
        args = SimpleNamespace(config=config, result=self.root/'result.json')
        with patch('util.cli.main', return_value=dict(run_dir='/a/run', metrics=dict(discovery_constraint_failures=1))):
            self.assertEqual(supplement.worker(args), 0)
        self.assertEqual(read(args.result)['metrics']['discovery_constraint_failures'], 1)

    def test_resume_recovers_a_worker_finished_after_parent_interruption_without_duplicate_inference(self):
        from scripts.legacy_submission_common import save
        run_dir = self.root/'finished_child';run_dir.mkdir()
        r = row('1-1_0000', next(iter(KNOWN)), 'known')
        write_csv(run_dir/'predictions.csv', list(r), [r])
        (run_dir/'unknown_features.npz').write_bytes(b'cache')
        save(run_dir/'run.json', dict(status='complete'))
        result = self.root/'child_result.json'
        save(result, dict(run_dir=str(run_dir), metrics={}))
        plan = self.root/'supplement_plan.json'
        entries = {i:dict(status='empty', inputs=[]) for i in ITEMS}
        entries['1-1'] = dict(status='running', result=str(result), inputs=[dict(sample='1-1_0000')])
        save(plan, dict(status='interrupted', items=entries))
        args = SimpleNamespace(plan=plan, wait_for_gpu=False, gpu_id='0', gpu_min_free_mib=16384)
        with patch('scripts.supplement_full_duration.check_frozen_inputs'), \
             patch('scripts.competition_pipeline.run_command') as launch, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(supplement.run(args), 0)
        launch.assert_not_called()
        self.assertEqual(read(plan)['items']['1-1']['status'], 'complete')

    def test_resume_refuses_to_launch_over_a_running_previous_worker(self):
        from scripts.legacy_submission_common import save
        child = self.root/'live_child';child.mkdir()
        save(child/'run.json', dict(status='running'))
        pointer = self.root/'pointer.json';save(pointer, dict(run_dir=str(child)))
        entries = {i:dict(status='empty', inputs=[]) for i in ITEMS}
        entries['1-1'] = dict(status='running', pointer=str(pointer), inputs=[dict(sample='1-1_0000')])
        plan = self.root/'plan.json';save(plan, dict(status='running', items=entries))
        args = SimpleNamespace(plan=plan, wait_for_gpu=False, gpu_id='0', gpu_min_free_mib=16384)
        with patch('scripts.supplement_full_duration.check_frozen_inputs'), \
             patch('scripts.competition_pipeline.run_command') as launch:
            with self.assertRaisesRegex(ValueError, 'previous child still reports running'):
                supplement.run(args)
        launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
