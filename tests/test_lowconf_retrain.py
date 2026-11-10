import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from prepare_submission import KNOWN, build_parser, build_plan, read_xlsx_rows
from scripts import low_confidence_retrain as flow
from scripts.legacy_submission_common import ITEMS, export_results, write_csv, save, read, read_csv
from util.open_set import FEATURE_VERSION, file_sha256, save_npz


class SharedNoSignalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name);self.predictions = {}
        for item in ITEMS:
            path = self.root/item/'predictions.csv'
            rows = [dict(path=f'/data/{item}_0000.dat', label='no_signal', status='no_signal', error='')]
            if item.startswith('1-'):
                rows += [dict(path=f'/data/{item}_0001.dat', label='Unknown_001', status='novel_candidate', error='')]
            write_csv(path, list(rows[0]), rows);self.predictions[item] = path

    def test_all_four_task_one_workbooks_use_fixed_24_and_task_two_empty_stays_minus_one(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code, summary = export_results(self.predictions, self.root/'out', task1_no_signal_code=24)
        self.assertEqual(code, 0);self.assertEqual(summary['workbooks'], 6)
        report = read(summary['report'])
        self.assertFalse(report['ready']);self.assertTrue(report['has_fallbacks'])
        self.assertEqual(report['task1_no_signal_placeholder']['code'], 24)
        for item in ITEMS[:4]:
            values = read_xlsx_rows(Path(summary['output'])/(item+'.xlsx'))
            self.assertEqual(values[1], [item+'_0000', '24'])
            self.assertEqual(values[2], [item+'_0001', '20'])
        values = read_xlsx_rows(Path(summary['output'])/'2-1/2-1_0000.xlsx')
        self.assertEqual(values[1][0], '-1')

    def test_placeholder_collision_blocks_output_instead_of_merging_into_a_real_unknown(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code, summary = export_results(self.predictions, self.root/'out', task1_no_signal_code=20)
        self.assertEqual(code, 2);self.assertEqual(summary['workbooks'], 0)
        self.assertTrue(any('collides' in e for e in read(summary['report'])['errors']))

    def test_fixed_placeholder_requires_explicit_preview_and_never_replaces_unassigned_unknowns(self):
        args = build_parser().parse_args(['--predictions',str(self.predictions['1-1']),
            '--output',str(self.root/'x'),'--task1-no-signal-code','24'])
        with self.assertRaisesRegex(ValueError, 'requires --preview'):
            build_plan(args)
        args.preview = True
        _, rows = read_csv(self.predictions['1-1'])
        rows[1]['label'] = 'Unknown_unassigned'
        write_csv(self.predictions['1-1'], list(rows[0]), rows)
        _, report = build_plan(args)
        self.assertTrue(any('Unknown_unassigned' in e for e in report['errors']))

    def test_ready_archive_contains_only_workbooks_and_keeps_report_outside(self):
        def export(arguments):
            target=Path(arguments[arguments.index('--output')+1]);target.mkdir()
            (target/'1-1.xlsx').write_bytes(b'workbook fixture')
            save(target.parent/(target.name+'_report.json'),dict(ready=True,written_workbooks=1,
                 errors=[],class_budget_errors=[],coverage_complete=True))
            return 0
        with patch('prepare_submission.main',side_effect=export),contextlib.redirect_stdout(io.StringIO()):
            _, summary=export_results(self.predictions,self.root/'strict')
        with ZipFile(summary['zip']) as archive:
            self.assertEqual(archive.namelist(),['1-1.xlsx'])
        self.assertTrue(Path(summary['report']).is_file())


class LowConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, **overrides):
        values = dict(config=flow.ROOT/'config/competition_lowconf_train.yaml', train_root=None,
            yolo_weights=None, confidence=.02, epochs=None, work_dir=self.root/'work',
            data_root=self.root/'data', dry_run=False, prepare_only=False, wait_for_gpu=False,
            gpu_id='0', gpu_min_free_mib=16384, retry_training=False, retry_export=False)
        values.update(overrides);return SimpleNamespace(**values)

    def test_lower_confidence_applies_to_training_profile_and_cover_inference_without_editing_template(self):
        path=flow.ROOT/'config/competition_lowconf_train.yaml';before=file_sha256(path)
        doc, _, _ = flow.configured(self.args(confidence=.01))
        self.assertEqual(doc['train']['yolo_conf_thres'], .01)
        self.assertEqual(doc['evaluation']['yolo_conf_thres'], .01)
        self.assertEqual(doc['experiment']['protocol'], 'all_known')
        self.assertEqual(doc['evaluation']['tail_policy'], 'cover')
        self.assertEqual(doc['evaluation']['max_slices'], 0)
        self.assertEqual(doc['train']['bbox_cache_mode'], 'off')
        self.assertEqual(doc['train']['singleton_min_confidence'], .5)
        self.assertEqual(file_sha256(path), before)

    def test_invalid_threshold_or_epoch_is_rejected_before_work_is_allocated(self):
        for threshold in [.05, 0, -1, float('nan')]:
            with self.assertRaises(ValueError):flow.configured(self.args(confidence=threshold))
        with self.assertRaises(ValueError):flow.configured(self.args(epochs=0))
        self.assertFalse((self.root/'work').exists())

    def test_dry_run_does_not_scan_raw_data_or_start_any_process(self):
        with patch.object(flow, 'frozen_inputs', side_effect=AssertionError('No raw data scan')), \
             patch.object(flow, 'execute_worker', side_effect=AssertionError('No worker')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(flow.run_workflow(self.args(dry_run=True)),0)
        self.assertFalse((self.root/'work').exists())

    def test_finished_needs_review_forward_pass_is_not_rerun_and_changed_csv_is_rejected(self):
        p=self.root/'run';p.mkdir()
        save(p/'run.json',dict(status='complete'))
        r=dict(path='/data/1-1_0000.dat', label='no_signal', status='no_signal',error='')
        write_csv(p/'predictions.csv',list(r),[r])
        save_npz(p/'unknown_features.npz',dict(kind='unknown_candidates',schema_version=1,
            feature_version=FEATURE_VERSION,signature={},records=[]),features=np.empty((0,4)))
        batch=dict(runs={'1-1':dict(run_dir=str(p),status='needs_review',input_count=1,
            csv_sha256=file_sha256(p/'predictions.csv'),cache_sha256=file_sha256(p/'unknown_features.npz'))})
        self.assertTrue(flow.finished_item(batch,'1-1'))
        r['label']='DJFPV';r['status']='known';write_csv(p/'predictions.csv',list(r),[r])
        with self.assertRaisesRegex(ValueError,'changed'):flow.finished_item(batch,'1-1')

    def test_training_class_inventory_requires_all_twenty_classes(self):
        from prepare_submission import KNOWN
        for name in KNOWN:
            p=self.root/name;p.mkdir();(p/'000.dat').write_bytes(b'IQIQ')
        self.assertEqual(flow.training_inventory(self.root)['files'],20)
        (self.root/next(iter(KNOWN))/'000.dat').unlink()
        with self.assertRaisesRegex(ValueError,'20 class'):flow.training_inventory(self.root)

    def test_task_two_burst_conflict_cannot_be_hidden_by_a_clean_compact_csv(self):
        p=self.root/'run';p.mkdir()
        save(p/'run.json',dict(status='completed_with_errors'))
        save(p/'failure_diagnostics.json',dict(files=[dict(burst_label_conflicts=1)]))
        r=dict(path='/data/2-1_0000.dat',label='no_signal',status='no_signal',error='')
        write_csv(p/'predictions.csv',list(r),[r])
        save_npz(p/'unknown_features.npz',dict(kind='unknown_candidates',schema_version=1,
            feature_version=FEATURE_VERSION,signature={},records=[]),features=np.empty((0,4)))
        batch=dict(runs={'2-1':dict(run_dir=str(p),status='failed',input_count=1)})
        self.assertFalse(flow.finished_item(batch,'2-1'))

    def test_experiment_pointer_is_available_before_training_callback(self):
        from util import run_context
        pointer=self.root/'pointer.json'
        args=SimpleNamespace(protocol='all_known',split_only=False)
        def callback(managed):
            self.assertEqual(Path(read(pointer)['run_dir']),managed.output_dir)
            self.assertEqual(read(pointer)['task'],'experiment')
            return dict(status='complete')
        with patch.object(run_context,'EXPERIMENT_ROOT',self.root/'experiment'), \
             patch.dict('os.environ',{'UAV_EXPERIMENT_RUN_POINTER':str(pointer)}), \
             contextlib.redirect_stderr(io.StringIO()):
            result=run_context.execute_run('experiment',args,callback)
        self.assertTrue(Path(result['run_dir']).is_dir())

    def test_prepare_only_records_a_workflow_without_launching_gpu_workers(self):
        args=self.args(prepare_only=True)
        frozen=dict(config={},sources={})
        with patch.object(flow,'frozen_inputs',return_value=frozen), \
             patch.object(flow,'execute_worker') as launch,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(flow.run_workflow(args),0)
        launch.assert_not_called()
        self.assertEqual(read(args.work_dir/'workflow.json')['status'],'prepared')

    def test_workflow_stops_after_training_failure_and_preserves_a_useful_error(self):
        args=self.args()
        with patch.object(flow,'frozen_inputs',return_value={}), \
             patch.object(flow,'execute_worker',side_effect=ValueError('training failure')) as launch:
            with self.assertRaisesRegex(ValueError,'training failure'):flow.run_workflow(args)
        self.assertEqual(launch.call_count,1)
        self.assertEqual(read(args.work_dir/'workflow.json')['status'],'failed')

    def test_nonstandard_export_is_reported_and_a_second_run_skips_both_gpu_phases(self):
        args=self.args();training=self.root/'new_training';training.mkdir()
        save(training/'experiment.json',dict(status='complete'))
        batch=self.root/'batch.json';save(batch,dict(runs={i:{} for i in ITEMS}))
        predictions={}
        for i in ITEMS:
            path=self.root/'predictions'/i/'predictions.csv'
            write_csv(path,['path','label','status','error'],[dict(path=f'{i}_0000.dat',label='no_signal',status='no_signal',error='')])
            predictions[i]=path
        def worker(root,phase,state,args,lock_fd,**kwargs):
            state[phase]=dict(status='complete',run_dir=str(training)) if phase=='training' else dict(status='complete',manifest=str(batch))
        def export(predictions,path,**kwargs):
            fallback=kwargs.get('task1_no_signal_code')==24
            return (0 if fallback else 2),dict(ready=False,workbooks=3504 if fallback else 0,report='dummy',output='dummy')
        with patch.object(flow,'frozen_inputs',return_value={}), \
             patch.object(flow,'execute_worker',side_effect=worker) as launch, \
             patch.object(flow,'validated_training',return_value={}), \
             patch.object(flow,'finished_item',return_value=True), \
             patch.object(flow,'sources_from_batch',return_value=({},{})), \
             patch.object(flow,'recluster',return_value=predictions), \
             patch.object(flow,'export_results',side_effect=export) as write, \
             patch.object(flow,'predictions_from_export',return_value=predictions), \
             patch.object(flow,'verify_export'),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(flow.run_workflow(args),0)
            self.assertEqual(launch.call_count,2)
            self.assertEqual(write.call_count,2)
            self.assertEqual(read(args.work_dir/'workflow.json')['status'],'complete_with_nonstandard_export')
            launch.reset_mock();write.reset_mock()
            self.assertEqual(flow.run_workflow(args),0)
            launch.assert_not_called();write.assert_not_called()


if __name__=='__main__':unittest.main()
