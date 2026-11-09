import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np
import yaml

import prepare_submission as submission
from scripts import competition_pipeline as pipeline
from util.run_context import metric_error_count


class SubmissionMainlineTests(unittest.TestCase):
    def test_known_micro_f1_counts_unknown_false_acceptances(self):
        from util.open_set_metrics import evaluate_open_set
        def row(path,truth,predicted):
            return dict(path=path,true_class_names=[truth],signals=[dict(track_id='T1',
                recognition_status='known',class_name=predicted,valid_votes=1,observation_count=1)])
        r=evaluate_open_set([row('a','A','A'),row('b','A','B'),row('u','U','A')],
            {'A':0,'B':1},{'estimated_unknown_class_count':0})['metrics']
        self.assertAlmostEqual(r['known_micro_precision'],1/3)
        self.assertEqual(r['known_micro_recall'],.5);self.assertEqual(r['known_micro_f1'],.4)
        self.assertEqual(r['unknown_ari_missing_recordings'],1)
    def test_full_task_two_empty_scenes_match_official_minus_one_sentinel(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t);source = p/'predictions.csv'
            with source.open('w', newline='') as stream:
                w = csv.writer(stream);w.writerow(['path','track_id','label','status','error','bursts_json'])
                w.writerows([[f'/data/2-1_{i:04}.dat','','no_signal','no_signal','',''] for i in range(1750)])
            with contextlib.redirect_stdout(io.StringIO()):
                code = submission.main(['--predictions',str(source),'--output',str(p/'out')])
            self.assertEqual(code,0)
            report=json.loads((p/'out_report.json').read_text())
            self.assertTrue(report['ready']);self.assertEqual(report['written_workbooks'],1750)
            self.assertEqual(report['datasets']['2-1']['output_signal_rows'],0)
            self.assertEqual(report['datasets']['2-1']['no_signal_files'],1750)
            with ZipFile(p/'out/2-1/2-1_0000.xlsx') as z:
                root=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
                rows=root.find(f'{{{submission.NS}}}sheetData')
                self.assertEqual([int(c.findtext(f'{{{submission.NS}}}v')) for c in rows[1]],[-1,0,0])

    def test_no_signal_cannot_hide_detected_track(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);source=p/'bad.csv'
            source.write_text('path,track_id,label,status,error,bursts_json\n/data/2-1_0000.dat,T1,no_signal,no_signal,,[]\n')
            with contextlib.redirect_stdout(io.StringIO()):
                code=submission.main(['--predictions',str(source),'--output',str(p/'out'),'--check'])
            r=json.loads((p/'out_report.json').read_text());self.assertEqual(code,2)
            self.assertFalse(r['valid_format']);self.assertFalse(r['ready'])

    def test_source_root_mismatch_changes_coverage_flag(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);source=p/'pred.csv';data=p/'data';data.mkdir()
            (data/'1-1_0000.dat').touch()
            source.write_text('path,label,status,error\n/data/1-1_0000.dat,Unknown_001,novel_candidate,\n/data/1-1_0001.dat,Unknown_002,novel_candidate,\n')
            with contextlib.redirect_stdout(io.StringIO()):
                submission.main(['--predictions',str(source),'--output',str(p/'out'),'--input-root',str(data),'--check'])
            self.assertFalse(json.loads((p/'out_report.json').read_text())['coverage_complete'])

    def test_task_one_event_diagnostic_does_not_fail_primary_record_status(self):
        self.assertEqual(metric_error_count(dict(submission_unit='recording',burst_label_conflicts=12)),0)
        self.assertEqual(metric_error_count(dict(submission_unit='burst',burst_label_conflicts=12)),12)
        self.assertEqual(metric_error_count(dict(submission_unit='recording',burst_label_conflicts=12,error_files=1)),1)

    def test_distinct_tracks_overlap_without_being_forced_duplicates(self):
        from util.burst_normalization import normalize_bursts
        a=dict(start_ms=1,end_ms=2,frequency_low_offset_mhz=1,frequency_high_offset_mhz=2,
               label='A',recognition_status='known',track_id='T1',observation_id='S0G0')
        b={**a,'track_id':'T2','observation_id':'S0G1','label':'B'}
        events,d=normalize_bursts([a,b]);self.assertEqual(len(events),2)
        self.assertEqual(d['label_conflicts'],0);self.assertEqual(d['overlapping_distinct_tracks'],1)
        b['observation_id']='S0G0'
        _,d=normalize_bursts([a,b]);self.assertEqual(d['label_conflicts'],1)

    def test_subpixel_or_outside_boxes_are_not_accepted_after_rounding(self):
        from util.event_detection import select_event_groups
        from unittest.mock import Mock
        cfg=SimpleNamespace(sampling_rate=122880000,valid_band_policy='source',event_context_gap_ms=0,
                            singleton_min_confidence=.5,singleton_min_contrast_z=1)
        pre=Mock();pre.select_signal_groups.return_value=[]
        groups,decisions=select_event_groups(pre,[[10,48,20,48.2]],np.zeros((512,750)),cfg,[.9],100e6)
        self.assertFalse(groups);self.assertEqual(decisions[0]['reason'],'empty_after_quantization')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.data=self.root/'data';self.data.mkdir()
        for name in pipeline.PIPELINE_SOURCES:
            path=self.root/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_text('# pipeline fixture\n')
        self.artifacts=dict(training_run=str(self.root/'training'),checkpoint=str(self.root/'best.pth'),
            profile=str(self.root/'profile.npz'),yolo=str(self.root/'yolo.pt'),
            checkpoint_sha256='ckpt',profile_sha256='profile',detector_sha256='yolo')
        self.config=self.root/'inference.yaml'
        self.config.write_text(yaml.safe_dump(dict(evaluation=dict(checkpoint_path=self.artifacts['checkpoint'],
            yolo_weights=self.artifacts['yolo'],signal_mode='multi'),infer=dict(inputs=['/placeholder'],
            open_set_profile=self.artifacts['profile']))))
        for target,replacement in [('scripts.competition_pipeline.ROOT',self.root),
            ('util.run_context.EXPERIMENT_ROOT',self.root/'experiment')]:
            patcher=patch(target,replacement);patcher.start();self.addCleanup(patcher.stop)
        patcher=patch('scripts.competition_pipeline.training_artifacts',return_value=self.artifacts)
        patcher.start();self.addCleanup(patcher.stop)

    def dat(self,item,i=0,size=40_000_000):
        folder=self.data/item;folder.mkdir(exist_ok=True)
        p=folder/f'{item}_{i:04}.dat'
        with p.open('wb') as stream:stream.truncate(size)
        return p

    def args(self,items,resume=None,dry=False):
        return SimpleNamespace(training_run=self.root/'training',data_root=self.data,items=items,
            config=self.config,yolo_weights=None,full=False,dry_run=dry,resume=resume)

    def runner(self,command,env,log):
        item=command[command.index('--test_item')+1]
        p=self.root/'experiment'/f'open_set_inference_{item}'
        p.mkdir(exist_ok=True)
        (p/'run.json').write_text(json.dumps(dict(status='complete')))
        (p/'unknown_features.npz').write_bytes(b'synthetic-cache')
        (p/'predictions.csv').write_text(f'path,label,status,error\n{self.data/item/(item+"_0000.dat")},DJFPV,known,\n')
        Path(env['UAV_COMPETITION_RUN_POINTER']).write_text(json.dumps(dict(run_dir=str(p))))
        self.assertIn('--no-known_recheck',command)
        self.assertEqual(command[command.index('--unknown_bounded_strategy')+1],'density_merge')
        return 0

    def test_batch_records_exact_paths_continues_and_resume_skips_completed(self):
        self.dat('1-1');self.dat('1-2')
        def run(command,env,log):
            if command[command.index('--test_item')+1]=='1-1':return 2
            return self.runner(command,env,log)
        with contextlib.redirect_stdout(io.StringIO()):
            code=pipeline.infer(self.args(['1-1','1-2']),runner=run)
        self.assertEqual(code,2)
        manifest=next((self.root/'experiment').glob('competition_batch_*/batch_manifest.json'))
        state=json.loads(manifest.read_text())
        self.assertEqual(state['runs']['1-1']['status'],'failed')
        self.assertEqual(state['runs']['1-2']['status'],'complete')
        calls=[]
        def resumed(command,env,log):
            calls.append(command[command.index('--test_item')+1]);return self.runner(command,env,log)
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.infer(self.args(['1-1','1-2'],manifest),runner=resumed)
        self.assertEqual(calls,['1-1'])

    def test_resume_detects_modified_cache(self):
        self.dat('1-1')
        with contextlib.redirect_stdout(io.StringIO()):pipeline.infer(self.args(['1-1']),runner=self.runner)
        manifest=next((self.root/'experiment').glob('competition_batch_*/batch_manifest.json'))
        state=json.loads(manifest.read_text());Path(state['runs']['1-1']['run_dir'],'predictions.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError,'changed'):
            pipeline.infer(self.args(['1-1'],manifest),runner=self.runner)

    def test_spectral_is_forwarded_to_inference_and_frozen_for_resume(self):
        self.dat('1-1');args=self.args(['1-1']);args.cluster_method='spectral'
        def runner(command,env,log):
            self.assertEqual(command[command.index('--unknown_cluster_backend')+1],'spectral')
            self.assertEqual(command[command.index('--unknown_bounded_strategy')+1],'spectral')
            self.assertEqual(command[command.index('--spectral_neighbors')+1],'10')
            folder=self.root/'experiment/spectral';folder.mkdir()
            (folder/'run.json').write_text(json.dumps(dict(status='complete')))
            (folder/'unknown_features.npz').write_bytes(b'cache')
            (folder/'predictions.csv').write_text('path,label,status,error\n/data/1-1_0000.dat,DJFPV,known,\n')
            Path(env['UAV_COMPETITION_RUN_POINTER']).write_text(json.dumps(dict(run_dir=str(folder))))
            return 0
        with contextlib.redirect_stdout(io.StringIO()):pipeline.infer(args,runner=runner)
        manifest=next((self.root/'experiment').glob('competition_batch_*/batch_manifest.json'))
        state=json.loads(manifest.read_text())
        self.assertEqual(state['fingerprint']['strategy'],'spectral')
        self.assertEqual(state['discovery']['spectral_options']['neighbors'],10)
        args.resume=manifest;args.spectral_neighbors=20
        with self.assertRaisesRegex(ValueError,'changed'):pipeline.infer(args,runner=runner)

    def test_gpu_wait_admits_the_checked_device_before_each_inference_worker(self):
        from contextlib import contextmanager
        from util.gpu_wait import GpuMemory
        self.dat('1-1');args=self.args(['1-1']);args.wait_for_gpu=True;args.gpu_id='0';args.gpu_min_free_mib=16384
        @contextmanager
        def ready(policy):
            manifest=next((self.root/'experiment').glob('competition_batch_*/batch_manifest.json'))
            self.assertEqual(json.loads(manifest.read_text())['runs']['1-1']['status'],'waiting')
            yield GpuMemory(0,'GPU-test',24576,23000)
        def worker(command,env,log):
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'GPU-test')
            return self.runner(command,env,log)
        with patch('util.gpu_wait.reserve_gpu',ready),contextlib.redirect_stdout(io.StringIO()):
            pipeline.infer(args,runner=worker)

    def test_preflight_rejects_wrong_duration_and_dry_run_creates_no_output(self):
        self.dat('1-1',size=4)
        with self.assertRaisesRegex(ValueError,'100-ms'):
            pipeline.inspect_inputs(self.data,['1-1'])
        self.dat('1-1')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.infer(self.args(['1-1'],dry=True)),0)
        self.assertFalse((self.root/'experiment').exists())

    def test_observed_short_capture_is_accepted_and_reported_without_rewriting(self):
        p=self.dat('1-1',size=39_845_888);details={}
        _,counts,_=pipeline.inspect_inputs(self.data,['1-1'],diagnostics=details)
        self.assertEqual(counts,{'1-1':1});self.assertEqual(p.stat().st_size,39_845_888)
        self.assertAlmostEqual(details['1-1']['min_duration_ms'],99.61472)
        self.assertEqual(details['1-1']['non_nominal_files'][0]['size_bytes'],39_845_888)
        for n in (pipeline.MIN_IQ_BYTES,pipeline.NOMINAL_IQ_BYTES):
            with self.subTest(size=n):
                self.dat('1-1',size=n)
                self.assertEqual(pipeline.inspect_inputs(self.data,['1-1'])[1],{'1-1':1})

    def test_partial_iq_pair_is_rejected_even_with_near_nominal_duration(self):
        self.dat('1-1',size=39_999_998)
        with self.assertRaisesRegex(ValueError,'whole int16 I/Q pairs.*39,999,998'):
            pipeline.inspect_inputs(self.data,['1-1'])

    def test_duration_compatibility_does_not_accept_grossly_short_or_long_captures(self):
        for n in (pipeline.MIN_IQ_BYTES-4,pipeline.NOMINAL_IQ_BYTES+4,20_000_000):
            with self.subTest(size=n):
                self.dat('1-1',size=n)
                with self.assertRaisesRegex(ValueError,'got .* bytes'):
                    pipeline.inspect_inputs(self.data,['1-1'])

    def test_item_id_validation_is_independent_of_iq_length(self):
        p=self.dat('1-1');p.rename(p.with_name('1-2_0000.dat'))
        with self.assertRaisesRegex(ValueError,'Expected a 1-1 file ID, got 1-2_0000'):
            pipeline.inspect_inputs(self.data,['1-1'])

    def test_short_capture_dry_run_exposes_drop_coverage_and_cover_plan(self):
        self.dat('1-1',size=39_845_888);out=io.StringIO();err=io.StringIO()
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
            self.assertEqual(pipeline.infer(self.args(['1-1'],dry=True)),0)
        data=json.loads(out.getvalue());gap=data['input_sampling']['1-1']['coverage_gaps'][0]
        self.assertAlmostEqual(gap['unobserved_ms'],49.61472)
        self.assertIn('49.61472 ms unobserved',err.getvalue())
        self.assertFalse((self.root/'experiment').exists())
        details=data['input_duration_diagnostics']
        coverage=pipeline.input_sampling_diagnostics(details,SimpleNamespace(iq_sample_rate=100_000_000,
            iq_stride_ms=50,tail_policy='cover',max_slices=0))
        self.assertEqual(coverage['1-1']['uncovered_file_count'],0)

    def test_large_burst_json_field_does_not_break_batch_csv_review(self):
        p=self.root/'multi.csv'
        bursts=json.dumps([dict(start_ms=i/1000,end_ms=i/1000+.0005,
            frequency_low_offset_mhz=1,frequency_high_offset_mhz=2) for i in range(2000)])
        self.assertGreater(len(bursts),128*1024)
        with p.open('w',newline='') as stream:
            w=csv.writer(stream);w.writerow(['path','label','status','error','bursts_json'])
            w.writerow(['/data/2-1_0000.dat','DJFPV','known','',bursts])
        self.assertEqual(pipeline.review_csv(p,'2-1',1),0)

    def test_incomplete_batch_cannot_submit(self):
        p=self.root/'batch.json';p.write_text(json.dumps(dict(runs={})))
        with self.assertRaisesRegex(ValueError,'All six'):
            pipeline.submit(SimpleNamespace(manifest=p,output=None,check=False))

    def test_six_item_manifest_joint_selection_and_3504_workbook_export(self):
        from util.open_set import save_npz, normalize_features, FEATURE_VERSION, file_sha256
        import subprocess
        real_root=Path(__file__).resolve().parents[1]
        real_run=subprocess.run
        runs={}
        for item in pipeline.ITEMS:
            folder=self.root/'experiment'/('run_'+item);folder.mkdir(parents=True)
            (folder/'run.json').write_text(json.dumps(dict(status='complete')))
            records=[];features=[];table=[]
            k=2 if item in ('1-1','1-3') else 3
            n=1750 if item.startswith('2') else k*7
            for i in range(n):
                path=self.dat(item,i)
                if item.startswith('2'):
                    table.append([str(path),'no_signal','no_signal',''])
                else:
                    c=i//7;table.append([str(path),f'Unknown_{c+1:03}','novel_candidate',''])
                    records.append(dict(path=str(path),recording_id=str(path),track_id='T1',
                        discovery_is_primary=True,discovery_primary_path=str(path),discovery_primary_track_id='T1'))
                    features.append(np.eye(5)[c]+np.random.default_rng(i).normal(0,.002,5))
            with (folder/'predictions.csv').open('w',newline='') as stream:
                w=csv.writer(stream);w.writerow(['path','label','status','error']);w.writerows(table)
            if features:
                save_npz(folder/'unknown_features.npz',dict(kind='unknown_candidates',schema_version=1,
                    feature_version=FEATURE_VERSION,signature=dict(sha256='fixture'),records=records),
                    features=normalize_features(features))
            else:
                (folder/'unknown_features.npz').write_bytes(b'empty synthetic task-two cache; not loaded by submit')
            runs[item]=dict(status='complete',run_dir=str(folder),csv_sha256=file_sha256(folder/'predictions.csv'),
                cache_sha256=file_sha256(folder/'unknown_features.npz'))
        _,_,signatures=pipeline.inspect_inputs(self.data,pipeline.ITEMS)
        for item in runs:runs[item]['input_signature']=signatures[item]
        batch=dict(runs=runs,fingerprint=dict(training_run=self.artifacts['training_run'],
            yolo=self.artifacts['yolo'],profile_sha256=self.artifacts['profile_sha256'],data_root=str(self.data),
            pipeline_sources=pipeline.source_fingerprint()),
            discovery=dict(min_cluster_size=5,min_recordings=2,min_samples=3,min_silhouette=.15))
        manifest=self.root/'batch.json';manifest.write_text(json.dumps(batch))
        def child(command,**kwargs):
            command=list(command);path=Path(command[1]);command[1]=str(real_root/path.relative_to(self.root))
            return real_run(command,cwd=real_root,check=True,capture_output=True,text=True)
        with patch('scripts.competition_pipeline.subprocess.run',side_effect=child):
            self.assertEqual(pipeline.submit(SimpleNamespace(manifest=manifest,output=self.root/'out',check=False)),0)
        report=json.loads((self.root/'out_report.json').read_text())
        self.assertTrue(report['ready']);self.assertEqual(report['written_workbooks'],3504)
        final=json.loads(manifest.read_text());selection=json.loads(Path(final['last_submission']['joint_selection']).read_text())
        self.assertEqual(selection['selected_counts'],{'1-1':2,'1-2':3,'1-3':2,'1-4':3})
        # A spectral batch must remain spectral during final joint selection.
        final['fingerprint']['strategy']='spectral'
        final['discovery'].update(strategy='spectral',spectral_options=dict(neighbors=10,seed=42,n_init=10))
        manifest.write_text(json.dumps(final))
        def spectral_child(command,**kwargs):
            if Path(command[1]).name=='recluster_competition_batch.py':
                self.assertEqual(command[command.index('--bounded-strategy')+1],'spectral')
                self.assertEqual(command[command.index('--spectral-neighbors')+1],'10')
            return child(command,**kwargs)
        with patch('scripts.competition_pipeline.subprocess.run',side_effect=spectral_child):
            self.assertEqual(pipeline.submit(SimpleNamespace(manifest=manifest,output=self.root/'spectral_out',check=True)),0)
        spectral_report=json.loads((self.root/'spectral_out_report.json').read_text())
        self.assertTrue(spectral_report['ready']);self.assertEqual(spectral_report['written_workbooks'],0)


class TrainingPreflightTests(unittest.TestCase):
    def test_complete_twenty_class_profile_and_weight_source_guards(self):
        from util.open_set import save_npz, FEATURE_VERSION, file_sha256
        from importlib.metadata import version
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);training=p/'training_run';(training/'training/models').mkdir(parents=True)
            (training/'open_set').mkdir();(p/'util').mkdir()
            checkpoint=training/'training/models/best.pth';checkpoint.write_bytes(b'synthetic checkpoint')
            yolo=p/'detector.pt';yolo.write_bytes(b'synthetic detector')
            for name in ('open_set.py','event_detection.py'):(p/'util'/name).write_text('# fixture\n')
            meta=dict(kind='open_set_profile',schema_version=1,feature_version=FEATURE_VERSION,
                classes=submission.KNOWN,score_method='maha_pp_shrink',calibration_unit='primary',
                signature=dict(details=dict(checkpoint_sha256=file_sha256(checkpoint),detector_sha256=file_sha256(yolo),
                    source_sha256={f'util/{name}':file_sha256(p/'util'/name) for name in ('open_set.py','event_detection.py')},
                    libraries=dict(numpy=version('numpy')))))
            save_npz(training/'open_set/known_profile.npz',meta,reference=np.eye(20))
            (training/'experiment.json').write_text(json.dumps(dict(status='complete',protocol='all_known')))
            (training/'effective_profile.json').write_text(json.dumps(dict(yolo_weights=str(yolo))))
            with patch('scripts.competition_pipeline.ROOT',p):
                result=pipeline.training_artifacts(training)
                self.assertEqual(result['checkpoint_sha256'],file_sha256(checkpoint))
                checkpoint.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'mismatched'):
                    pipeline.training_artifacts(training)


if __name__=='__main__':unittest.main()
