import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

import prepare_submission as submission
from util.burst_normalization import normalize_bursts, collect_raw_bursts, finalize_bursts
from util.competition_protocol import protocol_for_item, validate_counts, choose_joint_counts
from util.competition_metrics import evaluate_events
from util.bounded_discovery import bounded_partition
from util.unknown_discovery import discover_unknown
from util.eval_common import INFERENCE_DEFAULTS
from util.event_detection import select_event_groups
from util.preprocess import SignalPreprocessor


def burst(start=1,end=2,**extra):
    return dict(start_ms=start,end_ms=end,frequency_low_offset_mhz=1.,frequency_high_offset_mhz=2.,label='A',**extra)


class BurstTests(unittest.TestCase):
    def test_same_frequency_separate_times_are_three_events(self):
        out,d=normalize_bursts([burst(1,2),burst(3,4),burst(8,9)])
        self.assertEqual(len(out),3)
        self.assertEqual(d['duplicates_removed'],0)

    def test_duplicate_overlap_and_input_order(self):
        a=burst(10,20,window_start_ms=0,window_end_ms=50)
        b=burst(10,20,window_start_ms=5,window_end_ms=55)
        out,d=normalize_bursts([a,b],'file')
        reverse,_=normalize_bursts([b,a],'file')
        self.assertEqual(len(out),1);self.assertEqual(d['duplicates_removed'],1)
        self.assertEqual(out[0]['burst_id'],reverse[0]['burst_id'])

    def test_boundary_fragments_stitch(self):
        out,d=normalize_bursts([burst(48,50,window_start_ms=0,window_end_ms=50),
                               burst(50,53,window_start_ms=50,window_end_ms=100)])
        self.assertEqual([(e['start_ms'],e['end_ms']) for e in out],[(48.,53.)])
        self.assertEqual(d['boundary_fragments_joined'],1)

    def test_boundary_nearby_but_separated_events_stay_separate(self):
        out,_=normalize_bursts([burst(48,49.8,window_start_ms=0,window_end_ms=50),
                               burst(50.2,53,window_start_ms=50,window_end_ms=100)])
        self.assertEqual(len(out),2)

    def test_partial_observations_in_overlapping_windows_are_one_event(self):
        out,d=normalize_bursts([burst(40,50,window_start_ms=0,window_end_ms=50),
                               burst(40,60,window_start_ms=25,window_end_ms=75)])
        self.assertEqual(len(out),1);self.assertEqual(out[0]['end_ms'],60.)
        self.assertEqual(d['duplicates_removed'],1)

    def test_measurable_gap_at_window_boundary_is_not_stitched(self):
        out,_=normalize_bursts([burst(48,49.99,window_start_ms=0,window_end_ms=50),
                               burst(50.01,53,window_start_ms=50,window_end_ms=100)])
        self.assertEqual(len(out),2)

    def test_class_conflicts_do_not_merge(self):
        a=burst();b=burst();b['label']='B'
        out,d=normalize_bursts([a,b]);self.assertEqual(len(out),2);self.assertEqual(d['label_conflicts'],1)

    def test_invalid_times_numbers_and_source_window_rejected(self):
        for e in (burst(3,1),burst(-1,2),burst(1,float('nan')),burst(1,3,window_start_ms=0,window_end_ms=2)):
            with self.subTest(e=e),self.assertRaises(ValueError):normalize_bursts([e])

    def test_full_and_fast_provenance_finalization_match(self):
        source={'path':'x.dat','signals':[{'track_id':'T1','recognition_status':'known','display_label':'A'}]}
        rows=[{'slice_start_ms':0.,'slice_end_ms':50.,'groups':[{'track_id':'T1','observation_id':'S0G0','pulses':[burst()]}]}]
        a=copy.deepcopy(source);b=copy.deepcopy(source)
        finalize_bursts([a],{'x.dat':rows});collect_raw_bursts(b,rows);finalize_bursts([b])
        self.assertEqual(a['bursts'],b['bursts'])


class CountTests(unittest.TestCase):
    def candidates(self,count=2):
        items=[]
        for c in range(count):
            for i in range(7):
                f=np.eye(5)[c]+np.random.default_rng(c*100+i).normal(0,.002,5)
                items.append({'path':f'{c}-{i}.dat','recording_id':f'{c}-{i}', 'feature':f,
                              'track':{'track_id':'T1','discovery_is_primary':True}})
        return items

    def test_public_ranges_and_joint_relations(self):
        self.assertEqual((protocol_for_item('1-1').minimum,protocol_for_item('1-1').maximum),(2,3))
        self.assertTrue(validate_counts({'1-1':8}));self.assertTrue(validate_counts({'1-1':3,'1-2':3}))
        self.assertFalse(validate_counts({'1-1':2,'1-3':2,'1-2':4,'1-4':4,'2-1':5}))

    def test_joint_selection_uses_scores_without_ground_truth(self):
        out=choose_joint_counts({'1-1':{2:.9,3:.1},'1-2':{3:.2,4:.8},'1-3':{2:.8,3:.2},'1-4':{3:.3,4:.8}})
        self.assertEqual(out,{'1-1':2,'1-2':4,'1-3':2,'1-4':4})

    def test_bounded_discovery_uses_features_and_is_order_invariant(self):
        a=self.candidates();b=copy.deepcopy(list(reversed(a)))
        da=discover_unknown(a,input_mode='single',competition_profile='task1_n1')
        db=discover_unknown(b,input_mode='single',competition_profile='task1_n1')
        self.assertTrue(da['constraint_satisfied']);self.assertEqual(da['estimated_unknown_class_count'],2)
        self.assertEqual({c['path']:c['track']['unknown_cluster_id'] for c in a},{c['path']:c['track']['unknown_cluster_id'] for c in b})
        self.assertEqual(da['diagnostics']['selected_K'],db['diagnostics']['selected_K'])

    def test_identical_features_cannot_fake_multiple_classes(self):
        labels,d=bounded_partition(np.tile([1.,0.],(20,1)),[str(i) for i in range(20)],2,3)
        self.assertTrue(np.all(labels==-1));self.assertFalse(d['constraint_satisfied'])

    def test_duplicated_components_from_one_record_do_not_add_support(self):
        items=self.candidates();a=discover_unknown(items,competition_profile='task2')
        b=copy.deepcopy(items)
        for i in range(20):
            c=copy.deepcopy(items[0]);c['track']['track_id']=f'copy{i}';b.append(c)
        d=discover_unknown(b,competition_profile='task2')
        self.assertEqual(a['estimated_unknown_class_count'],d['estimated_unknown_class_count'])
        self.assertEqual(a['diagnostics']['unique_fit_components'],d['diagnostics']['unique_fit_components'])

    def test_empty_task_two_and_insufficient_task_one(self):
        self.assertTrue(discover_unknown([],competition_profile='task2')['constraint_satisfied'])
        self.assertFalse(discover_unknown([],input_mode='single',competition_profile='task1_n1')['constraint_satisfied'])

    def test_wrong_input_assumption_and_illegal_fixed_k(self):
        with self.assertRaises(ValueError):discover_unknown([],competition_profile='task1_n1')
        with self.assertRaises(ValueError):discover_unknown(self.candidates(),competition_profile='task2',fixed_k=8)


class EventDetectionTests(unittest.TestCase):
    def cfg(self,**extra):
        values={**INFERENCE_DEFAULTS,'event_mode':'physical','valid_band_policy':'source'};values.update(extra)
        return SimpleNamespace(**values)

    def test_isolated_burst_with_independent_evidence_survives(self):
        cfg=self.cfg();pre=SignalPreprocessor(cfg,Mock());spec=np.zeros((512,750));spec[250:260,100:150]=1.
        groups,decisions=select_event_groups(pre,[[100,250,150,260]],spec,cfg,[.95],100e6)
        self.assertEqual(len(groups),1);self.assertEqual(groups[0]['group_type'],'isolated_burst')

    def test_weak_singleton_not_promoted(self):
        cfg=self.cfg();pre=SignalPreprocessor(cfg,Mock());spec=np.ones((512,750))
        groups,_=select_event_groups(pre,[[100,250,150,260]],spec,cfg,[.1],100e6)
        self.assertFalse(groups)

    def test_outside_capture_band_rejected_before_classification(self):
        cfg=self.cfg();pre=SignalPreprocessor(cfg,Mock());spec=np.ones((512,750))
        groups,decisions=select_event_groups(pre,[[10,0,50,20]],spec,cfg,[.99],100e6)
        self.assertFalse(groups);self.assertEqual(decisions[0]['reason'],'outside_source_nyquist')

    def test_edge_boxes_remain_on_correct_resampled_grid(self):
        cfg=self.cfg(singleton_min_contrast_z=-100);pre=SignalPreprocessor(cfg,Mock());spec=np.ones((512,750))
        groups,_=select_event_groups(pre,[[10,40,50,80]],spec,cfg,[.99],100e6)
        self.assertTrue(groups)
        self.assertGreaterEqual(groups[0]['freq_edge_min'],-50.)
        self.assertEqual(cfg.sampling_rate,122880000.)

    def test_same_carrier_distant_contexts_remain_independent(self):
        cfg=self.cfg(event_context_gap_ms=5.);pre=SignalPreprocessor(cfg,Mock())
        spec=np.zeros((512,750));spec[250:260,10:70]=1;spec[250:260,500:560]=1
        groups,_=select_event_groups(pre,[[10,250,70,260],[500,250,560,260]],spec,cfg,[.99,.99],100e6)
        self.assertEqual(len(groups),2)
        self.assertTrue(all(len(g['boxes'])==1 for g in groups))

    def test_optional_appearance_gate_separates_same_carrier_sources(self):
        from tests.test_multi_signal_eval import observation
        from util.multi_signal_eval import FrequencyTracker
        tracker=FrequencyTracker(feature_cosine_min=.8)
        a=observation(0,100,[.9,.1,0]);a['embedding']=[1.,0.]
        b=observation(1,100,[.1,.9,0]);b['embedding']=[0.,1.]
        tracker.update([a]);tracker.update([b]);self.assertEqual(len(tracker.tracks),2)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def run_case(self,rows,extra=(),check=True):
        path=self.root/'predictions.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        with contextlib.redirect_stdout(io.StringIO()):
            code=submission.main(['--predictions',str(path),'--output',str(self.root/'out'),*(['--check'] if check else []),*extra])
        return code,json.loads((self.root/'out_report.json').read_text())

    def single(self,i=0,label='Unknown_001',status='novel_candidate'):
        return {'path':f'/data/1-1_{i:04}.dat','label':label,'status':status,'error':''}

    def multi(self,pulses):
        return {'path':'/data/2-1_0000.dat','track_id':'T1','label':'DJFPV','status':'known','error':'','bursts_json':json.dumps(pulses)}

    def test_eight_unknowns_are_not_ready(self):
        code,r=self.run_case([self.single(i,f'Unknown_{i+1:03}') for i in range(8)],['--expected-count','8'])
        self.assertEqual(code,2);self.assertFalse(r['class_budget_valid']);self.assertFalse(r['ready'])

    def test_shared_fallback_is_preview_only(self):
        code,r=self.run_case([self.single(status='no_signal',label='no_signal')],['--unresolved','shared','--preview','--expected-count','1'])
        self.assertEqual(code,0);self.assertTrue(r['has_fallbacks']);self.assertFalse(r['ready'])

    def test_complete_task_one_write_and_readback(self):
        code,r=self.run_case([self.single(),self.single(1,'Unknown_002')],['--expected-count','2'],check=False)
        self.assertEqual(code,0);self.assertTrue(r['ready']);self.assertEqual(r['written_workbooks'],1)
        self.assertEqual(len(submission.read_xlsx_rows(self.root/'out/1-1.xlsx')),3)

    def test_missing_coverage_is_not_ready(self):
        _,r=self.run_case([self.single(),self.single(1,'Unknown_002')])
        self.assertFalse(r['coverage_complete']);self.assertFalse(r['ready'])

    def test_same_frequency_bursts_and_duplicate_observation(self):
        p=[burst(1,2),burst(8,9),burst(1,2)]
        _,r=self.run_case([self.multi(p)],['--expected-count','1'])
        self.assertFalse(r['ready']);self.assertEqual(r['datasets']['2-1']['output_signal_rows'],2)

    def test_invalid_time_is_format_error(self):
        code,r=self.run_case([self.multi([burst(5,2)])],['--expected-count','1'])
        self.assertEqual(code,2);self.assertFalse(r['valid_format'])

    def test_full_infer_schema(self):
        rows=[{'path':f'/data/1-1_{i:04}.dat','status':'ok','error':'','single_signal_prediction':json.dumps({'display_label':f'Unknown_{i+1:03}','recognition_status':'novel_candidate'})} for i in range(2)]
        code,r=self.run_case(rows,['--expected-count','2']);self.assertEqual(code,0);self.assertTrue(r['ready'])


class MetricTests(unittest.TestCase):
    def event(self,i=0,label=20):return {'file_id':'a','class_id':label,'frequency_low_hz':1.,'frequency_high_hz':2.,'start_ms':i*10.,'end_ms':i*10.+2.}

    def test_one_prediction_cannot_match_three_same_frequency_truth_events(self):
        r=evaluate_events([self.event(i) for i in range(3)],[self.event()])
        self.assertEqual(r['detection']['tp'],1);self.assertEqual(r['detection']['fn'],2)

    def test_duplicate_prediction_is_false_alarm(self):
        r=evaluate_events([self.event()],[self.event(),self.event()])
        self.assertEqual(r['detection']['fp'],1);self.assertEqual(r['detection']['FAR'],.5)

    def test_time_gated_diagnostic_does_not_match_wrong_segment(self):
        r=evaluate_events([self.event()],[self.event(2)],time_threshold=.5)
        self.assertEqual(r['detection']['tp'],0)

    def test_missing_unknowns_remain_in_denominator(self):
        r=evaluate_events([self.event(),self.event(1)],[])
        self.assertEqual(r['unknown_detection']['recall'],0.);self.assertEqual(r['matched_unknown_coverage'],0.)

    def test_empty_scene_is_valid(self):
        r=evaluate_events([],[]);self.assertIsNone(r['detection']['CDR']);self.assertEqual(r['detection']['fp'],0)


class PipelineParityTests(unittest.TestCase):
    def test_joint_batch_reclusters_inconsistent_individual_counts(self):
        import sys
        from unittest.mock import patch
        from scripts.recluster_competition_batch import main as batch_main
        from util.open_set import save_npz,FEATURE_VERSION,normalize_features
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);tokens=['batch','--output',str(root/'joint'),'--min-cluster-size','2']
            for item,number in [('1-1',3),('1-2',3),('1-3',2),('1-4',3)]:
                records=[];features=[];rows=[]
                for c in range(number):
                    for j in range(7):
                        i=c*7+j;path=f'/data/{item}_{i:04}.dat'
                        records.append({'path':path,'recording_id':path,'track_id':'T1','discovery_is_primary':True,
                                        'discovery_primary_path':path,'discovery_primary_track_id':'T1'})
                        features.append(np.eye(5)[c]+np.random.default_rng(i).normal(0,.002,5))
                        rows.append({'path':path,'label':f'Unknown_{c+1:03}','status':'novel_candidate','error':''})
                archive=root/(item+'.npz');csv_path=root/(item+'.csv')
                save_npz(archive,{'kind':'unknown_candidates','schema_version':1,'feature_version':FEATURE_VERSION,
                    'signature':{'sha256':'fixture'},'records':records},features=normalize_features(features))
                with csv_path.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                tokens.extend(['--item',item,str(archive),str(csv_path)])
            with patch.object(sys,'argv',tokens),contextlib.redirect_stdout(io.StringIO()):batch_main()
            report=json.loads((root/'joint/joint_selection.json').read_text())
            self.assertFalse(validate_counts(report['selected_counts']))
            self.assertEqual(report['selected_counts']['1-1'],report['selected_counts']['1-3'])
            self.assertEqual(report['selected_counts']['1-1'],2)
            self.assertTrue(Path(report['predictions']['1-1']).is_file())

    def test_bounded_full_fast_cache_recluster_and_submission(self):
        from unittest.mock import patch
        from tests.test_fast_infer import FastInferenceTests,Predictor
        from tests.fixtures import parse_test_args
        from scripts.evaluate_single_signal import run
        from scripts.cluster_unknown_features import build_parser,run as cluster
        fixture=FastInferenceTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        paths=[]
        for i in range(2):
            p=fixture.root/f'2-1_{i:04d}.dat'
            with p.open('wb') as stream:stream.write(bytes([i+1]));stream.truncate(40_000_000)
            paths.append(p)
        ids=fixture.root/'ids.json';ids.write_text(json.dumps({str(p):str(i) for i,p in enumerate(paths)}))
        reports=[];arguments=[]
        for name,fast in [('full-v2',False),('fast-v2',True)]:
            args=fixture.args(name,fast=fast,extra=['--competition_profile','task2','--test_item','2-1',
                '--recording_ids_json',str(ids),*(['--save_discovery_cache'] if fast else [])])
            args.inputs=list(map(str,paths));arguments.append(args)
            with patch('util.multi_signal_eval.load_spectrum',return_value=(np.full((512,750),20,dtype=np.uint8),'image')):
                reports.append(run(args,Predictor()))
        self.assertTrue(reports[1]['discovery']['constraint_satisfied'])
        self.assertEqual([r['bursts'] for r in reports[0]['files']],[r['bursts'] for r in reports[1]['files']])
        fast=arguments[1].output_dir
        args=parse_test_args(build_parser(),['--features',str(fast/'unknown_features.npz'),
            '--predictions_csv',str(fast/'predictions.csv'),'--competition_profile','task2','--test_item','2-1',
            '--unknown_min_cluster_size','2','--unknown_min_samples','1','--no-save_feature_plots','--output_dir',str(fixture.root/'recluster')])
        found=cluster(args);self.assertTrue(found['constraint_satisfied'])
        with (fast/'predictions.csv').open(encoding='utf-8-sig') as stream:original=list(csv.DictReader(stream))
        with (args.output_dir/'predictions.csv').open(encoding='utf-8-sig') as stream:reclustered=list(csv.DictReader(stream))
        self.assertEqual(original,reclustered)
        alias=fixture.root/'aliases.json';alias.write_text(json.dumps({'known':'DJFPV'}))
        with contextlib.redirect_stdout(io.StringIO()):
            code=submission.main(['--predictions',str(args.output_dir/'predictions.csv'),'--output',str(fixture.root/'submission'),
                '--aliases',str(alias),'--preview'])
        self.assertEqual(code,0)
        report=json.loads((fixture.root/'submission_report.json').read_text())
        self.assertEqual(report['datasets']['2-1']['output_signal_rows'],8)
        self.assertFalse(report['ready'])  # two fixtures do not constitute a 1750-file test item


if __name__=='__main__':unittest.main()
