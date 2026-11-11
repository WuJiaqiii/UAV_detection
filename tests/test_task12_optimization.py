import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from model.representation import contrastive_mask_batch,SupervisedContrastiveLoss
from model.resnet import MaskImageClassifier
from util.eval_common import INFERENCE_DEFAULTS,resolve_config
from util.mask_input import build_mask_input,validate_classifier_input
from util.competition_metrics import evaluate_records,evaluate_events
from util.pipeline_diagnostics import capture_stages,write_diagnostics
from scripts.audit_open_set_predictions import audit
from scripts import task12_suite as suite


ROOT=Path(__file__).resolve().parents[1]


class DualInputTests(unittest.TestCase):
    def config(self,dual=True):
        return SimpleNamespace(**{**INFERENCE_DEFAULTS,'classes':{'A':0,'B':1},'num_classes':2,
            'classifier_input_mode':'mask_amplitude' if dual else 'mask','mask_in_chans':2 if dual else 1})

    def sample(self):
        spec=np.full((512,750),.02,dtype=np.float32);spec[100:120,100:180]=1.
        return spec,np.array([[100,100,180,120]])

    def test_same_support_translation_and_legacy_first_channel(self):
        spec,boxes=self.sample();dual,mask=build_mask_input(spec,boxes,self.config())
        old,old_mask=build_mask_input(spec,boxes,self.config(False))
        self.assertTrue(torch.equal(dual[:1],old));np.testing.assert_array_equal(mask,old_mask)
        self.assertEqual(tuple(dual.shape),(2,224,224));self.assertTrue(torch.isfinite(dual).all())
        self.assertTrue(torch.all(dual[1][dual[0]==0]==0));self.assertGreater(float(dual[1].max()),.9)
        shifted=np.full_like(spec,.02);shifted[120:140,100:180]=1.
        other,_=build_mask_input(shifted,boxes+np.array([0,20,0,20]),self.config())
        self.assertTrue(torch.equal(dual,other))

    def test_empty_boxes_are_zero_and_noncanonical_amplitude_fails(self):
        spec,_=self.sample();tensor,_=build_mask_input(spec,[],self.config())
        self.assertFalse(torch.any(tensor))
        for bad in [np.full_like(spec,np.nan),spec*256]:
            with self.assertRaises(ValueError):build_mask_input(bad,[[1,1,5,5]],self.config())

    def test_mode_channel_mismatch_and_checkpoint_override_are_rejected(self):
        cfg=self.config();cfg.mask_in_chans=1
        with self.assertRaisesRegex(ValueError,'mask_in_chans=2'):validate_classifier_input(cfg)
        checkpoint={'config':{'classes':{'A':0,'B':1}}}
        with self.assertRaisesRegex(ValueError,'match the checkpoint'):
            resolve_config(checkpoint,overrides={'classifier_input_mode':'mask_amplitude','mask_in_chans':2})
        self.assertEqual(resolve_config(checkpoint).classifier_input_mode,'mask')

    def test_dual_supcon_uses_real_amplitude_and_has_finite_gradients(self):
        spec,boxes=self.sample();cfg=self.config()
        metas=[{'boxes':boxes,'spec_shape':spec.shape,'spectrum':spec} for _ in range(2)]
        view=contrastive_mask_batch(metas,cfg)
        self.assertEqual(tuple(view.shape),(2,2,224,224));self.assertGreater(float(view[:,1].sum()),0.)
        conv=torch.nn.Conv2d(2,4,3,padding=1)
        features=conv(view).mean((2,3));loss=SupervisedContrastiveLoss()(torch.stack([features,features+.01],dim=1),torch.tensor([0,0]))
        loss.backward();self.assertTrue(torch.isfinite(conv.weight.grad).all())
        with self.assertRaisesRegex(ValueError,'original scientific spectrum'):
            contrastive_mask_batch([{'boxes':boxes,'spec_shape':spec.shape}],cfg)

    def test_resnet_accepts_two_channels_and_strict_checkpoint_round_trip(self):
        model=MaskImageClassifier(num_classes=2,in_chans=2,pretrained=False).eval()
        with torch.inference_mode():logits,features=model(torch.ones(1,2,64,64),return_features=True)
        self.assertEqual(tuple(logits.shape),(1,2));self.assertEqual(tuple(features.shape),(1,512))
        restored=MaskImageClassifier(num_classes=2,in_chans=2,pretrained=False)
        restored.load_state_dict(model.state_dict(),strict=True)

    def test_png_cannot_supply_scientific_amplitude(self):
        from util.group_predictor import GroupPredictor
        cfg=self.config();predictor=GroupPredictor(torch.nn.Identity(),None,None,cfg)
        with self.assertRaisesRegex(ValueError,'rendered PNG'):
            predictor.detect_groups(np.zeros((512,750),np.uint8),'image')


class FeedbackMetricsTests(unittest.TestCase):
    def rows(self):
        truth=[{'file_id':str(i),'class_id':c} for i,c in enumerate([0,0,1,1,3,3])]
        pred=[{'file_id':str(i),'class_id':c} for i,c in enumerate([0,None,0,1,20,20])]
        return truth,pred

    def test_macro_variants_and_abstentions_keep_full_denominators(self):
        truth,pred=self.rows();r=evaluate_records(truth,pred,[0,1])
        self.assertAlmostEqual(r['known_macro']['f1'],7/12)
        self.assertAlmostEqual(r['known_macro']['f1_from_macro_pr'],.6)
        self.assertEqual(r['abstained_records'],1);self.assertEqual(r['ari_sample_count'],2)
        self.assertEqual(r['unknown_detection']['recall'],1.)
        self.assertFalse(r['feedback_style_score']['official_equivalence_verified'])

    def test_heldout_id_inside_official_0_to_19_is_unknown(self):
        r=evaluate_records([{'file_id':'x','class_id':3}],[{'file_id':'x','class_id':20}],[0,1])
        self.assertEqual(r['unknown_detection']['tp'],1)
        self.assertIsNone(r['feedback_style_score']['value'])

    def test_missing_prediction_is_abstention_and_fake_cluster_is_rejected(self):
        r=evaluate_records([{'file_id':'x','class_id':0}],[],[0])
        self.assertEqual(r['known_macro']['recall'],0.);self.assertEqual(r['abstained_records'],1)
        for cls in ['no_signal','Unknown_unassigned','error']:
            with self.assertRaises(ValueError):evaluate_records([{'file_id':'x','class_id':0}],[{'file_id':'x','class_id':cls}],[0])

    def test_event_matching_and_classification_are_separate(self):
        def event(c,low=0):return dict(file_id='x',class_id=c,frequency_low_hz=low,frequency_high_hz=low+10)
        r=evaluate_events([event(0),event(0),event(3,20)],[event(1),event(20,20)],known_classes=[0,1])
        self.assertEqual(r['detection']['tp'],2);self.assertEqual(r['detection']['fn'],1)
        self.assertEqual(r['known_micro']['tp'],0);self.assertEqual(r['unknown_detection']['tp'],1)
        self.assertEqual(r['classification_diagnostics']['matched_known_wrong_class'],1)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def csv(self,rows):
        path=self.root/'predictions.csv'
        fields=['path','track_id','label','status','error','bursts_json']
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        return path

    def burst(self,start,end):return dict(start_ms=start,end_ms=end,frequency_low_offset_mhz=1.,frequency_high_offset_mhz=2.)

    def test_same_frequency_repeated_bursts_export_separately_and_empty_scene_kept(self):
        path=self.csv([dict(path='/data/2-1_0000.dat',track_id='T1',label='DJFPV',status='known',
            bursts_json=json.dumps([self.burst(1,2),self.burst(4,5)])),
            dict(path='/data/2-1_0001.dat',label='no_signal',status='no_signal')])
        r=audit([path],'task2',self.root/'audit')
        self.assertTrue(r['metric_input_complete']);self.assertEqual(r['input_files'],2)
        events=[json.loads(x) for x in (self.root/'audit/events.jsonl').read_text().splitlines()]
        self.assertEqual(len(events),2);self.assertEqual(events[0]['class_id'],2)
        self.assertEqual(events[0]['frequency_low_hz'],1000000)
        self.assertEqual(len(json.loads((self.root/'audit/file_inventory.json').read_text())),2)

    def test_track_only_or_unresolved_data_never_produces_partial_events(self):
        for i,row in enumerate([dict(label='DJFPV',status='known'),dict(label='Unknown_unassigned',status='novel_candidate',bursts_json=json.dumps([self.burst(1,2)]))]):
            path=self.csv([dict(path='/data/2-1_0000.dat',track_id='T1',**row)])
            output=self.root/str(i);r=audit([path],'task2',output)
            self.assertFalse(r['metric_input_complete']);self.assertFalse((output/'events.jsonl').exists())

    def test_task1_no_signal_remains_null_not_an_unknown_id(self):
        p=self.csv([dict(path='/data/1-1_0000.dat',label='no_signal',status='no_signal')])
        r=audit([p],'task1',self.root/'a');self.assertEqual(r['abstained_records'],1)
        self.assertIsNone(json.loads((self.root/'a/records.jsonl').read_text())['class_id'])

    def test_no_signal_and_detections_conflict(self):
        p=self.csv([dict(path='/data/2-1_0000.dat',label='no_signal',status='no_signal'),
            dict(path='/data/2-1_0000.dat',track_id='T1',label='DJFPV',status='known',bursts_json=json.dumps([self.burst(1,2)]))])
        r=audit([p],'task2',self.root/'a');self.assertFalse(r['metric_input_complete'])


class StageDiagnosticsTests(unittest.TestCase):
    def test_stage_reasons_and_truth_free_counts(self):
        result={'path':'x.dat','status':'no_signal','signals':[]}
        capture_stages(result,[{'status':'no_group','detected_boxes':2,'groups':[],
            'detection_decisions':[{'accepted_as_event':False,'reason':'insufficient_detection_evidence'}]}])
        with tempfile.TemporaryDirectory() as temp:
            r=write_diagnostics(temp,[result],{'A':0})
            self.assertEqual(r['first_observed_stages'],{'no_event_group':1})
            self.assertEqual(r['detection_gate_reasons']['insufficient_detection_evidence'],1)
            self.assertIsNone(r['single_record_truth_diagnostics'])


class ExperimentRecipeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def args(self,protocol='open_set'):
        inputs=self.root/'inputs';inputs.mkdir(exist_ok=True)
        weights=self.root/'detector.pt';weights.write_bytes(b'fixture')
        return SimpleNamespace(suite=self.root/'suite',inputs=[inputs],yolo_weights=weights,
            base_config=ROOT/'config/competition_train.yaml',protocol=protocol,unknown_classes=['U'] if protocol=='open_set' else None,seed=42,epochs=2)

    def test_prepare_is_a_controlled_factorial_and_load_checks_config_hash(self):
        a=self.args()
        with patch('subprocess.run',side_effect=AssertionError('training must not start')):suite.prepare(a)
        state=suite.load(a.suite);experiments=[]
        for name,item in state['cases'].items():
            cfg=yaml.safe_load(Path(item['config']).read_text());experiments.append(cfg['experiment'])
            self.assertEqual(cfg['experiment']['training_variant'],'configured')
            self.assertEqual(cfg['train']['train_group_policy'],'primary')
            self.assertEqual(cfg['train']['mask_in_chans'],2 if name.startswith('dual') else 1)
            self.assertEqual(cfg['train']['supcon_weight'],.1 if name.endswith('supcon') else 0.)
        self.assertTrue(all(x==experiments[0] for x in experiments))
        p=Path(state['cases']['mask_ce']['config']);p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'configuration changed'):suite.load(a.suite)

    def test_supcon_override_is_visible_and_primary_supcon_does_not_change_component_policy(self):
        from util.cli import main
        from util.open_set_experiment import stage_arguments
        base=yaml.safe_load((ROOT/'config/competition_train.yaml').read_text())
        base['experiment']['training_variant']='primary_ce';base['train']['supcon_weight']=.1
        path=self.root/'case.yaml';path.write_text(yaml.safe_dump(base))
        with contextlib.redirect_stdout(io.StringIO()),self.assertWarnsRegex(UserWarning,'overrides'):
            main(['experiment','--config',str(path),'--dry_run'])
        base['experiment']['training_variant']='primary_supcon';path.write_text(yaml.safe_dump(base))
        with contextlib.redirect_stdout(io.StringIO()):args=main(['experiment','--config',str(path),'--dry_run'])
        train,_=stage_arguments(args,path);self.assertEqual(train.supcon_weight,.1);self.assertEqual(train.train_group_policy,'primary')

    def completed_state(self):
        state={'protocol':'open_set','jobs':{}}
        for name in suite.CASES:
            run=self.root/name;run.mkdir();(run/'test').mkdir()
            manifest={'known_classes':['A'],'unknown_classes':['U'],'classes':{'A':0},'ratios':{},
                'records':[dict(path=f'/data/{i}.dat',recording_id=str(i),source_sha256=str(i),split='test_known' if i==0 else 'test_unknown') for i in range(3)]}
            suite.save(run/'split_manifest.json',manifest)
            suite.save(run/'run.json',{'status':'complete'})
            suite.save(run/'test/results.json',{'files':[dict(path=f'/data/{i}.dat',true_class_names=['A' if i==0 else 'U'],
                single_signal_prediction=dict(recognition_status='known' if i==0 else 'novel_candidate',
                    class_name='A' if i==0 else None,display_label='A' if i==0 else 'Unknown_001')) for i in range(3)]})
            state['jobs'][name]={'status':'complete','run_dir':str(run),'artifacts':{str(run/'run.json'):suite.digest(run/'run.json'),str(run/'split_manifest.json'):suite.digest(run/'split_manifest.json')}}
        return state

    def test_comparison_checks_original_record_splits_before_reporting(self):
        state=self.completed_state()
        with contextlib.redirect_stdout(io.StringIO()):suite.compare(self.root,state)
        result=json.loads((self.root/'comparison.json').read_text())
        self.assertEqual(result['cases'][0]['metrics']['known_macro']['f1'],1.)
        self.assertEqual(result['cases'][0]['metrics']['unknown_detection']['recall'],1.)
        run=Path(state['jobs']['dual_ce']['run_dir']);path=run/'split_manifest.json'
        manifest=json.loads(path.read_text());manifest['records'][0]['split']='train';suite.save(path,manifest)
        state['jobs']['dual_ce']['artifacts'][str(path)]=suite.digest(path)
        with self.assertRaisesRegex(ValueError,'splits differ'):suite.compare(self.root,state)

    def test_completed_jobs_are_skipped_only_after_hash_verification(self):
        state=self.completed_state()
        with patch('subprocess.run',side_effect=AssertionError('must not rerun completed jobs')):
            suite.run_cases(self.root,state,'mask_ce')
        path=Path(state['jobs']['mask_ce']['run_dir'])/'run.json';path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'artifact changed'):suite.run_cases(self.root,state,'mask_ce')


if __name__=='__main__':unittest.main()
