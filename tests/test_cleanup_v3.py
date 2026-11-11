import contextlib
import io
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts import incremental_experiment as inc
from util.checkpoint import save_training_checkpoint, load_classifier_checkpoint
from util.experiment_io import write_json, read_json


class CheckpointRetentionTests(unittest.TestCase):
    def test_pruning_keeps_best_recent_and_explicit_model_and_does_not_touch_other_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);model=torch.nn.Linear(2,2)
            cfg=SimpleNamespace(classes={'a':0,'b':1},checkpoint_keep_last=2,checkpoint_keep_epochs=[2],root=root)
            (root/'notes.txt').write_text('keep')
            for epoch in range(6):
                save_training_checkpoint(model,epoch,.5,root,config=cfg,is_best=epoch==1)
            self.assertEqual({p.name for p in root.glob('*.pth')},{'best.pth','epoch_2.pth','epoch_5.pth','epoch_6.pth'})
            saved=load_classifier_checkpoint(torch.nn.Linear(2,2),root/'best.pth',expected_classes=cfg.classes)
            self.assertEqual(saved['epoch'],1)
            self.assertEqual(saved['config']['root'],str(root))
            self.assertNotIn('optimizer_state_dict',saved)
            self.assertNotIn('scheduler_state_dict',saved)
            self.assertEqual((root/'notes.txt').read_text(),'keep')
            self.assertFalse(list(root.glob('*.tmp')))

    def test_invalid_retention_is_rejected_before_any_artifact_is_created(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'models'
            with self.assertRaises(ValueError):
                save_training_checkpoint(torch.nn.Linear(1,1),0,0.,root,config=SimpleNamespace(checkpoint_keep_last=0))
            self.assertFalse(root.exists())


class FreshIncrementalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.rows=[]
        for name in ['a','b','c','d']:
            directory=self.root/'data'/name;directory.mkdir(parents=True)
            for split,count in [('train',18),('val',6),('calibration',8),('test_known',8)]:
                for i in range(count):
                    path=directory/f'{split}{i}.dat';path.write_text(f'{name}-{split}-{i}')
                    from util.experiment_io import digest
                    sha=digest(path)
                    self.rows.append(dict(path=str(path),class_name=name,split=split,recording_id=sha,source_sha256=sha))
        self.detector=self.root/'yolo.pt';self.detector.write_bytes(b'detector fixture')

    def test_prepare_from_real_dat_metadata_without_any_old_experiment(self):
        # Sparse, uniquely identified 50-ms recordings exercise the actual split
        # and IQ metadata loader without running a detector or reading IQ arrays.
        data=self.root/'real_dat'
        for label,name in enumerate(['a','b','c','d']):
            directory=data/name;directory.mkdir(parents=True)
            for index in range(25):
                with (directory/f'{index:03}.dat').open('wb') as stream:
                    stream.write((label*100+index+1).to_bytes(4,'little'))
                    stream.truncate(20_000_000)
        work=self.root/'fresh'
        with patch('data.iq_dat_loader.read_iq_dat',side_effect=AssertionError('No IQ conversion')), \
             contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(inc.main(['prepare','--data-root',str(data),'--yolo-weights',str(self.detector),
                '--work',str(work),'--class-counts','1','2','3','--methods','replay','--no-wait-for-gpu']),0)
            self.assertEqual(inc.main(['check','--work',str(work)]),0)
        protocol=read_json(work/'protocol.json')
        for counts in protocol['counts'].values():
            self.assertEqual(counts,dict(train=12,val=3,calibration=5,test=5))

    def prepare(self):
        manifest=dict(classes={n:i for i,n in enumerate(['a','b','c','d'])},records=self.rows)
        with patch('util.experiment_split.split_dataset',return_value=manifest) as split,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(inc.main(['prepare','--work',str(self.root/'suite'),
                '--data-root',str(self.root/'data'),'--yolo-weights',str(self.detector),
                '--class-counts','1','2','3','--methods','replay','--no-wait-for-gpu']),0)
        self.assertEqual(split.call_args.args[0].inputs,[str(self.root/'data')])
        self.assertEqual(split.call_args.args[0].protocol,'all_known')
        self.assertEqual(split.call_args.args[0].test_ratio,.2)
        return inc.load_state((self.root/'suite').resolve())

    def test_preparation_has_no_prior_experiment_or_model_dependency_and_has_disjoint_inputs(self):
        state=self.prepare()
        self.assertEqual(inc.models(state),['base','replay_1','replay_2'])
        self.assertNotIn('source_suite',state);self.assertNotIn('source_experiment',state)
        labels=read_json(self.root/'suite/labels.json')
        self.assertEqual(set(labels),{r['path'] for r in state['records'] if r['split']=='test'})
        self.assertFalse(set(inc.subset(state,'train'))&set(inc.subset(state,'test')))
        for model in inc.models(state):
            cfg=inc.train_config(state,model,'/fresh/teacher.pth' if model!='base' else None)['train']
            cfg_names=state['class_order'][:state['class_counts'][inc.stage_of(model)]]
            self.assertEqual(cfg['exclude_classes'],state['class_order'][len(cfg_names):])
            profile=inc.profile_config(state,model,'/fresh/model.pth')
            self.assertTrue(all(Path(p).parent.name in cfg_names for p in profile['profile']['reference_inputs']))
            self.assertTrue(set(profile['profile']['calibration_inputs']).isdisjoint(profile['profile']['reference_inputs']))
            self.assertEqual(profile['evaluation']['tail_policy'],'cover')
            self.assertNotIn('checkpoint_path',cfg)
        first=inc.train_config(state,'replay_1','/fresh/base.pth')['train']
        second=inc.train_config(state,'replay_2','/fresh/stage1.pth')['train']
        self.assertEqual((first['lwf_replay_per_class'],first['lwf_replay_fraction']),(4,.5))
        self.assertEqual((second['lwf_replay_per_class'],second['lwf_replay_fraction']),(8,.8))
        self.assertEqual(second['lwf_temperature'],2.)
        self.assertEqual(second['lwf_kd_new_weight'],0.)

    def test_changed_prepared_manifest_is_rejected(self):
        self.prepare()
        (self.root/'suite/labels.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            inc.load_state((self.root/'suite').resolve())

    def test_previous_stage_teacher_must_exist_and_match_its_saved_hash(self):
        state=self.prepare()
        with self.assertRaisesRegex(ValueError,'Previous-stage'):
            inc.train(state,'replay_2',0)

    def test_non_waiting_worker_inherits_suite_lock_without_querying_gpu(self):
        state=self.prepare()
        def child(command,env,log,pass_fds):
            self.assertEqual(pass_fds,(123,))
            result=Path(command[command.index('--result')+1])
            write_json(result,dict(status='complete',run_dir='/fixture/job',artifacts={}))
            return 0
        with patch('scripts.competition_infer.run_command',side_effect=child), \
             patch.object(inc,'reserve_gpu',side_effect=AssertionError('No GPU query')), \
             contextlib.redirect_stdout(io.StringIO()):
            result=inc.invoke(state,'mock','train',{},123)
        self.assertEqual(result['status'],'complete')
