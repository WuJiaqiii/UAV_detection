import hashlib
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from data.incremental_replay import ReplayBatchSampler, select_replay_records
from model.resnet import MaskImageClassifier
from util.incremental_metrics import incremental_validation_metrics
from util.lwf_trainer import LwFTrainer
from util.single_signal_eval import resolve_config
from util.train_entry import build_parser, validate_training_options
from util.trainer import Trainer


class ReplayPartitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for name in ('a', 'b', 'c'):
            (self.root/name).mkdir()
            for split, count in (('train', 3), ('val', 1), ('calibration', 1), ('test', 1)):
                for i in range(count):
                    path = self.root/name/f'{split}{i}.dat'
                    data = f'{name}-{split}-{i}'.encode()
                    path.write_bytes(data)
                    sha = hashlib.sha256(data).hexdigest()
                    self.rows.append(dict(path=str(path), class_name=name, split=split,
                                          recording_id=sha, source_sha256=sha))
        self.manifest = self.root/'partitions.json'
        self.training = [dict(fp=r['path'], class_name=r['class_name']) for r in self.rows
                         if r['split']=='train' and r['class_name']=='c']
        self.validation = [dict(fp=r['path'], class_name=r['class_name']) for r in self.rows if r['split']=='val']

    def select(self, count=2):
        self.manifest.write_text(json.dumps({'records': self.rows}))
        return select_replay_records(self.manifest, {'a':0,'b':1,'c':2}, 2, count, 42,
                                     self.training, self.validation)

    def test_budget_determinism_and_train_only(self):
        first = self.select()
        self.assertEqual(first, self.select())
        self.assertEqual(len(first), 4)
        self.assertTrue(all(r['split']=='train' and r['class_name']!='c' for r in first))
        self.assertEqual(len({r['recording_id'] for r in first}), 4)

    def test_identity_aliases_cannot_cross_partitions(self):
        self.rows[3]['recording_id'] = self.rows[0]['recording_id']
        with self.assertRaisesRegex(ValueError, 'crosses'):
            self.select()

    def test_content_aliases_cannot_cross_partitions_even_with_different_ids(self):
        self.rows[3]['source_sha256'] = self.rows[0]['source_sha256']
        with self.assertRaisesRegex(ValueError, 'crosses'):
            self.select()

    def test_multiple_files_of_one_recording_do_not_inflate_budget(self):
        for row in self.rows[:3]:
            row['recording_id'] = 'same-original-recording'
        with self.assertRaisesRegex(ValueError, 'independent TRAIN'):
            self.select()

    def test_content_change_and_partition_mismatch_fail_closed(self):
        for row in self.rows:
            if row['split']=='train' and row['class_name']=='a':
                Path(row['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'content changed'):
            self.select()
        self.training[0]['fp'] = self.validation[0]['fp']
        with self.assertRaisesRegex(ValueError, 'matching manifest'):
            self.select()


class SamplerTests(unittest.TestCase):
    def test_dcg_accepts_column_shaped_numpy_inverse_indices(self):
        from util.dcg_discovery import cluster_dcg
        unique = np.unique
        def column_inverse(*args, **kwargs):
            values, inverse, counts = unique(*args, **kwargs)
            return values, inverse.reshape(-1,1), counts
        with patch('util.dcg_discovery.np.unique', side_effect=column_inverse):
            labels, info = cluster_dcg([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        self.assertEqual(labels.shape,(4,))
        self.assertEqual(labels[0],labels[1])
        self.assertNotEqual(labels[0],labels[2])
        self.assertEqual(info['raw_community_count'],2)

    def test_new_coverage_old_balance_batch_limits_and_epoch_reproducibility(self):
        labels = [0]*4+[1]*4+[2]*4+[3]*7+[4]*7
        sampler = ReplayBatchSampler(labels, 3, 8, .5, 42)
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        seen = [i for batch in first for i in batch if labels[i]>=3]
        self.assertEqual(sorted(seen), list(range(12,26)))
        self.assertTrue(all(len(b)<=8 and any(labels[i]<3 for i in b) and any(labels[i]>=3 for i in b) for b in first))
        counts = [sum(labels[i]==c for b in first for i in b) for c in range(3)]
        self.assertLessEqual(max(counts)-min(counts), 1)
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))


class DistillationTests(unittest.TestCase):
    def trainer(self):
        obj = LwFTrainer.__new__(LwFTrainer)
        obj.old_num_classes, obj.temperature, obj.kd_new_weight = 2, 2., 0.
        return obj

    def test_kd_preserves_old_logits_without_constraining_new_examples(self):
        obj = self.trainer()
        student = torch.randn(3,4,requires_grad=True)
        teacher = torch.randn(3,2,requires_grad=True)
        obj._kd_loss(student, teacher, torch.tensor([0,2,3])).backward()
        self.assertGreater(float(student.grad[0,:2].abs().sum()), 0.)
        self.assertEqual(float(student.grad[1:].abs().sum()), 0.)
        self.assertEqual(float(student.grad[:,2:].abs().sum()), 0.)
        self.assertIsNone(teacher.grad)

    def test_new_only_batch_has_connected_finite_zero(self):
        student = torch.randn(2,4,requires_grad=True)
        loss = self.trainer()._kd_loss(student, torch.randn(2,2), torch.tensor([2,3]))
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_default_kd_matches_original_temperature_scaled_kl(self):
        obj = self.trainer()
        student, teacher = torch.randn(5,4), torch.randn(5,2)
        expected = nn.functional.kl_div(nn.functional.log_softmax(student[:,:2]/2,dim=1),
            nn.functional.softmax(teacher/2,dim=1),reduction='batchmean')*4
        torch.testing.assert_close(obj._kd_loss(student,teacher), expected)

    def test_feature_preservation_only_backpropagates_on_old_examples(self):
        student, teacher = torch.randn(4,8,requires_grad=True), torch.randn(4,8,requires_grad=True)
        self.trainer()._feature_loss(student,teacher,torch.tensor([0,1,2,3]),None).backward()
        self.assertGreater(float(student.grad[:2].abs().sum()), 0.)
        self.assertEqual(float(student.grad[2:].abs().sum()), 0.)
        self.assertIsNone(teacher.grad)


class SelectionTests(unittest.TestCase):
    def test_replay_options_require_explicit_safe_configuration(self):
        parser = build_parser()
        for overrides in ({'lwf_replay_per_class':4}, {'lwf_kd_new_weight':-1},
                          {'lwf_replay_fraction':float('nan')}, {'lwf_feature_kd_weight':.1},
                          {'checkpoint_selection':'old_new_hmean'}, {'early_stop_min_epochs':1000}):
            args = parser.parse_args(['--yolo_weights','unused.pt'])
            for name,value in overrides.items():
                setattr(args,name,value)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_training_options(args)

    def test_misses_remain_in_denominator_and_new_zero_cannot_be_hidden(self):
        classes = {'a':0,'b':1,'c':2}
        m = incremental_validation_metrics(classes,2,{0:6,1:6,2:6},{0:6,1:6,2:0})
        self.assertEqual(m['old_new_hmean'],0.)
        m = incremental_validation_metrics(classes,2,{0:6,1:6,2:6},{0:6,1:6,2:3})
        self.assertAlmostEqual(m['old_new_hmean'],200/3)
        with self.assertRaisesRegex(ValueError,'every active class'):
            incremental_validation_metrics(classes,2,{0:6,1:6},{0:6,1:6})

    def test_training_selects_harmonic_checkpoint_instead_of_old_dominated_acc(self):
        obj = Trainer.__new__(Trainer)
        obj.config = SimpleNamespace(epochs=2,early_stop_patience=10,save_interval=1,
            checkpoint_selection='old_new_hmean',model_dir='unused',early_stop_min_epochs=0)
        obj.train_loader = obj.val_loader = [1]
        obj.start_epoch, obj.best_val_acc, obj.best_selection_score, obj.no_improve_epochs = 0,-1.,-1.,0
        obj.writer, obj.logger = Mock(), Mock()
        obj.classifier = obj.optimizer = obj.scheduler = Mock()
        obj.train_one_epoch = Mock(return_value=(1.,50.,1.))
        def validate(epoch):
            obj.last_validation_metrics = {'old_new_hmean': [0.,60.][epoch]}
            return 1., [85.,70.][epoch], 1.
        obj.validate = validate
        with patch('util.trainer.save_training_checkpoint') as save:
            obj.train()
        self.assertTrue(save.call_args_list[-1].kwargs['is_best'])
        self.assertEqual(obj.best_val_acc,70.)
        self.assertEqual(obj.best_selection_score,60.)


class TrainingIntegrationTests(unittest.TestCase):
    def test_replay_feature_training_updates_new_head_and_keeps_teacher_and_bn_fixed(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            teacher = MaskImageClassifier('resnet18',2,1,pretrained=False)
            checkpoint = root/'teacher.pth'
            torch.save({'config':{'classes':{'a':0,'b':1}},'model_state_dict':teacher.state_dict()},checkpoint)
            cfg = resolve_config({'config':{'classes':{'a':0,'b':1,'c':2},'mask_img_size':32}})
            for k,v in dict(device=torch.device('cpu'),lr=.001,weight_decay=0.,log_dir=str(root/'log'),
                result_dir=str(root/'result'),checkpoint_path=None,run_mode='train',eval_exclude_classes=[],
                val_detect_vis_ratio=0.,infer_detect_vis_ratio=0.,lwf_old_num_classes=2,
                lwf_teacher_checkpoint=str(checkpoint),lwf_temperature=2.,lwf_lambda_kd=1.,
                lwf_feature_kd_weight=.1,lwf_kd_new_weight=0.,lwf_backbone_lr_multiplier=.1,lwf_freeze_bn=True).items():
                setattr(cfg,k,v)
            boxes = np.array([[10,100,70,110],[150,100,210,110]])
            detector,preprocessor = Mock(),Mock()
            detector.detect.return_value = boxes
            preprocessor.select_signal_groups.return_value = [{'boxes':boxes,'score':2.}]
            batch = (torch.zeros(4,512,750),[[{'label':i,'has_frequency_label':False}] for i in [0,1,2,2]],
                     torch.zeros(4),['a','b','c','d'])
            student = MaskImageClassifier('resnet18',3,1,pretrained=False)
            obj = LwFTrainer(cfg,([batch],[batch]),logging.getLogger('replay-test'),detector,preprocessor,student)
            before = student.net.fc[-1].weight[2].detach().clone()
            bn = student.net.bn1.running_mean.clone()
            loss,_,_ = obj.train_one_epoch(0)
            obj.writer.close()
            self.assertTrue(np.isfinite(loss))
            self.assertFalse(torch.equal(before,student.net.fc[-1].weight[2]))
            torch.testing.assert_close(bn,student.net.bn1.running_mean,rtol=0,atol=0)
            self.assertEqual(sorted(g['initial_lr'] for g in obj.optimizer.param_groups),[.0001,.001])
            for name,value in obj.teacher.state_dict().items():
                torch.testing.assert_close(value,teacher.state_dict()[name],rtol=0,atol=0)
            self.assertTrue(all(p.grad is None for p in obj.teacher.parameters()))
            self.assertEqual(json.loads((root/'result/train_metrics/epoch_1.json').read_text())['new_instances'],2)

    def test_validation_includes_a_record_lost_by_the_detector(self):
        obj = Trainer.__new__(Trainer)
        obj.config = SimpleNamespace(classes={'a':0,'b':1,'c':2},lwf_old_num_classes=2,
            checkpoint_selection='old_new_hmean',save_val_detect_vis=False,result_dir='unused',eval_exclude_classes=[])
        obj.classifier = Mock()
        obj.classifier.return_value = torch.eye(3)*5
        obj.device, obj.criterion = 'cpu', nn.CrossEntropyLoss()
        obj.eval_exclude_label_ids, obj.inv_class_map = set(), {0:'a',1:'b',2:'c'}
        obj._build_single_instances = Mock(return_value=(torch.zeros(3,1,8,8),torch.tensor([0,1,2]),
            [{'fp':str(i),'target_idx':0,'group_idx':0} for i in range(3)],3,4,[]))
        batch = (torch.zeros(4,1,8,8),[[{'label':i}] for i in [0,1,2,2]],None,['a','b','c','d'])
        with patch('util.trainer.save_confusion_matrix'), patch('util.trainer.save_eval_summary'), patch('util.trainer.save_instance_csv'):
            obj.evaluate_single_loader([batch])
        self.assertEqual(obj.last_validation_metrics['new_macro_recall'],50.)
        self.assertAlmostEqual(obj.last_validation_metrics['old_new_hmean'],200/3)


if __name__ == '__main__':
    unittest.main()
