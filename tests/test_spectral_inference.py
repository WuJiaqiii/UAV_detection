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
import yaml

from prepare_submission import KNOWN
from util.open_set import FEATURE_VERSION,normalize_features,save_npz,file_sha256,load_npz
from util.spectral_discovery import spectral_partition,validate_spectral_options
from util.unknown_discovery import discover_unknown
from scripts import refresh_competition_profile as refresh
from scripts import competition_pipeline as pipeline
from scripts.inspect_competition_clusters import inspect


def populations(k=3,n=8):
    rng=np.random.default_rng(10)
    x=normalize_features(np.vstack([np.eye(6)[c]+rng.normal(0,.001,(n,6)) for c in range(k)]))
    return x,[f'record:{i}' for i in range(len(x))]


class SpectralPartitionTests(unittest.TestCase):
    def test_public_ranges_order_invariance_and_duplicate_record_support(self):
        x,ids=populations()
        y,d=spectral_partition(x,ids,3,4)
        self.assertTrue(d['constraint_satisfied']);self.assertEqual(d['selected_K'],3)
        reverse,_=spectral_partition(x[::-1],ids[::-1],3,4)
        np.testing.assert_array_equal(y,reverse[::-1])
        dup,dd=spectral_partition(np.vstack([x,np.tile(x[0],(20,1))]),ids+[ids[0]]*20,3,4)
        np.testing.assert_array_equal(y,dup[:len(y)])
        self.assertEqual(d['unique_fit_components'],dd['unique_fit_components'])

    def test_multi_record_keeps_different_component_classes(self):
        x,_=populations();ids=[f'r{i%8}' for i in range(len(x))]
        y,d=spectral_partition(x,ids,0,5)
        self.assertEqual(d['selected_K'],3)
        self.assertEqual(len({y[0],y[8],y[16]}),3)

    def test_empty_task_two_and_indistinguishable_classes_do_not_fabricate_ids(self):
        y,d=spectral_partition(np.empty((0,6)),[],0,5)
        self.assertTrue(d['constraint_satisfied']);self.assertEqual(len(y),0)
        y,d=spectral_partition(np.tile([1.,0,0],(40,1)),[str(i) for i in range(40)],2,3)
        self.assertFalse(d['constraint_satisfied']);self.assertTrue(np.all(y==-1))

    def test_fixed_k_and_invalid_options_are_checked(self):
        x,ids=populations()
        for bad in [2,5,True]:
            with self.assertRaises(ValueError):spectral_partition(x,ids,3,4,fixed_k=bad)
        for kwargs in [dict(neighbors=0),dict(n_init=0),dict(seed=-1),dict(minimum=4,maximum=3)]:
            with self.assertRaises(ValueError):validate_spectral_options(**kwargs)

    def test_shared_discovery_populates_clusters_and_keeps_single_secondary_tracks(self):
        x,ids=populations()
        records=[dict(path=f'1-2_{i:04}.dat',recording_id=r,feature=f,
                      track=dict(track_id='T1',discovery_is_primary=True)) for i,(f,r) in enumerate(zip(x,ids))]
        secondary=copy.deepcopy(records[0]);secondary['track'].update(track_id='T2',discovery_is_primary=False)
        records.append(secondary)
        d=discover_unknown(records,input_mode='single',competition_profile='task1_n2',backend='spectral',bounded_strategy='spectral')
        self.assertTrue(d['constraint_satisfied']);self.assertEqual(d['estimated_unknown_class_count'],3)
        self.assertEqual(records[0]['track']['unknown_cluster_id'],records[-1]['track']['unknown_cluster_id'])
        self.assertEqual(records[-1]['track']['cluster_assignment_source'],'recording_primary')
        self.assertIsNone(records[-1]['track']['cluster_membership'])

    def test_invalid_spectral_cli_rejected_before_model_loading(self):
        from util.cli import main
        with self.assertRaisesRegex(ValueError,'spectral neighbors'):
            main(['infer','--inputs','missing.dat','--checkpoint_path','missing.pth','--yolo_weights','missing.pt','--open_set_profile','missing.npz',
                '--unknown_cluster_backend','spectral','--unknown_bounded_strategy','spectral','--spectral_neighbors','0','--dry_run'])


class ProfileRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.training=self.root/'training';(self.training/'training/models').mkdir(parents=True)
        (self.training/'open_set').mkdir();self.checkpoint=self.training/'training/models/best.pth';self.checkpoint.write_bytes(b'original cnn')
        self.yolo=self.root/'yolo.pt';self.yolo.write_bytes(b'original detector')
        records=[]
        for c,name in enumerate(KNOWN):
            for split in ['train','val','calibration']:
                path=self.root/f'{c}_{split}.dat';path.write_bytes(f'{c}/{split}'.encode())
                digest=file_sha256(path);records.append(dict(path=str(path),class_name=name,split=split,
                    source_sha256=digest,recording_id='record:'+digest))
        self.manifest=dict(known_classes=list(KNOWN),unknown_classes=[],records=records)
        refresh.save(self.training/'split_manifest.json',self.manifest)
        refresh.save(self.training/'experiment.json',dict(status='complete',protocol='all_known'))
        refresh.save(self.training/'effective_profile.json',dict(yolo_weights=str(self.yolo),settings=[],
            iq_sample_rate=100000000,tail_policy='drop',iq_stride_ms=50,signal_mode='multi',
            output_dir='/old/managed/output',profile_score_method='maha_pp_shrink',profile_calibration_unit='primary'))
        self.meta=dict(kind='open_set_profile',schema_version=1,feature_version=FEATURE_VERSION,classes=KNOWN,
            score_method='maha_pp_shrink',calibration_unit='primary',signature=dict(details=dict(
                checkpoint_sha256=file_sha256(self.checkpoint),detector_sha256=file_sha256(self.yolo),
                source_sha256={name:file_sha256(pipeline.ROOT/name) for name in ['util/open_set.py','util/event_detection.py']})))
        save_npz(self.training/'open_set/known_profile.npz',self.meta,reference=np.eye(20))
        self.out=self.root/'refresh'

    def args(self,dry=False):
        return SimpleNamespace(training_run=self.training,output=self.out,tail_policy='cover',yolo_weights=None,
                               dry_run=dry,wait_for_gpu=False,gpu_id='0',gpu_min_free_mib=16384)

    def worker(self,command,**kwargs):
        self.assertEqual(command[2],'_worker');self.assertNotIn('train',command)
        result=Path(command[command.index('--result')+1]);profile_run=self.root/'new_profile';profile_run.mkdir()
        for name,split in [('reference','train'),('calibration','calibration')]:
            folder=profile_run/name;folder.mkdir()
            sources=[r for r in self.manifest['records'] if r['split']==split]
            save_npz(folder/'features.npz',dict(kind='track_features',schema_version=1,feature_version=FEATURE_VERSION,
                source_recordings=sources,signature=dict(sha256='new')),features=np.eye(20))
        save_npz(profile_run/'known_profile.npz',self.meta,reference=np.eye(20))
        refresh.save(result,dict(profile=str(profile_run/'known_profile.npz'),run_dir=str(profile_run),extraction_errors={}))

    def test_dry_run_only_reads_metadata_and_preserves_weights(self):
        before=file_sha256(self.checkpoint)
        with patch('subprocess.run',side_effect=AssertionError('No extraction in preflight')),contextlib.redirect_stdout(io.StringIO()):
            refresh.refresh(self.args(True))
        self.assertFalse(self.out.exists());self.assertEqual(file_sha256(self.checkpoint),before)

    def test_refresh_keeps_weights_and_emits_matched_inference_config(self):
        before=file_sha256(self.training/'open_set/known_profile.npz')
        with patch.object(refresh.subprocess,'run',side_effect=self.worker),contextlib.redirect_stdout(io.StringIO()):
            state=refresh.refresh(self.args())
        self.assertEqual(state['status'],'complete');self.assertEqual(state['checkpoint_sha256'],file_sha256(self.checkpoint))
        self.assertEqual(before,file_sha256(self.training/'open_set/known_profile.npz'))
        config=yaml.safe_load((self.out/'inference.yaml').read_text())['infer']
        self.assertEqual(config['tail_policy'],'cover');self.assertEqual(config['unknown_cluster_backend'],'spectral')
        self.assertEqual(config['open_set_profile'],state['profile']);self.assertNotIn('reference_inputs',config)
        self.assertEqual(pipeline.training_artifacts(self.training,profile_override=state['profile'])['profile'],state['profile'])

    def test_changed_source_bytes_block_publication_of_a_refreshed_profile(self):
        def worker(*args,**kwargs):
            self.worker(*args,**kwargs)
            path=self.root/'new_profile/reference/features.npz';metadata,arrays=load_npz(path,'track_features')
            metadata['source_recordings'][0]['source_sha256']='changed';path.unlink();save_npz(path,metadata,**arrays)
        with patch.object(refresh.subprocess,'run',side_effect=worker),self.assertRaisesRegex(ValueError,'data changed'):
            refresh.refresh(self.args())
        self.assertEqual(refresh.read(self.out/'refresh_manifest.json')['status'],'failed')
        self.assertFalse((self.out/'inference.yaml').exists())

    def test_switched_checkpoint_or_detector_requires_explicit_review(self):
        self.checkpoint.write_bytes(b'switched')
        with self.assertRaisesRegex(ValueError,'Checkpoint changed'):refresh.prepare_settings(self.training,self.out)

    def test_refresh_does_not_silently_change_a_different_rejection_method(self):
        path=self.training/'open_set/known_profile.npz';meta,arrays=load_npz(path,'open_set_profile')
        meta['score_method']='global_knn';path.unlink();save_npz(path,meta,**arrays)
        with self.assertRaisesRegex(ValueError,'preserves mainline'):refresh.prepare_settings(self.training,self.out)


class SixItemInspectionTests(unittest.TestCase):
    def test_joint_spectral_inspection_and_plots_preserve_original_caches(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);batch={'fingerprint':{'strategy':'spectral'},'runs':{},'discovery':dict(strategy='spectral',
                min_cluster_size=5,min_recordings=2,min_samples=3,min_silhouette=.15,
                spectral_options=dict(neighbors=10,seed=42,n_init=10))}
            for item in pipeline.ITEMS:
                folder=root/item;folder.mkdir();k=2 if item in ['1-1','1-3'] else 3;x,ids=populations(k,7)
                candidates=[dict(path=f'/data/{item}_{i:04}.dat',recording_id=r,feature=f,
                    track=dict(track_id='T1',discovery_is_primary=True)) for i,(f,r) in enumerate(zip(x,ids))]
                discovery=discover_unknown(candidates,input_mode='single' if item.startswith('1') else 'multi',
                    competition_profile='task1_n1' if item in ['1-1','1-3'] else 'task1_n2' if item.startswith('1') else 'task2',
                    backend='spectral',bounded_strategy='spectral')
                records=[dict(path=c['path'],recording_id=c['recording_id'],**c['track']) for c in candidates]
                save_npz(folder/'unknown_features.npz',dict(kind='unknown_candidates',schema_version=1,
                    feature_version=FEATURE_VERSION,signature=dict(sha256='fixture'),records=records),features=x)
                with (folder/'predictions.csv').open('w',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(['path','label','status','error'])
                    writer.writerows([[c['path'],c['track']['unknown_cluster_id'],'novel_candidate',''] for c in candidates])
                refresh.save(folder/'unknown_clusters.json',discovery)
                batch['runs'][item]=dict(status='complete',run_dir=str(folder),
                    csv_sha256=file_sha256(folder/'predictions.csv'),cache_sha256=file_sha256(folder/'unknown_features.npz'))
            source=root/'batch.json';refresh.save(source,batch)
            with contextlib.redirect_stdout(io.StringIO()):
                result=inspect(SimpleNamespace(manifest=source,output=root/'inspection',joint_task1=True,projection='pca',max_points=100))
            self.assertEqual(result['joint_status'],'complete')
            self.assertEqual(result['joint_selected_counts'],{'1-1':2,'1-2':3,'1-3':2,'1-4':3})
            for item in pipeline.ITEMS:
                self.assertTrue((root/'inspection'/item/'feature_report.html').is_file())
                self.assertEqual(file_sha256(root/item/'unknown_features.npz'),batch['runs'][item]['cache_sha256'])
            self.assertTrue((root/'inspection/cluster_sizes.csv').is_file())


if __name__=='__main__':unittest.main()
