import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from prepare_submission import KNOWN
from util.open_set import FEATURE_VERSION,save_npz
from util.clustering_benchmark import (PARAMETERS,SkipCase,build_cases,fit_partition,geometry,
    load_dataset,score_partition,selected_k_rows,spherical_kmeans,support_filter,transform_features)
from util.clustering_matrix_report import aggregate
from scripts import benchmark_unknown_clustering as runner


class MatrixAlgorithmTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        self.x=np.vstack([rng.normal(c,.015,(20,4)) for c in [[1,0,0,0],[0,1,0,0],[0,0,1,0]]])
        self.x/=np.linalg.norm(self.x,axis=1,keepdims=True)
        self.ids=[str(i) for i in range(len(self.x))]
        self.config={'min_cluster_size':3,'min_recordings':2,'k_candidates':[1,2,3,4]}

    def test_all_ten_families_produce_deterministic_valid_partitions(self):
        with threadpool_limits(limits=1):
            for family in PARAMETERS:
                case={'family':family,'params':{},'seed':42,'k':3 if family in {'kmeans','spherical_kmeans','agglomerative','gmm','spectral'} else None}
                if family=='density_merge':case['params']={'bootstrap_runs':0}
                with self.subTest(family=family):
                    first,_=fit_partition(self.x,self.ids,case,self.config)
                    second,_=fit_partition(self.x,self.ids,case,self.config)
                    np.testing.assert_array_equal(first,second)
                    self.assertEqual(first.shape,(60,));self.assertTrue(np.all(first>=-1))

    def test_spherical_kmeans_optimizes_unit_sphere_assignments(self):
        with threadpool_limits(limits=1):labels=spherical_kmeans(self.x,3,42,n_init=3)
        self.assertEqual(len(set(labels)),3)
        from sklearn.metrics import adjusted_rand_score
        self.assertEqual(adjusted_rand_score(np.repeat(range(3),20),labels),1.)

    def test_pca_is_clustering_only_deterministic_and_does_not_drop_zero_rows(self):
        original=self.x.copy();transformed,info=transform_features(self.x,2)
        np.testing.assert_array_equal(self.x,original)
        self.assertEqual(transformed.shape,(60,2));np.testing.assert_allclose(np.linalg.norm(transformed,axis=1),1.,atol=1e-6)
        self.assertGreater(info['explained_variance'],.9)
        with self.assertRaises(SkipCase):transform_features(self.x,65)
        with self.assertRaises(SkipCase):transform_features(np.ones((6,4)),2)

    def test_support_filter_uses_recordings_and_treats_removed_points_as_unassigned(self):
        x=self.x[:6];ids=['same','same','same','b','c','d']
        labels=support_filter(x,ids,np.array([0,0,0,1,1,1]),3,2)
        np.testing.assert_array_equal(labels,[-1,-1,-1,0,0,0])
        self.assertEqual(geometry(x,labels)['fit_coverage'],.5)

    def test_k_selection_ignores_truth_metrics_and_accounts_for_all_candidate_fits(self):
        rows=[]
        for k,score,accuracy in [(2,.3,1.),(3,.6,.1),(4,.4,.8)]:
            rows.append(dict(dataset='fixture',family='kmeans',params={},pca_dimension=0,seed=42,
                status='complete',mode='fixed_k_diagnostic',case_id=str(k),k=k,retained_k=k,
                selection_score=score,unknown_accuracy=accuracy,fit_seconds=1.))
        selected=selected_k_rows(rows)[0]
        self.assertEqual(selected['k'],3);self.assertEqual(selected['fit_seconds'],3.)
        for row in rows:row['unknown_accuracy']=1-row['unknown_accuracy']
        self.assertEqual(selected_k_rows(rows)[0]['k'],3)

    def test_grid_is_explicit_and_rejects_typos_or_duplicate_cases(self):
        config={'version':1,'pca_dimensions':[0,2],'seeds':[42,43],'k_candidates':[1,2,3],
            'min_cluster_size':3,'min_recordings':2,'algorithms':{'kmeans':{'n_init':[2]},'hdbscan_auto':{'min_samples':[3,5]}}}
        self.assertEqual(len(build_cases(config)),16)
        config['algorithms']['kmeans']['misspelled']=[1]
        with self.assertRaises(ValueError):build_cases(config)


class FrozenBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'source';self.source.mkdir()
        (self.source/'test').mkdir();self.known=list(KNOWN)[:16];self.unknown=list(KNOWN)[16:]
        manifest={'known_classes':self.known,'unknown_classes':self.unknown,'records':[]}
        files=[];records=[];features=[];rng=np.random.default_rng(3)
        for c,name in enumerate(KNOWN):
            if name in self.known:
                self.add_manifest(manifest,f'/data/{c}/train.dat',name,'train')
                indices=range(1)
            else:indices=range(6)
            for j in indices:
                path=f'/data/{c}/{j}.dat';row=self.add_manifest(manifest,path,name,'test_known' if name in self.known else 'test_unknown')
                rejected=name in self.unknown or c==0
                track=dict(track_id='T1',valid_votes=2,observation_count=2,
                    recognition_status='novel_candidate' if rejected else 'known',class_name=name,ood_score=2. if rejected else .1)
                files.append(dict(path=path,true_class_names=[name],signals=[track]))
                if rejected:
                    vector=np.zeros(8);vector[c%8]=1;vector+=rng.normal(0,.01,8)
                    features.append(vector)
                    records.append(dict(path=path,recording_id=row['recording_id'],track_id='T1',ood_score=2.))
        self.manifest=manifest
        (self.source/'split_manifest.json').write_text(json.dumps(manifest))
        self.report={'files':files,'effective_config':{'classes':dict(zip(self.known,range(16)))},'metrics':{}}
        (self.source/'test/results.json').write_text(json.dumps(self.report))
        save_npz(self.source/'test/unknown_features.npz',{'kind':'unknown_candidates','schema_version':1,
            'feature_version':FEATURE_VERSION,'signature':{'sha256':'fixture'},'records':records},features=np.asarray(features))
        self.config_path=self.root/'config.yaml'
        config={'version':1,'pca_dimensions':[0],'seeds':[42],'k_candidates':[1,2,3,4,5],
                'min_cluster_size':2,'min_recordings':2,'algorithms':{'kmeans':{'n_init':[2]}}}
        self.config_path.write_text(yaml.safe_dump(config))

    def add_manifest(self,manifest,path,name,split):
        digest=hashlib.sha256(path.encode()).hexdigest()
        row=dict(path=path,class_name=name,split=split,source_sha256=digest,recording_id='recording:'+digest)
        manifest['records'].append(row);return row

    def args(self,**kwargs):
        return SimpleNamespace(source_experiment=[self.source],config=self.config_path,preset='standard',
            algorithms=None,threads=1,output=self.root/'matrix',resume=False,retry_errors=False,plan_only=False,**kwargs)

    def test_manifest_audit_and_truth_are_separate_from_fitting(self):
        dataset=load_dataset(self.source)
        self.assertEqual(len(dataset['ids']),25)  # The false-rejected known sample remains.
        self.assertEqual(len(dataset['files']),40)
        with threadpool_limits(limits=1):
            labels,_=fit_partition(dataset['x'],dataset['ids'],dict(family='kmeans',params={'n_init':2},seed=42,k=4),
                                   {'min_cluster_size':2,'min_recordings':2,'k_candidates':[1,2,3,4]})
        scored=score_partition(dataset,labels)['metrics']
        self.assertEqual(scored['unknown_recordings'],24);self.assertEqual(scored['known_recordings'],16)
        self.assertEqual(scored['assigned_known_contamination_count'],1)
        zero=score_partition(dataset,np.full(25,-1))['metrics']
        self.assertEqual(zero['unknown_accuracy'],0.)
        self.assertEqual(zero['known_accuracy'],scored['known_accuracy'])
        self.assertEqual(zero['unknown_cluster_coverage'],0.)

    def test_leakage_incomplete_truth_and_wrong_twenty_class_partition_are_rejected(self):
        for alter in ['leak','missing','class_partition']:
            m=copy.deepcopy(self.manifest);report=copy.deepcopy(self.report)
            if alter=='leak':m['records'][-1]['split']='train'
            elif alter=='missing':report['files'].pop()
            else:m['unknown_classes']=m['unknown_classes'][:-1]
            (self.source/'split_manifest.json').write_text(json.dumps(m))
            (self.source/'test/results.json').write_text(json.dumps(report))
            with self.subTest(alter=alter),self.assertRaises(ValueError):load_dataset(self.source)

    def test_candidate_ingestion_order_cannot_leak_folder_order_into_fitting(self):
        first=load_dataset(self.source);path=self.source/'test/unknown_features.npz'
        from util.open_set import load_npz
        metadata,arrays=load_npz(path,'unknown_candidates')
        order=np.random.default_rng(42).permutation(len(metadata['records']))
        metadata['records']=[metadata['records'][i] for i in order]
        path.unlink();save_npz(path,metadata,features=arrays['features'][order])
        second=load_dataset(self.source)
        self.assertEqual(first['ids'],second['ids']);np.testing.assert_array_equal(first['x'],second['x'])

    def test_aggregate_reports_incomplete_seed_sets_instead_of_hiding_failures(self):
        base=dict(dataset='D',family='kmeans',params={},pca_dimension=0,mode='label_free_selected',
                  status='complete',case_id='A',seed=42,unknown_accuracy=.9)
        raw={**base,'mode':'fixed_k_diagnostic'}
        failed={**raw,'seed':43,'status':'error'}
        result=aggregate([base,raw,failed])
        self.assertEqual(result[0]['expected_repeats'],2)
        self.assertFalse(result[0]['complete_seed_coverage'])

    def test_runner_makes_real_csv_and_plots_and_resumes_only_verified_results(self):
        args=self.args()
        with contextlib.redirect_stdout(io.StringIO()):result=runner.run(args)
        self.assertEqual(result['status_counts'],{'complete':6})
        for filename in ['report.html','results_long.csv','aggregate_results.csv','summary.json','source_unknown_accuracy.png','source_winner_unknown_confusion.png']:
            self.assertTrue((args.output/filename).is_file(),filename)
        args.resume=True
        with patch.object(runner,'execute_case',side_effect=AssertionError('must not refit')),contextlib.redirect_stdout(io.StringIO()):runner.run(args)
        path=next((args.output/'cases/source').glob('*.json'));path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'result changed'):runner.run(args)

    def test_plan_only_does_not_create_output_and_resume_rejects_changed_source(self):
        args=self.args();args.plan_only=True
        with contextlib.redirect_stdout(io.StringIO()):runner.run(args)
        self.assertFalse(args.output.exists());args.plan_only=False
        with contextlib.redirect_stdout(io.StringIO()):runner.run(args)
        args.resume=True
        path=self.source/'test/results.json';path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'Frozen'):runner.run(args)


if __name__=='__main__':unittest.main()
