"""Frozen, single-recording clustering experiments; truth is used after fitting."""
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances, silhouette_score

from util.open_set import load_npz, normalize_features
from util.discovery_context import annotate_discovery_candidates
from util.open_set_metrics import evaluate_open_set


PARAMETERS = {
    'hdbscan_auto': {'min_cluster_size', 'min_samples'},
    'density_merge': {'min_cluster_size', 'min_samples', 'min_silhouette', 'bootstrap_runs'},
    'kmeans': {'n_init'}, 'spherical_kmeans': {'n_init'},
    'agglomerative': {'linkage'},
    'gmm': {'covariance_type', 'reg_covar'},
    'dbscan': {'min_samples', 'eps_quantile'},
    'optics': {'min_samples', 'min_cluster_size', 'xi'},
    'spectral': {'n_neighbors'},
    'dcg': {'topk', 'cosine_threshold', 'resolution'},
}
FIXED_K = {'kmeans', 'spherical_kmeans', 'agglomerative', 'gmm', 'spectral'}
STOCHASTIC = {'kmeans', 'spherical_kmeans', 'gmm', 'spectral', 'dcg'}


class SkipCase(ValueError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(root):
    """Audit class/record isolation using manifests; raw IQ is never opened."""
    from prepare_submission import KNOWN
    root=Path(root).expanduser().resolve()
    manifest=json.loads((root/'split_manifest.json').read_text(encoding='utf-8'))
    report=json.loads((root/'test/results.json').read_text(encoding='utf-8'))
    metadata,arrays=load_npz(root/'test/unknown_features.npz','unknown_candidates')
    features=normalize_features(arrays['features']);records=metadata['records']
    known=set(manifest['known_classes']);unknown=set(manifest['unknown_classes'])
    classes=report.get('effective_config',{}).get('classes',{})
    if known & unknown or known|unknown != set(KNOWN) or set(classes)!=known:
        raise ValueError('Require an exact official-20 class holdout and matching known classifier classes')
    by_path={};by_id={};by_hash={}
    for row in manifest['records']:
        if row['path'] in by_path:raise ValueError('Duplicate manifest path')
        by_path[row['path']]=row
        for lookup,key in [(by_id,row['recording_id']),(by_hash,row['source_sha256'])]:
            value=(row['split'],row['class_name'])
            if key in lookup and lookup[key]!=value:raise ValueError('Recording/content crosses partitions or truth labels')
            lookup[key]=value
        if row['class_name'] in unknown and row['split'] in {'train','val','calibration'}:
            raise ValueError('Unknown class leaked into training/validation/calibration')
    expected={p for p,r in by_path.items() if r['split'] in {'test_known','test_unknown'}}
    files=[];identities={};seen=set()
    fields=['track_id','valid_votes','observation_count','recognition_status','class_name','ood_score']
    for row in report['files']:
        path=row['path']
        if path in seen or path not in expected:raise ValueError('Test report/manifest paths differ')
        seen.add(path);truth=row.get('true_class_names')
        if truth != [by_path[path]['class_name']]:raise ValueError('Require matching single-device file truth')
        identities[path]=by_path[path]['recording_id']
        files.append({'path':path,'true_class_names':truth,
                      'signals':[{k:t.get(k) for k in fields} for t in row.get('signals',[])]})
    if seen!=expected:raise ValueError('Incomplete test report; preserve every test recording')
    if len(records)!=len(features) or not isinstance(metadata.get('signature'),dict):
        raise ValueError('Invalid candidate features/signature')
    candidates=[];keys=set()
    source_tracks={(r['path'],t['track_id']):t for r in files for t in r['signals']}
    for record in records:
        key=(record['path'],record['track_id'])
        if key in keys or key not in source_tracks or source_tracks[key]['recognition_status']!='novel_candidate':
            raise ValueError('Duplicate/mismatched candidate identity')
        if record['recording_id']!=identities.get(record['path']):raise ValueError('Candidate recording identity mismatch')
        if record.get('ood_score') is not None and not np.isclose(record['ood_score'],source_tracks[key]['ood_score']):
            raise ValueError('Candidate/source rejection score mismatch')
        keys.add(key)
        candidates.append({'path':record['path'],'recording_id':record['recording_id'],
                           'track':{'track_id':record['track_id']}})
    if keys!={k for k,t in source_tracks.items() if t['recognition_status']=='novel_candidate'}:
        raise ValueError('Candidate archive must contain the complete rejected pool')
    annotate_discovery_candidates(candidates,files,identities)
    # Folder/file ordering can correlate with truth and influence tied-neighbor
    # algorithms. Use a stable hash of identity, never class/path order.
    selected=sorted([i for i,c in enumerate(candidates) if c['track']['discovery_is_primary']],
                    key=lambda i:hashlib.sha256(candidates[i]['recording_id'].encode()).hexdigest())
    fit_ids=[candidates[i]['recording_id'] for i in selected]
    if len(set(fit_ids))!=len(fit_ids):raise ValueError('Multiple primary seeds for one recording')
    hashes={p:file_digest(root/p) for p in ['split_manifest.json','test/results.json','test/unknown_features.npz']}
    errors={k:report.get('metrics',{}).get(k) for k in ['error_files','files_with_slice_errors','burst_label_conflicts']}
    return {'name':root.name,'root':str(root),'source_hashes':hashes,'classes':classes,
            'unknown_classes':sorted(unknown),'files':files,'recording_ids':identities,
            'x':features[selected],'ids':fit_ids,'fit_records':[records[i] for i in selected],
            'upstream_errors':errors,'feature_signature':metadata['signature'],
            'scope':'single-device recording classification; not multi-device event validation'}


def build_cases(config):
    allowed={'version','pca_dimensions','seeds','k_candidates','min_cluster_size','min_recordings','algorithms'}
    if set(config)-allowed or config.get('version')!=1:raise ValueError('Invalid matrix configuration schema')
    for key in ['pca_dimensions','seeds','k_candidates']:
        values=config.get(key)
        lower=1 if key=='k_candidates' else 0
        if not isinstance(values,list) or not values or len(set(values))!=len(values) or any(type(v)!=int or v<lower for v in values):
            raise ValueError('Invalid '+key)
    for key in ['min_cluster_size','min_recordings']:
        if type(config.get(key))!=int or config[key]<(2 if key=='min_cluster_size' else 1):raise ValueError('Invalid '+key)
    if not config.get('algorithms'):raise ValueError('No algorithms configured')
    cases=[]
    for family,grid in config['algorithms'].items():
        if family not in PARAMETERS or not isinstance(grid,dict) or set(grid)-PARAMETERS[family]:
            raise ValueError('Unknown algorithm/parameter: '+family)
        if any(not isinstance(v,list) or not v for v in grid.values()):raise ValueError('Grid values must be nonempty lists')
        for values in itertools.product(*grid.values()):
            params=dict(zip(grid,values))
            for dim,seed in itertools.product(config['pca_dimensions'],config['seeds'] if family in STOCHASTIC else [42]):
                for k in config['k_candidates'] if family in FIXED_K else [None]:
                    case={'family':family,'params':params,'pca_dimension':dim,'seed':seed,'k':k,
                          'mode':'fixed_k_diagnostic' if k is not None else 'auto',
                          'k_information':'declared candidate range' if family in FIXED_K|{'density_merge'} else 'native automatic K',
                          'declared_k_candidates':config['k_candidates'] if family in FIXED_K|{'density_merge'} else None}
                    case['case_id']=fingerprint(case)[:16];cases.append(case)
    if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('Duplicate matrix cases')
    return cases


def transform_features(x,dimension):
    if not len(x):raise SkipCase('No rejected primary features')
    if dimension==0:return np.asarray(x,dtype=np.float64),{'explained_variance':1.}
    if dimension>min(len(x)-1,x.shape[1]):raise SkipCase('Requested PCA dimension exceeds sample/feature rank limit')
    from sklearn.decomposition import PCA
    model=PCA(n_components=dimension,svd_solver='full',whiten=False)
    values=model.fit_transform(x)
    if (np.linalg.norm(values,axis=1)<1e-12).any():raise SkipCase('PCA produces zero vectors; no samples silently discarded')
    return normalize_features(values).astype(np.float64),{'explained_variance':float(model.explained_variance_ratio_.sum())}


def spherical_kmeans(x,k,seed,n_init=10,max_iter=200):
    from sklearn.cluster import kmeans_plusplus
    rng=np.random.default_rng(seed);best=None
    for _ in range(n_init):
        centers,_=kmeans_plusplus(x,k,random_state=int(rng.integers(2**31-1)))
        centers=normalize_features(centers).astype(float);previous=None
        for _ in range(max_iter):
            similarity=x@centers.T;labels=similarity.argmax(1)
            if previous is not None and np.array_equal(previous,labels):break
            previous=labels.copy()
            for c in range(k):
                mean=x[labels==c].mean(0) if np.any(labels==c) else x[np.argmin(similarity.max(1))]
                length=np.linalg.norm(mean)
                centers[c]=mean/length if length>1e-12 else x[int(rng.integers(len(x)))]
        labels=(x@centers.T).argmax(1);objective=float(np.max(x@centers.T,axis=1).sum())
        if best is None or objective>best[0]:best=(objective,labels.copy())
    return best[1]


def fit_partition(x,ids,case,config):
    """No labels, class names, source report or oracle K enter this interface."""
    from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN,OPTICS,SpectralClustering
    from sklearn.mixture import GaussianMixture
    family=case['family'];p=case['params'];k=case['k'];seed=case['seed'];info={}
    if k is not None and len(x)<=k:raise SkipCase('Not enough samples for requested K')
    if k==1:return np.zeros(len(x),dtype=int),info
    if family=='kmeans':labels=KMeans(n_clusters=k,random_state=seed,n_init=p.get('n_init',10)).fit_predict(x)
    elif family=='spherical_kmeans':labels=spherical_kmeans(x,k,seed,p.get('n_init',10))
    elif family=='agglomerative':
        linkage=p.get('linkage','average')
        labels=AgglomerativeClustering(n_clusters=k,linkage=linkage,metric='euclidean' if linkage=='ward' else 'cosine').fit_predict(x)
    elif family=='gmm':
        model=GaussianMixture(n_components=k,covariance_type=p.get('covariance_type','diag'),
            reg_covar=p.get('reg_covar',1e-4),n_init=3,max_iter=200,random_state=seed)
        labels=model.fit_predict(x);info.update(converged=bool(model.converged_),bic=float(model.bic(x)))
    elif family=='dbscan':
        distances=np.clip(pairwise_distances(x,metric='cosine'),0,2);np.fill_diagonal(distances,0)
        neighbors=min(p.get('min_samples',5),len(x))
        radii=np.sort(distances,axis=1)[:,neighbors-1]
        eps=max(1e-8,float(np.quantile(radii,p.get('eps_quantile',.75))))
        labels=DBSCAN(eps=eps,min_samples=p.get('min_samples',5),metric='precomputed',n_jobs=1).fit_predict(distances)
        info['effective_eps']=eps
    elif family=='optics':
        labels=OPTICS(min_samples=p.get('min_samples',5),min_cluster_size=p.get('min_cluster_size',5),
            xi=p.get('xi',.05),metric='euclidean',n_jobs=1).fit_predict(x)
    elif family=='spectral':
        neighbors=min(p.get('n_neighbors',10),len(x)-1)
        labels=SpectralClustering(n_clusters=k,affinity='nearest_neighbors',n_neighbors=neighbors,
            assign_labels='kmeans',n_init=10,random_state=seed,n_jobs=1).fit_predict(x)
        info['effective_neighbors']=neighbors
    elif family=='dcg':
        from util.dcg_discovery import cluster_dcg
        labels,info=cluster_dcg(x,topk=p.get('topk',15),cosine_threshold=p.get('cosine_threshold',.85),
                               resolution=p.get('resolution',1.),mutual=True,seed=seed)
    elif family=='hdbscan_auto':
        from util.unknown_density import guarded_hdbscan
        labels,_,info=guarded_hdbscan(x,ids,p.get('min_cluster_size',5),p.get('min_samples',3),config['min_recordings'],True,'auto')
    elif family=='density_merge':
        from util.bounded_density import bounded_density_partition
        labels,info=bounded_density_partition(x,ids,min(config['k_candidates']),max(config['k_candidates']),
            p.get('min_cluster_size',5),config['min_recordings'],p.get('min_silhouette',.15),p.get('min_samples',3),
            bootstrap_runs=p.get('bootstrap_runs',5),seed=seed)
    else:raise ValueError('Unsupported algorithm')
    return np.asarray(labels,dtype=int),info


def support_filter(x,ids,labels,min_size,min_recordings):
    labels=np.asarray(labels,dtype=int).copy()
    if labels.shape!=(len(x),):raise ValueError('Partition length mismatch')
    for label in sorted(set(labels)-{-1}):
        positions=np.flatnonzero(labels==label)
        if (len(positions)<min_size or len({ids[i] for i in positions})<min_recordings or
                np.linalg.norm(x[positions].mean(0))<1e-12):labels[positions]=-1
    groups=sorted(set(labels)-{-1});mapping={v:i for i,v in enumerate(groups)}
    return np.array([mapping[v] if v>=0 else -1 for v in labels],dtype=int)


def geometry(x,labels):
    keep=labels>=0;k=len(set(labels[keep]));coverage=float(keep.mean()) if len(labels) else 0.
    silhouette=None
    if 1<k<int(keep.sum()):silhouette=float(silhouette_score(x[keep],labels[keep],metric='cosine'))
    # A declared geometry-only selector. K=1 has a zero-silhouette reference.
    score=coverage*((silhouette if silhouette is not None else 0.)+.1)-.01*max(k-1,0) if k else None
    return {'retained_k':k,'fit_coverage':coverage,'fit_silhouette':silhouette,'selection_score':score}


def score_partition(dataset,labels):
    """Inherit primary labels, then evaluate ALL records with frozen rejection."""
    assignments={identity:(f'Unknown_{label+1:03d}' if label>=0 else None) for identity,label in zip(dataset['ids'],labels)}
    files=copy.deepcopy(dataset['files'])
    for row in files:
        identity=dataset['recording_ids'][row['path']]
        for track in row['signals']:
            track['unknown_cluster_id']=assignments.get(identity) if track['recognition_status']=='novel_candidate' else None
    count=len(set(labels)-{-1})
    result=evaluate_open_set(files,dataset['classes'],{'estimated_unknown_class_count':count},dataset['recording_ids'])
    fit_truth={dataset['recording_ids'][r['path']]:r['true_class_names'][0] for r in files}
    contaminated=sum(label>=0 and fit_truth[identity] in dataset['classes'] for identity,label in zip(dataset['ids'],labels))
    result['metrics']['assigned_known_contamination_count']=int(contaminated)
    return result


def selected_k_rows(rows):
    """Select K before accessing any ground-truth metric."""
    groups={}
    for row in rows:
        if row.get('status')!='complete' or row['mode']!='fixed_k_diagnostic':continue
        key=fingerprint({k:row[k] for k in ['dataset','family','params','pca_dimension','seed']})
        groups.setdefault(key,[]).append(row)
    selected=[]
    for choices in groups.values():
        valid=[r for r in choices if r.get('converged',True) and r['retained_k']==r['k'] and r['selection_score'] is not None]
        if not valid:continue
        best=max(valid,key=lambda r:(r['selection_score'],-r['k']))
        row=copy.deepcopy(best);row.update(mode='label_free_selected',selected_case_id=best['case_id'])
        row['selected_candidate_fit_seconds']=row.get('fit_seconds',0.)
        row['fit_seconds']=sum(r.get('fit_seconds',0.) for r in choices)
        row['case_id']=fingerprint({'selected_from':best['case_id'],'selector':'coverage-cosine-v1'})[:16]
        selected.append(row)
    return selected
