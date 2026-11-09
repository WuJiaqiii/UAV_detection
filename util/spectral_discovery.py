"""Raw-L2, 10-neighbor spectral candidate with public-count-constrained K search."""
import hashlib
import math
import warnings

import numpy as np


VERSION='record-balanced-spectral-candidate-v1'


def validate_spectral_options(neighbors=10,seed=42,n_init=10,minimum=1,maximum=8):
    for name,value,lower in [('neighbors',neighbors,1),('seed',seed,0),('n_init',n_init,1),
                             ('minimum',minimum,0),('maximum',maximum,1)]:
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<lower:
            raise ValueError('Invalid spectral '+name)
    if minimum>maximum:raise ValueError('Spectral minimum must not exceed maximum')


def spectral_partition(features,identities,minimum=1,maximum=8,min_size=5,min_recordings=2,
                       min_silhouette=.15,fixed_k=None,neighbors=10,seed=42,n_init=10):
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import silhouette_samples
    from sklearn.neighbors import kneighbors_graph
    from threadpoolctl import threadpool_limits
    from util.recording_features import record_weights
    validate_spectral_options(neighbors,seed,n_init,minimum,maximum)
    values=np.asarray(features,dtype=float)
    if (values.ndim!=2 or len(values)!=len(identities) or not values.shape[1] or
            not np.isfinite(values).all() or any(not isinstance(r,str) or not r for r in identities)):
        raise ValueError('Invalid spectral features/recording identities')
    if len(values) and not np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-5):
        raise ValueError('Spectral discovery expects L2-normalized features')
    if (isinstance(min_size,bool) or not isinstance(min_size,(int,np.integer)) or min_size<2 or
            isinstance(min_recordings,bool) or not isinstance(min_recordings,(int,np.integer)) or min_recordings<1 or
            not math.isfinite(min_silhouette) or not -1<=min_silhouette<=1):
        raise ValueError('Invalid spectral support/separation settings')
    if fixed_k is not None and (isinstance(fixed_k,bool) or not isinstance(fixed_k,(int,np.integer)) or
                               not max(1,minimum)<=fixed_k<=maximum):
        raise ValueError('Spectral fixed K must be within declared bounds')
    keys=[(r,tuple(row)) for r,row in zip(identities,values)]
    unique=sorted(set(keys),key=lambda key:(hashlib.sha256(key[0].encode()).hexdigest(),key[1]))
    lookup={key:i for i,key in enumerate(unique)};inverse=np.array([lookup[k] for k in keys],dtype=int)
    x=np.array([k[1] for k in unique],dtype=float).reshape(-1,values.shape[1]);ids=[k[0] for k in unique]
    output=np.full(len(values),-1,dtype=int)
    info=dict(policy=VERSION,allowed_K=[minimum,maximum],neighbors=neighbors,effective_neighbors=min(neighbors,max(0,len(x)-1)),
              seed=seed,n_init=n_init,unique_fit_components=len(x),candidate_K_scores=[],
              constraint_satisfied=False,selected_K=None,prototype_space='original L2 features; no PCA/whitening',
              interpretation='Feature-only K selection; single-device benchmark candidate, multi-device transfer unvalidated.')
    if len(x)<min_size:
        info.update(reason='no_unknown_candidates' if not len(x) else 'insufficient_components',
                    constraint_satisfied=minimum==0 and not len(x))
        return output,info
    weights=record_weights([{'recording_id':r} for r in ids]);affinity=None;candidates=[]
    for k in ([fixed_k] if fixed_k is not None else range(max(1,minimum),maximum+1)):
        candidate=dict(K=int(k),valid=False);info['candidate_K_scores'].append(candidate)
        if len(x)<=k or len(x)<k*min_size:
            candidate['reason']='insufficient_components';continue
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
            warnings.simplefilter('always')
            if k==1:labels=np.zeros(len(x),dtype=int)
            else:
                if affinity is None:
                    graph=kneighbors_graph(x,n_neighbors=info['effective_neighbors'],include_self=True,n_jobs=1)
                    affinity=.5*(graph+graph.T)
                labels=SpectralClustering(n_clusters=k,affinity='precomputed',assign_labels='kmeans',
                    n_init=n_init,random_state=seed,n_jobs=1).fit_predict(affinity)
        candidate['warnings']=list(dict.fromkeys(str(w.message) for w in caught))
        sizes=[int(np.sum(labels==c)) for c in range(k)]
        support=[len({ids[i] for i in np.flatnonzero(labels==c)}) for c in range(k)]
        candidate.update(component_support=sizes,recording_support=support)
        if min(sizes)<min_size or min(support)<min_recordings:
            candidate['reason']='insufficient_independent_support';continue
        centers=[]
        for c in range(k):
            selected=np.flatnonzero(labels==c)
            center=np.average(x[selected],axis=0,weights=record_weights([{'recording_id':ids[i]} for i in selected]))
            centers.append(center)
        centers=np.asarray(centers);length=np.linalg.norm(centers,axis=1)
        if (length<1e-12).any():candidate['reason']='degenerate_center';continue
        centers/=length[:,None]
        if k>1 and np.max(centers@centers.T-2*np.eye(k))>1-1e-6:
            candidate['reason']='indistinguishable_centers';continue
        separation=None
        if k>1:
            selected=np.arange(len(x))
            if len(selected)>2000:
                mandatory=np.array([np.flatnonzero(labels==c)[0] for c in range(k)])
                available=np.setdiff1d(selected,mandatory)
                p=weights[available];p/=p.sum()
                chosen=np.random.default_rng(seed).choice(available,max(2000,k+1)-k,replace=False,p=p)
                selected=np.sort(np.r_[mandatory,chosen])
            with threadpool_limits(limits=1):
                samples=silhouette_samples(x[selected],labels[selected],metric='cosine')
            separation=float(record_weights([{'recording_id':ids[i]} for i in selected])@samples)
            candidate['silhouette_sample_components']=len(selected)
            if separation<min_silhouette:
                candidate.update(reason='weak_separation',silhouette=separation);continue
        score=(separation if separation is not None else 0.)+.1-.01*(k-1)
        candidate.update(valid=True,silhouette=separation,coverage=1.,score=float(score))
        candidates.append((score,-k,labels))
    if not candidates:
        info['reason']='no_supported_partition';return output,info
    _,negative_k,labels=max(candidates,key=lambda c:c[:2])
    info.update(constraint_satisfied=True,selected_K=-negative_k)
    return labels[inverse],info
