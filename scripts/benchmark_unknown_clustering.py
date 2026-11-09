#!/usr/bin/env python3
"""CPU-only algorithm/parameter matrix on audited official-20 class holdouts."""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import fcntl
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import yaml
from threadpoolctl import threadpool_limits

from util.clustering_benchmark import (SkipCase,build_cases,file_digest,fingerprint,fit_partition,
    geometry,load_dataset,score_partition,selected_k_rows,support_filter,transform_features)
from util.clustering_matrix_report import make_report


def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def configuration(path,preset,algorithms=None):
    config=yaml.safe_load(path.read_text(encoding='utf-8'))
    if preset=='quick':
        config['pca_dimensions']=[0,32];config['seeds']=[42];config['k_candidates']=[1,2,3,4,5,6]
        for grid in config['algorithms'].values():
            for key in grid:grid[key]=grid[key][:2]
    elif preset=='full':
        config['pca_dimensions']=[0,16,32,64,128];config['seeds']=[42,43,44,45,46]
        config['k_candidates']=list(range(1,11))
    if algorithms:
        if set(algorithms)-set(config['algorithms']):raise ValueError('Unknown requested algorithm')
        config['algorithms']={k:v for k,v in config['algorithms'].items() if k in algorithms}
    build_cases(config)
    return config


@contextmanager
def lock(output):
    with (output/'.matrix.lock').open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('This matrix is already running') from exc
        yield


def execute_case(dataset,x,case,config):
    started=time.perf_counter();row={**copy.deepcopy(case),'dataset':dataset['name'],'status':'complete'}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        labels,diagnostics=fit_partition(x,dataset['ids'],case,config)
        labels=support_filter(x,dataset['ids'],labels,config['min_cluster_size'],config['min_recordings'])
        row.update(geometry(x,labels));row['fit_seconds']=time.perf_counter()-started
        # No true labels are accessed until the partition and selector score exist.
        evaluation=score_partition(dataset,labels)
    row.update(evaluation['metrics']);row.update(labels=labels.tolist(),diagnostics=diagnostics,
        warnings=list(dict.fromkeys(str(w.message) for w in caught)),converged=diagnostics.get('converged',True))
    return row


def run(args):
    if args.threads<1:raise ValueError('threads must be positive')
    config=configuration(args.config,args.preset,args.algorithms);cases=build_cases(config)
    datasets=[load_dataset(p) for p in args.source_experiment]
    if len({d['name'] for d in datasets})!=len(datasets):raise ValueError('Dataset directory names must be unique')
    if args.plan_only:
        result={'fits_per_dataset':len(cases),'total_fits':len(cases)*len(datasets),
                'by_algorithm':dict(Counter(c['family'] for c in cases)),
                'datasets':[{k:d[k] for k in ['name','upstream_errors']}|{'primary_candidates':len(d['ids'])} for d in datasets]}
        print(json.dumps(result,ensure_ascii=False,indent=2));return result
    output=args.output.expanduser().resolve()
    if output.exists() and not args.resume:raise FileExistsError('Use a new output directory or --resume')
    output.mkdir(parents=True,exist_ok=True)
    modules=['util/clustering_benchmark.py','util/clustering_matrix_report.py','scripts/benchmark_unknown_clustering.py',
             'util/open_set.py','util/open_set_metrics.py','util/discovery_context.py','util/recording_features.py',
             'util/unknown_density.py','util/bounded_density.py','util/dcg_discovery.py','prepare_submission.py']
    snapshot={'config':config,'sources':[{k:d[k] for k in ['name','source_hashes']} for d in datasets],
              'code':{p:file_digest(ROOT/p) for p in modules},'threads':args.threads,
              'libraries':{p:version(p) for p in ['numpy','scipy','scikit-learn','networkx']}}
    with lock(output),threadpool_limits(limits=args.threads):
        frozen=output/'benchmark_manifest.json';progress_path=output/'progress.json'
        if frozen.exists():
            if json.loads(frozen.read_text())!=snapshot:raise ValueError('Frozen inputs/code/config/environment changed; use a new output')
        else:
            if args.resume:raise ValueError('Cannot resume without a benchmark manifest')
            save(frozen,snapshot)
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
        rows=[];total=len(cases)*len(datasets);done=0
        for dataset in datasets:
            directory=output/'cases'/dataset['name'];directory.mkdir(parents=True,exist_ok=True)
            reductions={}
            for case in cases:
                path=directory/(case['case_id']+'.json');key=path.relative_to(output).as_posix();done+=1
                if key in progress:
                    if not path.is_file() or file_digest(path)!=progress[key]:raise ValueError('Completed result changed: '+key)
                    old=json.loads(path.read_text())
                    if old['status']!='error' or not args.retry_errors:
                        rows.append(old);continue
                    backup=path.with_suffix(f'.failed_{time.time_ns()}.json');path.rename(backup)
                dimension=case['pca_dimension']
                try:
                    if dimension not in reductions:
                        try:reductions[dimension]=transform_features(dataset['x'],dimension)
                        except SkipCase as exc:reductions[dimension]=str(exc)
                    value=reductions[dimension]
                    if isinstance(value,str):raise SkipCase(value)
                    row=execute_case(dataset,value[0],case,config);row.update(value[1])
                except SkipCase as exc:
                    row={**case,'dataset':dataset['name'],'status':'skipped','reason':str(exc)}
                except Exception as exc:
                    row={**case,'dataset':dataset['name'],'status':'error','error':f'{type(exc).__name__}: {exc}',
                         'traceback':traceback.format_exc()}
                save(path,row);progress[key]=file_digest(path);save(progress_path,progress);rows.append(row)
                if done%25==0 or done==total:
                    print(f'[{done}/{total}] {dataset["name"]} {case["family"]} PCA={dimension} {row["status"]}',flush=True)
        selected=[]
        for dataset in datasets:
            selected+=selected_k_rows([r for r in rows if r['dataset']==dataset['name']])
        summary=make_report(output,rows+selected,datasets)
        print(json.dumps({'output':str(output),'status_counts':summary['status_counts'],
                          'raw_fits':len(rows),'label_free_selected':len(selected)},ensure_ascii=False,indent=2))
        return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-experiment',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'config/clustering_benchmark.yaml')
    parser.add_argument('--preset',choices=['quick','standard','full'],default='standard')
    parser.add_argument('--algorithms',nargs='+')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--retry-errors',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    result=run(parser.parse_args(argv))
    return 2 if result.get('status_counts',{}).get('error',0) else 0


if __name__=='__main__':raise SystemExit(main())
