#!/usr/bin/env python3
"""Prepare/check/run/compare a controlled mask x SupCon factorial experiment.

prepare/check never train. run delegates to the existing experiment pipeline,
with one subprocess per case. Existing completed, hash-verified jobs are reused.
"""
import argparse
from contextlib import contextmanager, nullcontext
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from util.gpu_wait import GpuWaitPolicy, is_cuda_allocation_failure, reserve_gpu, worker_environment
CASES={'mask_ce':('mask',0.),'mask_supcon':('mask',.1),
       'dual_ce':('mask_amplitude',0.),'dual_supcon':('mask_amplitude',.1)}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def sources():
    files=[REPO/'main.py',REPO/'prepare_submission.py']
    files += [p for directory in ['util','data','model','scripts','yolov5'] for p in (REPO/directory).rglob('*.py')]
    return {p.relative_to(REPO).as_posix():digest(p) for p in sorted(files)}


def prepare(a):
    output=a.suite.resolve()
    if output.exists():raise FileExistsError('Use a new suite directory')
    inputs=[p.expanduser().resolve() for p in a.inputs];weights=a.yolo_weights.expanduser().resolve()
    if not all(p.exists() for p in inputs) or not weights.is_file():raise FileNotFoundError('Verify input roots and YOLO weights')
    if a.protocol=='open_set' and not a.unknown_classes:raise ValueError('Explicit --unknown-classes required for development')
    if a.protocol=='all_known' and a.unknown_classes:raise ValueError('all_known must not hold out competition known classes')
    if a.seed<0 or a.epochs<1:raise ValueError('Invalid seed/epochs')
    base=yaml.safe_load(a.base_config.read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    cases={}
    for name,(mode,weight) in CASES.items():
        cfg=copy.deepcopy(base)
        cfg['experiment'].update(inputs=list(map(str,inputs)),protocol=a.protocol,seed=a.seed,
            training_variant='configured',val_ratio=.15,calibration_ratio=.2,
            test_ratio=.2 if a.protocol=='open_set' else 0.)
        cfg['experiment'].pop('known_classes',None);cfg['experiment'].pop('unknown_classes',None)
        if a.unknown_classes:cfg['experiment']['unknown_classes']=a.unknown_classes
        cfg['train'].update(yolo_weights=str(weights),epochs=a.epochs,train_group_policy='primary',
                            classifier_input_mode=mode,mask_in_chans=1 if mode=='mask' else 2,supcon_weight=weight,
                            early_stop_patience=a.epochs+1)
        cfg.setdefault('evaluation',{}).update(known_recheck=False,tail_policy='cover')
        # No held-out class count is supplied to the discovery algorithm.
        cfg.setdefault('discovery',{}).update(competition_profile='unconstrained',
            unknown_bounded_strategy='average',unknown_input_mode='single',save_visuals=False,
            save_feature_plots=False)
        path=output/(name+'.yaml');path.write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
        cases[name]={'config':str(path),'sha256':digest(path)}
    save(output/'suite.json',{'schema_version':1,'repo':str(REPO),'source_files':sources(),
        'seed':a.seed,'protocol':a.protocol,'detector':str(weights),'detector_sha256':digest(weights),
        'cases':cases,'jobs':{},'scope':'Four known-only training recipes; shared original-record split, detector, epochs and batch policy'})
    print('Prepared four configurations:',output)


def load(suite):
    state=json.loads((suite/'suite.json').read_text(encoding='utf-8'))
    if state['repo']!=str(REPO) or state['source_files']!=sources():
        raise ValueError('Prepared code changed or suite moved to another checkout; prepare a new suite')
    if digest(state['detector'])!=state['detector_sha256']:raise ValueError('Prepared detector changed')
    for item in state['cases'].values():
        if digest(item['config'])!=item['sha256']:raise ValueError('Prepared configuration changed; use a new suite')
    return state


@contextmanager
def locked(suite):
    with (suite/'.suite.lock').open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Another suite command is running') from exc
        yield


def artifacts(run):
    required=[run/'run.json',run/'split_manifest.json',run/'training/models/best.pth',run/'open_set/known_profile.npz']
    optional=[run/'inference.yaml',run/'test/results.json',run/'test/open_set_metrics.json',run/'test/predictions.csv']
    return required+[p for p in optional if p.is_file()]+sorted((run/'training/result/train_metrics').glob('epoch_*.json'))


def validate_gpu_case(config):
    """Workers are pinned to one physical UUID; model devices must stay local."""
    cfg=yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    if cfg.get('train',{}).get('yolo_device') not in (None,''):
        raise ValueError('GPU waiting requires train.yolo_device to be unset; select the physical card with --gpu-id')
    for section in ['evaluation','profile','discovery']:
        if cfg.get(section,{}).get('device','auto') not in ('auto','cuda','cuda:0'):
            raise ValueError('GPU waiting requires evaluation/profile/discovery.device=auto or cuda:0')


def worker(config,result_path):
    from util.cli import main as cli
    try:
        result=cli(['experiment','--config',str(config)])
    except BaseException as exc:
        save(result_path,{'status':'failed','error':f'{type(exc).__name__}: {exc}',
                          'cuda_allocation_failure':is_cuda_allocation_failure(exc)})
        raise
    save(result_path,{'status':'complete','run_dir':result['run_dir']})


def allocation_failure(attempt):
    try:
        result=json.loads((attempt/'result.json').read_text(encoding='utf-8'))
        return isinstance(result,dict) and result.get('cuda_allocation_failure') is True
    except (OSError,ValueError):
        return False


def run_cases(suite,state,selected,gpu_policy=None):
    for name in CASES if selected=='all' else [selected]:
        previous=state['jobs'].get(name)
        if previous and previous.get('status')=='complete':
            for p,h in previous['artifacts'].items():
                if digest(p)!=h:raise ValueError('Completed artifact changed: '+p)
            print('Verified; skip completed:',name);continue
        if gpu_policy:validate_gpu_case(state['cases'][name]['config'])
        history=copy.deepcopy((previous or {}).get('attempts',[]))
        if previous and previous.get('attempt') and not history:
            history.append({k:v for k,v in previous.items() if k!='attempts'})
        job={'status':'waiting' if gpu_policy else 'pending','attempts':history,'resource_retries':0}
        if gpu_policy:job['gpu_wait_policy']=vars(gpu_policy)
        state['jobs'][name]=job
        active=None
        try:
            while True:
                job['status']='waiting' if gpu_policy else 'pending';active=None
                save(suite/'suite.json',state)
                try:
                    with reserve_gpu(gpu_policy) if gpu_policy else nullcontext(None) as gpu:
                        attempt=suite/(name+'_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
                        attempt.mkdir();active={'status':'running','attempt':str(attempt)}
                        if gpu:active['gpu']=gpu.as_dict()
                        history.append(active);job.update(active);job.pop('error',None);save(suite/'suite.json',state)
                        command=[sys.executable,str(Path(__file__).resolve()),'_worker','--config',state['cases'][name]['config'],
                                 '--result',str(attempt/'result.json')]
                        options={'env':worker_environment(gpu)} if gpu else {}
                        subprocess.run(command,cwd=REPO,check=True,**options)
                        result=json.loads((attempt/'result.json').read_text());run=Path(result['run_dir'])
                        status=json.loads((run/'run.json').read_text())['status']
                        active.update(status=status,run_dir=str(run),artifacts={str(p):digest(p) for p in artifacts(run)})
                        job.update(active)
                        if status!='complete':raise RuntimeError('Run completed with upstream errors; review '+str(run))
                    break
                except subprocess.CalledProcessError as exc:
                    if active is None:raise
                    active.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                    if not (gpu_policy and allocation_failure(Path(active['attempt'])) and
                            job['resource_retries']<gpu_policy.oom_retries):raise
                    job.update(status='waiting',error=active['error'],resource_retries=job['resource_retries']+1)
                    save(suite/'suite.json',state)
                    print(f'[GPU retry] {name}: CUDA allocation failed; fresh experiment retry '
                          f'{job["resource_retries"]}/{gpu_policy.oom_retries} after '
                          f'{gpu_policy.poll_seconds:g}s. Failed attempt retained: {active["attempt"]}',flush=True)
                    time.sleep(gpu_policy.poll_seconds)
        except BaseException as exc:
            status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed'
            job.update(status=status,error=f'{type(exc).__name__}: {exc}')
            if active and active['status']=='running':active.update(status=status,error=job['error'])
            raise
        finally:save(suite/'suite.json',state)


def compare(suite,state):
    from util.competition_metrics import evaluate_records
    expected=None;rows=[]
    for name in CASES:
        job=state['jobs'].get(name,{})
        if job.get('status')!='complete':raise ValueError('Finish all cases before compare: '+name)
        for path,value in job['artifacts'].items():
            if digest(path)!=value:raise ValueError('Completed artifact changed: '+path)
        run=Path(job['run_dir']);manifest=json.loads((run/'split_manifest.json').read_text())
        shared={k:manifest[k] for k in ['known_classes','unknown_classes','classes','ratios','records']}
        signature=hashlib.sha256(json.dumps(shared,sort_keys=True).encode()).hexdigest()
        if expected is None:expected=signature
        if signature!=expected:raise ValueError('Data/classes/splits differ; comparison would not be controlled')
        epochs=[json.loads(p.read_text()) for p in (run/'training/result/train_metrics').glob('epoch_*.json')]
        row={'case':name,'run_dir':str(run),'split_sha256':signature,'metrics':None,
             'compute':{'epochs':len(epochs),'optimizer_steps':sum(e['optimizer_steps'] for e in epochs),
                        'valid_instances':sum(e['valid_instances'] for e in epochs),
                        'training_wall_seconds':sum(e['wall_seconds'] for e in epochs)}}
        if state['protocol']=='open_set':
            report=json.loads((run/'test/results.json').read_text());truth={};pred={}
            identities={r['path']:r['recording_id'] for r in manifest['records']}
            for item in report['files']:
                labels=item.get('true_class_names')
                if not isinstance(labels,list) or len(labels)!=1:raise ValueError('Single-device development truth required')
                key=identities[item['path']];decision=item['single_signal_prediction'];label=None
                if decision['recognition_status']=='known':label=decision['class_name']
                elif decision['recognition_status']=='novel_candidate' and str(decision['display_label']).startswith('Unknown_') and decision['display_label']!='Unknown_unassigned':label=decision['display_label']
                if key in truth and (truth[key]!=labels[0] or pred[key]!=label):raise ValueError('Original-record aliases disagree')
                truth[key]=labels[0];pred[key]=label
            row['metrics']=evaluate_records([{'file_id':k,'class_id':v} for k,v in truth.items()],
                [{'file_id':k,'class_id':v} for k,v in pred.items()],manifest['known_classes'])
        rows.append(row)
    result={'cases':rows,'selection_note':'Development results only. Lock a recipe before a fresh test. all_known has no held-out test metrics.'}
    save(suite/'comparison.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--suite',type=Path,required=True)
    prep.add_argument('--inputs',nargs='+',type=Path,required=True);prep.add_argument('--yolo-weights',type=Path,required=True)
    prep.add_argument('--base-config',type=Path,default=REPO/'config/competition_train.yaml')
    prep.add_argument('--protocol',choices=['open_set','all_known'],default='open_set')
    prep.add_argument('--unknown-classes',nargs='+');prep.add_argument('--seed',type=int,default=42);prep.add_argument('--epochs',type=int,default=100)
    for name in ['check','run','compare']:
        cmd=sub.add_parser(name);cmd.add_argument('--suite',type=Path,required=True)
        if name=='run':
            cmd.add_argument('--case',choices=[*CASES,'all'],default='all')
            cmd.add_argument('--wait-for-gpu',action='store_true',help='Wait before each case without creating a CUDA context')
            cmd.add_argument('--gpu-id',default='0',help='Physical nvidia-smi index or UUID; worker is pinned to this GPU')
            cmd.add_argument('--gpu-min-free-mib',type=int,default=8192)
            cmd.add_argument('--gpu-poll-seconds',type=float,default=30.)
            cmd.add_argument('--gpu-wait-timeout',type=float,default=0.,help='Per-admission wait limit in seconds; 0 waits indefinitely')
            cmd.add_argument('--gpu-oom-retries',type=int,default=1,help='Fresh case retries for explicit CUDA allocation failures only')
    worker_parser=sub.add_parser('_worker');worker_parser.add_argument('--config',type=Path,required=True);worker_parser.add_argument('--result',type=Path,required=True)
    a=p.parse_args(argv)
    if a.command=='prepare':return prepare(a)
    from util.cli import main as cli
    if a.command=='_worker':
        return worker(a.config,a.result)
    gpu_policy=None
    if a.command=='run' and a.wait_for_gpu:
        gpu_policy=GpuWaitPolicy(a.gpu_id,a.gpu_min_free_mib,a.gpu_poll_seconds,a.gpu_wait_timeout,a.gpu_oom_retries)
    suite=a.suite.resolve()
    with locked(suite):
        state=load(suite)
        if a.command=='check':
            for case in state['cases'].values():cli(['experiment','--config',case['config'],'--dry_run'])
            print('All four cases passed configuration preflight; no training/inference was run.')
        elif a.command=='run':run_cases(suite,state,a.case,gpu_policy)
        else:compare(suite,state)


if __name__=='__main__':main()
