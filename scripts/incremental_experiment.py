#!/usr/bin/env python3
"""Fresh, recording-disjoint 12->15->18 experiments: finetune, LwF and replay.

Folder names supply oracle labels. This evaluates incremental classifiers; it is
not a claim of autonomous new-class labeling or a complete online task-3 system.
"""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from util.experiment_io import read_json, write_json, digest, source_snapshot
from util.gpu_wait import GpuWaitPolicy, reserve_gpu, worker_environment

METHODS = ('finetune','lwf','replay')


def models(state):
    return ['base']+[f'{method}_{stage}' for stage in (1,2) for method in state['methods']]


def stage_of(model):
    return 0 if model == 'base' else int(model.rsplit('_',1)[1])


def subset(state, split, names=None):
    return [r['path'] for r in state['records'] if r['split']==split and
            (names is None or r['class_name'] in names)]


def train_config(state, model, teacher=None):
    stage = stage_of(model)
    counts, order = state['class_counts'], state['class_order']
    cfg = dict(state['train_settings'])
    cfg.update(input_type='iq_dat', train_dataset_path=subset(state,'train'),
        val_dataset_path=subset(state,'val'), class_names=order, exclude_classes=order[counts[stage]:],
        eval_exclude_classes=[], trainer_type='normal', incremental_split_mode='normal',
        incremental_new_classes=[], seed=state['seed'], epochs=state['epochs'], batch_size=state['batch_size'],
        num_workers=1, train_signal_mode='single', train_group_policy='primary', supcon_weight=0.,
        bbox_cache_mode='off', mask_pretrained=True, freeze_backbone=False,
        checkpoint_keep_last=2, checkpoint_keep_epochs=state['keep_epochs'],
        save_val_detect_vis=False, val_detect_vis_ratio=0.)
    if stage:
        if not teacher:
            raise ValueError('Incremental stages require their freshly trained previous-stage teacher')
        method = model.rsplit('_',1)[0]
        cfg.update(trainer_type='lwf', incremental_split_mode='incremental', mask_pretrained=False,
            incremental_new_classes=order[counts[stage-1]:counts[stage]],
            lwf_old_num_classes=counts[stage-1], lwf_teacher_checkpoint=str(teacher),
            lwf_temperature=4., lwf_lambda_kd=0. if method=='finetune' else 10.,
            lwf_replay_per_class=0, lwf_feature_kd_weight=0., lwf_kd_new_weight=1.,
            lwf_backbone_lr_multiplier=1., lwf_freeze_bn=False, lwf_freeze_backbone=False,
            checkpoint_selection='old_new_hmean', early_stop_min_epochs=min(30,state['epochs']))
        if method=='replay':
            cfg.update(lwf_replay_manifest=str(Path(state['work'])/'split_manifest.json'),
                lwf_replay_per_class=state['replay_per_class'][stage-1],
                lwf_replay_fraction=state['replay_fraction'][stage-1],
                lwf_temperature=2., lwf_lambda_kd=1., lwf_kd_new_weight=0.,
                lwf_feature_kd_weight=.1, lwf_backbone_lr_multiplier=.1, lwf_freeze_bn=True)
    return {'train':cfg}


def profile_config(state, model, checkpoint):
    names = set(state['class_order'][:state['class_counts'][stage_of(model)]])
    training = state['train_settings']
    return {'evaluation':dict(checkpoint_path=str(checkpoint), yolo_weights=state['yolo_weights'],
        yolo_conf_thres=training['yolo_conf_thres'], signal_mode='multi', unknown_input_mode='single',
        iq_sample_rate=100000000, iq_stride_ms=50., tail_policy='cover', max_slices=0,
        min_votes=1, track_feature_cosine_min=.8, known_recheck=False,
        recording_ids_json=str(Path(state['work'])/'recording_ids.json'),
        save_visuals=False, save_feature_plots=False, competition_profile='unconstrained',
        unknown_cluster_backend='spectral', unknown_bounded_strategy='spectral',
        unknown_min_cluster_size=5, unknown_min_recordings=2),
        'profile':dict(reference_inputs=subset(state,'train',names),
            calibration_inputs=subset(state,'calibration',names), label_mode='parent',
            profile_score_method='maha_pp_shrink', profile_calibration_unit='primary',
            profile_covariance_shrinkage=.1, min_calibration_per_class=5,
            known_false_reject_rate=.01, profile_recheck=False)}


def predict_config(state, model, checkpoint, profile):
    cfg = profile_config(state,model,checkpoint)
    cfg.pop('profile')
    cfg['predict'] = dict(inputs=subset(state,'test'), labels_json=str(Path(state['work'])/'labels.json'),
        open_set_profile=str(profile), fast=False, export_features=False)
    return cfg


def prepare(args):
    from util.cli import build_task_parser, settings_tokens, load_settings
    from util.experiment_split import split_dataset
    if not math.isfinite(args.confidence) or not 0<args.confidence<=1:
        raise ValueError('confidence must be finite and in (0,1]')
    if args.work.exists():
        raise ValueError('Use a new work directory; previous experiments are never overwritten')
    counts = args.class_counts
    if not (0 < counts[0] < counts[1] < counts[2]):
        raise ValueError('class-counts must be three strictly increasing positive counts')
    if args.epochs<1 or args.batch_size<2 or args.seed<0 or any(n<1 for n in args.keep_epochs):
        raise ValueError('Invalid epochs, batch-size, seed or keep-epochs')
    if len(set(args.methods)) != len(args.methods):
        raise ValueError('Duplicate comparison methods')
    if any(n<1 for n in args.replay_per_class) or any(not 0<f<1 for f in args.replay_fraction):
        raise ValueError('Positive replay budgets and replay fractions in (0,1) required')
    work = args.work.resolve()
    train_settings, _ = load_settings(args.config,'train')
    detector = args.yolo_weights.expanduser().resolve()
    if not detector.is_file():
        raise FileNotFoundError(detector)
    train_settings.update(yolo_weights=str(detector), yolo_conf_thres=args.confidence)
    parsed = build_task_parser('train').parse_args(settings_tokens(build_task_parser('train'),train_settings))
    from util.train_entry import validate_training_options
    validate_training_options(parsed)
    split = SimpleNamespace(protocol='all_known', inputs=[str(args.data_root.resolve())],
        input_type='iq_dat', recursive=True, exclude_classes=[], known_classes=None, unknown_classes=None,
        val_ratio=.15, calibration_ratio=.2, test_ratio=.2, recording_ids_json=args.recording_ids,
        seed=args.seed)
    manifest = split_dataset(split,parsed)
    order = sorted(manifest['classes'])
    if len(order)<=counts[-1]:
        raise ValueError('Reserve at least one always-unknown class after the final increment')
    random.Random(args.seed).shuffle(order)
    records = [{**row,'split':'test' if row['split']=='test_known' else row['split']} for row in manifest['records']]
    for name in order:
        group = [r for r in records if r['class_name']==name]
        per_split = {split:len({r['recording_id'] for r in group if r['split']==split})
                     for split in ('train','val','calibration','test')}
        if per_split['calibration']<5 or any(n<1 for n in per_split.values()):
            raise ValueError(f'Independent calibration>=5 and nonempty train/val/test required: {name}: {per_split}')
        if 'replay' in args.methods and per_split['train']<max(args.replay_per_class):
            raise ValueError(f'Replay budget exceeds independent TRAIN recordings: {name}: {per_split}')
    state = dict(schema_version=3, repo=str(ROOT), work=str(work), seed=args.seed,
        class_order=order, class_counts=counts, records=records, train_settings=train_settings,
        methods=args.methods, epochs=args.epochs, batch_size=args.batch_size, keep_epochs=args.keep_epochs,
        replay_per_class=args.replay_per_class, replay_fraction=args.replay_fraction,
        yolo_weights=str(detector), yolo_sha256=digest(detector), source_files=source_snapshot(ROOT),
        source_stats={r['path']:[Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns] for r in records},
        wait_for_gpu=args.wait_for_gpu, gpu_id=args.gpu_id, gpu_min_free_mib=args.gpu_min_free_mib,jobs={})
    work.mkdir(parents=True)
    write_json(work/'split_manifest.json',dict(schema_version=3,records=records,seed=args.seed))
    write_json(work/'labels.json',{r['path']:r['class_name'] for r in records if r['split']=='test'})
    write_json(work/'recording_ids.json',{r['path']:r['recording_id'] for r in records})
    write_json(work/'protocol.json',dict(class_order=order,class_counts=counts,always_unknown=order[counts[-1]:],
        counts={name:dict(Counter(r['split'] for r in records if r['class_name']==name)) for name in order},
        scope='Oracle incremental classifier comparison; all model stages train from this fresh split'))
    state['input_artifacts']={name:digest(work/name) for name in ('split_manifest.json','labels.json','recording_ids.json','protocol.json')}
    write_json(work/'suite.json',state)
    print('Prepared:',work,'\nNo pre-existing classifier or experiment is required.')
    return 0


@contextmanager
def locked(work):
    if not (work/'suite.json').is_file():
        raise ValueError('Run prepare first; suite.json is missing')
    with (work/'.lock').open('a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another parent or inherited worker still owns this suite') from exc
        yield stream.fileno()


def load_state(work):
    state = read_json(work/'suite.json')
    if state.get('schema_version')!=3 or state['work']!=str(work) or state['repo']!=str(ROOT):
        raise ValueError('Use a freshly prepared v3 suite in this checkout')
    if state['source_files']!=source_snapshot(ROOT) or state['yolo_sha256']!=digest(state['yolo_weights']):
        raise ValueError('Code or detector changed; prepare a fresh suite')
    for name,expected in state['input_artifacts'].items():
        if digest(work/name)!=expected:
            raise ValueError('Prepared input artifact changed: '+name)
    for path,expected in state['source_stats'].items():
        stat = Path(path).stat()
        if [stat.st_size,stat.st_mtime_ns]!=expected:
            raise ValueError('Source recording changed: '+path)
    for job in state['jobs'].values():
        if job.get('status')=='complete' and any(digest(path)!=sha for path,sha in job['artifacts'].items()):
            raise ValueError('Completed checkpoint/profile/report changed')
    return state


def invoke(state,key,task,config,lock_fd):
    work = Path(state['work'])
    previous = state['jobs'].get(key,{})
    if previous.get('status')=='complete':
        if read_json(Path(previous['run_dir'])/'run.json')['status']!='complete' or any(
                digest(path)!=sha for path,sha in previous['artifacts'].items()):
            raise ValueError('Completed job changed: '+key)
        return previous
    attempt = work/'jobs'/key/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    attempt.mkdir(parents=True)
    cfg = attempt/'config.yaml'
    cfg.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
    result = attempt/'result.json'
    state['jobs'][key]=dict(status='waiting',config=str(cfg),log=str(attempt/'worker.log'))
    write_json(work/'suite.json',state)
    policy = GpuWaitPolicy(state['gpu_id'],state['gpu_min_free_mib']) if state['wait_for_gpu'] else None
    from scripts.competition_infer import run_command
    try:
        with reserve_gpu(policy) if policy else nullcontext(None) as gpu:
            env = worker_environment(gpu) if gpu else dict(os.environ)
            command = [sys.executable,str(Path(__file__).resolve()),'_worker','--task',task,
                       '--config',str(cfg),'--result',str(result)]
            print('Running:',key,'log:',attempt/'worker.log',flush=True)
            code = run_command(command,env,attempt/'worker.log',pass_fds=(lock_fd,))
        if code!=0 or not result.is_file():
            raise RuntimeError(f'{key} failed; inspect {attempt/"worker.log"}')
        completed = read_json(result)
        state['jobs'][key].update(completed)
        write_json(work/'suite.json',state)
        return state['jobs'][key]
    except BaseException:
        state['jobs'][key]['status']='failed'
        write_json(work/'suite.json',state)
        raise


def checkpoint_for(state,model):
    job = state['jobs'].get('train_'+model,{})
    path = Path(job.get('run_dir',''))/'models/best.pth'
    if job.get('status')!='complete' or not path.is_file() or job['artifacts'].get(str(path))!=digest(path):
        raise ValueError('Previous-stage training must finish before use as teacher: '+model)
    return path


def train(state,model,lock_fd):
    stage = stage_of(model)
    teacher_model = None if not stage else ('base' if stage==1 else model.rsplit('_',1)[0]+'_1')
    teacher = checkpoint_for(state,teacher_model) if teacher_model else None
    return invoke(state,'train_'+model,'train',train_config(state,model,teacher),lock_fd)


def evaluate(state,model,lock_fd):
    checkpoint = checkpoint_for(state,model)
    profile = invoke(state,'profile_'+model,'profile',profile_config(state,model,checkpoint),lock_fd)
    return invoke(state,'eval_'+model,'predict',predict_config(state,model,checkpoint,
                  Path(profile['run_dir'])/'known_profile.npz'),lock_fd)


def worker(args):
    from util.cli import main as cli
    result = cli([args.task,'--config',str(args.config)])
    directory = Path(result['run_dir'])
    if read_json(directory/'run.json')['status']!='complete' or any(result.get('extraction_errors',{}).values()):
        raise RuntimeError('Upstream extraction/evaluation errors; inspect '+str(directory))
    names = {'train':['models/best.pth'],'profile':['known_profile.npz'],'predict':['open_set_metrics.json']}[args.task]
    write_json(args.result,dict(run_dir=str(directory),status='complete',
        artifacts={str(directory/name):digest(directory/name) for name in names}))
    return 0


def check(state):
    from util.cli import build_task_parser, settings_tokens
    for model in models(state):
        train_document = train_config(state,model,'/preflight/teacher.pth' if model!='base' else None)
        for task,document in [('train',train_document),('profile',profile_config(state,model,'/preflight/model.pth')),
            ('predict',predict_config(state,model,'/preflight/model.pth','/preflight/profile.npz'))]:
            parser = build_task_parser(task)
            settings = dict(document.get('evaluation',{}),**document.get(task,{}))
            args = parser.parse_args(settings_tokens(parser,settings))
            if task=='train':
                from util.train_entry import validate_training_options
                validate_training_options(args)
    print('CHECK PASSED: current CLI schemas and frozen data/code/artifact identities')


def summarize(state):
    rows = []
    expected = {r['recording_id'] for r in state['records'] if r['split']=='test'}
    for model in models(state):
        job = state['jobs'].get('eval_'+model,{})
        if job.get('status')!='complete':
            raise ValueError('Evaluation missing: '+model)
        report = read_json(Path(job['run_dir'])/'open_set_metrics.json')
        samples = report['samples']
        if report['skipped'] or len(samples)!=len(expected) or {r['recording_id'] for r in samples}!=expected:
            raise ValueError('Incomplete/duplicated recording coverage: '+model)
        old = set(state['class_order'][:state['class_counts'][0]])
        seen = set(state['class_order'][:state['class_counts'][stage_of(model)]])
        def accuracy(names):
            truth = [r for r in samples if r['true_class_name'] in names]
            correct = sum(r['recognition_status']=='known' and r['predicted_class']==r['true_class_name'] for r in truth)
            return correct/len(truth) if truth else None
        rows.append(dict(model=model,known_classes=len(seen),test_recordings=len(samples),
            old_accuracy=accuracy(old),new_accuracy=accuracy(seen-old),known_accuracy=accuracy(seen),
            metrics=report['metrics']))
    write_json(Path(state['work'])/'comparison.json',dict(rows=rows,
        scope='Same recording-disjoint development test; oracle class-folder labels; not official unseen-test accuracy'))
    print(json.dumps(rows,ensure_ascii=False,indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action',required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--yolo-weights',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'config/competition_train.yaml')
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--recording-ids',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--confidence',type=float,default=.02)
    p.add_argument('--class-counts',type=int,nargs=3,default=[12,15,18])
    p.add_argument('--methods',choices=METHODS,nargs='+',default=list(METHODS))
    p.add_argument('--replay-per-class',type=int,nargs=2,default=[4,8])
    p.add_argument('--replay-fraction',type=float,nargs=2,default=[.5,.8])
    p.add_argument('--keep-epochs',type=int,nargs='*',default=[5,10,15,20,30,35,59,60])
    p.add_argument('--wait-for-gpu',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--gpu-id',default='0');p.add_argument('--gpu-min-free-mib',type=int,default=8192)
    for action in ('check','train','evaluate','summarize','run'):
        p = commands.add_parser(action);p.add_argument('--work',type=Path,required=True)
        if action in ('train','evaluate'):
            p.add_argument('--model',default='all')
    p = commands.add_parser('_worker',help=argparse.SUPPRESS)
    p.add_argument('--task',choices=['train','profile','predict'],required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--result',type=Path,required=True)
    args = parser.parse_args(argv)
    if args.action=='prepare':return prepare(args)
    if args.action=='_worker':return worker(args)
    work = args.work.expanduser().resolve()
    with locked(work) as fd:
        state = load_state(work)
        if args.action=='check':check(state)
        elif args.action=='summarize':summarize(state)
        elif args.action in ('train','evaluate'):
            chosen = models(state) if args.model=='all' else [args.model]
            if any(m not in models(state) for m in chosen):raise ValueError('Unknown model for this prepared suite')
            for model in chosen:(train if args.action=='train' else evaluate)(state,model,fd)
        else:
            check(state)
            for model in models(state):
                train(state,model,fd);evaluate(state,model,fd)
            summarize(state)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,RuntimeError) as exc:
        print('ERROR:',exc,file=sys.stderr);raise SystemExit(2)
