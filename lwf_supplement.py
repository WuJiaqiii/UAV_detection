#!/usr/bin/env python3
"""Reproducible 12 -> 15 -> 18 class LwF comparison for UAV_detection 53a6975.

Uses existing public training recordings with oracle folder labels, not automatic
new-class annotation. Never uses the previous 20-class classifier as a teacher.
All RF training/inference is delegated to the repository's existing task CLI.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime

MODELS = ('base', 'kd0_1', 'lwf_1', 'kd0_2', 'lwf_2')
STAGES = {'base': 0, 'kd0_1': 1, 'lwf_1': 1, 'kd0_2': 2, 'lwf_2': 2}
COUNTS = (12, 15, 18)
SOURCE_DEFAULT = 'experiment/all_known_training_20261107_113635'
VERSION = '2026-11-08-profile-truth-fix'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(repo):
    names = ['main.py', 'scripts/evaluate_single_signal.py', 'config/competition_v2_train.yaml']
    names += [p.relative_to(repo).as_posix() for directory in ('util', 'data', 'model')
              for p in (repo/directory).glob('*.py')]
    names += [p.relative_to(repo).as_posix() for p in (repo/'yolov5').rglob('*.py')]
    return {name: digest(repo/name) for name in sorted(set(names))}


def partitions(source, seed):
    """Keep existing val/calibration; reserve eight ORIGINAL train IDs per class."""
    records = [dict(row) for row in source['records']]
    groups, content_ids = defaultdict(list), {}
    seen_paths = set()
    for row in records:
        path = str(Path(row['path']).expanduser().resolve())
        if path in seen_paths:
            raise ValueError('Duplicate source path: ' + path)
        seen_paths.add(path)
        row['path'] = path
        identity, sha = row['recording_id'], row['source_sha256']
        if not identity or len(sha) != 64:
            raise ValueError('Original recording identity / SHA256 required')
        if sha in content_ids and content_ids[sha] != identity:
            raise ValueError('Identical content has inconsistent original identities')
        content_ids[sha] = identity
        groups[identity].append(row)
    by_class = defaultdict(list)
    for identity, rows in groups.items():
        if len({row['class_name'] for row in rows}) != 1 or len({row['split'] for row in rows}) != 1:
            raise ValueError('Original recording crosses labels or source splits: ' + identity)
        name, split = rows[0]['class_name'], rows[0]['split']
        if split not in ('train', 'val', 'calibration'):
            raise ValueError('Expected an all_known source with train/val/calibration only')
        by_class[name].append((identity, rows))
    if len(by_class) != 20:
        raise ValueError('This protocol expects exactly 20 source classes')
    order = sorted(by_class)
    random.Random(seed).shuffle(order)
    result = []
    for name in order:
        train = [(identity, rows) for identity, rows in by_class[name] if rows[0]['split'] == 'train']
        train.sort(key=lambda pair: hashlib.sha256(f'{seed}:{name}:{pair[0]}'.encode()).hexdigest())
        if len(train) < 18:
            raise ValueError('Need at least 18 source training recordings per class: ' + name)
        test_ids = {identity for identity, _ in train[:8]}
        counts = Counter(rows[0]['split'] for _, rows in by_class[name])
        if counts['calibration'] < 5 or counts['val'] < 1:
            raise ValueError('Independent calibration >=5 and validation >=1 needed per class')
        for identity, rows in by_class[name]:
            for row in rows:
                row['source_split'] = row['split']
                if identity in test_ids:
                    row['split'] = 'test'
                result.append(row)
    return order, result


def subset(state, split, names=None):
    return [r['path'] for r in state['records'] if r['split'] == split and
            (names is None or r['class_name'] in names)]


def train_config(state, model, teacher=None):
    import yaml
    repo = Path(state['repo'])
    cfg = yaml.safe_load((repo/'config/competition_v2_train.yaml').read_text())['train'].copy()
    stage, order = STAGES[model], state['class_order']
    cfg.update(run_mode='train', input_type='iq_dat', trainer_type='normal',
        train_dataset_path=subset(state, 'train'), val_dataset_path=subset(state, 'val'),
        class_names=order, exclude_classes=order[COUNTS[stage]:], eval_exclude_classes=[],
        incremental_split_mode='normal', incremental_new_classes=[], checkpoint_path=None,
        lwf_teacher_checkpoint='', lwf_old_num_classes=0, seed=state['seed'],
        epochs=state['epochs'], batch_size=state['batch_size'], num_workers=1,
        train_signal_mode='single', train_group_policy='primary', supcon_weight=0.,
        bbox_cache_mode='off', mask_pretrained=True, freeze_backbone=False,
        lr=0.0001, weight_decay=0.01, early_stop_patience=20,
        save_val_detect_vis=False, val_detect_vis_ratio=0.)
    cfg['yolo_weights'] = state['yolo_weights']
    if stage:
        if not teacher:
            raise ValueError('An independently trained previous-stage teacher is required')
        cfg.update(trainer_type='lwf', incremental_split_mode='incremental', mask_pretrained=False,
            incremental_new_classes=order[COUNTS[stage-1]:COUNTS[stage]],
            lwf_old_num_classes=COUNTS[stage-1], lwf_teacher_checkpoint=str(teacher),
            lwf_temperature=4., lwf_lambda_kd=0. if model.startswith('kd0') else 10.,
            lwf_freeze_backbone=False)
    return {'train': cfg}


def profile_config(state, model, checkpoint):
    names = set(state['class_order'][:COUNTS[STAGES[model]]])
    work = Path(state['work'])
    return {'evaluation': dict(checkpoint_path=str(checkpoint), yolo_weights=state['yolo_weights'],
        signal_mode='multi', unknown_input_mode='single', iq_sample_rate=100000000,
        iq_stride_ms=50., tail_policy='drop', max_slices=0, min_votes=1,
        track_feature_cosine_min=0.8, known_recheck=False, novelty_policy='calibrated_track',
        recording_ids_json=str(work/'recording_ids.json'), save_visuals=False, save_feature_plots=False,
        competition_profile='unconstrained', unknown_bounded_strategy='average',
        unknown_cluster_policy='auto', unknown_cluster_backend='hdbscan',
        unknown_min_cluster_size=5, unknown_min_samples=3, unknown_min_recordings=2,
        unknown_allow_single_cluster=True),
        'profile': dict(reference_inputs=subset(state, 'train', names),
            calibration_inputs=subset(state, 'calibration', names), label_mode='parent',
            profile_score_method='maha_pp_shrink', profile_covariance_shrinkage=0.1,
            profile_calibration_unit='primary', known_false_reject_rate=0.01,
            min_calibration_per_class=5, max_reference_per_class=2000, profile_recheck=False)}


def predict_config(state, model, checkpoint, profile):
    cfg = profile_config(state, model, checkpoint)
    cfg.pop('profile')
    cfg['predict'] = dict(inputs=subset(state, 'test'), labels_json=str(Path(state['work'])/'labels.json'),
        open_set_profile=str(profile), fast=False, export_features=False)
    return cfg


def prepare(args):
    repo, source_dir, work = args.repo.resolve(), args.source_experiment.resolve(), args.work.resolve()
    if work.exists():
        raise FileExistsError('Use a new work directory; existing experiment is never overwritten: ' + str(work))
    source_path = source_dir/'split_manifest.json'
    source = read_json(source_path)
    if source.get('protocol') != 'all_known':
        raise ValueError('Use the existing all_known source experiment')
    order, records = partitions(source, args.seed)
    for row in records:
        path = Path(row['path'])
        size = path.stat().st_size
        if path.parent.name != row['class_name'] or size % 4 or size < 20000000:
            raise ValueError('Invalid class-folder IQ recording: ' + str(path))
    template = read_json(source_dir/'effective_train.json')
    yolo = str(Path(template['yolo_weights']).resolve())
    if not Path(yolo).is_file():
        raise FileNotFoundError(yolo)
    state = dict(schema_version=1, repo=str(repo), work=str(work), seed=args.seed,
        source_manifest=str(source_path), source_manifest_sha256=digest(source_path),
        epochs=args.epochs, batch_size=args.batch_size, yolo_weights=yolo, yolo_sha256=digest(yolo),
        source_stats={r['path']:[Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns] for r in records},
        class_order=order, records=records, source_files=sources(repo), jobs={})
    # No trained 20-class classifier is referenced or loaded by this suite.
    write_json(work/'suite.json', state)
    write_json(work/'labels.json', {r['path']:r['class_name'] for r in records})
    write_json(work/'recording_ids.json', {r['path']:r['recording_id'] for r in records})
    write_json(work/'protocol.json', dict(base=order[:12], increment1=order[12:15],
        increment2=order[15:18], always_unknown=order[18:],
        counts={name:dict(Counter(r['split'] for r in records if r['class_name']==name)) for name in order},
        training_note='Oracle class-folder labels; old validation labels retained; no old gradient replay',
        hash_note='Source content hashes reuse the original manifest; source files must remain immutable'))
    print('Prepared:', work)
    print((work/'protocol.json').read_text())


def load_state(work):
    state = read_json(work/'suite.json')
    if str(work.resolve()) != state['work']:
        raise ValueError('Do not move an experiment after preparation')
    if sources(Path(state['repo'])) != state['source_files']:
        raise ValueError('Repository feature/training code changed; use a new prepared work directory')
    if digest(state['yolo_weights']) != state['yolo_sha256']:
        raise ValueError('Detector checkpoint changed since preparation')
    for name, expected in state['source_stats'].items():
        stat = Path(name).stat()
        if [stat.st_size,stat.st_mtime_ns] != expected:
            raise ValueError('Source recording changed since preparation: ' + name)
    for job in state['jobs'].values():
        for name, expected in job.get('artifacts', {}).items():
            if digest(name) != expected:
                raise ValueError('Completed checkpoint/profile/report changed: ' + name)
    return state


@contextmanager
def locked(work):
    import fcntl
    with (work/'.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another suite step is already running in this work directory')
        yield


def write_yaml(path, cfg):
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')


def invoke(state, key, task, cfg):
    """Each repository task gets a separate process and its normal automatic run directory."""
    if key in state['jobs']:
        previous = state['jobs'][key]
        if read_json(Path(previous['run_dir'])/'run.json')['status'] != 'complete':
            raise RuntimeError('Recorded completed run was changed: ' + key)
        if any(digest(name) != sha for name, sha in previous.get('artifacts',{}).items()):
            raise RuntimeError('A completed checkpoint/profile/report was changed: ' + key)
        print('Already completed:', key, previous['run_dir'])
        return previous
    work = Path(state['work'])
    attempt = work/'jobs'/key/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    attempt.mkdir(parents=True)
    config, result = attempt/'config.yaml', attempt/'result.json'
    write_yaml(config, cfg)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '_worker',
               '--repo', state['repo'], '--task', task, '--config', str(config), '--result', str(result)]
    print('Running:', task, key, flush=True)
    subprocess.run(command, cwd=state['repo'], check=True)
    output = read_json(result)
    if output['status'] != 'complete':
        raise RuntimeError(f'{key}: upstream run did not complete: {output}')
    state['jobs'][key] = output
    write_json(work/'suite.json', state)
    return output


def checkpoint_for(state, model):
    key = 'train_'+model
    if key not in state['jobs']:
        raise RuntimeError('Run training first: ' + model)
    checkpoint = Path(state['jobs'][key]['run_dir'])/'models/best.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def train(state, model):
    previous = None if model == 'base' else ('base' if STAGES[model] == 1 else model.replace('_2', '_1'))
    teacher = checkpoint_for(state, previous) if previous else None
    invoke(state, 'train_'+model, 'train', train_config(state, model, teacher))
    print('Checkpoint:', checkpoint_for(state, model))


def evaluate(state, model):
    checkpoint = checkpoint_for(state, model)
    profile_run = invoke(state, 'profile_'+model, 'profile', profile_config(state, model, checkpoint))
    profile = Path(profile_run['run_dir'])/'known_profile.npz'
    invoke(state, 'eval_'+model, 'predict', predict_config(state, model, checkpoint, profile))


def worker(args):
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo.resolve()))
    from util.cli import main
    result = main([args.task, '--config', str(args.config)])
    run_dir = Path(result['run_dir'])
    run = read_json(run_dir/'run.json')
    if args.task == 'profile':
        for key in ('error_files', 'files_with_slice_errors'):
            if result.get('extraction_errors', {}).get(key, 0):
                raise RuntimeError('Profile extraction has upstream errors: ' + str(result['extraction_errors']))
    if args.task == 'predict':
        metrics = result.get('metrics', {})
        if metrics.get('error_files', 0) or metrics.get('files_with_slice_errors', 0):
            raise RuntimeError('Prediction has upstream errors; inspect ' + str(run_dir))
    artifacts = {'train':['models/best.pth'], 'profile':['known_profile.npz'],
                 'predict':['open_set_metrics.json']}[args.task]
    hashes = {str(run_dir/name):digest(run_dir/name) for name in artifacts}
    write_json(args.result, dict(run_dir=str(run_dir), status=run['status'], task=args.task, artifacts=hashes))


def check_truth_inputs(task, args, classes):
    """Use the real truth loader before GPU work; it validates ALL JSON entries."""
    from util.multi_signal_eval import load_multi_truth
    if task == 'profile':
        specifications = [(getattr(args, split+'_inputs'),
            getattr(args, split+'_labels_json') if args.label_mode == 'json' else None,
            args.label_mode == 'parent', args.label_mode == 'filename', False)
            for split in ('reference', 'calibration')]
    else:
        specifications = [(args.inputs, args.labels_json, args.label_from_parent,
                           args.label_from_filename, args.open_set_profile is not None)]
    for inputs, label_path, parent, filename, allow_unknown in specifications:
        paths = [Path(name).resolve() for name in inputs]
        truth = load_multi_truth(label_path, paths, classes, label_from_parent=parent,
                                 allow_unknown=allow_unknown, label_from_filename=filename)
        if any(truth.get(str(path)) != [path.parent.name] for path in paths):
            raise ValueError('Missing or inconsistent class-folder truth for '+task)


def check(state):
    """Schema and data-membership checks only: never read IQ arrays or build a detector."""
    repo = Path(state['repo'])
    sys.path.insert(0, str(repo))
    from util.cli import build_task_parser, settings_tokens
    from util.config import Config
    from util.train_entry import setup_incremental_class_split, validate_training_options
    from data.iq_dat_loader import RawIQDataset
    if digest(state['yolo_weights']) != state['yolo_sha256']:
        raise ValueError('Detector checkpoint changed since preparation')
    for model in MODELS:
        cfg = train_config(state, model, '/NOT_EXECUTED/previous_stage_best.pth' if model != 'base' else None)
        parser = build_task_parser('train')
        args = parser.parse_args(settings_tokens(parser, cfg['train']))
        validate_training_options(args)
        config = Config(args)
        setup_incremental_class_split(config)
        config.cache_dir = None
        training = RawIQDataset(config, None, dataset_path=config.train_dataset_path, split_name='train')
        validation = RawIQDataset(config, None, dataset_path=config.val_dataset_path, split_name='val')
        expected = state['class_order'][:COUNTS[STAGES[model]]]
        assert config.classes == {name:i for i,name in enumerate(expected)}
        new = set(expected if model == 'base' else expected[COUNTS[STAGES[model]-1]:])
        assert {r['class_name'] for r in training.samples} == new
        assert {r['class_name'] for r in validation.samples} == set(expected)
        print(model, 'classes=', len(expected), 'training_files=', len(training), 'validation_files=',len(validation))
        for task, document in [('profile', profile_config(state, model, '/NOT_EXECUTED/model.pth')),
                               ('predict', predict_config(state, model, '/NOT_EXECUTED/model.pth', '/NOT_EXECUTED/profile.npz'))]:
            settings = document['evaluation'].copy()
            settings.update(document.get(task, {}))
            p = build_task_parser(task)
            parsed = p.parse_args(settings_tokens(p, settings))
            check_truth_inputs(task, parsed, config.classes)
    print('CHECK PASSED: schemas, real truth loading, class prefixes, new-only training and disjoint source-record partitions')


def group_scores(samples, names):
    names = set(names)
    truth = [s for s in samples if s['true_class_name'] in names]
    predictions = [s for s in samples if s['recognition_status']=='known' and s['predicted_class'] in names]
    correct = sum(s['recognition_status']=='known' and s['predicted_class']==s['true_class_name'] for s in truth)
    return (correct/len(truth) if truth else None,
            2*correct/(len(truth)+len(predictions)) if truth or predictions else None)


def summarize(state):
    from sklearn.metrics import adjusted_rand_score
    expected_ids = {r['recording_id'] for r in state['records'] if r['split']=='test'}
    samples = {}
    for model in MODELS:
        key = 'eval_'+model
        if key not in state['jobs']:
            raise RuntimeError('Evaluation missing: ' + model)
        report = read_json(Path(state['jobs'][key]['run_dir'])/'open_set_metrics.json')
        rows = report['samples']
        if report['skipped'] or {s['recording_id'] for s in rows} != expected_ids or len(rows) != len(expected_ids):
            raise ValueError('Incomplete / duplicated test coverage: ' + model)
        samples[model] = rows
    result, order = [], state['class_order']
    base_acc = group_scores(samples['base'], order[:12])[0]
    for stage in (0,1,2):
        models = ['base'] if stage == 0 else ['base',f'kd0_{stage}',f'lwf_{stage}']
        known = set(order[:COUNTS[stage]])
        for model in models:
            rows = samples[model]
            unknown = [s for s in rows if s['true_class_name'] not in known]
            tp = sum(s['recognition_status']=='novel_candidate' for s in unknown)
            fp = sum(s['recognition_status']=='novel_candidate' for s in rows if s['true_class_name'] in known)
            assigned = [s for s in unknown if s.get('unknown_cluster_id')]
            old_acc, old_f1 = group_scores(rows, order[:12])
            row = dict(stage=stage, model='frozen_base' if model=='base' and stage else model,
                known_classes=len(known), remaining_unknown_classes=20-len(known), test_recordings=len(rows),
                old12_accuracy=old_acc, old12_micro_f1=old_f1,
                increment1_accuracy=group_scores(rows,order[12:15])[0],
                increment2_accuracy=group_scores(rows,order[15:18])[0],
                stage_known_micro_f1=group_scores(rows,known)[1],
                remaining_unknown_f1=2*tp/(len(unknown)+tp+fp) if len(unknown)+tp+fp else None,
                remaining_unknown_false_accept=sum(s['recognition_status']=='known' for s in unknown)/len(unknown),
                unknown_cluster_coverage=len(assigned)/len(unknown),
                unknown_ari_conditional=adjusted_rand_score([s['true_class_name'] for s in assigned],
                    [s['unknown_cluster_id'] for s in assigned]) if len(assigned)>=2 else None,
                no_signal=sum(s['recognition_status']=='no_signal' for s in rows),
                old12_forgetting_signed=base_acc-old_acc)
            if stage==2 and model!='base':
                first = samples[model.replace('_2','_1')]
                row['increment1_forgetting_signed'] = group_scores(first,order[12:15])[0]-row['increment1_accuracy']
            result.append(row)
    work = Path(state['work'])
    write_json(work/'comparison.json', dict(rows=result, stage_truth='Same stage-specific known/unknown sets for every comparator',
        scope='Oracle incremental classifier experiment; not a full autonomous task-three system'))
    columns = list(dict.fromkeys(k for r in result for k in r))
    with (work/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows(result)
    print('Results:', work/'comparison.csv')
    for row in result:
        print(json.dumps(row, ensure_ascii=False, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--source-experiment', type=Path, default=Path(SOURCE_DEFAULT))
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    for command in ('check','train','evaluate','summarize','run'):
        p = commands.add_parser(command)
        p.add_argument('--work', type=Path, required=True)
        if command in ('train','evaluate'):
            p.add_argument('--model', choices=(*MODELS,'all'), default='all')
    p = commands.add_parser('_worker', help=argparse.SUPPRESS)
    for key in ('repo','config','result'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--task', choices=('train','profile','predict'),required=True)
    args = parser.parse_args()
    if args.command == '_worker':
        return worker(args)
    if args.command == 'prepare':
        if args.seed<0 or args.epochs<1 or args.batch_size<1:
            raise ValueError('seed>=0, epochs>=1 and batch-size>=1 required')
        return prepare(args)
    work = args.work.resolve()
    with locked(work):
        state = load_state(work)
        if args.command == 'check':
            check(state)
        elif args.command == 'summarize':
            summarize(state)
        elif args.command in ('train','evaluate'):
            selected = MODELS if args.model == 'all' else [args.model]
            for model in selected:
                (train if args.command=='train' else evaluate)(state,model)
        else:
            check(state)
            for model in MODELS:
                train(state,model)
                evaluate(state,model)
            summarize(state)


if __name__ == '__main__':
    main()
