#!/usr/bin/env python3
"""Prepare -> resume full-duration supplemental inference -> merge/recluster/export.

No training. A cover-calibrated profile is required. Pending mode refreshes ALL
original novel-file features, preventing a mixture of drop/cover feature caches.
"""
import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.legacy_submission_common import (
    ITEMS, read, save, read_csv, write_csv, checked_rows, sources_from_batch,
    verify_sources, recluster, export_results)
from util.open_set import file_sha256, load_npz, save_npz
from prepare_submission import identify


@contextmanager
def plan_lock(plan):
    with plan.with_suffix('.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another operation is using this supplemental plan') from exc
        yield


def select_pending(rows, scope='pending'):
    """Return sample -> reasons; complete files, not individual multi-signal rows."""
    selected = defaultdict(set)
    grouped = defaultdict(list)
    for row in rows:
        sample, _ = identify(row['path'])
        grouped[sample].append(row)
        if scope == 'all':
            selected[sample].add('all_files')
        if row['status'] == 'no_signal':
            selected[sample].add('no_signal')
        if row['status'] == 'novel_candidate':
            selected[sample].add('refresh_entire_unknown_feature_pool')
        if row.get('error', '').strip() or row['status'] in {'error', 'uncertain'}:
            selected[sample].add('upstream_or_uncertain')
    for sample, group in grouped.items():
        if sample.startswith('2-') and len({r['label'] for r in group if r['status'] != 'no_signal'}) > 3:
            selected[sample].add('more_than_three_classes')
    return {sample: sorted(reasons) for sample, reasons in sorted(selected.items())}


def check_frozen_inputs(plan):
    verify_sources(plan['sources'])
    from scripts.competition_pipeline import inspect_inputs
    _, _, signatures = inspect_inputs(plan['data_root'], ITEMS)
    if signatures != plan['input_signatures']:
        raise ValueError('Source DAT inventory changed since preparation; create a new plan')
    for name, digest in plan['code_hashes'].items():
        if file_sha256(ROOT/name) != digest:
            raise ValueError('Code changed since preparation: '+name+'; create a new plan')
    for key, digest in plan['artifact_hashes'].items():
        if file_sha256(key) != digest:
            raise ValueError('Weights/profile/config changed since preparation: '+key)
    for item in ITEMS:
        for source in plan['items'][item]['inputs']:
            stat = Path(source['path']).stat()
            if stat.st_size != source['size'] or stat.st_mtime_ns != source['mtime_ns']:
                raise ValueError('Selected DAT changed since preparation: '+source['path'])


def prepare(args):
    from scripts.competition_pipeline import training_artifacts, inspect_inputs, input_sampling_diagnostics, source_fingerprint
    from util.cli import load_settings, build_task_parser, settings_tokens
    from util.competition_protocol import protocol_for_item
    import yaml
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError('Output must be a new plan directory')
    batch, sources = sources_from_batch(args.batch, allow_errors=True)
    refreshed = read(args.profile_refresh/'refresh_manifest.json')
    if refreshed.get('status') != 'complete' or refreshed.get('tail_policy') != 'cover':
        raise ValueError('Use a completed cover profile refresh (not an old drop profile)')
    training = Path(batch['fingerprint']['training_run']).resolve()
    if Path(refreshed['training_run']).resolve() != training:
        raise ValueError('Profile refresh belongs to a different training run')
    if file_sha256(refreshed['profile']) != refreshed['profile_sha256']:
        raise ValueError('Refresh profile changed after completion')
    artifacts = training_artifacts(training, profile_override=refreshed['profile'])
    for key in ['checkpoint_sha256', 'detector_sha256']:
        if artifacts[key] != batch['fingerprint'][key]:
            raise ValueError('Supplement must use the original CNN and YOLO weights')
    data_root = args.data_root or Path(batch['fingerprint']['data_root'])
    durations = {}
    data_root, counts, signatures = inspect_inputs(data_root, ITEMS, diagnostics=durations)
    if any(counts[item] != sources[item]['input_count'] for item in ITEMS):
        raise ValueError('Source DAT counts differ from original batch')
    if args.scope == 'pending' and any(
            signatures[item] != batch['runs'][item].get('input_signature') for item in ITEMS):
        raise ValueError('Original DAT inventory changed; pending mode cannot reuse old known predictions. Use --scope all.')
    settings, _ = load_settings(Path(refreshed['inference_config']), 'infer')
    parser = build_task_parser('infer')
    frozen = parser.parse_args(settings_tokens(parser, settings))
    if frozen.tail_policy != 'cover' or frozen.max_slices != 0:
        raise ValueError('Full-duration inference requires tail_policy=cover and max_slices=0')
    sampling = input_sampling_diagnostics(durations, frozen)
    if any(v['uncovered_file_count'] for v in sampling.values()):
        raise ValueError('Configuration still leaves unobserved samples')
    plan = dict(schema_version=1, status='prepared', batch=str(args.batch.resolve()),
                scope=args.scope, data_root=str(data_root), sources=sources, items={},
                profile=artifacts['profile'], training_run=str(training), artifacts=artifacts,
                input_signatures=signatures, input_sampling=sampling,
                interpretation='Hybrid predictions in pending mode: old unselected known decisions plus cover re-inference; all novel cache features are new. Unknown->known recheck disabled.')
    configs = {}
    for item in ITEMS:
        _, rows = checked_rows(sources[item]['predictions'], item, counts[item], allow_errors=True)
        selected = select_pending(rows, args.scope)
        available = {p.stem: p.resolve() for p in (data_root/item).rglob('*')
                     if p.is_file() and p.suffix.lower() == '.dat'}
        inputs = []
        for sample, reasons in selected.items():
            path = available[sample]
            stat = path.stat()
            inputs.append(dict(sample=sample, path=str(path), size=stat.st_size,
                               mtime_ns=stat.st_mtime_ns, reasons=reasons))
        config = output/'configs'/(item+'.yaml')
        protocol = protocol_for_item(item)
        effective = {**settings, 'inputs': [x['path'] for x in inputs],
                     'checkpoint_path': artifacts['checkpoint'], 'yolo_weights': artifacts['yolo'],
                     'open_set_profile': artifacts['profile'], 'tail_policy': 'cover', 'max_slices': 0,
                     'signal_mode': 'multi', 'fast': True, 'save_discovery_cache': True,
                     'save_visuals': False, 'save_feature_plots': False, 'known_recheck': False,
                     'unknown_input_mode': protocol.input_mode, 'competition_profile': protocol.name,
                     'test_item': item, 'unknown_cluster_backend': 'spectral',
                     'unknown_bounded_strategy': 'spectral', 'unknown_cluster_policy': 'auto',
                     'unknown_fixed_k': None, 'spectral_neighbors': 10, 'spectral_seed': 42, 'spectral_n_init': 10}
        if inputs:
            parser.parse_args(settings_tokens(parser, effective))
        configs[config] = effective
        plan['items'][item] = dict(status='pending' if inputs else 'empty', inputs=inputs,
                                    selected_files=len(inputs), original_files=counts[item], config=str(config))
    output.mkdir(parents=True)
    for config, effective in configs.items():
        config.parent.mkdir(exist_ok=True)
        config.write_text(yaml.safe_dump({'infer': effective}, allow_unicode=True, sort_keys=False), encoding='utf-8')
    artifact_paths = [artifacts['checkpoint'], artifacts['yolo'], artifacts['profile'],
                      *[str(p) for p in configs]]
    plan['artifact_hashes'] = {p: file_sha256(p) for p in artifact_paths}
    plan['code_hashes'] = {**source_fingerprint(), **{name: file_sha256(ROOT/name) for name in
                          ['scripts/legacy_submission_common.py', 'scripts/supplement_full_duration.py']}}
    save(output/'supplement_plan.json', plan)
    print('Plan:', output/'supplement_plan.json', flush=True)
    for item, entry in plan['items'].items():
        print(f"{item}: {entry['selected_files']}/{entry['original_files']} files selected", flush=True)
    print('Preparation only: no GPU inference has started.', flush=True)
    return 0


def verify_child(entry):
    run_dir = Path(entry['run_dir'])
    if read(run_dir/'run.json')['status'] not in {'complete', 'completed_with_errors'}:
        raise ValueError('Supplemental inference did not finish')
    _, rows = checked_rows(run_dir/'predictions.csv', entry['item'])
    from prepare_submission import identify
    actual = {identify(r['path'])[0] for r in rows}
    if actual != {x['sample'] for x in entry['inputs']}:
        raise ValueError('Supplemental CSV does not cover the exact selected file set')
    for filename, key in [('predictions.csv', 'csv_sha256'), ('unknown_features.npz', 'cache_sha256')]:
        digest = file_sha256(run_dir/filename)
        if entry.get(key) and entry[key] != digest:
            raise ValueError('Supplemental output changed after completion')
        entry[key] = digest


def run(args):
    from util.gpu_wait import GpuWaitPolicy, reserve_gpu, worker_environment
    from scripts.competition_pipeline import run_command
    plan_path = args.plan.expanduser().resolve()
    with plan_lock(plan_path):
        plan = read(plan_path)
        check_frozen_inputs(plan)
        policy = GpuWaitPolicy(args.gpu_id, args.gpu_min_free_mib) if args.wait_for_gpu else None
        plan['status'] = 'running'; save(plan_path, plan)
        try:
            for item in ITEMS:
                entry = plan['items'][item]
                entry['item'] = item
                if entry['status'] == 'empty':
                    continue
                if entry['status'] == 'complete':
                    verify_child(entry)
                    print(item+': complete; skipped', flush=True)
                    continue
                previous_result = entry.get('result')
                if previous_result and Path(previous_result).is_file() and entry['status'] != 'failed':
                    completed = read(previous_result)
                    entry.update(run_dir=completed['run_dir'], metrics=completed['metrics'])
                    verify_child(entry)
                    entry['status'] = 'complete'; save(plan_path, plan)
                    print(item+': recovered completed worker; skipped', flush=True)
                    continue
                # A previous killed parent may leave its worker alive. Never launch
                # a duplicate while the recorded child run is still running.
                previous_pointer = entry.get('pointer')
                if previous_pointer and Path(previous_pointer).is_file():
                    old_run = Path(read(previous_pointer)['run_dir'])
                    if read(old_run/'run.json')['status'] == 'running':
                        raise ValueError(f'{item}: previous child still reports running: {old_run}')
                attempt = entry.get('attempt', 0)+1
                result = plan_path.parent/f'{item}.attempt{attempt}.result.json'
                log = plan_path.parent/f'{item}.attempt{attempt}.log'
                pointer = plan_path.parent/f'{item}.attempt{attempt}.pointer.json'
                for stale in ['csv_sha256', 'cache_sha256', 'run_dir', 'metrics']:
                    entry.pop(stale, None)
                entry.update(status='waiting', attempt=attempt, pointer=str(pointer), log=str(log), result=str(result))
                save(plan_path, plan)
                command = [sys.executable, str(Path(__file__).resolve()), '_worker',
                           '--config', entry['config'], '--result', str(result)]
                with reserve_gpu(policy) if policy else nullcontext(None) as gpu:
                    env = worker_environment(gpu) if gpu else dict(os.environ)
                    env['UAV_COMPETITION_RUN_POINTER'] = str(pointer)
                    entry['status'] = 'running'; save(plan_path, plan)
                    code = run_command(command, env, log)
                if code != 0 or not result.is_file():
                    entry['status'] = 'failed'; save(plan_path, plan)
                    raise ValueError(f'{item}: worker failed; inspect {log}; rerun the same run command to retry')
                completed = read(result)
                entry.update(run_dir=completed['run_dir'], metrics=completed['metrics'])
                verify_child(entry)
                entry['status'] = 'complete'; save(plan_path, plan)
            plan['status'] = 'complete'; save(plan_path, plan)
        except BaseException as exc:
            plan['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            plan['error'] = f'{type(exc).__name__}: {exc}'; save(plan_path, plan)
            raise
    return 0


def merge_rows(original, fresh, selected):
    from prepare_submission import identify
    expected = set(selected)
    if {identify(r['path'])[0] for r in fresh} != expected:
        raise ValueError('Fresh rows do not match the selected complete-file set')
    if any(r['status'] == 'novel_candidate' and identify(r['path'])[0] not in expected for r in original):
        raise ValueError('Old novel files remain outside the refresh set; incompatible cache merge refused')
    grouped = defaultdict(list)
    for row in fresh:
        grouped[identify(row['path'])[0]].append(row)
    combined, replaced = [], set()
    for row in original:
        sample = identify(row['path'])[0]
        if sample not in expected:
            combined.append(row)
        elif sample not in replaced:
            combined.extend(grouped[sample]); replaced.add(sample)
    if replaced != expected:
        raise ValueError('Selected samples absent from original predictions')
    return combined


def export(args):
    import numpy as np
    from util.open_set import FEATURE_VERSION
    plan_path = args.plan.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError('Output must be a new directory')
    with plan_lock(plan_path):
        plan = read(plan_path)
        if plan['status'] != 'complete':
            raise ValueError('Finish all supplemental inference items before export')
        check_frozen_inputs(plan)
        profile_meta, profile_arrays = load_npz(plan['profile'], 'open_set_profile')
        merged_sources = {}
        for item in ITEMS:
            entry = plan['items'][item]
            source = plan['sources'][item]
            columns, original = checked_rows(source['predictions'], item, source['input_count'], allow_errors=True)
            if entry['status'] == 'complete':
                verify_child(entry)
                directory = Path(entry['run_dir'])
                fresh_columns, fresh = read_csv(directory/'predictions.csv')
                if columns != fresh_columns:
                    raise ValueError('Historical/current canonical CSV schemas differ; explicit migration required')
                metadata, arrays = load_npz(directory/'unknown_features.npz', 'unknown_candidates')
                if metadata['signature'] != profile_meta['signature']:
                    raise ValueError('Fresh features do not match the cover profile')
                merged = merge_rows(original, fresh, [x['sample'] for x in entry['inputs']])
            else:
                if any(r['status'] == 'novel_candidate' for r in original):
                    raise ValueError('Cannot keep old novel features in an empty supplemental item')
                merged = original
                metadata = dict(kind='unknown_candidates', schema_version=1, feature_version=FEATURE_VERSION,
                                signature=profile_meta['signature'], records=[])
                arrays = dict(features=np.empty((0, profile_arrays['reference'].shape[1]), dtype=np.float32))
            target = output/'merged'/item
            write_csv(target/'predictions.csv', columns, merged)
            save_npz(target/'unknown_features.npz', metadata, **arrays)
            checked_rows(target/'predictions.csv', item, source['input_count'])
            merged_sources[item] = dict(predictions=str(target/'predictions.csv'), features=str(target/'unknown_features.npz'))
        save(output/'merge_provenance.json', dict(plan=str(plan_path), sources=plan['sources'],
             supplemental_runs={i: e.get('run_dir') for i, e in plan['items'].items()},
             scope=plan['scope'], feature_policy='Only fresh cover-profile candidates; no old feature concatenation',
             prediction_policy='Replace entire selected file; preserve unselected old decisions'))
        predictions = recluster(merged_sources, output/'reclustered')
        code, summary = export_results(predictions, output, extra_no_signal=args.no_signal_extra_class,
                                       data_root=plan['data_root'], check_only=args.check_only)
        plan['last_export'] = summary; save(plan_path, plan)
        return code


def worker(args):
    import signal
    def interrupt(signum, frame):
        raise KeyboardInterrupt('Supplemental worker terminated')
    signal.signal(signal.SIGTERM, interrupt)
    from util.cli import main as cli
    result = cli(['infer', '--config', str(args.config)])
    metrics = result['metrics']
    # Discovery may fail in the subset; final discovery is performed on the merged
    # pool. File/slice extraction errors and task-two burst conflicts are fatal.
    import yaml
    single = yaml.safe_load(args.config.read_text())['infer']['unknown_input_mode'] == 'single'
    fatal = ['error_files', 'files_with_slice_errors'] + ([] if single else ['burst_label_conflicts'])
    if any(metrics.get(key, 0) for key in fatal):
        raise ValueError('Supplemental extraction errors: '+str(metrics))
    save(args.result, dict(run_dir=result['run_dir'], metrics=metrics))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--profile-refresh', type=Path, required=True)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scope', choices=['pending', 'all'], default='pending')
    p.set_defaults(function=prepare)
    p = sub.add_parser('run')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--wait-for-gpu', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--gpu-id', default='0')
    p.add_argument('--gpu-min-free-mib', type=int, default=16384)
    p.set_defaults(function=run)
    p = sub.add_parser('export')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--no-signal-extra-class', action='store_true', help='Explicit nonstandard fallback, same as first script')
    p.add_argument('--check-only', action='store_true')
    p.set_defaults(function=export)
    p = sub.add_parser('_worker', help=argparse.SUPPRESS)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p.set_defaults(function=worker)
    args = parser.parse_args(argv)
    return args.function(args)


if __name__ == '__main__':
    raise SystemExit(main())
