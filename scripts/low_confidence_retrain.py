#!/usr/bin/env python3
"""One controlled run: all-known CNN retraining -> fresh profile -> six tests -> exports.

The detector is not retrained. Its confidence is applied consistently to training,
reference/calibration extraction and deployment. Same work directory resumes stages.
"""
import argparse
from contextlib import contextmanager, nullcontext, redirect_stdout
import copy
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from scripts.legacy_submission_common import (ITEMS, read, save, checked_rows, sources_from_batch,
    recluster, export_results, predictions_from_export)
from scripts.competition_pipeline import training_artifacts, inspect_inputs, run_command
from util.open_set import file_sha256, load_npz
from util.gpu_wait import GpuWaitPolicy, reserve_gpu, worker_environment


def configured(args):
    if not math.isfinite(args.confidence) or not 0 < args.confidence < .05:
        raise ValueError('This lower-confidence experiment requires 0 < confidence < 0.05')
    path = args.config.expanduser().resolve()
    doc = copy.deepcopy(yaml.safe_load(path.read_text(encoding='utf-8')))
    train_root = Path(args.train_root or doc['experiment']['inputs'][0]).expanduser()
    train_root = (path.parent/train_root).resolve() if not train_root.is_absolute() else train_root.resolve()
    detector = Path(args.yolo_weights or doc['train']['yolo_weights']).expanduser()
    detector = (path.parent/detector).resolve() if not detector.is_absolute() else detector.resolve()
    doc['experiment'].update(protocol='all_known', inputs=[str(train_root)], input_type='iq_dat',
                             exclude_classes=[], test_ratio=0., training_variant='primary_ce')
    doc['train'].update(yolo_weights=str(detector), yolo_conf_thres=args.confidence,
                        bbox_cache_mode='off', train_group_policy='primary', supcon_weight=0.)
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError('epochs must be positive')
        doc['train']['epochs'] = args.epochs
    doc.setdefault('evaluation', {}).update(yolo_weights=str(detector), yolo_conf_thres=args.confidence,
        tail_policy='cover', iq_stride_ms=50., max_slices=0, known_recheck=False)
    doc.setdefault('profile', {}).update(profile_recheck=False)
    doc.setdefault('discovery', {}).update(unknown_cluster_backend='spectral', unknown_bounded_strategy='spectral',
        unknown_cluster_policy='auto', spectral_neighbors=10, spectral_seed=42, spectral_n_init=10)
    # Use the actual public parser before allocating a long GPU run.
    from util.cli import main as cli
    with tempfile.TemporaryDirectory() as temp:
        config = Path(temp)/'experiment.yaml'
        config.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding='utf-8')
        with redirect_stdout(io.StringIO()):
            cli(['experiment', '--config', str(config), '--dry_run'])
    return doc, train_root, detector


def training_inventory(root):
    from prepare_submission import KNOWN, normalized
    classes = {p.name for p in root.iterdir() if p.is_dir() and
               any(q.is_file() and q.suffix.lower() == '.dat' for q in p.rglob('*'))}
    if {normalized(c) for c in classes} != {normalized(c) for c in KNOWN}:
        raise ValueError('Training input must contain exactly the official 20 class folders')
    digest = hashlib.sha256()
    files = 0
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() == '.dat':
            stat = p.stat();files += 1
            digest.update(f'{p.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    return dict(files=files, inventory_sha256=digest.hexdigest())


def frozen_inputs(doc, train_root, detector, data_root):
    from importlib.metadata import version
    _, counts, signatures = inspect_inputs(data_root, ITEMS)
    # Freeze all model/data/pipeline Python code, including YOLO; no signed source
    # can silently change between training, profiling and deployment.
    sources = {str(p.relative_to(ROOT)): file_sha256(p)
               for directory in ['data', 'model', 'util', 'scripts', 'yolov5']
               for p in sorted((ROOT/directory).rglob('*.py'))}
    sources['prepare_submission.py'] = file_sha256(ROOT/'prepare_submission.py')
    return dict(config=doc, train_root=str(train_root), data_root=str(data_root.resolve()),
        detector_sha256=file_sha256(detector), training_inventory=training_inventory(train_root),
        input_counts=counts, input_signatures=signatures, sources=sources,
        libraries={name:version(name) for name in ['torch', 'torchvision', 'numpy', 'scikit-learn']})


@contextmanager
def workflow_lock(root):
    with (root/'workflow.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('This workflow or its GPU worker is still running') from exc
        # Inherit this descriptor into workers. An orphan worker retains the lock
        # until it exits, preventing duplicate launches after parent termination.
        yield lock.fileno()


def validated_training(directory, confidence):
    artifacts = training_artifacts(directory)
    meta, _ = load_npz(artifacts['profile'], 'open_set_profile')
    details = meta['signature']['details']
    if not math.isclose(details['config']['yolo_conf_thres'], confidence, rel_tol=0, abs_tol=1e-12):
        raise ValueError('New profile does not use the requested YOLO confidence')
    if details['options']['tail_policy'] != 'cover' or details['options']['max_slices'] != 0:
        raise ValueError('New profile is not calibrated for complete-duration inference')
    return artifacts


def finished_item(batch, item):
    entry = batch.get('runs', {}).get(item, {})
    if not entry.get('run_dir'):
        return False
    directory = Path(entry['run_dir'])
    if not (directory/'run.json').is_file() or read(directory/'run.json')['status'] not in {'complete', 'completed_with_errors'}:
        return False
    diagnostic = directory/'failure_diagnostics.json'
    if item.startswith('2-'):
        if diagnostic.is_file():
            if any(r.get('burst_label_conflicts', 0) for r in read(diagnostic).get('files', [])):
                return False
        elif read(directory/'run.json')['status'] != 'complete':
            # Do not treat an unexplained failed event export as a mere K issue.
            return False
    try:
        checked_rows(directory/'predictions.csv', item, entry['input_count'])
        load_npz(directory/'unknown_features.npz', 'unknown_candidates')
    except (ValueError, OSError, KeyError):
        return False
    for filename, field in [('predictions.csv', 'csv_sha256'), ('unknown_features.npz', 'cache_sha256')]:
        if entry.get(field) and file_sha256(directory/filename) != entry[field]:
            raise ValueError(f'{item}: completed inference output changed')
    # needs_review is a finished extraction with unassigned/no_signal labels.
    # It must be exported/reviewed, not repeatedly forwarded through the GPU.
    return True


def execute_worker(root, phase, state, args, lock_fd, *, batch=None):
    attempt = state.get(phase, {}).get('attempt', 0)+1
    pointer = root/f'{phase}.attempt{attempt}.pointer.json'
    result = root/f'{phase}.attempt{attempt}.result.json'
    log = root/f'{phase}.attempt{attempt}.log'
    state[phase] = dict(attempt=attempt, pointer=str(pointer), result=str(result), log=str(log), status='waiting')
    save(root/'workflow.json', state)
    command = [sys.executable, str(Path(__file__).resolve()), '_worker', '--work-dir', str(root),
               '--phase', phase, '--pointer', str(pointer), '--result', str(result),
               '--lock-fd', str(lock_fd),
               '--gpu-id', args.gpu_id, '--gpu-min-free-mib', str(args.gpu_min_free_mib)]
    command += ['--wait-for-gpu' if args.wait_for_gpu else '--no-wait-for-gpu']
    if batch:
        command += ['--batch', str(batch)]
    policy = GpuWaitPolicy(args.gpu_id, args.gpu_min_free_mib) if phase == 'training' and args.wait_for_gpu else None
    print(f'{phase}: log={log}', flush=True)
    with reserve_gpu(policy) if policy else nullcontext(None) as gpu:
        env = worker_environment(gpu) if gpu else dict(os.environ)
        if phase == 'training':
            env['UAV_EXPERIMENT_RUN_POINTER'] = str(pointer)
        state[phase]['status'] = 'running';save(root/'workflow.json', state)
        code = run_command(command, env, log, pass_fds=(lock_fd,))
    if code != 0 or not result.is_file():
        state[phase]['status'] = 'failed';save(root/'workflow.json', state)
        raise ValueError(f'{phase} failed; inspect {log}. Later stages were not started.')
    state[phase].update(status='complete', **read(result));save(root/'workflow.json', state)


def verify_export(summary):
    report = read(summary['report'])
    for name, digest in report.get('xlsx_sha256', {}).items():
        if file_sha256(Path(summary['output'])/name) != digest:
            raise ValueError('Export workbook changed; use --retry-export to regenerate in a new directory')
    if summary.get('workbooks'):
        with ZipFile(summary['zip']) as archive:
            if archive.testzip() is not None:
                raise ValueError('Export archive is damaged; use --retry-export')


def run_workflow(args):
    doc, train_root, detector = configured(args)
    root = args.work_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    if args.dry_run:
        print(json.dumps(dict(preflight='configuration passed; no model loaded or data scanned',
            work_dir=str(root), train_root=str(train_root), data_root=str(data_root),
            confidence=args.confidence, epochs=doc['train']['epochs'], tail_policy='cover', protocol='all_known'), indent=2))
        return 0
    frozen = frozen_inputs(doc, train_root, detector, data_root)
    if root.exists() and not (root/'workflow.json').is_file():
        raise ValueError('Existing work directory has no workflow manifest; choose a new directory')
    root.mkdir(parents=True, exist_ok=True)
    with workflow_lock(root) as lock_fd:
        target = root/'workflow.json'
        if target.is_file():
            state = read(target)
            if state['frozen'] != frozen:
                raise ValueError('Configuration, code, weights or input inventory changed; use a new work directory')
            if file_sha256(root/'training.yaml') != state['config_sha256']:
                raise ValueError('Generated training config changed')
        else:
            config = root/'training.yaml'
            config.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding='utf-8')
            state = dict(schema_version=1, status='prepared', frozen=frozen, confidence=args.confidence,
                         config_sha256=file_sha256(config))
            save(target, state)
        if args.prepare_only:
            print('Prepared:', root/'training.yaml', '\nWorkflow:', target)
            return 0
        try:
            previous = state.get('training', {})
            directory = previous.get('run_dir')
            if not directory and previous.get('pointer') and Path(previous['pointer']).is_file():
                directory = read(previous['pointer'])['run_dir']
            complete = directory and (Path(directory)/'experiment.json').is_file() and read(Path(directory)/'experiment.json').get('status') == 'complete'
            if not complete:
                if directory and not args.retry_training:
                    raise ValueError(f'Previous training/profile run is incomplete: {directory}. Inspect its state; --retry-training explicitly starts another training run.')
                execute_worker(root, 'training', state, args, lock_fd)
                directory = state['training']['run_dir']
            artifacts = validated_training(directory, args.confidence)
            if previous.get('artifacts') and previous['artifacts'] != artifacts:
                raise ValueError('Completed model/profile artifacts changed')
            state['training'].update(status='complete', run_dir=str(directory), artifacts=artifacts);save(target, state)
            batch_path = state.get('inference', {}).get('manifest')
            if not batch_path:
                pointer = state.get('inference', {}).get('pointer')
                if pointer and Path(pointer).is_file():
                    batch_path = read(pointer)['manifest']
            batch = read(batch_path) if batch_path and Path(batch_path).is_file() else None
            if batch is None or not all(finished_item(batch, i) for i in ITEMS):
                execute_worker(root, 'inference', state, args, lock_fd, batch=batch_path if batch else None)
                batch_path = state['inference']['manifest'];batch = read(batch_path)
            if not all(finished_item(batch, i) for i in ITEMS):
                raise ValueError('Some inference items have upstream errors; inspect batch logs before export')
            state['inference'].update(status='complete', manifest=str(batch_path));save(target, state)
            previous_exports = state.get('exports', {})
            if previous_exports.get('finished') and not args.retry_export:
                predictions_from_export(previous_exports['result_dir'])
                for summary in previous_exports['summaries'].values():
                    verify_export(summary)
            else:
                attempt = previous_exports.get('attempt', 0)+1
                results = root/f'results_attempt{attempt}'
                state['exports'] = dict(attempt=attempt, result_dir=str(results), finished=False);save(target, state)
                _, sources = sources_from_batch(batch_path)
                predictions = recluster(sources, results/'reclustered')
                save(results/'final_predictions.json', dict(predictions={k:str(v) for k,v in predictions.items()},
                     sha256={k:file_sha256(v) for k,v in predictions.items()}, batch=str(batch_path)))
                _, strict = export_results(predictions, results/'strict', data_root=data_root)
                summaries = {'strict': strict}
                if not strict['ready']:
                    _, fallback = export_results(predictions, results/'shared_no_signal', data_root=data_root,
                                                  task1_no_signal_code=24)
                    summaries['shared_no_signal'] = fallback
                    if not fallback['workbooks']:
                        raise ValueError('Format/burst errors still prevent export; inspect reports; no rows were dropped')
                state['exports'].update(finished=True, summaries=summaries);save(target, state)
            strict = state['exports']['summaries']['strict']
            state['status'] = 'complete' if strict['ready'] else 'complete_with_nonstandard_export'
            save(target, state)
            print(json.dumps(dict(status=state['status'], training_run=directory, batch_manifest=batch_path,
                                  exports=state['exports']['summaries']), ensure_ascii=False, indent=2))
            return 0
        except BaseException as exc:
            state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}')
            save(target, state)
            raise


def worker(args):
    def stop(signum, frame):
        raise KeyboardInterrupt('Worker terminated')
    signal.signal(signal.SIGTERM, stop)
    state = read(args.work_dir/'workflow.json')
    if args.phase == 'training':
        from util.cli import main as cli
        result = cli(['experiment', '--config', str(args.work_dir/'training.yaml')])
        if result.get('status') != 'complete':
            raise ValueError('Training/profile did not complete without extraction errors')
        save(args.result, dict(run_dir=result['run_dir']))
    else:
        from scripts.competition_pipeline import infer
        previous = read(args.batch) if args.batch else None
        items = [i for i in ITEMS if previous is None or not finished_item(previous, i)]
        a = SimpleNamespace(training_run=Path(state['training']['run_dir']), data_root=Path(state['frozen']['data_root']),
            items=items, config=None, yolo_weights=None, profile=None, profile_refresh=None,
            cluster_method='spectral', spectral_neighbors=10, spectral_seed=42,
            wait_for_gpu=args.wait_for_gpu, gpu_id=args.gpu_id, gpu_min_free_mib=args.gpu_min_free_mib,
            full=False, dry_run=False, resume=args.batch, manifest_pointer=args.pointer)
        infer(a, runner=lambda command, env, log: run_command(command, env, log, pass_fds=(args.lock_fd,)))
        # A batch may need label review while all forward passes are finished.
        manifest = Path(read(args.pointer)['manifest']);batch = read(manifest)
        if not all(finished_item(batch, i) for i in ITEMS):
            raise ValueError('Inference has failed/missing upstream items')
        save(args.result, dict(manifest=str(manifest)))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '_worker':
        p = argparse.ArgumentParser()
        p.add_argument('--work-dir', type=Path, required=True)
        p.add_argument('--phase', choices=['training', 'inference'], required=True)
        p.add_argument('--pointer', type=Path, required=True)
        p.add_argument('--result', type=Path, required=True)
        p.add_argument('--lock-fd', type=int, required=True)
        p.add_argument('--batch', type=Path)
        p.add_argument('--wait-for-gpu', action=argparse.BooleanOptionalAction, default=True)
        p.add_argument('--gpu-id', default='0');p.add_argument('--gpu-min-free-mib', type=int, default=16384)
        return worker(p.parse_args(argv[1:]))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'config/competition_lowconf_train.yaml')
    p.add_argument('--train-root', type=Path);p.add_argument('--yolo-weights', type=Path)
    p.add_argument('--data-root', type=Path, default=Path('/media/kaneki/05E4235537EC7AC2/dataset_wjq'))
    p.add_argument('--confidence', type=float, default=.02);p.add_argument('--epochs', type=int)
    p.add_argument('--wait-for-gpu', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--gpu-id', default='0');p.add_argument('--gpu-min-free-mib', type=int, default=16384)
    p.add_argument('--dry-run', action='store_true');p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--retry-training', action='store_true');p.add_argument('--retry-export', action='store_true')
    return run_workflow(p.parse_args(argv))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Completed stages are preserved; repeat the same command to resume.', file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, OSError, KeyError) as exc:
        print('ERROR: '+str(exc), file=sys.stderr)
        raise SystemExit(2)
