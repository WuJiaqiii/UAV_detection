#!/usr/bin/env python3
"""Mainline A: verified 20-class training -> six bounded runs -> strict XLSX.

The manifest records exact child run directories; directory timestamps are never
guessed. Only explicit inference/submit commands execute work, never training.
"""
import argparse
import codecs
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ITEMS = ('1-1', '1-2', '1-3', '1-4', '2-1', '2-2')
PIPELINE_SOURCES = ('util/bounded_density.py','util/unknown_discovery.py','util/unknown_density.py',
    'util/burst_normalization.py','util/competition_protocol.py','util/fast_infer.py',
    'scripts/evaluate_single_signal.py','scripts/recluster_competition_batch.py',
    'scripts/competition_pipeline.py','prepare_submission.py','util/run_context.py')


def source_fingerprint():
    from util.open_set import file_sha256
    return {name:file_sha256(ROOT/name) for name in PIPELINE_SOURCES}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def training_artifacts(directory, yolo=None):
    from importlib.metadata import version, PackageNotFoundError
    from util.open_set import load_npz, file_sha256
    from prepare_submission import KNOWN, normalized
    directory = Path(directory).expanduser().resolve()
    state = read(directory/'experiment.json')
    if state.get('status') != 'complete' or state.get('protocol') != 'all_known':
        raise ValueError('Use a completed all_known training run, not a 16-class development model')
    profile = directory/'open_set/known_profile.npz'
    meta, _ = load_npz(profile, 'open_set_profile')
    if {normalized(n) for n in meta['classes']} != {normalized(n) for n in KNOWN}:
        raise ValueError('Competition model must contain all official 20 known classes')
    if meta.get('score_method') != 'maha_pp_shrink' or meta.get('calibration_unit') != 'primary':
        raise ValueError('Mainline A requires the original Maha++ primary-calibrated profile')
    effective = directory/'effective_profile.json'
    if not effective.is_file():
        effective = directory/'effective_discover.json'
    settings = read(effective)
    detector = Path(yolo or settings['yolo_weights']).expanduser().resolve()
    checkpoint = directory/'training/models/best.pth'
    details = meta['signature']['details']
    for path, key in ((checkpoint, 'checkpoint_sha256'), (detector, 'detector_sha256')):
        if not path.is_file() or file_sha256(path) != details[key]:
            raise ValueError(f'Missing or profile-mismatched weights: {path}')
    hashes = details['source_sha256']
    if not {'util/open_set.py', 'util/event_detection.py'} <= set(hashes):
        raise ValueError('Profile has an incomplete source signature')
    for name, digest in hashes.items():
        if not (ROOT/name).is_file() or file_sha256(ROOT/name) != digest:
            raise ValueError(f'Profile source mismatch: {name}; rebuild training/profile with this version')
    yolo_sources = details.get('yolov5_source_sha256', {})
    for name, digest in yolo_sources.items():
        if not (ROOT/name).is_file() or file_sha256(ROOT/name) != digest:
            raise ValueError(f'YOLO source mismatch: {name}')
    for name, expected in details.get('libraries', {}).items():
        try:
            installed = version(name)
        except PackageNotFoundError as exc:
            raise ValueError(f'Missing profile dependency: {name}') from exc
        if installed != expected:
            raise ValueError(f'Profile library mismatch for {name}: expected {expected}')
    return dict(training_run=str(directory), checkpoint=str(checkpoint), profile=str(profile), yolo=str(detector),
        checkpoint_sha256=file_sha256(checkpoint), profile_sha256=file_sha256(profile),
        detector_sha256=file_sha256(detector))


def inspect_inputs(root, items):
    from prepare_submission import identify
    root = Path(root).expanduser().resolve()
    import hashlib
    counts, signatures = {}, {}
    for item in items:
        folder = root/item
        if not folder.is_dir():
            raise ValueError(f'Missing input directory: {folder}')
        paths = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() == '.dat')
        ids = []
        digest = hashlib.sha256()
        for path in paths:
            sample, dataset = identify(path.name)
            if dataset != item or path.stat().st_size != 40_000_000:
                raise ValueError(f'Expected {item} 100-ms, 100-MSps int16 IQ file: {path}')
            ids.append(int(sample.split('_')[1]))
            stat = path.stat()
            digest.update(f'{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
        if not ids or sorted(ids) != list(range(len(ids))):
            raise ValueError(f'Empty, duplicate, missing or noncontiguous input IDs in {folder}')
        if item.startswith('2') and len(ids) != 1750:
            raise ValueError(f'{item} must contain 1750 DAT files, got {len(ids)}')
        counts[item] = len(ids)
        signatures[item] = digest.hexdigest()
    return root, counts, signatures


def run_command(command, env, log):
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    with log.open('wb') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            while True:
                chunk = process.stdout.read1(4096)
                if not chunk:
                    break
                stream.write(chunk);stream.flush()
                sys.stdout.write(decoder.decode(chunk));sys.stdout.flush()
            sys.stdout.write(decoder.decode(b'', final=True))
            return process.wait()
        except BaseException:
            process.terminate();process.wait()
            raise
        finally:
            process.stdout.close()


def review_csv(path, item, expected):
    from prepare_submission import identify, UNKNOWN, label_mapping, normalized
    csv.field_size_limit(16*1024*1024)
    mapping = label_mapping()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    samples = set()
    unresolved = 0
    for row in rows:
        sample, dataset = identify(row['path'])
        if dataset != item:
            raise ValueError('Prediction test item differs from batch manifest')
        samples.add(sample)
        novel = UNKNOWN.fullmatch(row['label'])
        valid = (row['status'] == 'known' and normalized(row['label']) in mapping) or (
            row['status'] == 'novel_candidate' and novel is not None and int(novel[1]) > 0)
        valid = valid or (item.startswith('2') and row['status'] == 'no_signal' and row['label'] == 'no_signal')
        unresolved += not valid or bool(row.get('error', '').strip())
    if (samples != {f'{item}_{i:04}' for i in range(expected)} or
            item.startswith('1') and len(rows) != expected):
        raise ValueError(f'{item}: CSV covers {len(samples)} files, expected {expected}')
    return int(unresolved)


def infer(args, runner=run_command):
    from util.open_set import file_sha256
    from util.cli import build_task_parser, load_settings, settings_tokens
    from util.competition_protocol import protocol_for_item
    from util.run_context import create_directory
    artifacts = training_artifacts(args.training_run, args.yolo_weights)
    data, counts, input_signatures = inspect_inputs(args.data_root, args.items)
    config = (args.config or Path(artifacts['training_run'])/'inference.yaml').resolve()
    # Parse the real schema now, before any long-running model work.
    settings, _ = load_settings(config, 'infer')
    parser = build_task_parser('infer')
    frozen = parser.parse_args([*settings_tokens(parser, settings), '--checkpoint_path', artifacts['checkpoint'],
        '--open_set_profile', artifacts['profile'], '--yolo_weights', artifacts['yolo']])
    fingerprint = {**artifacts, 'data_root': str(data), 'config_sha256': file_sha256(config),
                   'strategy': 'density_merge', 'known_recheck': False, 'full': args.full,
                   'pipeline_sources': source_fingerprint()}
    if args.dry_run:
        print(json.dumps(dict(preflight='passed', **fingerprint, input_counts=counts), ensure_ascii=False, indent=2))
        return 0
    if args.resume:
        manifest = args.resume.expanduser().resolve();batch = read(manifest)
        if batch['fingerprint'] != fingerprint:
            raise ValueError('Resume inputs/weights/config changed; start a new batch instead')
        if manifest.parent.parent != (ROOT/'experiment').resolve():
            raise ValueError('Resume manifest must be inside this repository experiment directory')
        for item in args.items:
            previous = batch['runs'].get(item, {})
            if previous.get('status') == 'complete':
                if previous.get('input_signature') != input_signatures[item]:
                    raise ValueError(f'{item}: source DAT names/sizes/timestamps changed; start a new batch')
                folder = Path(previous['run_dir'])
                if (file_sha256(folder/'predictions.csv') != previous['csv_sha256'] or
                        file_sha256(folder/'unknown_features.npz') != previous['cache_sha256']):
                    raise ValueError(f'{item}: cached predictions/features changed after inference')
    else:
        manifest = create_directory('competition_batch')/'batch_manifest.json'
        batch = dict(schema_version=1, fingerprint=fingerprint, config=str(config), runs={}, status='running',
            discovery=dict(min_cluster_size=frozen.unknown_min_cluster_size,
                min_recordings=frozen.unknown_min_recordings, min_samples=frozen.unknown_min_samples,
                min_silhouette=frozen.unknown_min_silhouette, allow_single_cluster=frozen.unknown_allow_single_cluster))
    # OS releases this lock even if the user interrupts a batch.
    import fcntl
    with manifest.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another operation is using this batch manifest') from exc
        batch['status'] = 'running';save(manifest, batch)
        print(f'Batch manifest: {manifest}', flush=True)
        for item in args.items:
            previous = batch['runs'].get(item, {})
            if previous.get('status') == 'complete':
                print(f'{item}: already complete; skipped', flush=True);continue
            attempt = previous.get('attempt', 0)+1
            pointer = manifest.parent/f'{item}.attempt_{attempt}.pointer.json'
            log = manifest.parent/f'{item}.attempt_{attempt}.log'
            protocol = protocol_for_item(item)
            command = [sys.executable, str(ROOT/'main.py'), 'infer', '--config', str(config),
                '--checkpoint_path', artifacts['checkpoint'], '--open_set_profile', artifacts['profile'],
                '--yolo_weights', artifacts['yolo'], '--inputs', str(data/item), '--test_item', item,
                '--competition_profile', protocol.name, '--unknown_input_mode', protocol.input_mode,
                '--unknown_bounded_strategy', 'density_merge', '--unknown_cluster_backend', 'hdbscan',
                '--unknown_cluster_policy', 'auto', '--no-known_recheck', '--save_discovery_cache',
                '--no-fast' if args.full else '--fast']
            entry = dict(status='running', attempt=attempt, input_count=counts[item], log=str(log),
                         input_signature=input_signatures[item])
            batch['runs'][item] = entry;save(manifest, batch)
            env = dict(os.environ, UAV_COMPETITION_RUN_POINTER=str(pointer))
            try:
                code = runner(command, env, log)
                entry['return_code'] = code
                if not pointer.is_file():
                    raise ValueError('Child did not allocate a run; inspect its batch log')
                run_dir = Path(read(pointer)['run_dir']).resolve()
                if run_dir.parent != (ROOT/'experiment').resolve():
                    raise ValueError('Child run directory is outside the repository experiment root')
                entry['run_dir'] = str(run_dir)
                status = read(run_dir/'run.json')['status']
                entry['status'] = 'failed'
                if code == 0 and status == 'complete':
                    unresolved = review_csv(run_dir/'predictions.csv', item, counts[item])
                    entry.update(unresolved_rows=unresolved, status='needs_review' if unresolved else 'complete',
                        csv_sha256=file_sha256(run_dir/'predictions.csv'),
                        cache_sha256=file_sha256(run_dir/'unknown_features.npz'))
                else:
                    entry['error'] = f'Child exit {code}, run status {status}'
            except (OSError, ValueError, KeyError, csv.Error) as exc:
                entry.update(status='failed', error=str(exc))
            except BaseException:
                entry['status'] = 'interrupted';batch['status'] = 'interrupted';save(manifest, batch);raise
            save(manifest, batch)
            print(f"{item}: {entry['status']} (log: {log})", flush=True)
        batch['missing_items'] = [item for item in ITEMS if item not in batch['runs']]
        batch['status'] = ('complete' if not batch['missing_items'] and
            all(r['status'] == 'complete' for r in batch['runs'].values()) else 'incomplete')
        save(manifest, batch)
    print(f'Batch status: {batch["status"]}; manifest: {manifest}')
    return 0 if batch['status'] == 'complete' else 2


def submit(args):
    from util.run_context import planned_directory
    from util.open_set import file_sha256
    manifest = args.manifest.expanduser().resolve();batch = read(manifest)
    if any(batch['runs'].get(item, {}).get('status') != 'complete' for item in ITEMS):
        raise ValueError('All six items must be complete with no unresolved CSV rows before submission')
    import fcntl
    with manifest.with_suffix('.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Inference is still using this manifest') from exc
        artifacts = training_artifacts(batch['fingerprint']['training_run'], batch['fingerprint']['yolo'])
        if source_fingerprint() != batch['fingerprint']['pipeline_sources']:
            raise ValueError('Pipeline source changed after inference; review before creating a new batch')
        if artifacts['profile_sha256'] != batch['fingerprint']['profile_sha256']:
            raise ValueError('Training/profile changed after inference')
        data, _, signatures = inspect_inputs(batch['fingerprint']['data_root'], ITEMS)
        for item in ITEMS:
            folder = Path(batch['runs'][item]['run_dir'])
            if read(folder/'run.json')['status'] != 'complete':
                raise ValueError(f'{item} source run is not complete')
            entry = batch['runs'][item]
            if (signatures[item] != entry['input_signature'] or
                    file_sha256(folder/'predictions.csv') != entry['csv_sha256'] or
                    file_sha256(folder/'unknown_features.npz') != entry['cache_sha256']):
                raise ValueError(f'{item}: source data or cached predictions/features changed')
        joint = manifest.parent/(planned_directory('joint_task1').name+'_'+uuid4().hex[:8])
        command = [sys.executable, str(ROOT/'scripts/recluster_competition_batch.py'),
            '--output', str(joint), '--bounded-strategy', 'density_merge']
        command.append('--allow-single-cluster' if batch['discovery'].get('allow_single_cluster',True)
                       else '--no-allow-single-cluster')
        for key, flag in [('min_cluster_size','--min-cluster-size'), ('min_recordings','--min-recordings'),
                          ('min_samples','--min-samples'), ('min_silhouette','--min-silhouette')]:
            command += [flag, str(batch['discovery'][key])]
        for item in ITEMS[:4]:
            folder = Path(batch['runs'][item]['run_dir'])
            command += ['--item', item, str(folder/'unknown_features.npz'), str(folder/'predictions.csv')]
        subprocess.run(command, cwd=ROOT, check=True)
        final = read(joint/'joint_selection.json')['predictions']
        csvs = [final[item] if item in final else str(Path(batch['runs'][item]['run_dir'])/'predictions.csv') for item in ITEMS]
        output = args.output.expanduser().resolve() if args.output else ROOT/'submission'/(
            planned_directory('competition').name+'_'+uuid4().hex[:8])
        command = [sys.executable, str(ROOT/'prepare_submission.py'), '--predictions', *csvs,
            '--input-root', *[str(data/item) for item in ITEMS], '--output', str(output)]
        if args.check:
            command.append('--check')
        subprocess.run(command, cwd=ROOT, check=True)
        batch['last_submission'] = dict(output=str(Path(output).resolve()), joint_selection=str(joint/'joint_selection.json'),
            report=str(Path(output).resolve().with_name(Path(output).name+'_report.json')), check_only=args.check)
        save(manifest, batch)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('infer')
    p.add_argument('training_run', type=Path);p.add_argument('data_root', type=Path)
    p.add_argument('--items', nargs='+', choices=ITEMS, default=list(ITEMS))
    p.add_argument('--config', type=Path);p.add_argument('--yolo-weights', type=Path)
    p.add_argument('--full', action='store_true');p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', type=Path)
    p = sub.add_parser('submit');p.add_argument('manifest', type=Path)
    p.add_argument('--output', type=Path);p.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.action == 'infer' and len(set(args.items)) != len(args.items):
        parser.error('Duplicate test items')
    return infer(args) if args.action == 'infer' else submit(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr);raise SystemExit(2)
