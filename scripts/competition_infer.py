#!/usr/bin/env python3
"""Mainline A: verified 20-class training -> six bounded runs -> strict XLSX.

The manifest records exact child run directories; directory timestamps are never
guessed. Only explicit inference/submit commands execute work, never training.
"""
import argparse
import codecs
from contextlib import nullcontext
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.experiment_io import read_json as read, write_json as save

ITEMS = ('1-1', '1-2', '1-3', '1-4', '2-1', '2-2')
NOMINAL_IQ_BYTES = 40_000_000
# Converted captures may end at the last acquisition block (observed: 99.61472 ms).
# This is an input-compatibility allowance, not an official competition tolerance.
MIN_IQ_BYTES = 39_600_000  # 99 ms; keep the official 100-ms upper bound.
PIPELINE_SOURCES = ('util/bounded_density.py','util/unknown_discovery.py','util/unknown_density.py','util/spectral_discovery.py',
    'util/burst_normalization.py','util/competition_protocol.py','util/fast_infer.py',
    'scripts/evaluate_single_signal.py','scripts/recluster_competition_batch.py',
    'scripts/competition_infer.py','scripts/submission_pipeline.py','scripts/export_submission.py',
    'prepare_submission.py','util/run_context.py','util/experiment_io.py')


def source_fingerprint():
    from util.open_set import file_sha256
    return {name:file_sha256(ROOT/name) for name in PIPELINE_SOURCES}






def training_artifacts(directory, yolo=None, profile_override=None):
    from importlib.metadata import version, PackageNotFoundError
    from util.open_set import load_npz, file_sha256
    from prepare_submission import KNOWN, normalized
    directory = Path(directory).expanduser().resolve()
    state = read(directory/'experiment.json')
    if state.get('status') != 'complete' or state.get('protocol') != 'all_known':
        raise ValueError('Use a completed all_known training run, not a 16-class development model')
    profile = Path(profile_override).expanduser().resolve() if profile_override else directory/'open_set/known_profile.npz'
    meta, _ = load_npz(profile, 'open_set_profile')
    if {normalized(n) for n in meta['classes']} != {normalized(n) for n in KNOWN}:
        raise ValueError('Competition model must contain all official 20 known classes')
    if meta.get('score_method') != 'maha_pp_shrink' or meta.get('calibration_unit') != 'primary':
        raise ValueError('Mainline A requires the original Maha++ primary-calibrated profile')
    effective = directory/'effective_profile.json'
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


def inspect_inputs(root, items, *, diagnostics=None):
    from prepare_submission import identify
    root = Path(root).expanduser().resolve()
    import hashlib
    counts, signatures = {}, {}
    for item in items:
        folder = root/item
        if not folder.is_dir():
            raise ValueError(f'Missing input directory: {folder}')
        paths = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() == '.dat')
        ids, non_nominal, sizes = [], [], []
        digest = hashlib.sha256()
        for path in paths:
            sample, dataset = identify(path.name)
            stat = path.stat()
            if dataset != item:
                raise ValueError(f'Expected a {item} file ID, got {sample}: {path}')
            if stat.st_size % 4:
                raise ValueError(f'DAT must contain whole int16 I/Q pairs (4 bytes/pair); '
                                 f'got {stat.st_size:,} bytes: {path}')
            if not MIN_IQ_BYTES <= stat.st_size <= NOMINAL_IQ_BYTES:
                raise ValueError(f'Expected a nominal 100-ms, 100-MSps int16 IQ file '
                    f'({MIN_IQ_BYTES:,}..{NOMINAL_IQ_BYTES:,} bytes; 99..100 ms); '
                    f'got {stat.st_size:,} bytes ({stat.st_size/400_000:.8g} ms): {path}')
            sizes.append(stat.st_size)
            if stat.st_size != NOMINAL_IQ_BYTES:
                non_nominal.append(dict(path=str(path), size_bytes=stat.st_size,
                                        duration_ms=stat.st_size/400_000))
            ids.append(int(sample.split('_')[1]))
            digest.update(f'{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
        if not ids or sorted(ids) != list(range(len(ids))):
            raise ValueError(f'Empty, duplicate, missing or noncontiguous input IDs in {folder}')
        if item.startswith('2') and len(ids) != 1750:
            raise ValueError(f'{item} must contain 1750 DAT files, got {len(ids)}')
        counts[item] = len(ids)
        signatures[item] = digest.hexdigest()
        if diagnostics is not None:
            diagnostics[item] = dict(nominal_duration_ms=100, minimum_accepted_ms=99,
                min_duration_ms=min(sizes)/400_000, max_duration_ms=max(sizes)/400_000,
                size_histogram={str(n): sizes.count(n) for n in sorted(set(sizes))},
                non_nominal_files=non_nominal)
    return root, counts, signatures


def input_sampling_diagnostics(inputs, frozen):
    """Report actual coverage without silently changing a signed profile policy."""
    from util.eval_common import plan_iq_slices
    # The public parser leaves unspecified frontend fields as None; resolve_config
    # supplies this same source-rate default before actual inference.
    sample_rate = 100_000_000 if frozen.iq_sample_rate is None else frozen.iq_sample_rate
    if sample_rate != 100_000_000:
        raise ValueError('Competition IQ input requires iq_sample_rate=100000000')
    result = {}
    for item, details in inputs.items():
        gaps = []
        for size, count in details['size_histogram'].items():
            _, sampling = plan_iq_slices(int(size)//4, sample_rate,
                frozen.iq_stride_ms, frozen.tail_policy, frozen.max_slices)
            if sampling['unobserved_samples']:
                gaps.append(dict(size_bytes=int(size), files=count,
                    unobserved_ms=sampling['unobserved_samples']/100_000))
        result[item] = dict(tail_policy=frozen.tail_policy, uncovered_file_count=sum(g['files'] for g in gaps),
                            coverage_gaps=gaps)
    return result


def run_command(command, env, log, *, pass_fds=()):
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    with log.open('wb') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   pass_fds=pass_fds)
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
    artifacts = training_artifacts(args.training_run,args.yolo_weights,getattr(args,'profile',None))
    input_details = {}
    data, counts, input_signatures = inspect_inputs(args.data_root, args.items, diagnostics=input_details)
    config = (args.config or Path(artifacts['training_run'])/'inference.yaml').resolve()
    # Parse the real schema now, before any long-running model work.
    settings, _ = load_settings(config, 'infer')
    parser = build_task_parser('infer')
    frozen = parser.parse_args([*settings_tokens(parser, settings), '--checkpoint_path', artifacts['checkpoint'],
        '--open_set_profile', artifacts['profile'], '--yolo_weights', artifacts['yolo']])
    method=getattr(args,'cluster_method',None) or ('spectral' if frozen.unknown_bounded_strategy=='spectral' else 'density_merge')
    spectral=dict(neighbors=frozen.spectral_neighbors if getattr(args,'spectral_neighbors',None) is None else args.spectral_neighbors,
                  seed=frozen.spectral_seed if getattr(args,'spectral_seed',None) is None else args.spectral_seed,
                  n_init=frozen.spectral_n_init)
    if method=='spectral':
        from util.spectral_discovery import validate_spectral_options
        validate_spectral_options(**spectral)
    gpu_policy=None
    if getattr(args,'wait_for_gpu',False):
        from util.gpu_wait import GpuWaitPolicy
        gpu_policy=GpuWaitPolicy(args.gpu_id,args.gpu_min_free_mib)
    sampling_details = input_sampling_diagnostics(input_details, frozen)
    for item, details in input_details.items():
        if details['non_nominal_files']:
            print(f"WARNING: {item}: {len(details['non_nominal_files'])} file(s) have "
                f"99..100-ms aligned IQ captures; actual lengths are retained.", file=sys.stderr)
        for gap in sampling_details[item]['coverage_gaps']:
            print(f"WARNING: {item}: tail_policy={frozen.tail_policy}, max_slices={frozen.max_slices} "
                f"leaves {gap['unobserved_ms']:.8g} ms unobserved in {gap['files']} file(s) "
                f"of {gap['size_bytes']:,} bytes. Full tail coverage requires a matching "
                "cover profile; do not change signed settings alone.", file=sys.stderr)
    fingerprint = {**artifacts, 'data_root': str(data), 'config_sha256': file_sha256(config),
                   'strategy':method,'spectral_options':spectral if method=='spectral' else None,
                   'known_recheck': False, 'full': args.full,
                   'pipeline_sources': source_fingerprint()}
    if args.dry_run:
        print(json.dumps(dict(preflight='passed', **fingerprint, input_counts=counts,
            input_duration_diagnostics=input_details, input_sampling=sampling_details), ensure_ascii=False, indent=2))
        return 0
    if args.resume:
        manifest = args.resume.expanduser().resolve();batch = read(manifest)
        if batch.get('schema_version')!=3:
            raise ValueError('Start a fresh v3 batch')
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
        batch = dict(schema_version=3, fingerprint=fingerprint, config=str(config), runs={}, status='running',
            discovery=dict(min_cluster_size=frozen.unknown_min_cluster_size,
                min_recordings=frozen.unknown_min_recordings, min_samples=frozen.unknown_min_samples,
                min_silhouette=frozen.unknown_min_silhouette, allow_single_cluster=frozen.unknown_allow_single_cluster,
                strategy=method,spectral_options=spectral))
    pointer = getattr(args, 'manifest_pointer', None)
    if pointer:
        pointer = Path(pointer).expanduser().resolve()
        if pointer.exists():
            if Path(read(pointer)['manifest']).resolve() != manifest.resolve():
                raise ValueError('Batch pointer already refers to a different manifest')
        else:
            with pointer.open('x', encoding='utf-8') as stream:
                json.dump({'manifest': str(manifest.resolve())}, stream)
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
                '--unknown_bounded_strategy',method,'--unknown_cluster_backend','spectral' if method=='spectral' else 'hdbscan',
                '--unknown_cluster_policy', 'auto', '--no-known_recheck', '--save_discovery_cache',
                '--no-fast' if args.full else '--fast']
            if method=='spectral':
                command+=['--spectral_neighbors',str(spectral['neighbors']),'--spectral_seed',str(spectral['seed']),
                          '--spectral_n_init',str(spectral['n_init'])]
            entry = dict(status='waiting' if gpu_policy else 'running', attempt=attempt, input_count=counts[item], log=str(log),
                         input_signature=input_signatures[item], input_duration=input_details[item],
                         input_sampling=sampling_details[item])
            batch['runs'][item] = entry;save(manifest, batch)
            try:
                from util.gpu_wait import reserve_gpu,worker_environment
                with reserve_gpu(gpu_policy) if gpu_policy else nullcontext(None) as gpu:
                    env=worker_environment(gpu) if gpu else dict(os.environ)
                    env['UAV_COMPETITION_RUN_POINTER']=str(pointer)
                    entry['status']='running'
                    if gpu:entry['gpu']=gpu.as_dict()
                    save(manifest,batch)
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
            except (OSError, ValueError, KeyError, csv.Error, RuntimeError) as exc:
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






def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('training_run',type=Path);p.add_argument('data_root',type=Path)
    p.add_argument('--items',nargs='+',choices=ITEMS,default=list(ITEMS))
    p.add_argument('--config',type=Path);p.add_argument('--yolo-weights',type=Path)
    p.add_argument('--profile',type=Path)
    p.add_argument('--cluster-method',choices=['density_merge','spectral'])
    p.add_argument('--spectral-neighbors',type=int);p.add_argument('--spectral-seed',type=int)
    p.add_argument('--wait-for-gpu',action='store_true');p.add_argument('--gpu-id',default='0')
    p.add_argument('--gpu-min-free-mib',type=int,default=8192)
    p.add_argument('--full',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',type=Path);p.add_argument('--manifest-pointer',type=Path)
    args = p.parse_args(argv)
    if len(set(args.items))!=len(args.items):p.error('Duplicate test items')
    return infer(args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr);raise SystemExit(2)
