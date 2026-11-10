"""Isolated legacy-result recovery. Never rewrites source runs or weakens submit.sh."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from prepare_submission import UNKNOWN, identify
from util.open_set import file_sha256, load_npz

ITEMS = ('1-1', '1-2', '1-3', '1-4', '2-1', '2-2')
csv.field_size_limit(16 * 1024 * 1024)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def checked_rows(path, item, expected=None, *, allow_errors=False):
    columns, rows = read_csv(path)
    if not rows or not {'path', 'label', 'status', 'error'} <= set(columns):
        raise ValueError(f'{item}: missing canonical CSV columns or empty CSV')
    seen = set()
    for row in rows:
        sample, dataset = identify(row['path'])
        if dataset != item or (not allow_errors and (row.get('error', '').strip() or row['status'] == 'error')):
            raise ValueError(f'{item}: upstream error or wrong item: {row}')
        key = (sample, row.get('track_id', ''))
        if key in seen:
            raise ValueError(f'{item}: duplicate file/track identity {key}')
        seen.add(key)
    samples = {sample for sample, _ in seen}
    if expected is not None and samples != {f'{item}_{i:04d}' for i in range(expected)}:
        raise ValueError(f'{item}: missing, extra or noncontiguous sample IDs')
    return columns, rows


def sources_from_batch(batch_path, run_root=None, *, allow_errors=False):
    batch_path = Path(batch_path).expanduser().resolve()
    batch = read(batch_path)
    sources = {}
    signatures = []
    for item in ITEMS:
        entry = batch['runs'][item]
        if entry['status'] in {'running', 'waiting', 'interrupted'}:
            raise ValueError(f'{item}: source is still running or interrupted')
        directory = Path(entry['run_dir'])
        if run_root:
            directory = Path(run_root).expanduser().resolve()/directory.name
        csv_path = directory/'predictions.csv'
        cache = directory/'unknown_features.npz'
        for name, path in [('csv_sha256', csv_path), ('cache_sha256', cache)]:
            if entry.get(name) and file_sha256(path) != entry[name]:
                raise ValueError(f'{item}: source hash changed: {path}')
        run = read(directory/'run.json')
        if run['status'] not in {'complete', 'completed_with_errors'}:
            raise ValueError(f'{item}: source did not finish')
        checked_rows(csv_path, item, entry['input_count'], allow_errors=allow_errors)
        metadata, _ = load_npz(cache, 'unknown_candidates')
        signatures.append(metadata['signature'])
        sources[item] = dict(run_dir=str(directory.resolve()), predictions=str(csv_path.resolve()),
                             features=str(cache.resolve()), input_count=entry['input_count'],
                             csv_sha256=file_sha256(csv_path), cache_sha256=file_sha256(cache))
    if any(s != signatures[0] for s in signatures):
        raise ValueError('Source caches have different feature signatures')
    return batch, sources


def verify_sources(sources):
    for item, source in sources.items():
        for key, hash_key in [('predictions', 'csv_sha256'), ('features', 'cache_sha256')]:
            if file_sha256(source[key]) != source[hash_key]:
                raise ValueError(f'{item}: source changed after plan creation')


def recluster(sources, output):
    """All novel labels are regenerated, including formerly assigned unknowns."""
    from scripts.cluster_unknown_features import build_parser, run
    output = Path(output)
    command = [sys.executable, str(ROOT/'scripts/recluster_competition_batch.py'),
               '--output', str(output/'task1'), '--bounded-strategy', 'spectral']
    for item in ITEMS[:4]:
        command += ['--item', item, sources[item]['features'], sources[item]['predictions']]
    output.mkdir(parents=True, exist_ok=False)
    with (output/'task1.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    predictions = {k: Path(v) for k, v in read(output/'task1/joint_selection.json')['predictions'].items()}
    for item in ITEMS[4:]:
        args = build_parser().parse_args([
            '--features', sources[item]['features'], '--predictions_csv', sources[item]['predictions'],
            '--competition_profile', 'task2', '--test_item', item, '--unknown_input_mode', 'multi',
            '--unknown_cluster_backend', 'spectral', '--unknown_bounded_strategy', 'spectral',
            '--spectral_neighbors', '10', '--spectral_seed', '42', '--spectral_n_init', '10',
            '--unknown_min_cluster_size', '5', '--unknown_min_recordings', '2',
            '--unknown_min_silhouette', '.15', '--no-save_feature_plots'])
        args.output_dir = output/item
        discovery = run(args)
        if not discovery['constraint_satisfied']:
            raise ValueError(f'{item}: no supported partition; inspect {output/item}')
        predictions[item] = output/item/'predictions.csv'
    return predictions


def add_no_signal_class(predictions, output):
    output = Path(output)
    changed, mapping = {}, {}
    for item in ITEMS:
        columns, rows = read_csv(predictions[item])
        if item.startswith('1-'):
            labels = {r['label'] for r in rows if UNKNOWN.fullmatch(r['label'])}
            width = max([len(UNKNOWN.fullmatch(label)[1]) for label in labels] or [3])
            number = max([int(UNKNOWN.fullmatch(label)[1]) for label in labels] or [0])+1
            extra = f'Unknown_{number:0{width}d}'
            if labels and extra <= max(labels):
                extra = max(labels)+'9'
            count = 0
            for row in rows:
                if row['status'] == 'no_signal' and row['label'] == 'no_signal' and not row['error'].strip():
                    row.update(label=extra, status='novel_candidate')
                    count += 1
            if count:
                mapping[item] = dict(label=extra, rows=count, meaning='explicit no_signal placeholder; no signal evidence')
        path = output/item/'predictions.csv'
        write_csv(path, columns, rows)
        changed[item] = path
    return changed, mapping


def export_results(predictions, output, *, extra_no_signal=False, data_root=None, check_only=False,
                   task1_no_signal_code=None):
    from prepare_submission import main as export
    output = Path(output)
    policy = {}
    if extra_no_signal and task1_no_signal_code is not None:
        raise ValueError('Choose either a fixed no_signal code or per-dataset extra classes')
    nonstandard = extra_no_signal or task1_no_signal_code is not None
    if extra_no_signal:
        predictions, policy = add_no_signal_class(predictions, output/'no_signal_policy_csv')
    target = output/('submission_no_signal_extra' if nonstandard else 'submission')
    arguments = ['--predictions', *[str(predictions[i]) for i in ITEMS], '--output', str(target)]
    if data_root:
        arguments += ['--input-root', *[str(Path(data_root)/item) for item in ITEMS]]
    if nonstandard:
        arguments += ['--preview']
    if task1_no_signal_code is not None:
        arguments += ['--task1-no-signal-code', str(task1_no_signal_code)]
    if check_only:
        arguments += ['--check']
    output.mkdir(parents=True, exist_ok=True)
    with (output/'export.log').open('w', encoding='utf-8') as log:
        previous = sys.stdout
        try:
            sys.stdout = log
            code = export(arguments)
        finally:
            sys.stdout = previous
    report_path = target.parent/(target.name+'_report.json')
    report = read(report_path)
    if task1_no_signal_code is not None:
        policy = {item: dict(submission_code=task1_no_signal_code, rows=count,
                            meaning='shared no_signal placeholder; no signal evidence')
                  for item, count in report['task1_no_signal_placeholder']['datasets'].items()}
        report.update(ready=False, interpretation='Requested fixed no_signal placeholder export; review official constraints.')
        save(report_path, report)
    if extra_no_signal:
        for item, entry in policy.items():
            canonical = f"Unknown_{int(UNKNOWN.fullmatch(entry['label'])[1]):06d}"
            entry['submission_code'] = report['unknown_labels'][item][canonical]
        report.update(ready=False, preview=True, has_fallbacks=bool(policy) or report['has_fallbacks'],
                      explicit_no_signal_extra_classes=policy,
                      interpretation='Requested nonstandard export; not a compliant official submission.')
        report['warnings'].append('Task-one no_signal is encoded as an artificial extra class by explicit request; task-two empty scenes remain -1.')
        save(report_path, report)
    summary = dict(status='exported_nonstandard' if nonstandard and report['written_workbooks'] else
                          ('ready' if report['ready'] else 'needs_review'),
                   ready=report['ready'], output=str(target), report=str(report_path),
                   workbooks=report['written_workbooks'], no_signal_extra_classes=policy,
                   errors=len(report['errors']), class_budget_errors=len(report['class_budget_errors']),
                   coverage_complete=report['coverage_complete'])
    save(output/'export_summary.json', summary)
    if report['written_workbooks']:
        archive = target.with_suffix('.zip')
        with ZipFile(archive, 'w', ZIP_DEFLATED) as bundle:
            for path in sorted(target.rglob('*.xlsx')):
                bundle.write(path, path.relative_to(target))
            if nonstandard:
                bundle.write(report_path, 'REPORT_请先阅读.json')
        summary['zip'] = str(archive)
        save(output/'export_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return code, summary


def predictions_from_export(directory):
    """Load final joint-selected CSVs, including relocatable local snapshots."""
    directory = Path(directory).expanduser().resolve()
    manifest_path = directory/'final_predictions.json'
    manifest = read(manifest_path) if manifest_path.is_file() else None
    joint = read(directory/'reclustered/task1/joint_selection.json')
    if set(joint['predictions']) != set(ITEMS[:4]):
        raise ValueError('Export source must contain all four joint task-one predictions')
    predictions = {}
    for item in ITEMS:
        if item.startswith('1-'):
            stage = Path(joint['predictions'][item]).parent.name
            if stage not in {'initial', 'joint'}:
                raise ValueError('Unexpected joint prediction layout')
            path = directory/'reclustered/task1'/item/stage/'predictions.csv'
        else:
            path = directory/'reclustered'/item/'predictions.csv'
        if manifest and file_sha256(path) != manifest['sha256'][item]:
            raise ValueError(f'{item}: final prediction changed after selection')
        checked_rows(path, item)
        predictions[item] = path
    return predictions
