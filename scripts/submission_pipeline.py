"""Validated six-item result collection, joint clustering and submission export."""
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

from util.experiment_io import read_json as read, write_json as save

ITEMS = ('1-1', '1-2', '1-3', '1-4', '2-1', '2-2')
csv.field_size_limit(16 * 1024 * 1024)






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


def checked_rows(path, item, expected=None):
    columns, rows = read_csv(path)
    if not rows or not {'path', 'label', 'status', 'error'} <= set(columns):
        raise ValueError(f'{item}: missing canonical CSV columns or empty CSV')
    seen = set()
    for row in rows:
        sample, dataset = identify(row['path'])
        if dataset != item or (row.get('error', '').strip() or row['status'] == 'error'):
            raise ValueError(f'{item}: upstream error or wrong item: {row}')
        key = (sample, row.get('track_id', ''))
        if key in seen:
            raise ValueError(f'{item}: duplicate file/track identity {key}')
        seen.add(key)
    samples = {sample for sample, _ in seen}
    if expected is not None and samples != {f'{item}_{i:04d}' for i in range(expected)}:
        raise ValueError(f'{item}: missing, extra or noncontiguous sample IDs')
    return columns, rows


def sources_from_batch(batch_path):
    batch_path = Path(batch_path).expanduser().resolve()
    batch = read(batch_path)
    sources = {}
    signatures = []
    for item in ITEMS:
        entry = batch['runs'][item]
        if entry['status'] in {'running', 'waiting', 'interrupted'}:
            raise ValueError(f'{item}: source is still running or interrupted')
        directory = Path(entry['run_dir'])
        csv_path = directory/'predictions.csv'
        cache = directory/'unknown_features.npz'
        for name, path in [('csv_sha256', csv_path), ('cache_sha256', cache)]:
            if entry.get(name) and file_sha256(path) != entry[name]:
                raise ValueError(f'{item}: source hash changed: {path}')
        run = read(directory/'run.json')
        if run['status'] not in {'complete', 'completed_with_errors'}:
            raise ValueError(f'{item}: source did not finish')
        checked_rows(csv_path, item, entry['input_count'])
        metadata, _ = load_npz(cache, 'unknown_candidates')
        signatures.append(metadata['signature'])
        sources[item] = dict(run_dir=str(directory.resolve()), predictions=str(csv_path.resolve()),
                             features=str(cache.resolve()), input_count=entry['input_count'],
                             csv_sha256=file_sha256(csv_path), cache_sha256=file_sha256(cache))
    if any(s != signatures[0] for s in signatures):
        raise ValueError('Source caches have different feature signatures')
    return batch, sources




def recluster(sources, output, strategy='spectral', spectral=None, discovery=None):
    """All novel labels are regenerated, including formerly assigned unknowns."""
    from scripts.cluster_unknown_features import build_parser, run
    output = Path(output)
    spectral = spectral or dict(neighbors=10, seed=42, n_init=10)
    discovery = discovery or dict(min_cluster_size=5, min_recordings=2, min_samples=3, min_silhouette=.15)
    command = [sys.executable, str(ROOT/'scripts/recluster_competition_batch.py'),
               '--output', str(output/'task1'), '--bounded-strategy', strategy,
               '--min-cluster-size', str(discovery['min_cluster_size']),
               '--min-recordings', str(discovery['min_recordings']),
               '--min-samples', str(discovery['min_samples']),
               '--min-silhouette', str(discovery['min_silhouette'])]
    if strategy=='spectral':
        command += ['--spectral-neighbors',str(spectral['neighbors']),'--spectral-seed',str(spectral['seed']),
                    '--spectral-n-init',str(spectral['n_init'])]
    for item in ITEMS[:4]:
        command += ['--item', item, sources[item]['features'], sources[item]['predictions']]
    output.mkdir(parents=True, exist_ok=False)
    with (output/'task1.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    predictions = {k: Path(v) for k, v in read(output/'task1/joint_selection.json')['predictions'].items()}
    for item in ITEMS[4:]:
        metadata, _ = load_npz(sources[item]['features'], 'unknown_candidates')
        if not metadata['records']:
            columns, rows = checked_rows(sources[item]['predictions'], item, sources[item]['input_count'])
            if any(r['status']=='novel_candidate' for r in rows):
                raise ValueError(f'{item}: novel rows have no candidate features')
            path = output/item/'predictions.csv'
            write_csv(path,columns,rows)
            predictions[item] = path
            continue
        args = build_parser().parse_args([
            '--features', sources[item]['features'], '--predictions_csv', sources[item]['predictions'],
            '--competition_profile', 'task2', '--test_item', item, '--unknown_input_mode', 'multi',
            '--unknown_cluster_backend', 'spectral' if strategy=='spectral' else 'hdbscan',
            '--unknown_bounded_strategy', strategy,
            '--spectral_neighbors', str(spectral['neighbors']), '--spectral_seed', str(spectral['seed']),
            '--spectral_n_init', str(spectral['n_init']),
            '--unknown_min_cluster_size', str(discovery['min_cluster_size']),
            '--unknown_min_recordings', str(discovery['min_recordings']),
            '--unknown_min_samples', str(discovery['min_samples']),
            '--unknown_min_silhouette', str(discovery['min_silhouette']), '--no-save_feature_plots'])
        args.output_dir = output/item
        discovery = run(args)
        if not discovery['constraint_satisfied']:
            raise ValueError(f'{item}: no supported partition; inspect {output/item}')
        predictions[item] = output/item/'predictions.csv'
    return predictions




def export_results(predictions, output, *, data_root=None, check_only=False,
                   task1_no_signal_code=None):
    from prepare_submission import main as export
    output = Path(output)
    policy = {}
    nonstandard = task1_no_signal_code is not None
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
    directory = Path(directory).expanduser().resolve()
    manifest = read(directory/'final_predictions.json')
    if manifest.get('schema_version')!=3 or set(manifest['predictions'])!=set(ITEMS):
        raise ValueError('Use a complete v3 final_predictions.json manifest')
    predictions = {item:Path(path) for item,path in manifest['predictions'].items()}
    for item,path in predictions.items():
        if file_sha256(path)!=manifest['sha256'][item]:
            raise ValueError(f'{item}: final prediction changed after selection')
        checked_rows(path,item)
    return predictions
