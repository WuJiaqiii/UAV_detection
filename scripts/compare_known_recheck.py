#!/usr/bin/env python3
"""Paired baseline/recheck evaluation from one genuine unknown-class holdout.

Replays only baseline-rejected primary features. No YOLO/CNN rerun or test-label
fitting: accepted baseline predictions stay fixed; discovery is rerun for B.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False, default=str)+"\n", encoding="utf-8")


def run_comparison(source, output, config):
    import numpy as np
    from util.open_set import (build_profile, load_npz, save_npz, file_sha256,
                              OpenSetProfile, normalize_features, refresh_open_set_results)
    from util.known_recheck import RecheckConfig
    from util.unknown_discovery import discover_unknown, discovery_kwargs, save_unknown_prototypes
    from util.discovery_context import annotate_discovery_candidates, attach_recording_predictions, candidate_record
    from util.open_set_metrics import evaluate_open_set
    source, output = Path(source).resolve(), Path(output).resolve()
    policy = RecheckConfig(**config).validate()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Comparison output must be new or empty")
    manifest = read(source/'split_manifest.json')
    if manifest.get('protocol') != 'open_set' or not manifest.get('unknown_classes'):
        raise ValueError('Comparison requires a genuine open_set training-class holdout')
    classes = manifest['classes']
    held_out = set(manifest['unknown_classes'])
    if held_out & set(classes) or any(r['class_name'] in held_out and
        r['split'] not in ('test_unknown', 'excluded') for r in manifest['records']):
        raise ValueError('Unknown holdout leaked into training/reference/calibration')
    settings = read(source/'effective_discover.json')
    if settings.get('known_recheck', False) or settings['unknown_input_mode'] != 'single':
        raise ValueError('A must be a baseline single-device experiment with known_recheck disabled')
    baseline = read(source/'test/results.json')
    expected_profile_hash = baseline.get('open_set', {}).get('profile_sha256')
    if expected_profile_hash and expected_profile_hash != file_sha256(source/'open_set/known_profile.npz'):
        raise ValueError('Baseline profile changed after A inference')
    if baseline['effective_config']['classes'] != classes:
        raise ValueError('Baseline head differs from manifest')
    if baseline.get('arguments', {}).get('known_recheck', False):
        raise ValueError('A results already use known recheck')
    archive = source/'test/unknown_features.npz'
    metadata, arrays = load_npz(archive, 'unknown_candidates')
    # Require the same installed implementation for this paired comparison.
    hashes = metadata['signature'].get('details', {}).get('source_sha256', {})
    for name in ('util/open_set.py', 'util/known_recheck.py'):
        if hashes.get(name) != file_sha256(ROOT/name):
            raise ValueError('Run A with this code version before comparing; source signature differs: '+name)
    if len(metadata['records']) != len(arrays['features']):
        raise ValueError('Baseline candidate features/records are misaligned')
    output.mkdir(parents=True, exist_ok=True)
    target = output/'known_profile_recheck.npz'
    profile_meta = build_profile(source/'open_set/reference/features.npz',
        source/'open_set/calibration/features.npz', target,
        k=settings['knn_k'], known_false_reject_rate=settings['known_false_reject_rate'],
        min_calibration_per_class=settings['min_calibration_per_class'],
        max_reference_per_class=settings['max_reference_per_class'],
        ignore_classes=settings.get('ignore_classes', []),
        score_method=settings['profile_score_method'],
        scale_prior_records=settings['profile_scale_prior_records'],
        covariance_shrinkage=settings['profile_covariance_shrinkage'],
        calibration_unit=settings['profile_calibration_unit'], recheck_config=policy)
    profile = OpenSetProfile(target, metadata['signature'], classes)
    # Prove A and B differ only in the added gate, not the original detector.
    original_meta, original_arrays = load_npz(source/'open_set/known_profile.npz', 'open_set_profile')
    if original_meta.get('known_recheck') is not None or original_meta['threshold'] != profile.threshold:
        raise ValueError('A must use the same original profile without a recheck bank')
    _, enhanced_arrays = load_npz(target, 'open_set_profile')
    for key in original_arrays:
        if key not in enhanced_arrays or not np.array_equal(original_arrays[key], enhanced_arrays[key]):
            raise ValueError('Original profile changed in comparison: '+key)
    report = copy.deepcopy(baseline)
    results = report['files']
    tracks = {(r['path'], t['track_id']): t for r in results for t in r['signals']}
    records = metadata['records']
    source_keys = {(r['path'], r['track_id']) for r in records}
    if len(source_keys) != len(records) or source_keys != {
        key for key, t in tracks.items() if t['recognition_status'] == 'novel_candidate'}:
        raise ValueError('Candidate cache does not match baseline novel track identities')
    identities = dict(baseline.get('recording_ids', {}))
    for r in records:
        if r['path'] in identities and identities[r['path']] != r['recording_id']:
            raise ValueError('Conflicting source recording IDs')
        identities[r['path']] = r['recording_id']
    # Annotate from original track votes, not truth or previous cluster labels.
    candidates = [{'path': r['path'], 'recording_id': r['recording_id'],
                   'feature': normalize_features(arrays['features'][i:i+1])[0],
                   'track': tracks[(r['path'], r['track_id'])]} for i, r in enumerate(records)]
    annotate_discovery_candidates(candidates, results, identities)
    decisions = []
    for candidate in candidates:
        t = candidate['track']
        if not t['discovery_is_primary']:
            continue
        decision = profile.rechecker.decide([candidate['feature']], [t.get('known_candidate_class')])[0]
        t['known_recheck'] = decision
        decisions.append(dict(path=candidate['path'], track_id=t['track_id'], **decision))
        if decision['accepted']:
            name = decision['candidate_class']
            t.update(status=t.get('closed_set_status', 'ok'), recognition_status='known', class_name=name,
                class_id=classes[name], display_label=name, unknown_cluster_id=None,
                cluster_status=None, cluster_similarity=None, cluster_membership=None,
                cluster_assignment_source=None, unknown_reason=None, known_recheck_recovered=True)
    candidates = [c for c in candidates if c['track']['recognition_status'] == 'novel_candidate']
    annotate_discovery_candidates(candidates, results, identities)
    # Explicit args are copied from A. No true test labels or unknown class count
    # select discovery policy, reference scales, thresholds, or recovered labels.
    from util.unknown_discovery import add_discovery_arguments
    defaults = argparse.ArgumentParser(add_help=False)
    add_discovery_arguments(defaults)
    discovery_options = {**vars(defaults.parse_args([])), **settings}
    discovery = discover_unknown(candidates, **discovery_kwargs(SimpleNamespace(**discovery_options)))
    refresh_open_set_results(results, {r['path']: [] for r in results})
    attach_recording_predictions(results, identities, 'single')
    # Strip stale event-level artifacts: this replay is for single-device records.
    for row in results:
        for field in ('bursts', 'burst_diagnostics', 'class_evidence', 'class_set_correct'):
            row.pop(field, None)
        for t in row['signals']:
            for field in ('submission_bursts', '_submission_bursts'):
                t.pop(field, None)
    report['discovery'] = discovery
    report['arguments']['known_recheck'] = True
    report['arguments']['open_set_profile'] = str(target)
    report['arguments']['output_dir'] = str(output)
    from datetime import datetime, timezone
    report['created_at'] = datetime.now(timezone.utc).isoformat()
    report.pop('feature_visualization', None)
    report.pop('feature_archive', None)
    report['open_set'] = dict(profile=str(target), profile_sha256=file_sha256(target),
                              replay_source=str(source), scope='primary single-device replay; no event export')
    evaluation = evaluate_open_set(results, classes, discovery, identities)
    original_eval = evaluate_open_set(baseline['files'], classes, baseline['discovery'], identities)
    if evaluation['metrics']['skipped_recordings'] or original_eval['metrics']['skipped_recordings']:
        raise ValueError('Comparison needs complete single-label test truth')
    old = {r['recording_id']: r for r in original_eval['samples']}
    new = {r['recording_id']: r for r in evaluation['samples']}
    if old.keys() != new.keys():
        raise ValueError('Paired test recording identities changed')
    accepted = [r for key, r in new.items() if old[key]['recognition_status'] == 'novel_candidate'
                and r['recognition_status'] == 'known']
    report['metrics'] = dict(open_set_evaluation=evaluation['metrics'],
        known_recheck_evaluated=len(decisions), known_recheck_recovered=len(accepted),
        discovery_constraint_failures=int(discovery.get('constraint_satisfied') is False))
    write(output/'results.json', report)
    write(output/'open_set_metrics.json', evaluation)
    write(output/'known_recheck_decisions.json', decisions)
    write(output/'unknown_clusters.json', discovery)
    updated = [candidate_record(c) for c in candidates]
    features = np.stack([c['feature'] for c in candidates]) if candidates else np.empty((0, profile.reference.shape[1]))
    save_npz(output/'unknown_features.npz', {**metadata, 'records': updated,
        'discovery_parameters': discovery['parameters']}, features=features)
    save_unknown_prototypes(output/'unknown_prototypes.npz', discovery, metadata['signature'], profile.reference.shape[1])
    with (output/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'label', 'status', 'error'])
        writer.writeheader()
        for row in results:
            d = row['single_signal_prediction']
            writer.writerow(dict(path=row['path'], label=d['display_label'], status=d['recognition_status'],
                error=row.get('error') or (f"{row['error_slices']} slice(s) failed" if row.get('error_slices') else '')))
    metrics_a, metrics_b = original_eval['metrics'], evaluation['metrics']
    comparison = dict(source_experiment=str(source), paired_recordings=len(old),
        source_manifest_sha256=file_sha256(source/'split_manifest.json'),
        source_results_sha256=file_sha256(source/'test/results.json'),
        source_profile_sha256=file_sha256(source/'open_set/known_profile.npz'),
        source_features_sha256=file_sha256(archive), feature_signature=metadata['signature']['sha256'],
        same_checkpoint=True, same_features=True, same_test_recordings=True,
        A=metrics_a, B=metrics_b, gate_calibration=profile_meta['known_recheck'],
        recovered_known_correct=sum(not r['is_unknown'] and r['predicted_class'] == r['true_class_name'] for r in accepted),
        recovered_known_wrong=sum(not r['is_unknown'] and r['predicted_class'] != r['true_class_name'] for r in accepted),
        newly_false_accepted_unknown=sum(r['is_unknown'] for r in accepted),
        interpretation='One paired held-out-class test; no test-dependent threshold fitting or automatic winner selection.')
    from util.run_context import metric_error_count
    comparison['source_error_count'] = metric_error_count(baseline.get('metrics', {}))
    comparison['comparison_has_upstream_errors'] = bool(comparison['source_error_count'])
    keys = ('error_files', 'files_with_slice_errors', 'visualization_errors', 'feature_plot_errors',
            'discovery_constraint_failures', 'burst_label_conflicts')
    comparison['source_error_breakdown'] = {
        key: int(baseline.get('metrics', {}).get(key, 0) or 0) for key in keys}
    comparison['source_non_event_error_count'] = (comparison['source_error_count'] -
        comparison['source_error_breakdown']['burst_label_conflicts'])
    comparison['source_burst_conflict_files'] = [
        dict(path=row['path'], label_conflicts=row['burst_diagnostics']['label_conflicts'])
        for row in baseline['files'] if row.get('burst_diagnostics', {}).get('label_conflicts', 0)]
    comparison['primary_record_comparison_completed'] = True
    comparison['event_export_valid'] = not bool(comparison['source_error_breakdown']['burst_label_conflicts'])
    write(output/'comparison.json', comparison)
    with (output/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['metric', 'A_baseline', 'B_known_recheck', 'B_minus_A'])
        for key in ('known_accuracy', 'known_false_reject_rate', 'known_abstention_rate',
                    'unknown_false_accept_rate', 'unknown_detection_recall', 'unknown_detection_f1',
                    'unknown_cluster_coverage', 'clustered_unknown_ari', 'all_accuracy',
                    'estimated_unknown_class_count', 'unknown_class_count_absolute_error'):
            a, b = metrics_a[key], metrics_b[key]
            writer.writerow([key, a, b, b-a if a is not None and b is not None else ''])
    return comparison


def comparison_exit_code(result, allow_event_conflicts=False):
    """An explicit scope exception never suppresses IO/discovery/plot errors."""
    if not result['comparison_has_upstream_errors']:
        return 0
    if (allow_event_conflicts and result['source_non_event_error_count'] == 0 and
            result['source_error_breakdown']['burst_label_conflicts'] > 0):
        return 0
    return 2


def main():
    import yaml
    from util.run_context import create_directory
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_experiment', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT/'config/known_recheck_b.yaml')
    p.add_argument('--output', type=Path, help='New/empty output; default experiment/known_recheck_comparison_TIMESTAMP')
    p.add_argument('--allow-event-conflicts', action='store_true',
        help='Explicitly allow event-conflict-only warnings for this primary-record comparison; flags remain in report')
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or set(config) != {'recheck'} or not isinstance(config['recheck'], dict):
        raise ValueError('Comparison config must contain only the recheck mapping')
    output = args.output or create_directory('known_recheck_comparison')
    print('Comparison directory:', output, flush=True)
    result = run_comparison(args.source_experiment, output, config['recheck'])
    code = comparison_exit_code(result, args.allow_event_conflicts)
    result['event_conflicts_explicitly_allowed'] = args.allow_event_conflicts
    result['exit_code'] = code
    result['comparison_status'] = ('completed_with_upstream_errors' if code else
        'completed_with_event_conflict_warnings' if result['comparison_has_upstream_errors'] else 'complete')
    write(Path(output)/'comparison.json', result)
    print(json.dumps({k: result[k] for k in ('recovered_known_correct', 'recovered_known_wrong',
          'newly_false_accepted_unknown', 'comparison_has_upstream_errors', 'source_error_breakdown',
          'primary_record_comparison_completed', 'event_export_valid', 'comparison_status', 'exit_code')},
          ensure_ascii=False, indent=2))
    if result['comparison_has_upstream_errors']:
        print('Primary-record comparison completed; source diagnostics remain in comparison.json. '
              'Event conflicts must be resolved before event-level submission.', file=sys.stderr)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
