#!/usr/bin/env python3
"""One six-item export entry: frozen v3 batch or verified final predictions."""
import argparse
import fcntl
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.submission_pipeline import (ITEMS, predictions_from_export, sources_from_batch, recluster, export_results)
from scripts import competition_infer as pipeline
from util.experiment_io import read_json as read, write_json as save, digest


def validate_batch(path, data_root=None):
    batch = read(path)
    if batch.get('schema_version')!=3:
        raise ValueError('Use a fresh v3 batch; old result recovery has been removed')
    if set(batch.get('runs',{}))!=set(ITEMS):
        raise ValueError('All six inference items must exist before export')
    fingerprint = batch['fingerprint']
    artifacts = pipeline.training_artifacts(fingerprint['training_run'],fingerprint['yolo'],fingerprint.get('profile'))
    if artifacts['profile_sha256']!=fingerprint['profile_sha256'] or pipeline.source_fingerprint()!=fingerprint['pipeline_sources']:
        raise ValueError('Model/profile/pipeline changed after inference')
    root = Path(data_root or fingerprint['data_root']).expanduser().resolve()
    if root!=Path(fingerprint['data_root']).resolve():
        raise ValueError('Data root must match the frozen batch')
    _,_,signatures = pipeline.inspect_inputs(root,ITEMS)
    if any(batch['runs'][item]['input_signature']!=signatures[item] for item in ITEMS):
        raise ValueError('Input recordings changed after inference')
    _,sources = sources_from_batch(path)
    return batch,sources,root


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--batch',type=Path)
    source.add_argument('--source-results',type=Path,help='v3 results directory containing final_predictions.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--no-signal-code',type=int,help='Explicit nonstandard task-one placeholder; for example 24')
    p.add_argument('--check-only',action='store_true')
    args = p.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError('Choose a new output directory; source runs are never overwritten')
    if args.no_signal_code is not None and args.no_signal_code<20:
        raise ValueError('no-signal-code must be >=20')
    if args.batch:
        batch_path = args.batch.expanduser().resolve()
        with batch_path.with_suffix('.lock').open('a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('Inference still owns this batch') from exc
            batch,sources,data_root = validate_batch(batch_path,args.data_root)
            output.mkdir(parents=True)
            discovery = batch['discovery']
            predictions = recluster(sources,output/'reclustered',strategy=discovery['strategy'],
                spectral=discovery.get('spectral_options'),discovery=discovery)
    else:
        predictions = predictions_from_export(args.source_results)
        data_root = args.data_root
        output.mkdir(parents=True)
    save(output/'final_predictions.json',dict(schema_version=3,predictions={k:str(v) for k,v in predictions.items()},
        sha256={k:digest(v) for k,v in predictions.items()}))
    code,summary = export_results(predictions,output,data_root=data_root,check_only=args.check_only,
                                 task1_no_signal_code=args.no_signal_code)
    return code


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError) as exc:
        print('ERROR:',exc,file=sys.stderr);raise SystemExit(2)
