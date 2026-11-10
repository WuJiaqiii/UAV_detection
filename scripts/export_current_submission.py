#!/usr/bin/env python3
"""Export already selected six-item results with one fixed task-one no_signal class."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.legacy_submission_common import predictions_from_export, export_results, save
from util.open_set import file_sha256


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-export', type=Path, required=True,
                   help='Completed result directory containing reclustered/task1/joint_selection.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--no-signal-code', type=int, default=24)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args(argv)
    output = a.output.expanduser().resolve()
    if output.exists():
        raise ValueError('Output must be a new directory; source results are never overwritten')
    predictions = predictions_from_export(a.source_export)
    code, summary = export_results(predictions, output, data_root=a.data_root,
        task1_no_signal_code=a.no_signal_code, check_only=a.check_only)
    save(output/'source_manifest.json', dict(source_export=str(a.source_export.resolve()),
         predictions={k:str(v) for k,v in predictions.items()},
         sha256={k:file_sha256(v) for k,v in predictions.items()}, shared_no_signal_code=a.no_signal_code))
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print('ERROR: '+str(exc), file=sys.stderr)
        raise SystemExit(2)
