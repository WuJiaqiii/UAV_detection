#!/usr/bin/env python3
"""Recluster six legacy caches; explicitly export task-one no_signal as an extra class."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.legacy_submission_common import sources_from_batch, recluster, export_results, save


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True, help='Original six-item batch_manifest.json')
    parser.add_argument('--output', type=Path, required=True, help='New directory; original runs remain untouched')
    parser.add_argument('--run-root', type=Path, help='Relocate source run directories under this snapshot root')
    parser.add_argument('--data-root', type=Path, help='Actual six DAT directories; enables raw-file coverage audit')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError('Output must be a new directory')
    _, sources = sources_from_batch(args.batch, args.run_root)
    predictions = recluster(sources, output/'reclustered')
    save(output/'source_manifest.json', dict(batch=str(args.batch.resolve()), sources=sources,
         interpretation='Frozen-cache spectral recovery; all unknown IDs regenerated; known predictions unchanged.'))
    code, _ = export_results(predictions, output, extra_no_signal=True,
                              data_root=args.data_root, check_only=args.check_only)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
