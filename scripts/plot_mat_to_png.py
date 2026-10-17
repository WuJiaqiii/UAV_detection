#!/usr/bin/env python3
"""Compatibility command: export MAT detector images through the shared frontend.

Unlike the old plotting script this exports detector tiles, a source-group split,
metadata and the exact preprocessing manifest. No percentile/log enhancement.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.export_iq_yolo_dataset import parser, export, finalize


def main():
    p = parser()
    p.set_defaults(input_type="mat")
    args = p.parse_args()
    if args.finalize:
        finalize(args.output_dir)
    elif args.input_dir is None:
        p.error("--input_dir is required")
    else:
        export(args)


if __name__ == "__main__":
    main()
