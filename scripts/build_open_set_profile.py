#!/usr/bin/env python3
"""Calibrate a frozen known-only profile from disjoint labeled track archives."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.open_set import build_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_features", type=Path, required=True)
    parser.add_argument("--calibration_features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .npz file; never overwritten")
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--known_false_reject_rate", type=float, default=.01)
    parser.add_argument("--min_calibration_per_class", type=int, default=20, help="Minimum distinct original known recordings per class")
    parser.add_argument("--max_reference_per_class", type=int, default=2000)
    parser.add_argument("--ignore_classes", nargs="*", default=[], help="Optional abstention classes, e.g. Background; their predictions become uncertain")
    args = parser.parse_args()
    metadata = build_profile(args.reference_features, args.calibration_features, args.output,
        args.knn_k, args.known_false_reject_rate, args.min_calibration_per_class, args.max_reference_per_class, args.ignore_classes)
    summary = {key: value for key, value in metadata.items() if key != "signature"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
