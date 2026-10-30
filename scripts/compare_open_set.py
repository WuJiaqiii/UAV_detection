#!/usr/bin/env python3
"""Controlled frozen-model OSR experiments on one completed experiment directory."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser():
    from util.ood_scores import METHODS
    from util.unknown_discovery import add_discovery_arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source_experiment", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--calibration_units", nargs="+", choices=["primary", "record_max", "record_balanced"],
                        default=["primary", "record_max", "record_balanced"])
    parser.add_argument("--known_false_reject_rates", nargs="+", type=float, default=[.01, .05, .1])
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--profile_scale_prior_records", type=float, default=5.)
    parser.add_argument("--covariance_shrinkage", type=float, default=.1)
    parser.add_argument("--vim_dim", type=int, default=64)
    parser.add_argument("--replay_masks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--replay_cache", type=Path)
    parser.add_argument("--checkpoint_path", type=Path, help="Replay only; defaults to source training/models/best.pth")
    parser.add_argument("--replay_tolerance", type=float, default=.005)
    parser.add_argument("--replay_batch_size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--require_all_methods", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster_score_method", choices=METHODS, default="record_knn")
    parser.add_argument("--cluster_calibration_unit", choices=["primary", "record_max", "record_balanced"], default="primary")
    parser.add_argument("--cluster_frr", type=float, default=.05)
    parser.add_argument("--cluster_backends", nargs="+", choices=["hdbscan", "dcg", "smooth_hdbscan"],
                        default=["hdbscan", "dcg", "smooth_hdbscan"])
    parser.add_argument("--smooth_k", type=int, default=5)
    parser.add_argument("--smooth_blend", type=float, default=.2)
    add_discovery_arguments(parser)
    return parser


if __name__ == "__main__":
    from util.open_set_ablation import run
    print(run(build_parser().parse_args()))
