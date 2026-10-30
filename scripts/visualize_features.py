#!/usr/bin/env python3
"""Render existing track/candidate archives without loading models or reclustering."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser():
    from util.feature_visualization import add_feature_plot_arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--results_json", type=Path, help="Optional previous results.json; file labels are joined by exact source path for coloring only")
    add_feature_plot_arguments(parser)
    return parser


def run(args):
    from util.run_context import check_output_directory, execute_run
    if not getattr(args, "output_dir", None):
        return execute_run("visualize", args, run)
    from util.open_set import load_npz
    from util.feature_visualization import attach_file_truth, write_feature_plots, write_plot_gallery, plot_options
    import numpy as np
    # Read only the archive kind first; load_npz performs schema/representation validation.
    with np.load(args.features, allow_pickle=False) as archive:
        kind = json.loads(str(archive["metadata"].item())).get("kind")
    if kind not in {"inference_features", "unknown_candidates", "track_features"}:
        raise ValueError("Expected inference_features, unknown_candidates or track_features archive")
    metadata, arrays = load_npz(args.features, kind)
    settings = plot_options(args)
    records = attach_file_truth(metadata["records"], args.results_json)
    if kind == "unknown_candidates":
        records = [{**row, "recognition_status": row.get("recognition_status", "novel_candidate")} for row in records]
    output = check_output_directory(args)
    output.mkdir(parents=True, exist_ok=True)
    if args.save_feature_plots:
        summary = write_feature_plots(output/"feature_plots", arrays["features"], records, **settings)
    else:
        summary = {"status": "skipped", "reason": "Feature plots disabled", "plots": []}
    summary.update(source_archive=str(args.features.resolve()), source_kind=kind)
    (output/"results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    write_plot_gallery(output, summary)
    return summary


if __name__ == "__main__":
    from util.run_context import execute_run
    print(json.dumps(execute_run("visualize", build_parser().parse_args(), run), ensure_ascii=False, indent=2))
