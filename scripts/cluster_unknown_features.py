#!/usr/bin/env python3
"""Compare discovery backends on a fixed candidate archive, without running YOLO/CNN."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.open_set import file_sha256, load_npz, normalize_features, save_npz
from util.unknown_discovery import add_discovery_arguments, discover_unknown, discovery_kwargs, save_unknown_prototypes


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--features", type=Path, required=True, help="unknown_features.npz from open-set evaluation")
    add_discovery_arguments(parser)
    parser.add_argument("--results_json", type=Path, help="Optional source results.json for file-truth colors only")
    from util.feature_visualization import add_feature_plot_arguments
    add_feature_plot_arguments(parser)
    return parser


def run(args):
    if not getattr(args, "output_dir", None):
        from util.run_context import execute_run
        return execute_run("cluster", args, run)
    from util.run_context import check_output_directory
    from util.feature_visualization import plot_options
    plot_options(args)
    output = check_output_directory(args)
    metadata, arrays = load_npz(args.features, "unknown_candidates")
    features = normalize_features(arrays["features"])
    records = metadata["records"]
    if len(records) != len(features) or not isinstance(metadata.get("signature"), dict):
        raise ValueError("Candidate records/features/signature must be aligned")
    candidates, identities = [], set()
    for row, feature in zip(records, features):
        for key in ("path", "recording_id", "track_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Candidate metadata requires nonempty {key}")
        identity = (row["path"], row["track_id"])
        if identity in identities:
            raise ValueError("Duplicate candidate identity")
        identities.add(identity)
        candidates.append({"path": row["path"], "recording_id": row["recording_id"], "feature": feature,
                           "track": {"track_id": row["track_id"], "recognition_status": "novel_candidate"}})
    discovery = discover_unknown(candidates, **discovery_kwargs(args))
    discovery.update(candidate_archive_sha256=file_sha256(args.features), prototype_archive="unknown_prototypes.npz")
    updated = [{**row, **candidate["track"]} for row, candidate in zip(records, candidates)]
    output.mkdir(parents=True, exist_ok=True)
    save_unknown_prototypes(output/"unknown_prototypes.npz", discovery, metadata["signature"], features.shape[1])
    save_npz(output/"unknown_features.npz", {**metadata, "records": updated}, features=features)
    (output/"unknown_clusters.json").write_text(json.dumps(discovery, ensure_ascii=False, indent=2,
                                                        allow_nan=False)+"\n", encoding="utf-8")
    columns = ["path", "recording_id", "track_id", "display_label", "recognition_status", "unknown_cluster_id",
               "cluster_status", "cluster_similarity", "cluster_membership", "start_ms", "end_ms",
               "frequency_center_offset_mhz", "frequency_low_offset_mhz", "frequency_high_offset_mhz",
               "frequency_bands_offset_mhz", "rf_center_mhz", "frequency_center_absolute_mhz"]
    columns += sorted({key for row in updated for key in row} - set(columns))
    with (output/"unknown_assignments.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in updated:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})
    if args.save_feature_plots:
        from util.feature_visualization import attach_file_truth, write_feature_plots, write_plot_gallery, plot_options
        plot_records = attach_file_truth(updated, args.results_json)
        summary = write_feature_plots(output/"feature_plots", features, plot_records, **plot_options(args))
        write_plot_gallery(output, summary)
        discovery["feature_visualization"] = summary
        (output/"unknown_clusters.json").write_text(json.dumps(discovery, ensure_ascii=False, indent=2,
            allow_nan=False)+"\n", encoding="utf-8")
    return discovery


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps({key: value for key, value in result.items() if key != "clusters"}, ensure_ascii=False, indent=2))
