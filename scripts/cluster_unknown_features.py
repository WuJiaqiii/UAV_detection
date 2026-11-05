#!/usr/bin/env python3
"""Compare discovery backends on a fixed candidate archive, without running YOLO/CNN."""
import argparse
import copy
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
    parser.add_argument("--results_json", type=Path,
                        help="Source results.json: primary selection context, optional post-hoc scoring and truth colors")
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
                           "track": {**row, "recognition_status": "novel_candidate"}})
    from util.discovery_context import annotate_discovery_candidates, candidate_record, source_recording_ids
    source = json.loads(args.results_json.read_text(encoding="utf-8")) if args.results_json else None
    recording_ids = {}
    if source and any("signals" in row for row in source.get("files", [])):
        recording_ids = source_recording_ids(source, records, args.results_json)
        annotate_discovery_candidates(candidates, source["files"], recording_ids)
    if source and source.get("effective_config", {}).get("classes"):
        source_novel = {(r["path"], t["track_id"]) for r in source["files"] for t in r.get("signals", [])
                        if t.get("recognition_status") == "novel_candidate"}
        if identities != source_novel:
            raise ValueError("Candidate archive/source novel track sets differ; use the matching complete archive")
    discovery = discover_unknown(candidates, **discovery_kwargs(args))
    discovery.update(candidate_archive_sha256=file_sha256(args.features), prototype_archive="unknown_prototypes.npz")
    updated = [{**row, **candidate_record(candidate)} for row, candidate in zip(records, candidates)]
    output.mkdir(parents=True, exist_ok=True)
    save_unknown_prototypes(output/"unknown_prototypes.npz", discovery, metadata["signature"], features.shape[1])
    save_npz(output/"unknown_features.npz", {**metadata, "records": updated,
             "discovery_parameters": discovery["parameters"]}, features=features)
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
    if source and source.get("effective_config", {}).get("classes"):
        # Scoring happens only after discovery. Labels never enter its density,
        # primary selection, guard, support rules or estimated number of classes.
        from util.open_set_metrics import evaluate_open_set
        from util.discovery_context import attach_recording_predictions
        refreshed = copy.deepcopy(source)
        tracks = {(r["path"], t["track_id"]): t for r in refreshed["files"] for t in r.get("signals", [])}
        for row in updated:
            track = tracks.get((row["path"], row["track_id"]))
            if track is None or track.get("recognition_status") != "novel_candidate":
                raise ValueError("Candidate archive is inconsistent with source results")
            for key in ("ood_score", "ood_threshold"):
                if row.get(key) is not None and not np.isclose(row[key], track.get(key, float("nan"))):
                    raise ValueError("Candidate archive/source rejection scores differ")
            for key in ("unknown_cluster_id", "display_label", "cluster_status", "cluster_membership",
                        "cluster_similarity", "discovery_role", "cluster_assignment_source", "discovery_is_primary"):
                track[key] = row.get(key)
            track["display_label"] = row["unknown_cluster_id"] or "Unknown_unassigned"
        for row in refreshed["files"]:
            row["predicted_unknown_classes"] = sorted({t["unknown_cluster_id"] for t in row.get("signals", [])
                                                       if t.get("unknown_cluster_id")})
        attach_recording_predictions(refreshed["files"], recording_ids, discovery["input_mode"])
        from util.class_evidence import summarize_class_evidence
        type_context = {r["path"]: [{"groups": [
            {"track_id": t["track_id"], "group_type": kind}
            for t in r.get("signals", []) for kind in (t.get("component_types") or ["unspecified"])]}]
            for r in refreshed["files"]}
        summarize_class_evidence(refreshed["files"], type_context)
        evaluation = evaluate_open_set(refreshed["files"], refreshed["effective_config"]["classes"],
                                       discovery, recording_ids)
        refreshed.update(discovery=discovery, recording_ids=recording_ids,
            reclustering={"source_results": str(args.results_json.resolve()),
                          "source_results_sha256": file_sha256(args.results_json),
                          "source_candidate_sha256": file_sha256(args.features),
                          "discovery_parameters": discovery["parameters"],
                          "previews": "Original slice previews are not regenerated by offline clustering."})
        refreshed["metrics"]["open_set_evaluation"] = evaluation["metrics"]
        refreshed["metrics"].update(discovery_fit_tracks=discovery["fit_tracks"],
                                      discovery_inherited_tracks=discovery["inherited_secondary_tracks"])
        refreshed["metrics"].update({k: discovery[k] for k in ("unknown_candidate_tracks", "estimated_unknown_class_count",
                                                             "clustered_unknown_tracks", "unassigned_unknown_tracks")})
        (output/"open_set_metrics.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2,
                                                              allow_nan=False)+"\n", encoding="utf-8")
        (output/"reclustered_results.json").write_text(json.dumps(refreshed, ensure_ascii=False, indent=2,
                                                                 allow_nan=False)+"\n", encoding="utf-8")
        discovery["rescored_metrics"] = evaluation["metrics"]
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
