"""Post-hoc single-label recording metrics; never used for prediction or tuning."""
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score


def evaluate_open_set(results, classes, discovery, recording_ids=None):
    """Select one primary track per recording using votes, independently of truth.

    Multi-label/conflicting records are explicitly excluded: file-level label sets
    cannot establish per-signal truth. Missing/uncertain/unassigned predictions
    remain in the end-to-end ACC denominators. ARI/NMI are conditional on coverage.
    """
    grouped = defaultdict(list)
    for row in results:
        grouped[(recording_ids or {}).get(row["path"], row["path"])].append(row)
    samples, skipped = [], []
    for identity, rows in sorted(grouped.items()):
        labels = [r.get("true_class_names") for r in rows]
        if any(label is None or len(label) != 1 for label in labels) or len({label[0] for label in labels}) != 1:
            skipped.append({"recording_id": identity, "reason": "missing_multi_or_conflicting_truth"})
            continue
        tracks = [(row["path"], track) for row in rows for track in row.get("signals", [])]
        tracks.sort(key=lambda item: (-item[1].get("valid_votes", 0), -item[1].get("observation_count", 0),
                                      item[0], str(item[1].get("track_id", ""))))
        path, track = tracks[0] if tracks else (rows[0]["path"], {})
        status = track.get("recognition_status", "no_signal")
        samples.append({"recording_id": identity, "paths": [r["path"] for r in rows],
                        "primary_path": path, "primary_track_id": track.get("track_id"),
                        "true_class_name": labels[0][0], "is_unknown": labels[0][0] not in classes,
                        "recognition_status": status, "predicted_class": track.get("class_name") if status == "known" else None,
                        "unknown_cluster_id": track.get("unknown_cluster_id") if status == "novel_candidate" else None,
                        "ood_score": track.get("ood_score")})
    known = [s for s in samples if not s["is_unknown"]]
    novel = [s for s in samples if s["is_unknown"]]
    true_names = sorted({s["true_class_name"] for s in novel})
    cluster_names = sorted({s["unknown_cluster_id"] for s in samples if s["unknown_cluster_id"]})
    matrix = np.zeros((len(true_names), len(cluster_names)), dtype=np.int64)
    for sample in novel:
        if sample["unknown_cluster_id"]:
            matrix[true_names.index(sample["true_class_name"]), cluster_names.index(sample["unknown_cluster_id"])] += 1
    mapping = {}
    if matrix.size:
        rows, columns = linear_sum_assignment(-matrix)
        mapping = {cluster_names[c]: true_names[r] for r, c in zip(rows, columns) if matrix[r, c] > 0}
    known_correct = sum(s["predicted_class"] == s["true_class_name"] for s in known)
    novel_correct = sum(mapping.get(s["unknown_cluster_id"]) == s["true_class_name"] for s in novel)
    clustered = [s for s in novel if s["unknown_cluster_id"]]
    tp = sum(s["recognition_status"] == "novel_candidate" for s in novel)
    fp = sum(s["recognition_status"] == "novel_candidate" for s in known)
    fn = len(novel) - tp
    ratio = lambda value, total: value / total if total else None
    scored = [s for s in samples if s["ood_score"] is not None and np.isfinite(s["ood_score"])]
    auc = (float(roc_auc_score([s["is_unknown"] for s in scored], [s["ood_score"] for s in scored]))
           if len({s["is_unknown"] for s in scored}) == 2 else None)
    estimated = discovery["estimated_unknown_class_count"]
    complete_truth = bool(samples) and not skipped
    metrics = {"unit": "one primary track per labeled recording (file if no recording ID)",
               "primary_rule": "most valid votes, then observations, then path/track ID; independent of truth",
               "evaluated_recordings": len(samples), "skipped_recordings": len(skipped),
               "known_recordings": len(known), "unknown_recordings": len(novel),
               "known_accuracy": ratio(known_correct, len(known)),
               "unknown_accuracy": ratio(novel_correct, len(novel)),
               "all_accuracy": ratio(known_correct + novel_correct, len(samples)),
               "known_false_reject_rate": ratio(fp, len(known)),
               "known_abstention_rate": ratio(sum(s["recognition_status"] != "known" for s in known), len(known)),
               "unknown_detection_precision": ratio(tp, tp + fp),
               "unknown_detection_recall": ratio(tp, len(novel)),
               "unknown_detection_f1": ratio(2 * tp, 2 * tp + fp + fn),
               "unknown_cluster_coverage": ratio(len(clustered), len(novel)),
               "clustered_unknown_accuracy": ratio(novel_correct, len(clustered)),
               "clustered_unknown_ari": float(adjusted_rand_score([s["true_class_name"] for s in clustered],
                                                                 [s["unknown_cluster_id"] for s in clustered])) if len(clustered) >= 2 else None,
               "clustered_unknown_nmi": float(normalized_mutual_info_score([s["true_class_name"] for s in clustered],
                                                                          [s["unknown_cluster_id"] for s in clustered])) if len(clustered) >= 2 else None,
               "ood_auroc": auc, "ood_scored_recordings": len(scored),
               "class_count_comparison_scope": "complete_single_label_truth" if complete_truth else "incomplete_or_unsupported_truth",
               "observed_labeled_unknown_class_count": len(true_names) if samples else None,
               "true_unknown_class_count": len(true_names) if complete_truth else None,
               "estimated_unknown_class_count": estimated,
               "unknown_class_count_error": estimated - len(true_names) if complete_truth else None,
               "unknown_class_count_absolute_error": abs(estimated - len(true_names)) if complete_truth else None,
               "unknown_cluster_to_truth_for_scoring_only": mapping}
    return {"metrics": metrics, "samples": samples, "skipped": skipped}
