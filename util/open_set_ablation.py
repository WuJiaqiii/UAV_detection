"""Frozen-model controlled experiments: label, calibration, OOD score, clustering."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from sklearn.metrics import average_precision_score

from util.ablation_data import load_split, read_json, replay_classifier, validate_splits
from util.ood_scores import FittedScore, RAW_METHODS, calibrate
from util.open_set import normalize_features, file_sha256
from util.open_set_metrics import evaluate_open_set
from util.unknown_discovery import discover_unknown, discovery_kwargs


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def restore_head(results, classes):
    for row in results:
        for track in row.get("signals", []):
            if track.get("recognition_status") == "known":
                name = track.get("known_candidate_class") or track.get("candidate_class_name")
                if name not in classes:
                    raise ValueError("Missing original voting-head class; cannot isolate label override")
                track.update(class_name=name, class_id=classes[name], display_label=name)


def extra_metrics(evaluation, prior=.1):
    samples = [s for s in evaluation["samples"] if s["ood_score"] is not None and np.isfinite(s["ood_score"])]
    if not samples or len({s["is_unknown"] for s in samples}) < 2:
        return {}
    y = np.asarray([s["is_unknown"] for s in samples], dtype=bool)
    scores = np.asarray([s["ood_score"] for s in samples])
    correct = np.asarray([not s["is_unknown"] and s["true_class_name"] == s.get("curve_class") for s in samples])
    # Accept increasing novelty scores as the threshold is relaxed. Process ties together.
    order = np.argsort(scores, kind="stable")
    endpoints = np.r_[np.flatnonzero(np.diff(scores[order])), len(order)-1]
    known_correct = np.r_[0., np.cumsum(correct[order])[endpoints] / (~y).sum()]
    unknown_accepted = np.r_[0., np.cumsum(y[order])[endpoints] / y.sum()]
    weights = np.where(y, prior/y.sum(), (1-prior)/(~y).sum())
    rejected = np.asarray([s["recognition_status"] == "novel_candidate" for s in samples])
    tpr, fpr = float(rejected[y].mean()), float(rejected[~y].mean())
    denominator = prior*tpr + (1-prior)*fpr
    return {"ood_aupr_unknown": float(average_precision_score(y, scores)),
            "oscr_scored": float(trapezoid(known_correct, unknown_accepted)),
            "oscr_unit": "scored primary tracks; configured label policy; threshold-sweep diagnostic",
            "prior_adjusted_unknown_fraction": prior,
            "prior_adjusted_ood_aupr_unknown": float(average_precision_score(y, scores, sample_weight=weights)),
            "prior_adjusted_unknown_precision": prior*tpr/denominator if denominator else None,
            "prior_adjustment_scope": "score/decision reweighting only; clustering pool is unchanged"}


def enrich_evaluation(evaluation, test, label_policy="head"):
    lookup = {(r["path"], r["track_id"]): r for r in test.records}
    for sample in evaluation["samples"]:
        row = lookup.get((sample["primary_path"], sample["primary_track_id"]), {})
        sample["head_class"] = row.get("head_class")
        sample["curve_class"] = (row.get("profile_class_name") or row.get("head_class")) if label_policy == "saved" else row.get("head_class")
    evaluation["metrics"].update(extra_metrics(evaluation))
    return evaluation


def smooth_candidates(candidates, k=5, blend=.2):
    """One conservative mean-shift step with distinct-record neighbors. Not CMS training."""
    if len(candidates) < 2:
        return candidates
    features = normalize_features([c["feature"] for c in candidates])
    similarities = features @ features.T
    result = []
    for i, candidate in enumerate(candidates):
        seen = {candidate["recording_id"]}
        neighbors = []
        for j in np.argsort(-similarities[i], kind="stable"):
            identity = candidates[j]["recording_id"]
            if identity not in seen:
                seen.add(identity)
                neighbors.append(j)
                if len(neighbors) == k:
                    break
        feature = features[i]
        if neighbors:
            mixed = (1-blend)*feature + blend*features[neighbors].mean(0)
            feature = normalize_features(mixed[None])[0]
        result.append({**candidate, "feature": feature})
    return result


def predict_case(test, scores, threshold, args, backend, cluster_cache):
    # Shallow-copy nested track metadata; mutations affect only the new top-level tracks.
    results = [{"path": f["path"], "true_class_names": f.get("true_class_names"),
                "signals": [dict(t) for t in f.get("signals", [])]} for f in test.report["files"]]
    tracks = {(f["path"], t["track_id"]): t for f in results for t in f["signals"]}
    candidates = []
    for r, feature, score in zip(test.records, test.features, scores):
        track = tracks[(r["path"], r["track_id"])]
        novel = bool(score > threshold)
        name = r["head_class"]
        if name not in test.classes:
            raise ValueError("Valid feature track has no known voting-head class")
        track.update(ood_score=float(score), ood_threshold=float(threshold), unknown_cluster_id=None,
                     status="unknown" if novel else "ok", recognition_status="novel_candidate" if novel else "known",
                     class_name=None if novel else name, class_id=None if novel else test.classes[name],
                     display_label="Unknown_unassigned" if novel else name)
        if novel:
            candidates.append({"path": r["path"], "recording_id": r["recording_id"], "track": track,
                               "feature": feature})
    # A fixed candidate set is the cache key. Clustering never receives truth labels.
    key = (backend, tuple((c["path"], c["track"]["track_id"]) for c in candidates))
    if key in cluster_cache:
        discovery, assignments = copy.deepcopy(cluster_cache[key])
        for c, update in zip(candidates, assignments):
            c["track"].update(update)
    else:
        options = discovery_kwargs(args)
        options["backend"] = "hdbscan" if backend == "smooth_hdbscan" else backend
        clustering_inputs = smooth_candidates(candidates, args.smooth_k, args.smooth_blend) if backend == "smooth_hdbscan" else candidates
        discovery = discover_unknown(clustering_inputs, **options)
        discovery["candidate_pool_sha256"] = hashlib.sha256(json.dumps(key[1]).encode()).hexdigest()
        if backend == "smooth_hdbscan":
            discovery.update(method="one-step record-neighbor smoothing + sklearn.HDBSCAN",
                             smoothing={"k": args.smooth_k, "blend": args.smooth_blend,
                                        "interpretation": "frozen-feature adaptation, not a CMS reproduction"})
        fields = ("unknown_cluster_id", "cluster_status", "cluster_membership", "cluster_similarity", "display_label")
        assignments = [{k: c["track"].get(k) for k in fields} for c in candidates]
        cluster_cache[key] = copy.deepcopy((discovery, assignments))
    return results, discovery


def save_case(output, case, results, evaluation, discovery):
    directory = output / case["name"]
    directory.mkdir()
    metrics = dict(evaluation["metrics"])
    metrics["evaluation_unit"] = metrics.pop("unit")
    write_json(directory / "metrics.json", {**case, **metrics})
    write_json(directory / "evaluation.json", evaluation)
    write_json(directory / "clusters.json", discovery)
    columns = ["path", "track_id", "recognition_status", "class_name", "known_candidate_class", "ood_score", "ood_threshold",
               "unknown_cluster_id", "cluster_status", "start_ms", "end_ms", "frequency_center_offset_mhz",
               "frequency_low_offset_mhz", "frequency_high_offset_mhz", "frequency_bands_offset_mhz"]
    with (directory / "assignments.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        for f in results:
            for track in f.get("signals", []):
                row = {**track, "path": f["path"]}
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    return {**case, **metrics}


def attach_replay_cache(splits, cache):
    cache = Path(cache)
    expected = splits[0].metadata["signature"]["details"]["checkpoint_sha256"]
    with np.load(cache / "classifier_head.npz", allow_pickle=False) as z:
        metadata = json.loads(z["metadata"].item())
        if metadata["checkpoint_sha256"] != expected:
            raise ValueError("Replay cache checkpoint mismatch")
        head = {k: z[k].copy() for k in ("weight", "bias")}
    evidence = []
    for name, split in zip(("reference", "calibration", "test"), splits):
        with np.load(cache / f"{name}.npz", allow_pickle=False) as z:
            m = json.loads(z["metadata"].item())
            if (m["signature"] != split.metadata["signature"] or
                [(r["path"], r["track_id"]) for r in m["records"]] !=
                    [(r["path"], r["track_id"]) for r in split.records] or
                not np.array_equal(z["features"], split.features)):
                raise ValueError("Replay cache and source track archive differ")
            split.logits, split.raw_features = z["logits"].copy(), z["raw_features"].copy()
            if (split.logits.shape != (len(split.records), len(split.classes)) or
                split.raw_features.shape != split.features.shape or
                not np.isfinite(split.logits).all() or not np.isfinite(split.raw_features).all()):
                raise ValueError("Invalid cached raw-feature/logit arrays")
            parity = m["replay"]
            if parity["max_feature_l2_error"] > parity["tolerance"] or parity["max_probability_error"] > parity["tolerance"]:
                raise ValueError("Replay cache failed its declared parity tolerance")
            evidence.append(parity)
    return head, evidence


def run(args):
    if not getattr(args, "output_dir", None):
        from util.run_context import execute_run
        return execute_run("ablate", args, run)
    validate_options(args)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        return _run_frozen(args)


def _run_frozen(args):
    from util.run_context import check_output_directory
    from util.ablation_report import write_report
    output = check_output_directory(args)
    output.mkdir(parents=True, exist_ok=True)
    source = args.source_experiment.expanduser().resolve()
    reference = load_split(source / "open_set/reference", "features.npz")
    calibration = load_split(source / "open_set/calibration", "features.npz")
    test = load_split(source / "test", "inference_features.npz")
    splits = (reference, calibration, test)
    audit = validate_splits(*splits)
    identities_path = source / "recording_ids.json"
    identities = read_json(identities_path) if identities_path.is_file() else {}
    identities.update({r["path"]: r["recording_id"] for s in splits for r in s.records})
    for s in splits:
        s.source_recording_ids = identities
    from importlib.metadata import version
    implementation_root = Path(__file__).resolve().parents[1]
    implementation_files = ["util/ood_scores.py", "util/ablation_data.py", "util/open_set_ablation.py",
                            "util/known_reference.py", "util/open_set_metrics.py", "util/unknown_discovery.py",
                            "util/dcg_discovery.py", "util/mask_input.py", "model/resnet.py"]
    audit.update(source_experiment=str(source), development_base_commit="fe85debb0c7d723d82a659c6ed4bada12772f7eb",
                 implementation_sha256={name: file_sha256(implementation_root/name) for name in implementation_files},
                 libraries={name: version(name) for name in ("numpy", "scipy", "scikit-learn", "torch", "torchvision")},
                 source_archive_sha256={name: file_sha256(s.directory/filename)
                     for name, s, filename in zip(("reference", "calibration", "test"), splits,
                                                 ("features.npz", "features.npz", "inference_features.npz"))},
                 source_result_sha256=file_sha256(source / "test/results.json"),
                 selection="All configured cases are reported; no test-label method/threshold selection",
                 fitting="Known reference and independent known calibration only; frozen encoder",
                 arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if not k.startswith("_")})
    head, evidence = None, []
    if args.replay_cache:
        head, evidence = attach_replay_cache(splits, args.replay_cache)
        audit["replay_cache"] = str(args.replay_cache)
    elif args.replay_masks:
        checkpoint = args.checkpoint_path or source / "training/models/best.pth"
        head, evidence = replay_classifier(splits, checkpoint, output / "replay_cache", args.replay_batch_size,
                                          args.device, args.threads, args.replay_tolerance)
    audit["mask_replay"] = evidence
    write_json(output / "audit.json", audit)
    rows, evaluations, cluster_cache, skipped = [], {}, {}, []
    cases_output = output / "cases"
    cases_output.mkdir()
    # Exact saved baseline and isolated head-label ablation preserve original clusters/decisions.
    for name, restore in (("R0_saved", False), ("R1_saved_head", True)):
        results = copy.deepcopy(test.report["files"])
        if restore:
            restore_head(results, test.classes)
        discovery = test.report["discovery"]
        evaluation = enrich_evaluation(evaluate_open_set(results, test.classes, discovery, identities), test,
                                      "head" if restore else "saved")
        case = {"name": name, "stage": "R1" if restore else "R0", "method": "saved_profile", "unit": "saved",
                "target_frr": None, "backend": discovery["backend"], "label_policy": "head" if restore else "saved"}
        rows.append(save_case(cases_output, case, results, evaluation, discovery))
        evaluations[name] = evaluation
    methods_output = output / "scores"
    methods_output.mkdir()
    for method in args.methods:
        if method in RAW_METHODS and reference.logits is None:
            reason = {"method": method, "reason": "Original logits/raw features unavailable; enable replay_masks or replay_cache"}
            if args.require_all_methods:
                raise ValueError(reason["reason"])
            skipped.append(reason)
            continue
        scorer = FittedScore(method, reference, args.knn_k, args.covariance_shrinkage,
                             args.profile_scale_prior_records, args.vim_dim, head)
        cal_scores, test_scores = scorer.score(calibration), scorer.score(test)
        if not np.isfinite(cal_scores).all() or not np.isfinite(test_scores).all():
            raise ValueError(f"Non-finite scores for {method}")
        np.savez_compressed(methods_output / f"{method}.npz", calibration=cal_scores, test=test_scores)
        thresholds = []
        for unit in args.calibration_units:
            for rate in args.known_false_reject_rates:
                fitted = calibrate(cal_scores, calibration, unit, rate)
                thresholds.append(fitted)
                name = f"{method}__{unit}__frr_{rate:g}__hdbscan"
                case = {"name": name, "stage": "R2" if method == "record_class_knn" else "R3", "method": method,
                        "unit": unit, "target_frr": rate, "backend": "hdbscan", "label_policy": "head",
                        **fitted}
                results, discovery = predict_case(test, test_scores, fitted["threshold"], args, "hdbscan", cluster_cache)
                evaluation = enrich_evaluation(evaluate_open_set(results, test.classes, discovery, identities), test)
                rows.append(save_case(cases_output, case, results, evaluation, discovery))
                evaluations[name] = evaluation
        write_json(methods_output / f"{method}.json", {"fit": scorer.info, "calibration": thresholds})
        print(f"Completed score ablations: {method}", flush=True)
    # R4: chosen in configuration, not from test accuracy; same rejection pool for every backend.
    clustering_scorer = FittedScore(args.cluster_score_method, reference, args.knn_k, args.covariance_shrinkage,
                                   args.profile_scale_prior_records, args.vim_dim, head)
    fitted = calibrate(clustering_scorer.score(calibration), calibration, args.cluster_calibration_unit, args.cluster_frr)
    scores = clustering_scorer.score(test)
    pool_hashes = set()
    for backend in args.cluster_backends:
        name = f"R4_{backend}"
        case = {"name": name, "stage": "R4", "method": args.cluster_score_method,
                "unit": args.cluster_calibration_unit, "target_frr": args.cluster_frr,
                "backend": backend, "label_policy": "head", **fitted}
        results, discovery = predict_case(test, scores, fitted["threshold"], args, backend, cluster_cache)
        pool_hashes.add(discovery["candidate_pool_sha256"])
        evaluation = enrich_evaluation(evaluate_open_set(results, test.classes, discovery, identities), test)
        rows.append(save_case(cases_output, case, results, evaluation, discovery))
        evaluations[name] = evaluation
    if len(pool_hashes) > 1:
        raise AssertionError("R4 clustering candidates differ")
    summary = {"schema_version": 1, "protocol": "frozen known-only OSR and transductive batch clustering",
               "completed_cases": len(rows), "skipped_methods": skipped, "cases": rows,
               "r4_candidate_pool_sha256": next(iter(pool_hashes), None),
               "interpretation": "Descriptive comparisons on an already inspected test set; validate choices on pseudo-unknown development classes",
               "not_run": ["encoder retraining on additional components", "amplitude/IQ dual-branch training",
                           "GCD representation adaptation using an unlabeled target pool"]}
    write_report(output, summary, evaluations)
    return {"metrics": {"completed_cases": len(rows), "skipped_methods": len(skipped)},
            "summary_csv": str(output / "summary.csv"), "report": str(output / "report.html")}


def validate_options(args):
    if not args.methods or len(set(args.methods)) != len(args.methods):
        raise ValueError("methods must be nonempty and unique")
    if not args.calibration_units or len(set(args.calibration_units)) != len(args.calibration_units):
        raise ValueError("calibration_units must be nonempty and unique")
    if not args.cluster_backends or len(set(args.cluster_backends)) != len(args.cluster_backends):
        raise ValueError("cluster_backends must be nonempty and unique")
    if (not args.known_false_reject_rates or len(set(args.known_false_reject_rates)) != len(args.known_false_reject_rates)
            or any(not 0 < r < 1 for r in args.known_false_reject_rates)):
        raise ValueError("known_false_reject_rates must be unique values in (0,1)")
    if not 0 < args.cluster_frr < 1 or not 0 <= args.smooth_blend <= 1:
        raise ValueError("Invalid cluster_frr/smooth_blend")
    if min(args.knn_k, args.smooth_k, args.replay_batch_size, args.threads, args.vim_dim) < 1:
        raise ValueError("Counts, threads and vim_dim must be positive")
    if not np.isfinite(args.replay_tolerance) or not 0 < args.replay_tolerance <= .02:
        raise ValueError("replay_tolerance must be in (0,.02]")
    if not np.isfinite(args.covariance_shrinkage) or not 0 < args.covariance_shrinkage <= 1:
        raise ValueError("covariance_shrinkage must be in (0,1]")
    if not np.isfinite(args.profile_scale_prior_records) or args.profile_scale_prior_records < 0:
        raise ValueError("profile_scale_prior_records must be finite and nonnegative")
