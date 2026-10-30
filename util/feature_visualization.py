"""Post-hoc projections of frozen features. Never used for recognition/clustering."""
import hashlib
import argparse
import inspect
import json
from pathlib import Path
import textwrap

import numpy as np


def add_feature_plot_arguments(parser):
    parser.add_argument("--save_feature_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature_plot_method", choices=["pca", "tsne", "both"], default="pca")
    parser.add_argument("--feature_plot_max_points", type=int, default=3000)
    parser.add_argument("--feature_plot_seed", type=int, default=42)


def plot_options(args):
    limit = getattr(args, "feature_plot_max_points", 3000)
    if limit < 1:
        raise ValueError("feature_plot_max_points must be positive")
    method = getattr(args, "feature_plot_method", "pca")
    if method not in {"pca", "tsne", "both"}:
        raise ValueError("Invalid feature_plot_method")
    return {"method": method, "max_points": limit, "seed": getattr(args, "feature_plot_seed", 42)}


def attach_file_truth(records, report_path):
    if not report_path:
        return [dict(row) for row in records]
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if not isinstance(report.get("files"), list):
        raise ValueError("results_json must be a multi-signal results.json containing files")
    truth = {row["path"]: row.get("true_class_names") for row in report["files"]}
    return [{**row, "true_class_names": truth.get(row["path"], row.get("true_class_names"))} for row in records]


def project_features(features, method, seed):
    """Only numerical features enter projection; labels never enter this function."""
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    n, dimension = features.shape
    if n < 2 or np.allclose(features, features[:1], rtol=0., atol=1e-12):
        return np.zeros((n, 2)), {"method": "degenerate", "note": "One point or identical features; no separability can be displayed"}
    with threadpool_limits(limits=1):
        count = min(2, n, dimension)
        pca = PCA(n_components=count, svd_solver="full")
        coordinates = pca.fit_transform(features)
        if method == "pca" or n < 4:
            if count == 1:
                coordinates = np.column_stack([coordinates[:, 0], np.zeros(n)])
            info = {"method": "pca", "explained_variance_ratio": pca.explained_variance_ratio_.tolist()}
            if method == "tsne":
                info["note"] = "Fewer than four points: PCA fallback"
            return coordinates, info
        from sklearn.manifold import TSNE
        reduced = PCA(n_components=min(50, n-1, dimension), svd_solver="full").fit_transform(features)
        kwargs = dict(n_components=2, random_state=seed, perplexity=min(30., (n-1)/3.),
                      init="pca" if reduced.shape[1] >= 2 else "random", learning_rate="auto", n_jobs=1)
        # sklearn 1.3/1.4 use n_iter; 1.5+ renamed it max_iter.
        kwargs["max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"] = 1000
        model = TSNE(**kwargs)
        coordinates = model.fit_transform(reduced)
        return coordinates, {"method": "tsne", "perplexity": kwargs["perplexity"],
                             "kl_divergence": float(model.kl_divergence_), "pca_input_dimensions": reduced.shape[1]}


def _prediction(row):
    status = row.get("recognition_status")
    if status == "known":
        return row.get("class_name") or row.get("display_label") or "Known", "o"
    if status == "novel_candidate" or row.get("unknown_cluster_id"):
        label = row.get("unknown_cluster_id") or "Unknown_unassigned"
        return label, "^" if row.get("unknown_cluster_id") else "x"
    if status == "uncertain":
        return "Uncertain", "x"
    return row.get("display_label") or row.get("class_name") or row.get("known_candidate_class") or "Unspecified", "o"


def _truth(row):
    names = row.get("true_class_names")
    if names is None and row.get("true_class_name"):
        names = [row["true_class_name"]]
    if isinstance(names, list) and len(names) == 1:
        return str(names[0])
    return "Unlabeled / multi-label file"


def _scatter(path, coordinates, labels, markers, title, subtitle, axes_label):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib import colormaps
    names = sorted(set(labels))
    figure = Figure(figsize=(12, max(6., 1.7 + len(names)*.23)), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    palette = colormaps["tab20"] if len(names) <= 20 else colormaps["hsv"]
    # Prefer the saturated entries before pastel variants for small category sets.
    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    colors = {name: palette(order[i]/19. if len(names) <= 20 else i/max(1, len(names)))
              for i, name in enumerate(names)}
    colors.update({"Uncertain": "#777777", "Unknown_unassigned": "#222222",
                   "Unlabeled / multi-label file": "#aaaaaa"})
    for name in names:
        shown = False
        for marker in sorted(set(markers)):
            indices = [i for i, (label, symbol) in enumerate(zip(labels, markers)) if label == name and symbol == marker]
            if indices:
                ax.scatter(coordinates[indices, 0], coordinates[indices, 1], s=25, alpha=.75,
                           marker=marker, color=colors[name], linewidths=.5,
                           label=textwrap.fill(name, 30) if not shown else None)
                shown = True
    ax.set_title(title + "\n" + subtitle, fontsize=12, pad=12)
    ax.set_xlabel(axes_label + " 1")
    ax.set_ylabel(axes_label + " 2")
    ax.grid(alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.), fontsize=8, frameon=False)
    figure.savefig(path, dpi=170)
    figure.clear()


def write_feature_plots(output, features, records, method="pca", max_points=3000, seed=42):
    output = Path(output)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not features.shape[1] or len(features) != len(records) or not np.isfinite(features).all():
        raise ValueError("Plot records and finite [N,D] features must be aligned")
    if max_points < 1 or method not in {"pca", "tsne", "both"}:
        raise ValueError("Invalid projection settings")
    output.mkdir(parents=True, exist_ok=True)
    summary = {"status": "complete", "unit": "one valid track feature per point; tracks are not independent recordings",
               "total_tracks": len(records), "max_points": max_points, "seed": seed, "requested_method": method,
               "truth_scope": "Single-label FILE truth only; multi-signal instance labels are not inferred",
               "interpretation": "Display only. Coordinates and apparent cluster sizes are not recognition metrics.",
               "plots": [], "projections": []}
    if not records:
        summary.update(status="skipped", reason="No valid track features", plotted_tracks=0)
    else:
        # Stable subsampling by identity; neither prediction nor truth affects inclusion.
        rank = lambda i: hashlib.sha256(f"{seed}|{records[i].get('path', '')}|{records[i].get('track_id', '')}".encode()).digest()
        chosen = sorted(sorted(range(len(records)), key=rank)[:max_points])
        sampled = [records[i] for i in chosen]
        methods = ["pca", "tsne"] if method == "both" else [method]
        summary.update(plotted_tracks=len(chosen), omitted_tracks=len(records)-len(chosen))
        for requested in methods:
            coordinates, info = project_features(features[chosen], requested, seed)
            prefix = requested
            np.savez_compressed(output/(prefix+"_coordinates.npz"), coordinates=coordinates, source_indices=np.asarray(chosen))
            points = [{**row, "source_index": original, "x": float(point[0]), "y": float(point[1])}
                      for row, original, point in zip(sampled, chosen, coordinates)]
            (output/(prefix+"_points.json")).write_text(json.dumps(points, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            info.update(requested_method=requested, points=prefix+"_points.json", coordinates=prefix+"_coordinates.npz")
            summary["projections"].append(info)
            prediction, markers = zip(*[_prediction(row) for row in sampled])
            caption = f"Projection fit: {len(sampled)} / {len(records)} tracks; seed {seed}; {info['method'].upper()}"
            if info["method"] == "pca":
                caption += f"; explained variance {sum(info['explained_variance_ratio']):.1%}"
            def draw(suffix, indices, labels, symbols, title):
                name = prefix+"_"+suffix+".png"
                _scatter(output/name, coordinates[indices], labels, symbols, title,
                         f"{len(indices)} displayed; "+caption, info["method"].upper())
                summary["plots"].append({"file": name, "title": title, "tracks": len(indices), "projection": requested})
            draw("predictions", list(range(len(sampled))), list(prediction), list(markers), "Features by final prediction / recognition status")
            candidates = [i for i, row in enumerate(sampled) if row.get("recognition_status") == "novel_candidate" or row.get("unknown_cluster_id")]
            if candidates:
                draw("unknown_clusters", candidates, [prediction[i] for i in candidates], [markers[i] for i in candidates], "Unknown candidate clusters (same projection as all tracks)")
            truths = [_truth(row) for row in sampled]
            if any(label != "Unlabeled / multi-label file" for label in truths):
                draw("truth", list(range(len(sampled))), truths, ["o"]*len(sampled), "File ground-truth classes (same coordinates; labels used for coloring only)")
    (output/"feature_plots.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return summary


def write_plot_gallery(output, summary):
    """Small standalone gallery for offline plotting and reclustering."""
    from html import escape
    parts = ['<!doctype html><meta charset="utf-8"><title>Feature projections</title>',
             '<style>body{font:15px system-ui;margin:30px auto;max-width:1250px;padding:16px}img{max-width:100%}</style>',
             '<h1>Feature projections</h1><p>Each point is a valid track. These projections do not change clustering or estimate accuracy.</p>']
    for plot in summary.get("plots", []):
        parts.append(f'<h2>{escape(plot["title"])}</h2><img src="feature_plots/{escape(plot["file"], quote=True)}">')
    if not summary.get("plots"):
        parts.append('<p>'+escape(summary.get("reason", "No plots generated"))+'</p>')
    (Path(output)/"feature_report.html").write_text("\n".join(parts), encoding="utf-8")
