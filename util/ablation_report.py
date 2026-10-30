"""CSV/JSON, standalone scientific figures and a local comparison report."""
import csv
from html import escape
import json
from pathlib import Path

import numpy as np


def write_report(output, summary, evaluations):
    output = Path(output)
    rows = summary["cases"]
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    columns = ["name", "stage", "method", "unit", "target_frr", "backend", "known_accuracy", "known_false_reject_rate",
               "unknown_detection_recall", "unknown_cluster_coverage", "clustered_unknown_accuracy", "unknown_accuracy",
               "all_accuracy", "ood_auroc", "ood_aupr_unknown", "oscr_scored", "estimated_unknown_class_count"]
    columns += sorted(set().union(*(r.keys() for r in rows)) - set(columns))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["case", "class_name", "is_unknown", "recordings", "detected_unknown", "clustered",
                  "known_correct", "unknown_correct"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for name, evaluation in evaluations.items():
            mapping = evaluation["metrics"]["unknown_cluster_to_truth_for_scoring_only"]
            for truth in sorted({r["true_class_name"] for r in evaluation["samples"]}):
                samples = [r for r in evaluation["samples"] if r["true_class_name"] == truth]
                writer.writerow({"case": name, "class_name": truth, "is_unknown": samples[0]["is_unknown"],
                    "recordings": len(samples), "detected_unknown": sum(r["recognition_status"] == "novel_candidate" for r in samples),
                    "clustered": sum(bool(r["unknown_cluster_id"]) for r in samples),
                    "known_correct": sum(not r["is_unknown"] and r["predicted_class"] == truth for r in samples),
                    "unknown_correct": sum(r["is_unknown"] and mapping.get(r["unknown_cluster_id"]) == truth for r in samples)})
    _plots(output, rows, evaluations)
    visible = [r for r in rows if r["stage"] in {"R0", "R1", "R4"} or r["unit"] == "primary"]
    def percent(value):
        return "—" if value is None else f"{100*value:.2f}%"
    header = ["实验", "已知准确率", "已知误拒", "未知召回", "聚类覆盖", "条件聚类准确率", "未知端到端", "估计 K"]
    lines = ["# 冻结模型开集对照实验", "", "阈值只由独立已知校准记录拟合。所有配置均报告，不按测试标签自动选出部署方法。",
             "", "完整矩阵见 summary.csv；每组的 predictions、指标和聚类组成见 cases。", "",
             "| " + " | ".join(header) + " |", "|" + "---|"*len(header)]
    table_rows = []
    for row in visible:
        values = [row["name"], *[percent(row.get(k)) for k in ("known_accuracy", "known_false_reject_rate",
                   "unknown_detection_recall", "unknown_cluster_coverage", "clustered_unknown_accuracy", "unknown_accuracy")],
                  str(row["estimated_unknown_class_count"])]
        lines.append("| " + " | ".join(values) + " |")
    for row in rows:
        values = [f'<a href="cases/{escape(row["name"])}/metrics.json">{escape(row["name"])}</a>',
                  *[percent(row.get(k)) for k in ("known_accuracy", "known_false_reject_rate", "unknown_detection_recall",
                       "unknown_cluster_coverage", "clustered_unknown_accuracy", "unknown_accuracy")],
                  str(row["estimated_unknown_class_count"])]
        table_rows.append("<tr>"+"".join(f"<td>{v}</td>" for v in values)+"</tr>")
    lines += ["", "R4 在完全相同的候选池上比较聚类；smooth_hdbscan 是一小步邻域平滑，不是完整 CMS/GPC 复现。",
              "", "10% 先验调整指标只重新加权拒识结果，并未把聚类批次变为 10% 未知。",
              "", "未运行："+"；".join(summary["not_run"]), ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>开集对照实验</title>
<style>body{font:15px system-ui;margin:32px;line-height:1.6;color:#20262d}table{border-collapse:collapse;width:100%}
th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}
img{max-width:100%;display:block;margin:20px 0}a{color:#096c9b}.scroll{overflow:auto}</style>
<h1>冻结模型开集对照实验</h1><p>所有分数以“越大越未知”为方向。参考库统计只拟合已知参考记录，
阈值只拟合独立已知校准记录，测试真实标签仅用于最后评分。此测试集已用于前期诊断，结果应另用伪未知开发划分及独立数据验证。</p>
<p>R0 保留原结果；R1 只恢复分类头标签；R2 改变校准单位；R3 比较分数；R4 固定候选池比较聚类。
不同目标误拒率必须同时查看实际已知误拒，不能只按未知召回排名。</p>
<p>10% 先验调整 AUPR/精确率只重新加权拒识结果，未改变聚类池。无信号记录保留在端到端分母中；ROC/OSCR 仅含有有效分数的记录。</p>
<p><a href="summary.csv">完整 CSV</a> · <a href="per_class.csv">逐类别结果</a> · <a href="audit.json">来源与重放误差</a> · <a href="summary.json">完整 JSON</a></p>
<img src="performance.png" alt="各阶段性能"><img src="roc.png" alt="拒识曲线">
<h2>完整配置矩阵</h2><div class="scroll"><table><thead><tr>"""
    html += "".join("<th>"+escape(h)+"</th>" for h in header)+"</tr></thead><tbody>"+"".join(table_rows)+"</tbody></table></div>"
    html += "<p>未运行的训练方向："+escape("；".join(summary["not_run"]))+"</p>"
    html += "<p>跳过的方法："+escape(json.dumps(summary["skipped_methods"], ensure_ascii=False))+"</p></html>"
    (output / "report.html").write_text(html, encoding="utf-8")


def _plots(output, rows, evaluations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    primary = [r for r in rows if r["unit"] == "primary" and r["target_frr"] == .01 and r["stage"] != "R4"]
    if not primary:
        primary = [r for r in rows if r["unit"] == "primary" and r["stage"] != "R4"]
    primary = list({r["method"]: r for r in primary}.values())
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), layout="constrained")
    selected = rows[:2] + primary
    labels = [r["name"] if r["stage"] in {"R0", "R1"} else r["method"] for r in selected]
    x = np.arange(len(selected))
    for delta, key, label in ((-.18, "known_accuracy", "Known accuracy"), (.18, "unknown_accuracy", "Unknown end-to-end")):
        axes[0, 0].bar(x+delta, [100*(r.get(key) or 0) for r in selected], width=.36, label=label)
    axes[0, 0].set_xticks(x, labels, rotation=38, ha="right", fontsize=8)
    axes[0, 0].set(title="Primary calibration, 1% target (or first configured rate)", ylabel="Accuracy (%)", ylim=(0, 105))
    axes[0, 0].legend(fontsize=8)
    for row in primary:
        axes[0, 1].scatter(100*(row.get("known_false_reject_rate") or 0), 100*(row.get("unknown_detection_recall") or 0), s=45, label=row["method"])
    axes[0, 1].set(xlabel="Actual known false rejection (%)", ylabel="Unknown recall (%)", title="Equal calibration target does not ensure equal test cost", ylim=(0, 105))
    maximum_frr = max([100*(r.get("known_false_reject_rate") or 0) for r in primary] or [1.])
    axes[0, 1].set_xlim(-.1, max(2.2, maximum_frr*1.9))
    if primary:
        axes[0, 1].legend(fontsize=7, loc="lower right")
    for unit in ("primary", "record_max", "record_balanced"):
        group = sorted([r for r in rows if r["method"] == "record_class_knn" and r["unit"] == unit and r["stage"] != "R4"], key=lambda r:r["target_frr"])
        axes[1, 0].plot([100*r["target_frr"] for r in group], [100*(r["unknown_detection_recall"] or 0) for r in group], marker="o", label=unit)
    axes[1, 0].set(xlabel="Target known false rejection (%)", ylabel="Unknown recall (%)", title="R2: same class-scaled distance, calibration unit changes", ylim=(0, 105))
    axes[1, 0].legend(fontsize=8)
    clustering = [r for r in rows if r["stage"] == "R4"]
    x = np.arange(len(clustering))
    for delta, key, label in ((-.24, "unknown_cluster_coverage", "Coverage"), (0., "clustered_unknown_accuracy", "Conditional accuracy"), (.24, "unknown_accuracy", "End-to-end")):
        axes[1, 1].bar(x+delta, [100*(r.get(key) or 0) for r in clustering], width=.24, label=label)
    axes[1, 1].set_xticks(x, [r["backend"]+f" (K={r['estimated_unknown_class_count']})" for r in clustering], fontsize=8)
    axes[1, 1].set(title="R4: identical candidate pool", ylabel="Percent", ylim=(0, 105))
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(output / "performance.png", dpi=170)
    fig.savefig(output / "performance.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for row in primary:
        samples = [s for s in evaluations[row["name"]]["samples"] if s["ood_score"] is not None]
        y = np.asarray([s["is_unknown"] for s in samples])
        if len(set(y)) != 2:
            continue
        fpr, tpr, _ = roc_curve(y, [s["ood_score"] for s in samples])
        for ax in axes:
            ax.step(100*fpr, 100*tpr, where="post", label=f"{row['method']} ({row['ood_auroc']:.3f})")
    axes[0].set(xlim=(0, 100), ylim=(0, 102), title="Scored primary recordings: full ROC")
    axes[1].set(xlim=(0, 10), ylim=(0, 102), title="Low known false-rejection region")
    for ax in axes:
        ax.set(xlabel="Known false rejection (%)", ylabel="Unknown recall (%)")
        ax.grid(alpha=.2)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=7, loc="lower right")
    fig.savefig(output / "roc.png", dpi=170)
    fig.savefig(output / "roc.pdf")
    plt.close(fig)
