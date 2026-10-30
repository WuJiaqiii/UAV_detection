# 开集特征分布与聚类可视化

开集 experiment、discover、带 profile 的 predict/infer 默认生成 PCA 图片，并嵌入原 report.html；cluster 离线重聚类也生成图片和 feature_report.html。已有实验可使用 visualize 入口补画，不加载 YOLO/CNN，也不重新聚类。

```yaml
discovery: # 完整 experiment / discover 使用此段；infer 可放入 infer 段
  save_feature_plots: true
  feature_plot_method: both  # pca / tsne / both；默认 pca
  feature_plot_max_points: 3000
  feature_plot_seed: 42
```

`--no-save_feature_plots` 可关闭；与频谱检测图片的 save_visuals 相互独立。PCA、t-SNE、抽样仅用于事后显示，原始高维特征仍用于拒识、聚类和评分。PCA 记录二维解释方差比例；t-SNE 先按需 PCA 降至不超过 50 维，固定种子，按点数调整 perplexity。少于四点时 t-SNE 显式退回 PCA；单点/相同向量显示重合点并记录原因。没有有效特征时记录 skipped，不伪造数据点。

## 输出

```text
test/                         # 独立 infer/predict 则是本次运行根目录
  report.html
  inference_features.npz      # 所有有有效特征的测试轨迹，不限于未知候选
  feature_plots/
    pca_predictions.png       # 按最终已知类别、未知簇、uncertain、未归簇着色
    pca_unknown_clusters.png  # 未知候选子集，使用与全体图相同的坐标
    pca_truth.png             # 有单标签文件真值时生成；复用同一坐标
    pca_coordinates.npz      # 坐标与源归档索引
    pca_points.json           # 每个点对应的文件、轨迹、状态、坐标和可用频带
    feature_plots.json        # 方法、种子、总数、抽样数、方差或 t-SNE 参数
    tsne_*.png                # method=tsne/both 时生成同类输出
```

一个点表示一条具有有效特征的轨迹，不是一个脉冲，也不是一个独立原始记录。原始投票不足、无检测/无分组的对象没有有效特征，不强行绘制；报告记录 tracks_without_features。一个 DAT 的遥控、图传链路可以是两个点。绘图超过上限时按文件/轨迹身份和种子确定抽样，既不使用真值，也不根据簇数抽样；NPZ 中仍保留全部有效轨迹。

真值仅用于着色，不进入抽样、投影、拒识或聚类。一个文件具有多个类别时，无法为它的各条轨迹确定实例真值，真值图将其标为 Unlabeled / multi-label file；如果全批次都没有可用的单标签真值，则不生成真值图。单无人机文件标签也不等于逐链路人工标注。

二维图中的间距、团块数量和视觉紧密程度不能替代 unknown_accuracy、clustered_unknown_accuracy、覆盖率和 ARI/NMI；尤其 t-SNE 的全局距离、簇大小不宜直接作协议数量结论。不同运行或不同投影图的坐标轴也不保证可比。

在线推理绘图失败会保留识别结果和原始特征，报告 feature_plot_errors 并返回非零状态；离线命令的绘图异常会作为失败记录在 run.json/error.log。源码显示阶段不加入特征签名，因此仅添加绘图功能不要求重建与当前算法匹配的 profile。更早算法版本的 profile 仍按原有签名检查迁移。

## 给已有实验补图

优先使用新实验的全体轨迹归档：

```bash
bash run.sh visualize --features /path/to/test/inference_features.npz --feature_plot_method both
```

旧实验只有未知候选归档时：

```bash
bash run.sh visualize \
  --features /path/to/test/unknown_features.npz \
  --results_json /path/to/test/results.json \
  --feature_plot_method both
```

results_json 按原文件 path 精确关联文件真值，仅用于补充颜色。旧候选归档只能显示已进入未知池的轨迹，无法恢复被接受为已知的未知信号或无特征对象。也支持参考/校准的 track_features 归档，但其预测颜色表示归档中的分类头候选，不能当成最终开集路由。

输出自动创建 `experiment/feature_visualization_北京时间/`，包含 feature_report.html、results.json、run.json 和 feature_plots；不修改原实验目录。可使用 config/feature_plots.yaml，并修改 features/results_json 路径。`cluster --results_json ...` 同样可以补充真值颜色，聚类算法仍不读取该真值。
