# 开发验证与聚类矩阵

## 用已知型号模拟未知类

从训练数据中按型号留出若干完整类别，使用`config/experiment_iq.yaml`：配置known_classes或unknown_classes之一，另一个取补集；按原始记录划分训练、验证、校准与测试。测试目录标签只用于事后评分。正式20类训练使用`competition_train.yaml`，其校准集也不能作为未知类开发测试集。

```bash
bash run.sh experiment --config config/experiment_iq.yaml --dry_run
bash run.sh experiment --config config/experiment_iq.yaml
```

配置路径和真实类名应先按数据目录修改。对照实验至少固定划分、种子、检测器、窗口、输入表示和拒识阈值，分别报告known accuracy/FRR、unknown precision/recall/F1、聚类coverage/ARI/NMI及类别数量误差。记录级别相同内容、切片和导出别名不能跨训练/校准/测试。

## 不重训的聚类矩阵

```bash
python scripts/benchmark_unknown_clustering.py --help
```

使用已完成的带真值开发实验提供的冻结`unknown_features.npz`及对应标签/拒识记录。`config/clustering_benchmark.yaml`列出10种方法及PCA维度、候选K、种子和参数网格；先使用入口的计划选项确认矩阵规模，再正式运行。它不重跑YOLO或CNN。

选择方法时同时看ARI/NMI、覆盖率、缺失记录、噪声比例、最小簇支持、数量误差与时间。条件ARI只衡量成功分簇的那部分未知，丢弃难样本会使它虚高。诊断矩阵可用真值评分；正式测试K必须依赖公开范围和无标签诊断，不能拿真实未知类标签做部署决策。

`config/clustering_spectral_candidate.yaml`保存当前候选范围。选择方法后在新划分/新种子上冻结验证；同一开发集多轮择优只可称开发结论。

## 输入表示与组件对照

`scripts/task12_suite.py`保留mask/mask_amplitude与CE/SupCon的四配置比较。可先查看其`prepare --help`，使用真实原始数据根目录与检测器路径，并指定留出的完整未知类别。该工具与完整20类训练分开，避免把16类开发模型误用于正式提交。

`scripts/audit_open_set_predictions.py`检查未决标签、覆盖和错误来源；`scripts/evaluate_competition.py`按任务单位计算指标；`scripts/inspect_competition_clusters.py`查看六项聚类诊断；`run.sh visualize`从归档特征生成PCA/t-SNE。这些是验证工具，不应只因没有被训练入口导入而删除。

## 失败状态

GPU资源不足可使用等待选项；配置错误和驱动查询错误应修正后再启动。`completed_with_errors`不是干净完成，需检查文件/切片错误和突发冲突。某些聚类约束失败不要求重新训练，可从冻结未知特征重新聚类；输入、特征生产代码或模型改变则需重新提特征及校准。
