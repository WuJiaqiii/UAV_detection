# 实验目录和存储策略

所有单任务使用项目内`experiment/<类型>_<时间>/`。完整工作流另有自己的work目录保存状态、任务指针和最终导出来源，工作流目录不一定包含全部实际训练文件；查找结果应读取manifest。

| 产物 | 用途与保留建议 |
| --- | --- |
| `run.json`、`config.yaml`、有效配置JSON | 参数、版本与运行状态；保留 |
| `split_manifest.json`、标签和记录ID | 复现划分、防泄漏；保留 |
| `training/models/best.pth`或`models/best.pth` | 部署与增量教师；保留 |
| `known_profile.npz`及参考/校准features | 已知拒识复现；保留 |
| `predictions.csv`、`unknown_features.npz` | 提交、离线重聚类和审查；保留 |
| `final_predictions.json`、导出报告与XLSX | 最终输出来源和校验；保留 |
| `result/`中的指标/混淆矩阵与日志 | 结果分析证据；保留 |
| `cache/iq_validation` | 可再生的固定验证频谱；不用于随机训练窗 |
| 逐样本检测图 | 按需开启，保留代表样本，避免每轮全量保存 |

## v3权重

训练产物保存模型、类别/前端配置、epoch和选模指标，不保存优化器与调度器。用于预测及LwF教师，不用于恢复普通训练。新配置中的Path/device转为普通字符串，不再使用旧跨系统pickle补丁。

默认`checkpoint_keep_last=2`：保留最近两个已经保存的epoch快照，以及独立的best。`checkpoint_keep_epochs`中的1-based轮次额外保留；例如需要之后比较第15/20轮，应在训练前指定`checkpoint_keep_epochs: [15,20]`。仅增大save_interval不能避免“每次改善都保存”的累积，保留策略才是最终上限。

清理只作用于当前新训练输出的`epoch_N.pth`文件，不遍历旧实验、不删除best、其他文件或软链接。只要每轮有新保存，非最近且未指定保留的中间权重会被移除。增量实验为事后验证设置了额外epoch清单，详见[incremental](incremental.md)。

默认关闭逐样本训练验证检测图，训练曲线和指标仍保留。需要图时明确开启`--save_val_detect_vis --val_detect_vis_ratio 0.05`，并考虑每类最少采样数带来的数量增长。

## 历史记录

本次代码重构不自动删除历史实验或报告。旧版备份应与当前实验分组，保留最终模型、关键选择轮次、划分、指标及报告来源。移动旧目录会使其绝对路径失效；同盘移动也不会释放空间。不要使用git clean或扩展名通配符对整个项目进行清理。

旧checkpoint/profile/批次/suite没有迁移兼容保证；新版本请使用新工作目录重新训练。新版本内部已完成阶段可按校验值复用，未完成的训练阶段重新从头运行。
