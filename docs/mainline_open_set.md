# 主线：主分组 CE + Maha++ 适配 + HDBSCAN

正式 `experiment / profile / discover / infer / predict --open_set_profile` 已支持该方案，推理直接读取已保存的统计量和阈值。训练只使用已知类；未知数据和未知类别数量不参与训练或校准。

| 环节 | 主线设置 |
|---|---|
| 检测和分组 | 原 YOLO 权重、hopping + shape，形状容差 0.50 |
| 已知训练 | ResNet18、ImageNet 初始化、224 mask、`train_group_policy=primary`、`supcon_weight=0` |
| 参考统计 | 保留参考记录的有效组件；每条原始记录总权重为 1 |
| 拒识分数 | L2 轨迹特征到类别均值的最小平方 Mahalanobis 距离，共享协方差向各向同性矩阵固定收缩 0.1 |
| 校准 | 独立已知校准集，每条原始记录选一个主轨迹；经验上分位数 `method=higher`，目标误拒率 0.01 |
| 已知类别 | 保留分类头投票类别；最近统计类别只写入 `profile_class_name` 供诊断 |
| DAT 多切片 | 每个有效观测 L2 → 轨迹均值 → L2，再判定轨迹是否未知；默认不附加逐切片一致性否决 |
| 未知发现 | HDBSCAN，min_cluster_size=5、min_samples=3、min_recordings=2、allow_single_cluster=true |

这是项目的记录平衡、小样本协方差收缩适配。其数学实现由正式 profile 和离线 `ablate` 共用；不是原论文基准的直接复现。原有框分组、轨迹关联和频率计算保持原实现。多信号推理继续输出每条轨迹的类别、时间、频率和匿名未知簇。

## 从头运行 DAT 实验

先修改 `config/experiment_iq.yaml`：

- `experiment.inputs`：包含实际类别子目录的 DAT 根目录。
- `experiment.known_classes`：作为已知类训练的实际目录名；不设置 unknown_classes 时，其余发现的类别作为未知测试类。
- `train.yolo_weights`：现有检测权重。

DAT 类名从父目录读取。known_classes 用于实验划分，训练标签 ID 由生成的已知类别空间确定。

```bash
bash run.sh experiment --config config/experiment_iq.yaml
```

该配置已选择 `training_variant: primary_ce`。默认每类 40 条时，训练/验证/校准/已知测试为 20/6/8/6；未知类的记录只进入测试。同一 DAT 的所有切片保持在同一集合。仅检查划分和生效配置可追加 `--split_only`。

运行自动保存在 `experiment/open_set_experiment_<北京时间>/`。主要结果：

```text
training/models/best.pth       # 按已知验证准确率选择
open_set/known_profile.npz     # 均值、精度矩阵、阈值和签名
open_set/reference/features.npz
open_set/calibration/features.npz
test/results.json             # 汇总、主记录级评估及预测
test/tracks.csv               # 轨迹类别/未知簇/时间/频率
test/class_evidence.csv        # 同类别组件证据
test/feature_plots/            # PCA，可配置 t-SNE
inference.yaml                # 已导出的配套无标签推理参数
```

## 训练完成后推理

使用该实验生成的配置，覆盖待测目录即可：

```bash
bash run.sh infer --config experiment/open_set_experiment_YYYYMMDD_HHMMSS/inference.yaml \
  --inputs /path/to/unlabeled_test
```

输入可以是目录、多个 MAT/PNG/DAT 文件或通配符。无标签 infer 不从父目录推断真实类别。分数方法、类别空间和阈值来自 profile，不需要再次传入 profile_score_method，不需要 `ablate`。

新一批输入的未知候选会一起聚类；Unknown_001 等是当前批次匿名簇。无有效组、不足投票、拒识后未入簇有各自状态，不用未知比例或真实 K 强行分配类别。一个 DAT 可以输出多条信号及各自频率；同类别组件汇总不等同于证明它们属于同一物理设备。

## 已有分类器权重时重新校准

编辑 `config/open_set.yaml` 的 checkpoint/yolo/reference/calibration 路径。DAT 默认 `label_mode: parent`，PNG/MAT 协议文件名可改为 filename，也可提供 JSON 标签。

```bash
bash run.sh profile --config config/open_set.yaml
```

把返回的 `known_profile.npz` 和同一 checkpoint、YOLO 权重填入 `config/infer_open_set.yaml`，再运行 infer。需要同时给出带标签测试集时，可以使用 discover。参考和校准必须是独立原始记录，特征设置、切片和投票参数需在所有阶段一致。

主线 profile 参数：

```yaml
profile:
  profile_score_method: maha_pp_shrink
  profile_covariance_shrinkage: 0.1
  profile_calibration_unit: primary
  known_false_reject_rate: 0.01
  min_calibration_per_class: 5
  max_reference_per_class: 2000
```

Maha++ 至少需要每类 2 条独立参考记录；实际应充分利用已知训练记录。max_reference_per_class 限制每类独立参考记录数。knn_k 和 profile_scale_prior_records 用于兼容 kNN 方法，不影响 Maha++ 分数。

主轨迹按有效票数、观测数、路径、轨迹 ID 选择，选择不读取真值。特征归档记录 is_primary；如果选中的主轨迹没有有效特征，会跳过该记录并记录数量，不拿次轨迹替代。校准最低数量检查针对实际有效的校准单位。

1% 是全体有效主校准记录上的经验目标，不是每个类别或每个次组件的误拒保证。多组件推理时应同时查看主记录指标及次组件误拒。每类只有少量校准录波时，尾部阈值仍有波动。record_max 和 record_balanced 可作为显式校准对照。

## 兼容与对照

旧分类器参数键保持兼容，可用已有权重重新生成参考、校准和 profile。旧 profile/特征归档缺少主轨迹标记或源码签名不匹配时，请重新提取并校准。保持签名检查。

旧方法通过显式配置保留：

```yaml
profile:
  profile_score_method: record_class_knn
  profile_calibration_unit: record_max
```

`global_knn` 对应 track 校准；`ablate` 继续支持九种分数和各种校准/聚类对照。修改标准配置中的 supcon_weight 时，若 experiment.training_variant 仍为 primary_ce，该配方会强制使用 primary CE；自定义训练设置需选择 `training_variant: configured`，组件对照则用 all_ce / all_supcon。

HDBSCAN 保留允许单簇：实际未知可能只有一类。关闭单簇的历史诊断只缓解了某些表示下的整体合并，不能保证真实类别数。历史对照见 [组件训练结果](component_training_results_20261103.md)。当前实验配置中未知测试比例仍由类别和文件划分决定，不等同于实际 10% 未知应用场景。
