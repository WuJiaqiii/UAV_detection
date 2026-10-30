# 已知训练后的开集识别与离线类别发现

当前默认已升级为 `record_class_knn`：原始记录平衡的多组件参考库、类别尺度和每记录一次校准；默认轨迹拒识不再使用逐观测 60% 否决门槛。具体定义与多链路输出见 [记录平衡参考库](record_balanced_profiles.md)。旧 `global_knn` 保留为消融。

训练阶段只有已知类，未知仅在最终混合测试出现。先冻结编码器，用已知参考库和独立已知校准数据确定距离阈值，然后对跨文件未知候选轨迹做 HDBSCAN / DCG-Louvain。未知比例和类别数不作为输入。

## 一套配置运行

编辑 `config/open_set.yaml` 的权重和数据路径，再选一种方式：

```bash
# 一次运行：参考/校准特征、profile、混合输入发现。
bash run.sh discover --config config/open_set.yaml
# 或先 profile，稍后用同一模型评估新测试；两个任务使用各自的输出目录。
bash run.sh profile --config config/open_set.yaml
bash run.sh predict --config config/open_set.yaml
```

以上是两种替代方式。每次命令自动创建 `experiment/<任务类型>_<北京时间>/`；分步运行时，把 profile 返回的实际 known_profile.npz 路径填入预测配置。profile 默认不保存参考/校准图片；discover 的 `save_visuals` 控制最终测试图片。直接使用核心 Python 脚本也遵循相同目录规则。

配置段 evaluation 对三个特征/推理阶段共用，确保权重、前端、分组、mask、切片、投票和关联一致。profile 段只处理已知参考与校准；discovery 段的 test_labels_json / test_label_mode 只用于混合测试报告，不会进入 profile。

## 数据与独立性

- reference_inputs：已知训练参考，可复用训练集合。
- calibration_inputs：同一组已知类别的独立原始记录；建议不参与 CNN 拟合和选模型。
- test_inputs / predict.inputs：整批已知/未知混合输入，可为目录、多个文件或带引号的通配符；recursive 控制目录递归。

profile 输入必须能为每条轨迹确定一个已知协议真值。使用干净单类别文件；不同协议混在同文件的列表标签不能当成逐轨迹真值。

label_mode 选择 filename（项目 Protocol-[...] 名称）、parent（类别父目录）、json（需要 reference_labels_json 与 calibration_labels_json）。JSON 的相对键以 JSON 所在目录为基准。测试真值可用 test_labels_json / predict.labels_json，也可显式启用文件名/父目录标签；未传真值时精度指标为空。

同一采集的 Figure 切片、SNR 增强、MAT/PNG/DAT 导出必须在同一划分。项目 -Figure-N 会按前缀自动统一身份，无法自动判断的同源数据用 evaluation.recording_ids_json 指定。内容哈希或 recording_id 的参考/校准重叠会报错；测试隔离仍需实验者管理。

每类参考轨迹至少为 knn_k（默认 5）；每类校准独立记录默认至少 20 条。20 是运行下限，不保证稳定估计 1% 尾部。known_false_reject_rate=0.01 是已知校准轨迹的目标误拒比例，不保证部署误拒或未知召回。

class_names / checkpoint 中若保留了 Background，但它没有有效组，可显式 `ignore_classes: [Background]`。其预测成为 uncertain，不进入未知池。若训练排除了 Background，则 `ignore_classes: []`，不能忽略不存在的类别。

## 表示与路由

ResNet18/34 使用分类头前 512 维特征，MobileNetV3-small 使用 1024 维。每个有效观测先 L2 标准化，在轨迹内求均值再标准化；一个轨迹只有一个发现特征。默认 score 为各类别的第 k 个独立参考记录距离除以类别尺度，再取最小值；global_knn 消融使用原始参考轨迹的第 k 近邻距离。

| 状态 | 含义 | 进入未知聚类 |
| --- | --- | --- |
| known | 距离不超过已知阈值，保留原投票类别 | 否 |
| novel_candidate | 轨迹异常分数超阈值；可选 observation_guard 才额外检查观测比例 | 是 |
| uncertain | 投票不足、不稳定、拒识不一致或命中忽略类 | 否 |
| no_signal | 无检测或无有效分组 | 否 |
| error | 数据或流程错误 | 否 |

无检测不是新类别。Unknown_001 是当前批次匿名簇；Unknown_unassigned 是已拒识但未通过聚类/支持条件的候选，不额外算一个类。最终 K 可以为 0 或 1。

## 发现后端与重新聚类

HDBSCAN 使用 sklearn.cluster.HDBSCAN。DCG-Louvain 使用归一化特征的稀疏余弦近邻图，不做测试时白化，不设总 K，不训练网络；它是 AMR_NCD 图发现思想的适配，不是其整个单阶段训练器。

```bash
bash run.sh cluster --config config/cluster.yaml
```

两后端共享 unknown_min_cluster_size（默认 5 条轨迹）与 unknown_min_recordings（默认 2 条独立记录）。同一个 DAT 多条轨迹不增加独立记录数。只有一个 DAT 时可显式降低记录要求探索，但不应据此确认可靠的新类数量。

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| unknown_min_samples | 3 | HDBSCAN 密度尺度，增大通常更保守 |
| unknown_allow_single_cluster | true | HDBSCAN 允许单簇 |
| dcg_topk | 15 | 图近邻数量 |
| dcg_cosine_threshold | 0.85 | 实际余弦门限；降低通常增加合并 |
| dcg_resolution | 1.0 | Louvain 粒度，增大通常更细 |
| dcg_mutual | true | 互为近邻；关闭会增加连通性和错并风险 |
| dcg_seed | 42 | 图社区随机种子 |

完全重复特征合并后构图再展开支持，避免并列近邻随机漏分。无边孤立单点保持未归簇；后端异常明确报错，不伪造 K=1。原型与查询始终在同一个归一化空间。

## 输出、频率与兼容

`tracks.csv` 的 display_label / recognition_status 是最终结果；逐组 class_name 仍为已知头候选。unknown_clusters.json 保存估计 K、支持与代表；unknown_features.npz 保留候选及时间/频率供离线复用；unknown_prototypes.npz 保存归一化中心。cluster_similarity 是余弦相似度，HDBSCAN membership 也不是协议后验概率；图片中的 known p 来自原已知头。

时间和频率使用未居中原始框。rf_center_mhz / rf_centers_json 提供绝对 MHz，否则输出频偏。DAT 默认 100 MHz、50 ms、iq_stride_ms=50；122.88 MHz 数据要在所有阶段统一设置 iq_sample_rate。max_slices=0 表示使用所有可用切片。

profile 绑定权重、有效设置、表示版本及关键源码。本次整理新增 eval_common.py / group_predictor.py 后，旧 profile 需要重新提取和校准；不关闭签名检查。旧模型参数键保持兼容。仅更换发现后端可直接复用同一候选归档，不重新检测，也不修改原 HTML/图片。

原 file_unknown_presence_metrics 的未知 precision/recall 是文件级是否含 novel_candidate。现在额外提供单类别原始记录的主轨迹 All/Known/Unknown ACC、未知匈牙利匹配、ARI/NMI、覆盖率及类别数误差；多信号文件的实例真值仍需额外标注。完整训练/推理入口和指标分母见 [开集实验指南](open_set_experiments.md)。未知簇不自动注册为永久协议名称。
