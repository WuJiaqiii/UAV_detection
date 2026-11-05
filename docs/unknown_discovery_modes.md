# 单信号与多信号未知发现

当前默认 `unknown_cluster_policy: auto`。HDBSCAN 的密度邻居按独立原始记录计数；出现单个大簇时，只有在存在足够支持、覆盖和几何分离的子簇时才采用子簇。真实只有一个未知类时仍可以返回 K=1；不传入真实类别数或未知比例。

## 两种输入假设

| 参数 | single | multi（infer/predict/cluster 默认） |
|---|---|---|
| unknown_input_mode | 每条原始记录只有一个设备/类别，可包含遥控、图传等组件 | 一条记录可以有多个不同信号/类别 |
| 聚类拟合单位 | 每条记录的一条主轨迹 | 全部有效未知候选轨迹 |
| 主轨迹选择 | 在全部轨迹中按有效票数、观测数、路径、轨迹ID选择；不使用真值 | 无需过滤主轨迹 |
| 次组件 | 同记录的未知次轨迹继承已入簇主轨迹的簇ID；已知和不确定组件保留原判定 | 独立参与发现，保留各自类别、频率和时间 |
| 输出 | single_signal_prediction 给出记录级主判定；signals 保留全部轨迹 | signals 保留每条独立轨迹 |

没有自动根据轨迹数量猜测输入假设：同一无人机的遥控/图传也会形成多条轨迹。无法确认每条记录只有一个设备时使用 multi。

这里的 `unknown_input_mode` 与旧 `signal_mode` 不同。开集和频率输出仍使用 `signal_mode: multi` 的轨迹引擎；不要将旧闭集文件投票器的 `signal_mode: single` 用于开集任务。

如果选中的主轨迹被判为已知、未通过投票或缺少特征，single 不把某条次轨迹提升为主轨迹。相应次候选保持 Unknown_unassigned，不据此新增类别。single 假设不会强行改写已被接受为已知的组件类别。

## 无标签推理

沿用现有配套 checkpoint、YOLO 和 Maha++ profile：

```bash
# 单设备/单类别记录组成的一个批次
bash run.sh infer --config experiment/open_set_experiment_实际时间/inference.yaml \
  --inputs /data/unlabeled --recursive --unknown_input_mode single

# 同一文件可能含多个不同信号
bash run.sh infer --config experiment/open_set_experiment_实际时间/inference.yaml \
  --inputs /data/unlabeled --recursive --unknown_input_mode multi
```

或修改 `config/infer_open_set.yaml`：

```yaml
infer:
  inputs: [/data/unlabeled]
  open_set_profile: /models/known_profile.npz
  unknown_input_mode: multi  # single / multi
  unknown_cluster_policy: auto
  unknown_cluster_backend: hdbscan
  unknown_allow_single_cluster: true
  unknown_min_cluster_size: 5
  unknown_min_samples: 3
  unknown_min_recordings: 2
```

一次输入整个待测批次，使未知候选有独立记录支持。只输入一条文件仍可得到已知分类或未知候选；默认支持量不足时保持 Unknown_unassigned，不把一次采集确定为新类别。single 的 min_cluster_size=5 要求至少5条主记录种子。

`experiment` 对单设备带标签记录默认选择 single；config/experiment_iq.yaml 已显式设置。训练后导出的 inference.yaml 在 infer 段显式选择 multi，以保留部署场景的全部信号；需要单信号时用上述参数覆盖。训练、Maha++参考拟合、阈值校准和现有轨迹关联没有改变。

## 防止全部并入一个簇

auto 的 HDBSCAN 流程为：

1. L2轨迹特征构成欧氏距离矩阵。计算某条轨迹的核心距离时，同一个原始记录只提供一个最近组件距离；min_samples 包含查询自身的原始记录，小批次有效邻居数不超过实际记录数。
2. 根据这些核心距离构造相互可达距离，再做 HDBSCAN。多信号轨迹都保留，不使用频率近邻或类别真值进行过滤。簇原型按簇内原始记录等权求均值再L2。
3. 若只产生一个有效簇，计算不选全局根簇的层次候选。仅当至少有两个子簇分别达到轨迹/独立记录支持、记录平衡覆盖率不低于0.5、记录平衡 silhouette 不低于0.25时，采用该层次。
4. 证据不足时保留单簇。噪声、支持不足、缺少主轨迹上下文都不伪造成额外类别。后端异常明确报错。

上述常数与实际诊断均写入 unknown_clusters.json 的 diagnostics.single_cluster_check；不使用真实K，不按本批次标签合并/拆分。silhouette 只在原始L2特征上检查；超过2000点时固定种子抽样计算这个诊断，聚类拟合仍使用全部选中轨迹。

独立记录核心距离减少了组件重复带来的密度偏置，但多信号层次的稳定性和语义类别数仍受组件模式影响。该几何检查不能保证估计簇数始终等于真实设备类别数。距离矩阵为O(N²)内存，应在大规模批次中关注候选数量。

DCG仍可选择，支持相同single/multi采样和支持量规则；HDBSCAN核心距离及单根检查只作用于HDBSCAN。保留历史对照：`--unknown_cluster_policy legacy --unknown_input_mode multi`。config/ablation.yaml 显式保留legacy，以复核已发布历史矩阵。

## 不重新训练，重聚类已有实验

本次仅改变发现及报告，不改变被profile绑定的特征源码和分数。当前主线Maha++ profile在相同权重、前端、采样及运行环境下可直接复用，仍执行原有签名检查。

旧 unknown_features.npz 可直接用 multi；single 需要完整源 results.json 提供全部轨迹上下文。新归档已保存discovery_is_primary及主轨迹来源，可不提供源报告完成single聚类。

```bash
bash run.sh cluster \
  --features /path/to/open_set_experiment_20261104_074313/test/unknown_features.npz \
  --results_json /path/to/open_set_experiment_20261104_074313/test/results.json \
  --unknown_input_mode single --unknown_cluster_policy auto

# 多信号策略的归档对照
bash run.sh cluster \
  --features /path/to/open_set_experiment_20261104_074313/test/unknown_features.npz \
  --results_json /path/to/open_set_experiment_20261104_074313/test/results.json \
  --unknown_input_mode multi --unknown_cluster_policy auto
```

结果自动另存至 experiment/unknown_clustering_时间/。完整源报告同时允许保存重新评分后的open_set_metrics.json、reclustered_results.json；真值只在发现结束后评分或着色。Linux源数据路径在离线操作中只是身份元数据，无需原始DAT或网络权重文件可访问。原实验报告及预览不覆盖，离线结果不重新绘制原谱检测框。

## 输出解释

- unknown_candidate_tracks：全部被拒识的候选轨迹数。
- fit_tracks / clustered_fit_tracks：实际用于拟合的种子数及入簇种子数。
- inherited_secondary_tracks：single模式按同记录假设继承簇的次候选数。
- discovery_role：track、recording_primary或recording_secondary。
- cluster_assignment_source：direct或recording_primary。继承的次轨迹没有HDBSCAN membership，不能把主轨迹的密度置信度当成次轨迹的独立证据。
- single_signal_prediction：single的记录级主判定，不覆盖signals中的坐标及组件级证据。

## 20261104_074313归档回查

使用同一训练模型和阈值，仅重聚类：legacy/multi为1簇、未知端到端25.00%；auto/single为6簇、91.25%；auto/multi保留483条候选，得到9簇、88.75%。已知准确率88.54%和未知检出98.75%不变。

这是观察本测试集后进行的修正与回查，真实未知仍为4类。auto/multi在这里是在单设备记录数据上验证轨迹策略；真实多设备同场性能需要相应的实例标注或额外数据评估。该结果不替代新的随机种子、未知类别划分和实际10%未知批次验证。
