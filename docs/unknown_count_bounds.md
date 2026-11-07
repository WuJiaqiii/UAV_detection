# 利用公开未知类别数量范围：受限密度簇合并

本更新应用在 known_recheck 版本之上。默认仍是原 average 策略；只有显式选择 density_merge 才启用新方法。修改在独立源码目录完成，没有修改或推送远程 main，没有训练新模型或重跑 YOLO/CNN。

## 为什么需要另一种受限策略

既有 competition_profile 已提供公开数量范围：task1_n1 为 2–3，task1_n2 为 3–4，task2 为 0–5；任务一的四项还要满足共享 N1/N2 与 N1<N2。

原 bounded average 方法直接对全部候选做指定 K 的层次聚类。少量异常点可能先占据一个分支，导致真实类别被合并；范围合法不等于型号正确。使用当前 B 缓存、仅给定 [3,4] 时，原方法选择 K=3，总准确率降到 66.32%。K=4 的分区含单记录簇，不能通过支持数检查。

新方法先保留密度支持充分的核心簇，再在允许的数量范围内比较合并结果，不按簇大小直接删类或无条件并入最近中心。

## 算法

1. 使用原有独立记录密度 HDBSCAN，得到受支持核心簇；同一记录的相同特征组件先去重。噪声仍为未分配，不伪装成新类别。
2. 每一轮枚举当前所有簇对，计算合并后整个分区的记录均衡余弦轮廓系数，选择最高者。这里使用整体分区质量，而不是只比较中心距离。
3. 保存允许范围内每个可达 K 的候选；每个候选使用 5 次按原始记录保留 85% 的子采样重跑密度提取与合并，计算与原分区的 ARI，一并考虑共同覆盖率。
4. 使用现有评分形式：silhouette + 0.1*stability - 0.02*(K-1)，只在合法且有支持的候选中选择。任务二 K=1 使用现有固定参考分数 0.35。
5. 最终仍保留原最小簇大小、独立记录数、分离程度和非退化中心检查。联合任务一继续比较 (N1,N2)=(2,3)、(2,4)、(3,4)。

合并路径是贪心层级搜索，不是对所有可能分区的数学全局最优求解。当前实现最多处理 32 个初始核心簇；超过会明确失败并给出原因。轮廓诊断最多使用固定随机种子的 2000 个规范化组件样本，所有成员仍获得输出标签。

新方法只合并核心簇，不会凭空拆出不存在的类别。密度核心数低于公开下界时明确失败。它不保证消除所有小簇：当小簇受到支持且保留它的整体评分更好时，仍可以保留。最小簇大小 5 是开发规则，不是官方类别出现次数。

## 当前缓存的开发演示

输入为同一伪未知类别留出实验，288 个独立测试记录。下表中的真值只用于完成选择后的评分；选择函数只读取冻结特征、记录 ID 与公开范围 [3,4]，不传入精确 K=4 或测试标签。

| 方案 | 估计未知类数 | 总准确率 | 未知误收率 |
|---|---:|---:|---:|
| A：原拒识＋无范围聚类 | 5 | 75.35% | 25.00% |
| A：原拒识＋新范围策略 | 4 | 77.43% | 25.00% |
| B：已知复核＋无范围聚类 | 5 | 74.31% | 28.13% |
| B：已知复核＋新范围策略 | 4 | 76.39% | 28.13% |

B 的 25 条与 6 条两个核心簇在候选 K=4 下被合并，最终主记录簇大小为 31、39、14、32；合并后的 31 条中仍有一条来自其他未知型号，已知误拒混入和未知误收也仍存在。因此数量约束改善了这次簇分裂，没有解决所有识别问题。

更值得下一轮验证的是 A＋新范围策略，暂不启用已知复核。本轮已参与算法设计，应视为开发演示，不能作为独立最终性能结论。后续须冻结策略，用新的类别留出/独立测试划分验证。仅改变随机种子能检查记录划分敏感性，不能代替不同未知类别的验证。

## 安装

将发布包的 UAV_density_bounds.patch 应用在上一版 known_recheck 代码的新分支上：

```bash
git switch -c codex/density-bounds
git apply --check /path/to/UAV_density_bounds.patch
git apply /path/to/UAV_density_bounds.patch
```

不需要新增第三方依赖。本次没有修改任何被 inference_signature 绑定的特征源码；与当前 known_recheck 代码匹配的模型/profile 可继续使用。不代表更早的 a436201 profile 自动兼容 known_recheck 版本。

## 先复用现有 A 缓存验证

在服务器原环境、仓库根目录执行：

```bash
BASE="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/experiment/open_set_experiment_20261107_085954"

bash run.sh cluster \
  --features "$BASE/test/unknown_features.npz" \
  --predictions_csv "$BASE/test/predictions.csv" \
  --results_json "$BASE/test/results.json" \
  --competition_profile task1_n2 \
  --unknown_input_mode single \
  --unknown_bounded_strategy density_merge \
  --no-save_feature_plots
```

输出自动位于新 experiment/unknown_clustering_时间戳。unknown_clusters.json 记录合法候选、K 评分、密度核心数、噪声数、合并路径及失败原因；open_set_metrics.json 是事后评分；predictions.csv 是新分配结果。merge_path 包括探索较小 K 的路径，最终使用 selected_K 对应的分区，不是执行整条路径直到最少 K。

这里的 task1_n2 仅为本开发演示提供 [3,4] 的范围，未声明它是正式测试项 1-2，因此没有设置 test_item。若使用 B 缓存，把三个路径改到 B 的 unknown_features.npz、predictions.csv、results.json 即可；B 已误收为已知的记录不会通过重聚类恢复。

本步骤不重训、不运行模型、不覆盖原输入。A 源数据仍有事件标签冲突；离线主记录聚类完成不等于事件提交有效。示例 CSV 使用训练数据集文件名，不是正式竞赛提交文件。

## 真实六项推理

在已经能够正常运行的当前推理配置/命令中追加：

```bash
--unknown_bounded_strategy density_merge
```

它要求有界 competition_profile、unknown_cluster_backend=hdbscan、unknown_cluster_policy=auto。平均链接和非受限模式仍保持原行为，错误组合会在开始前拒绝。

测试项对应规则不变：

| 项目 | competition_profile | unknown_input_mode |
|---|---|---|
| 1-1、1-3 | task1_n1 | single |
| 1-2、1-4 | task1_n2 | single |
| 2-1、2-2 | task2 | multi |

任务二 [0,5] 是上限约束，不能据此推断每批必须有 5 类，也不保证必然合并一个仍在合法范围内的小簇。空候选仍允许 0 类。

## 任务一联合选择

在原 submit.sh 的联合选择命令中追加 --bounded-strategy density_merge，保证联合步骤继续使用同一策略：

```bash
python scripts/recluster_competition_batch.py \
  --item 1-1 "$R11/unknown_features.npz" "$R11/predictions.csv" \
  --item 1-2 "$R12/unknown_features.npz" "$R12/predictions.csv" \
  --item 1-3 "$R13/unknown_features.npz" "$R13/predictions.csv" \
  --item 1-4 "$R14/unknown_features.npz" "$R14/predictions.csv" \
  --bounded-strategy density_merge \
  --min-cluster-size 5 \
  --min-recordings 2 \
  --min-samples 3 \
  --min-silhouette 0.15 \
  --output "$JOINT"
```

R11–R22 和 JOINT 按原说明填写实际目录，JOINT 使用新目录。若开发时明确采用其他支持数门槛，需要保持推理和联合步骤一致。导出继续读取 joint_selection.json 指向的最终 CSV，并要求严格 ready=true。未解决的噪声不会因数量范围而自动获得标签。

## 验证

新增测试覆盖有界核心合并、稳定编号、独立记录去重、噪声保留、证据不足失败、相同特征不能伪造类别、固定 K 边界、错误组合、实际密度后端与联合 CSV 输出。全部回归共 314 项：313 项通过，1 项实际 YOLO 环境测试跳过。另用原始缓存进行了无模型的数值演示；没有启动远程训练或新射频推理实验。
