# 开集对照实验

`ablate` 在同一完成的开集实验上冻结检测、分组、轨迹和分类器，分开比较类别输出、校准单位、拒识分数及聚类。输出固定放在 `experiment/open_set_ablation_<北京时间>/`，不提供自定义输出目录参数。不会修改输入实验。

## 直接运行

```bash
bash run.sh ablate --config config/ablation.yaml \
  --source_experiment /path/to/open_set_experiment_YYYYMMDD_HHMMSS
```

配置的 `replay_masks: true` 会读取该实验的 `training/models/best.pth`，用已保存分组框重建二值 mask，从而补齐原始 logits 和归一化前特征。这里不需要原始 DAT 或 YOLO 权重；不支持由框重建频谱幅度/IQ 输入。checkpoint 必须是生成旧特征的同一个文件，通过 SHA256 检查。只加载你信任的项目 checkpoint。

需要的文件：

```text
source_experiment/
  recording_ids.json
  training/models/best.pth
  open_set/reference/{features.npz,results.json,slices.jsonl}
  open_set/calibration/{features.npz,results.json,slices.jsonl}
  test/{inference_features.npz,results.json,slices.jsonl}
```

`inference_features.npz` 必须包含全部有效轨迹，不能用仅含未知候选的 `unknown_features.npz` 替代。当前特征可视化流程默认保留该归档。归档需来自同一个权重、前端、分组和轨迹配置；参考、校准和测试记录不得重叠。对照指标需要有标签测试实验；真实标签不输入分数拟合、阈值或聚类。

CPU 与原 GPU 输出存在小数值差异。脚本校验重放概率与已保存概率、重放的归一化轨迹向量与旧向量，超过容差则停止。距离方法始终使用原有归一化特征；原始 logits/原始特征仅供需要它们的方法使用。原始投票标签与轨迹保持固定，以隔离拒识因素。重放中的 argmax 差异、概率误差、向量误差记录在 `audit.json`。

重放一次后可以复用缓存：

```bash
bash run.sh ablate --config config/ablation.yaml \
  --source_experiment /path/to/original_experiment \
  --replay_cache /path/to/previous_ablation/replay_cache --no-replay_masks
```

已有独立重放目录时，传入其中含 `reference.npz`、`calibration.npz`、`test.npz`、`classifier_head.npz` 的目录即可。缓存会再次核对签名、轨迹顺序、原有归一化特征和权重哈希。

## 实验矩阵

默认完成 86 组：R0/R1 两组；九种分数 × 三种校准单位 × 三个目标误拒率，共 81 组；固定候选池的三组聚类。所有配置均输出，不根据测试准确率自动选择方法或阈值。

| 组 | 改变什么 | 保持什么 |
|---|---|---|
| R0 | 无，重新评分保存的原结果 | 原类别、拒识、簇分配 |
| R1 | 被接受为已知时恢复原分类头投票类别 | 原拒识决定和簇分配 |
| R2 | 同一 `record_class_knn` 的校准单位/目标误拒率 | 模型、特征、距离、分类头和 HDBSCAN 参数 |
| R3 | 不同拒识分数 | 模型、输入轨迹、标签策略及配置中的校准目标 |
| R4 | HDBSCAN、DCG、一小步平滑后 HDBSCAN | 同一个预先指定的拒识候选池，保存并核对候选池哈希 |

三种校准单位：

- `primary`：按有效投票数、观测数、路径、轨迹 ID 选择主轨迹，与主指标相同。主轨迹缺少特征时记录跳过，不拿次轨迹替代。
- `record_max`：每条独立记录的所有有效组件分数取最大，再拟合高分位数，保持当前部署 profile 的校准定义。
- `record_balanced`：所有有效轨迹进入经验分布，但每条记录的总权重为 1；使用加权经验 CDF 的逆分位数。

`primary` 与 `record_max` 使用 NumPy 的 `quantile(..., method="higher")`。目标误拒率是已知校准集上的经验目标，不能当作部署保证。尤其每类只有约 8 条校准记录时，应查看重复划分的波动。

## 分数定义和论文适配

所有分数越大越未知，拒识条件统一为 `score > threshold`。统计量只拟合已知参考记录，阈值只拟合独立已知校准记录。

| 名称 | 定义 | 适配说明 |
|---|---|---|
| `record_class_knn` | 每类第 k 个独立记录距离除以该类参考尺度，再取最小 | 复用部署 `RecordingClassBank` 和相同的参考尺度拟合 |
| `record_class_knn_unscaled` | 相同类内独立记录近邻距离，不除类别尺度 | 与上一行配合隔离尺度影响 |
| `record_knn` | 全库第 k 个独立记录距离；同记录多组件取最近距离 | 独立记录支持量，不是第 k 个切片 |
| `global_knn` | 全库第 k 个轨迹欧氏距离 | 保留旧全局 kNN 对照 |
| `maha_pp_shrink` | L2 特征的类别均值 + 共享协方差 Mahalanobis 距离 | 按记录赋权，固定 0.1 向各向同性协方差收缩；是小样本适配版本 |
| `maxlogit` | 原始轨迹平均 logits 的最大值取负 | 使用真实 logits，不由 softmax 猜测 |
| `energy` | 原始轨迹平均 logits 的 logsumexp 取负，T=1 | 与所有方法共用同一校准机制 |
| `nnguide_record` | 查询 logsumexp 乘参考能量加权余弦近邻的平均指导值，再取负 | 同一记录最多提供一个近邻，属于记录平衡适配 |
| `vim` | 已知主空间外残差 × 参考尺度 − 已知 logsumexp | 原始特征/对应线性头；参考主空间默认 64 维，参考矩按记录赋权 |

轨迹原始特征和 logits 都取其原先接受观测的平均，因此对应线性头的关系仍成立；原有近邻特征仍为“逐观测 L2 → 平均 → L2”，二者不混用。ViM 的分数是虚拟未知 logit 的 log-odds，与原方法虚拟未知概率单调对应。小样本协方差收缩、记录平衡和轨迹汇总均为本项目适配，不应直接声称复现论文基准。

参考：[Deep kNN](https://proceedings.mlr.press/v162/sun22d.html)、[MaxLogit 强基线](https://arxiv.org/abs/2110.06207)、[NNGuide 官方实现](https://github.com/roomo7time/nnguide)、[Mahalanobis++ 官方实现](https://github.com/mueller-mp/maha-norm/)、[ViM 官方实现](https://github.com/haoqiwang/vim)。

R4 的 `smooth_hdbscan` 只做一次小幅近邻平滑，近邻来自不同原始记录；默认 k=5、混合比例 0.2。它不是完整的 CMS 对比训练或 GPC 拆分合并，不能以这两篇论文的方法名报告。R4 池由配置的 `record_knn / primary / 0.05` 预先固定，禁止从测试结果自动挑出“最好”的候选池。

## 查看输出

- `report.html`：完整矩阵、图、每组指标链接。
- `summary.csv` / `summary.json`：所有组的参数及已知/未知/端到端指标。
- `per_class.csv`：各真实类别的记录数、未知检测数、聚类数、匹配正确数。
- `scores/*.npz`、`scores/*.json`：校准与测试分数、仅由已知数据拟合的阈值及参数。
- `cases/<case>/assignments.csv`：每条轨迹的类别、拒识、簇编号、时间与频率位置。
- `cases/<case>/evaluation.json`：包括无信号在内的主记录级评分。
- `cases/<case>/clusters.json`：簇支持量、原型、候选池哈希。
- `audit.json`：输入来源、记录独立性、模型签名和重放误差。

ROC/AUPR/OSCR 仅使用有有效分数的主轨迹；端到端准确率保留无信号记录。匿名未知簇与真实类别的一一匹配只用于评分。不同候选集的条件聚类准确率不能单独比较；须同时看覆盖率与未知端到端准确率。

`prior_adjusted_*` 指标把拒识结果重新赋权为 10% 未知先验，不改变实际聚类池，不能当作真实 10% 混合批次的聚类实验。

## 需要重新训练的组件对照

```bash
bash run.sh experiment --config config/component_ablation.yaml --training_variant primary_ce
bash run.sh experiment --config config/component_ablation.yaml --training_variant all_ce
bash run.sh experiment --config config/component_ablation.yaml --training_variant all_supcon
```

先修改数据和 YOLO 路径。三组使用相同 seed 和记录划分：

1. `primary_ce`：只训练原主分组，CE。
2. `all_ce`：训练单设备记录的所有非空最终分组，CE。
3. `all_supcon`：同上，增加 0.1 权重的两视图监督对比损失。

对每个 DAT 采样输入，其组件 CE 总权重为 1；SupCon 的锚点、正样本和分母也按组件记录权重处理，避免组件多的记录主导训练。验证仍使用原主分组；推理保留多信号和频率输出。默认 `train_group_policy=primary`，已有训练行为保持不变。

`all` 仅适用于已确认单无人机的训练记录；多个目标或其他设备干扰不能自动共享同一真值。当前仅支持 normal 已知类训练，不与 LwF 蒸馏混用。所有分组先经过既有分组器的质量过滤；若最终分组含明显伪组，应先审核数据，不把它们强行加上设备类别。

每组完成后再用 `ablate --source_experiment <该组输出>` 比较同样的拒识配置。编码器改变后不能复用旧参考/校准特征。原始 DAT、YOLO 权重和训练算力不在结果缓存中，冻结实验不代表这三组已完成完整训练。

本版本抽取了共享参考尺度函数；由于 profile 签名包含该源文件，正常推理使用历史 profile 时应按新代码重新生成 profile。历史实验的离线对照不依赖当前代码签名，也不会改写旧 profile。

## 2026-10-31 实测：19:10 历史实验

同一原模型、319 条有效已知参考记录、127 条有效已知校准记录、96 条已知测试记录与 160 条未知测试记录。原始归一化特征用于所有距离方法。九种方法、三个校准单位和三个目标误拒率的完整矩阵均运行，不按测试标签调参。

以下新分数组统一使用主轨迹校准、目标误拒率 1%、HDBSCAN；R0/R1 保持原校准与聚类。目标相同不代表测试误拒相同。

| 方法 | 已知准确率 | 实际已知误拒 | 未知召回 | 未知端到端准确率 | 估计簇数 |
|---|---:|---:|---:|---:|---:|
| R0 原结果 | 85.42% | 1.04% | 29.38% | 23.13% | 5 |
| R1 仅恢复分类头标签 | 90.63% | 1.04% | 29.38% | 23.13% | 5 |
| 类别尺度距离 | 90.63% | 1.04% | 47.50% | 41.25% | 7 |
| 独立记录全局 kNN | 90.63% | 1.04% | 49.38% | 38.13% | 10 |
| Mahalanobis++ 记录平衡/收缩适配 | 91.67% | 0.00% | 97.50% | 85.00% | 10 |
| MaxLogit | 91.67% | 0.00% | 54.38% | 22.50% | 1 |
| Energy | 91.67% | 0.00% | 45.00% | 20.63% | 1 |
| NNGuide 记录平衡适配 | 91.67% | 0.00% | 52.50% | 25.00% | 1 |
| ViM 轨迹适配 | 90.63% | 1.04% | 88.13% | 77.50% | 10 |

Mahalanobis++ 适配的逐未知类别结果：DEVO 检出/最终正确 40/40；AVATA2 检出 39、最终正确 22；MAVIC3 PRO 检出 40、最终正确 37；X14 检出/最终正确 37/37。仍存在 AVATA 碎簇、少量噪声和组件模式过多的问题。估计 10 个簇不能当作找到了 10 个真实设备类别。

上述“已知误拒 0”仅指主轨迹指标：该方法的未知池还有 14 条来自已知类别文件的组件轨迹，并非所有链路均零误拒。10 个簇中 AVATA 占多个簇，MAVIC3 PRO 也被拆成 3 个簇；应继续审核组件质量和跨链路表示，而不是为了使 K=4 直接调参合并。

固定 R4 候选池中，HDBSCAN 未知端到端准确率为 85.00%，DCG 为 71.25%，小步平滑后 HDBSCAN 为 66.88%；这批数据不支持直接采用该平滑方案。

CPU 重放的最大轨迹 L2 误差约 0.00225，最大概率误差约 0.00186；参考观测中 1 个 argmax 与旧 GPU 输出不同，校准和测试观测为 0。原始投票标签全部保留。Mahalanobis++ 使用旧特征，因此其结果不依赖这部分重放数值差异。

这些是已被反复观察的测试集上的描述性对照，尚未经过新的伪未知类别划分或独立采集场次检验；本次 0/96 的已知误拒不能解释为部署误拒率为零。完整组件重训、频谱双分支和无标签 GCD 自适应尚未运行。
