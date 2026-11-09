# 二十类模型：六项谱聚类推理与检查

本版本将开发矩阵选出的“原始L2特征＋10邻居谱聚类”接入普通/fast推理、任务一联合数量选择和提交转换。已知端仍使用原Maha++拒识，关闭known_recheck；训练YAML和CNN输入模式沿用既有checkpoint，不需要重训分类器。

谱聚类是两份16已知＋4伪未知开发缓存上的候选。十邻居、种子42、n_init=10，不使用PCA或whitening。对真实多设备场景的迁移仍待本次六项测试验证；聚类图的形状不能替代带真值的准确率。

## 1. 取得temp代码，保留原训练目录

推荐在新的工作目录运行，既有同名目录时换一个名字：

```bash
cd /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code
git fetch origin temp
git worktree add ../code_spectral_infer origin/temp
cd ../code_spectral_infer
git submodule update --init --recursive
conda activate myconda

TRAIN=/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/experiment/all_known_training_20261107_113635
DATA=/media/kaneki/05E4235537EC7AC2/dataset_wjq
REFRESH="$PWD/experiment/profile_refresh_all20_spectral"
```

2026-11-09只读检查确认该TRAIN是完成的二十类all_known实验，权重及旧profile都存在。旧profile与当前代码有4处特征源码签名差异：model/resnet.py、util/eval_common.py、util/group_predictor.py、util/mask_input.py。本轮新增聚类代码本身没有修改这些特征模块，但旧模型跨此前版本迁移时需要重新提取参考/校准；不能改写签名来绕过验证。

## 2. 仅刷新profile，复用原CNN和YOLO权重

```bash
python scripts/refresh_competition_profile.py "$TRAIN" --output "$REFRESH" --dry-run

python -u scripts/refresh_competition_profile.py "$TRAIN" --output "$REFRESH" \
  --wait-for-gpu --gpu-id 0 --gpu-min-free-mib 16384
```

预检只检查配置、文件和权重摘要，不提取信号或训练。实际步骤使用原split_manifest中的520条训练参考、160条独立校准记录，调用profile任务，不运行train；保留原checkpoint、检测器、原训练目录和旧profile。重新提取后核对所有源文件内容摘要仍与原划分一致，失败时不会发布新的推理配置。

默认使用tail_policy=cover、50ms步长、max_slices=0，配套重建profile，覆盖99..100ms转换文件的尾段。保持原drop采样的研究对照可以显式加 `--tail-policy drop`，随后推理也必须使用同一次刷新生成的配置。刷新只拟合已知参考/校准统计，不使用六项官方测试数据。

成功后 `$REFRESH/refresh_manifest.json` 记录新profile路径和对应的 `$REFRESH/inference.yaml`。profile实际保存在当前工作目录下新建的 `experiment/known_profile_时间/known_profile.npz`。若输出目录已经存在，选择新的REFRESH名字，避免覆盖前次记录。

`--wait-for-gpu`会在启动特征提取子进程之前等待资源；等待不占GPU显存。实际提取仍需要CNN推理算力，其耗时不是重新训练耗时。

如果你已有与当前代码完全匹配的二十类profile，可省略刷新，用 `bash infer.sh "$TRAIN" "$DATA" --cluster-method spectral --dry-run` 验证后直接运行。不要使用16类开发模型替代二十类部署模型。

## 3. 预检并运行六项推理

```bash
bash infer.sh "$TRAIN" "$DATA" \
  --profile-refresh "$REFRESH" --cluster-method spectral --dry-run

bash infer.sh "$TRAIN" "$DATA" \
  --profile-refresh "$REFRESH" --cluster-method spectral \
  --wait-for-gpu --gpu-id 0 --gpu-min-free-mib 16384
```

默认运行1-1、1-2、1-3、1-4、2-1、2-2全部六项，不需要手写六份YAML。使用fast模式保存predictions.csv、unknown_features.npz、unknown_clusters.json和诊断；不批量导出每条信号的时频预览。确实需要完整逐切片结果时追加 `--full`，但查看聚类本身用下一步离线工具即可。

上述等待选项在每一项启动前检查GPU空闲显存，不满足时持续排队；默认不加此选项时直接执行。它不暂停已经运行的模型，也不把其他代码错误伪装成等待。

需要单项试运行时增加 `--items 1-1`；这仍只形成一个不完整批次，不能直接提交。正式六项完成后按日志打印的实际 `Batch manifest:` 路径设置BATCH，不能凭目录时间猜测。

| 项目 | 输入聚类单位 | 公开K范围 |
|---|---|---|
| 1-1、1-3 | 每原始记录一条主轨迹，次未知组件继承 | N1=2–3 |
| 1-2、1-4 | 同上 | N2=3–4 |
| 2-1、2-2 | 保留每条未知信号轨迹 | 0–5 |

任务一的联合步骤继续满足相同N1、相同N2及N1<N2。各项只在对应范围内以记录均衡的余弦轮廓分数选择K，不把开发集的真实K=4带入正式测试。任务二K=0只在没有未知候选时成立；非空但证据不足不会被伪装为零类。

谱图按轨迹构建，支持数及几何评分按原始记录均衡；同记录完全重复特征先去重，异类特征保留。最小簇大小5、最少独立记录2及分离门槛0.15仍生效。谱方法对通过检查的分区给出硬分配，并不具备HDBSCAN的原生噪声剔除；已知误拒污染仍可能影响分区。原频率、时间、逐突发坐标保持不变，cluster_membership不伪造为概率。

切换算法/邻居数/种子会绑定到批次清单；恢复同一批次时必须使用原参数。默认不指定谱方案的旧用法仍运行density_merge。

## 4. 一次查看六项聚类

```bash
BATCH="$PWD/experiment/competition_batch_实际时间/batch_manifest.json"
python scripts/inspect_competition_clusters.py "$BATCH" --projection both
```

该步骤只读特征缓存，在CPU上完成任务一联合选N1/N2并生成新的检查目录，不再运行YOLO/CNN，也不覆盖原批次。输出：

- `index.html`：六项入口和联合选择状态。
- `cluster_sizes.csv`：每簇轨迹数、独立记录数和拟合种子数。
- `inspection.json`：每项K候选、轮廓分数、支持数、警告、未分配数量和来源。
- `1-1/feature_report.html` 至 `2-2/feature_report.html`：PCA/t-SNE聚类图；对应points JSON可查每点原文件、轨迹、标签和位置。
- `joint_task1/joint_selection.json`：任务一联合选择结果及新的CSV路径。

如果部分推理失败但已保存缓存，仍可检查现有项；源状态、联合失败及未分配会保留，检查报告不是提交通过证明。`--no-joint-task1`可只看各项独立分区，用于排错。`--projection pca`可缩短绘图时间。不同测试项的Unknown_001不代表同一个型号。

## 5. 校验与导出

```bash
bash submit.sh "$BATCH" --check
bash submit.sh "$BATCH"
```

提交入口读取批次中冻结的谱参数重新做任务一联合选择，不会切回density_merge。它仍要求六项完整、没有未分配/上游错误、类别预算合法、任务二逐突发标签没有冲突。`--check`不写正式工作簿；不合格时保留诊断，不能用占位类别编号绕过。

失败项恢复：

```bash
bash infer.sh "$TRAIN" "$DATA" --profile-refresh "$REFRESH" \
  --cluster-method spectral --wait-for-gpu --gpu-min-free-mib 16384 --resume "$BATCH"
```

仅hash一致且完成的项会跳过。因源码签名或批次源码指纹改变而拒绝恢复时，应建立新批次，不手工修改摘要。

## 手动编辑配置时

`config/competition_spectral_infer.yaml` 是单项模板。核心字段为：

```yaml
unknown_cluster_backend: spectral
unknown_bounded_strategy: spectral
unknown_cluster_policy: auto
spectral_neighbors: 10
spectral_seed: 42
spectral_n_init: 10
```

训练配置保持原样。checkpoint、profile、前端、采样和投票设置必须成套使用，推荐直接使用刷新工具生成的inference.yaml；`--profile-refresh`会自动加载它，避免从通用模板遗漏旧模型的有效设置。无数量范围的开发实验才使用spectral_min_k/spectral_max_k，正式六项由公开profile覆盖其范围。
