# 回放增强 LwF 与对比实验

本版本针对 20261108 的 12→15→18 类实验中新类学习不足、旧类占多数导致选模偏差的问题。实现的是项目适配的 replay LwF，不宣称复现 DER++、SRIL 或 Smart Replay。参数是待验证的初始配置，不能预先保证准确率提高。

## 训练变化

- 原始 LwF 的默认行为保留：不回放、所有输入均进行旧类 logits 蒸馏、温度 4、蒸馏系数 10。
- 新配置每个旧类保留 4 条独立的**训练原始记录**。每个完整批次默认一半新记录、一半旧记录；旧记录按类别轮换。每轮全部新文件恰好遍历一次，最后一个批次可较小。重复回放时仍重新随机截取 50 ms IQ，不缓存随机裁剪的检测框。
- 所有样本参与全局分类头的交叉熵。新配置只在旧样本上进行旧类 logits KL 蒸馏，并对旧样本增加归一化特征余弦约束。新增类别不必模仿旧教师对新信号的错误解释。
- 分类头学习率 1e-4，骨干学习率 1e-5；保留 BN 运行统计，BN 仿射参数仍可学习。教师始终冻结并处于 eval 模式。

对于新数据 D 和旧回放 M，训练目标为：

\[
L=L_{CE}(D\cup M)+\lambda T^2\mathbb{E}_{x\in M}
  KL(p_T^{teacher}(x)\Vert p_T^{student,old}(x))
  +\mu\mathbb{E}_{x\in M}[1-\cos(f_s(x),f_t(x))].
\]

默认改进配置为 T=2、λ=1、μ=0.1。KL 的两个概率分布均在旧类别集合内归一化；旧类与新类的全局竞争由混合样本的 CE 监督。若配置 all-component 训练，蒸馏也使用每条原始记录的组件权重。

## 选模与数据隔离

`checkpoint_selection: old_new_hmean` 使用旧类别宏平均召回率与本轮新增类别宏平均召回率的调和平均。验证分母包含检测与分组漏失，因此不能通过丢弃困难样本提高选模分数。原有 `acc` 指标仍记录，但不再作为该配置的最佳模型选择标准。每个活跃类别必须有验证记录；至少训练 30 轮后才允许早停。

每次保存的 checkpoint 增加 `selection_metric`、`selection_score`、`validation_metrics`，`best_val_acc` 表示已选最佳模型的原始验证准确率。每轮验证 JSON 包含新旧类别召回、调和平均和每类计数。不要把 `selection_score` 当作测试准确率。

启用回放必须提供包含 `records` 列表的 JSON。每项需有 `path`、`class_name`、`split`、`recording_id`、`source_sha256`。允许 split 为 train/val/calibration/test，只有旧类 train 能成为回放候选。新类训练路径和全部验证路径必须出现在相应分区。身份或内容哈希存在别名时进行合并；跨分区或跨标签的原始记录会直接报错。每类预算按合并后的记录数计算，并验证选中文件的 SHA256。

训练输出 `result/replay_selection.json`，记录清单哈希、选中文件、记录身份和批次配置。校准及测试文件始终不参与梯度更新。构建开集档案仍使用该阶段已知类的训练参考集与独立校准集；每个新 checkpoint 必须重新构建档案。

## 推荐的同划分 A/B 实验

在服务器仓库根目录、myconda 环境中执行。请先将这次 temp 提交更新到服务器的工作分支，保留已有实验目录；如存在本地修改，应先保存，避免强制覆盖。

```bash
cd /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code

WORK=experiment/lwf_replay_20261108_s42_m4

python scripts/lwf_replay_experiment.py prepare \
  --source-suite experiment/lwf_supplement_20261108_s42 \
  --work "$WORK" --replay-per-class 4 --epochs 100 --batch-size 32

python scripts/lwf_replay_experiment.py check --work "$WORK"

python -u scripts/lwf_replay_experiment.py run --work "$WORK"
```

`prepare` 复用上一轮**完全相同的记录划分、类别顺序、随机种子与 12 类基础模型**，不会加载包含 20 类知识的分类器。基础模型训练无需重跑。`check` 只检查真实文件、清单、配置和标签，不执行训练、YOLO 推理或新数据实验。

运行顺序为：重建基础模型档案并测试；A 第一阶段；B 第一阶段；A 第二阶段；B 第二阶段。各模型分别重建开集档案并在同一测试集上评估。A 为原始新类输入蒸馏（λ=10、T=4、无旧梯度回放）；B 为上文回放增强方案。**两组都使用新的调和平均选模规则**，所以 A 与上一轮按总准确率选模的结果不完全相同。

脚本会跳过已完成且哈希未变化的作业，中断后可再次运行相同的 `run`。代码、检测器或源数据变化后应新建 work 目录。可用 `train --work "$WORK" --model replay_1` 等命令分步执行，但第二阶段必须先有对应第一阶段模型。

结果写入 `$WORK/comparison.csv`、`comparison.json` 与 `suite.json`；各训练、profile、predict 的独立实验地址在 suite 中。重点检查初始 12 类、第一次新增 3 类、第二次新增 3 类准确率，以及剩余未知类 F1、误接收率、聚类覆盖与条件 ARI。未检测到信号的文件保留在准确率分母中。

每轮实际优化步数、新旧实例曝光数量和损失分量保存在训练目录的 `result/train_metrics/epoch_N.json`。另修复了原 DCG 后端在部分 NumPy 环境中因 `unique(return_inverse=True, axis=0)` 的索引维度差异而产生二维标签的问题，兼容一维与列向量索引。

可把 `--replay-per-class 4` 改为 8，并使用新的 work 目录进行预算比较。只从训练记录中选样，不根据测试结果调整超参数。回放使每轮更新次数增加，且本版本同时改变学习率、BN 与蒸馏；这是整套策略 A/B，对单个组件的贡献需后续消融，并同时报告时间与更新步数。它仍是使用真实文件夹标签的增量分类实验，不能作为自主未知类标注注册已经完成的证据。

## 单独使用训练配置

修改 `config/train_incremental_replay.yaml` 的路径与实际类别名称后：

```bash
python main.py train --config config/train_incremental_replay.yaml
```

`train_dataset_path` 可以只包含新类训练文件，旧类来源由回放清单提供；验证路径必须包含全部已见类别。IQ 的类名发现仍依据显式训练、验证目录共同完成。默认参数保持兼容原配置；启用回放仅支持 IQ、显式 train/val、`sample_ratio=1`、批量大小至少 2。
