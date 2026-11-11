# 从头运行增量识别实验

入口为`lwf.sh`，所有实验直接从类别文件夹构建，不再依赖旧的all_known实验、旧suite或20类分类器。文件夹提供人工真值，属于可评价的增量分类器实验，不等于已经完成自动命名、数据库治理和在线回滚等整个任务三系统。

## 准备和执行

```bash
bash lwf.sh prepare \
  --data-root /path/to/train_dataset \
  --yolo-weights /path/to/detector.pt \
  --work "$PWD/experiment/incremental_v3_s42" \
  --seed 42 --class-counts 12 15 18 \
  --methods finetune lwf replay \
  --wait-for-gpu --gpu-min-free-mib 16384
bash lwf.sh check --work "$PWD/experiment/incremental_v3_s42"
bash lwf.sh run --work "$PWD/experiment/incremental_v3_s42"
```

`prepare`只索引和哈希文件，不训练CNN。默认按原始记录划分训练45%、验证15%、校准20%、测试20%；比例取整时以实际manifest为准。每类40个独立记录时分别为18/6/8/8。至少保留5个独立校准记录，回放预算不得超过每类独立训练记录数。类顺序按种子确定，12→15→18之外的类别始终作为未知测试类；原始20类数据相应保留2类。

原始记录有多个不同内容的导出时，用`--recording-ids /path/to/recording_ids.json`指定共同ID，避免划分泄漏。`protocol.json`记录类顺序与各集合数量；校准记录、验证记录和测试记录从不加入梯度回放。

## 比较方法

| 方法 | 梯度训练数据与约束 |
| --- | --- |
| finetune | 仅新增类，蒸馏权重0，作为遗忘对照 |
| lwf | 仅新增类，温度4、logit蒸馏权重10，作为普通LwF对照 |
| replay | 旧训练记录回放＋旧样本logit蒸馏＋特征保持，主线方法 |

base只训练一次。每个比较方法的第二阶段必须使用该方法第一阶段的新模型作为教师，不共享其他方法的增量教师，也不加载20类教师。冻结base的预测输出可作为后续阶段对照依据；比较时必须按同一阶段的已知/未知类集合重算指标。

主线replay保留此前较好实验的设置：第一阶段每旧类4条、旧样本比例0.5；第二阶段每旧类8条、比例0.8；温度2、logit蒸馏系数1、特征保持系数0.1、仅旧回放样本承受蒸馏、新样本蒸馏权重0、骨干学习率为分类头的0.1、冻结BN统计量。可用`--replay-per-class 4 8`和`--replay-fraction 0.5 0.8`显式改变预算。

训练以新旧类宏召回调和平均选择best；完整时长profile/测试读取该best。旧实验的“第20轮最优”不能作为新数据、种子的固定结论。为便于另做完整100ms验证，默认额外保留5/10/15/20/30/35/59/60轮快照；`--keep-epochs`可改写该列表。验证集选模后才在测试集评价，禁止按测试分数挑epoch。

只跑主线时用`--methods replay`，可将训练数量降至base、replay_1、replay_2三次。需要新种子验证时使用新work目录，依次运行43、44等种子，报告均值与离散程度。

## 输出和阶段操作

`suite.json`保存任务目录和完成产物哈希；`jobs/`保存生成的配置与日志；`comparison.json`汇总各模型实际已知类数、旧/新/整体准确率和开集指标。旧类保持率应与base比较，新类效果应与同阶段finetune比较，聚类ARI同时查看覆盖率，不能只看条件ARI。

```bash
bash lwf.sh train --work /path/to/suite --model replay_1
bash lwf.sh evaluate --work /path/to/suite --model replay_1
bash lwf.sh summarize --work /path/to/suite
```

同一新suite可复用已完成阶段，锁会传给子进程以防父进程退出后重复启动。数据、代码、准备文件或完成产物变化会阻止继续复用。训练失败重启该阶段会从头训练，不恢复优化器。
