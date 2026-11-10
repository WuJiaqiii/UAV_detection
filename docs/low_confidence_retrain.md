# 当前结果统一占位类别，以及降低YOLO阈值后的重训与六项测试

本次包含两个独立操作。第一个只处理已经完成的结果，第二个从全20类重新训练识别模型。代码发布到temp；不要覆盖原目录未提交的gcd.sh，也不要在其他实验运行时切换其源码。

## 获取代码并保留原实验目录

服务器原目录有未提交的gcd.sh，建议单独建立工作目录：

```bash
REPO=/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code
git -C "$REPO" fetch origin temp
git -C "$REPO" worktree add --detach "${REPO}_lowconf_20261110" origin/temp
cd "${REPO}_lowconf_20261110"
```

如果这个工作目录已创建，进入它即可，不重复worktree add。使用原myconda环境；数据和现有YOLO权重仍通过绝对路径读取，不复制进Git。

## 操作一：最新补推理结果中的no_signal统一为24

```bash
DATA=/media/kaneki/05E4235537EC7AC2/dataset_wjq

python scripts/export_current_submission.py \
  --source-export "$REPO/experiment/full_duration_submission_20261109" \
  --data-root "$DATA" \
  --no-signal-code 24 \
  --output "$PWD/experiment/current_shared24_20261110"
```

必须使用新输出目录。本命令不启动GPU，也不重新聚类或更改已选中的未知类别；读取联合选择后的最终CSV。

- 任务一四组的所有no_signal均使用同一个数字24，共3078条。已知类别仍为0–19，有特征的未知簇仍从20起编码。
- 任务二空场景保留官方示例的 `[-1,0,0]`。
- 24必须与真实未知簇编码区分。发生碰撞时拒绝输出，避免把空缺和实际信号混成一类。
- 所有频带与逐段突发行保留；不为满足三类限制丢弃轨迹。

输出：`submission_no_signal_extra/`、`submission_no_signal_extra.zip`、同名校验报告及来源散列。共4+1750+1750=3504个Excel。

这版是明确要求的占位导出，`preview=true,ready=false`。任务一增加额外类别后仍超出公开类别范围，任务二原有106/261个超类文件也仍存在。返回0只表示导出成功，不表示满足正式提交条件。

## 操作二：降低检测阈值，重新训练全20类并测试六项

默认对照设置：

| 内容 | 配置 |
| --- | --- |
| YOLO候选框阈值 | 0.05降至0.02；可通过参数改为0.01 |
| YOLO权重 | 沿用exp10/best.pt，重训的是20类识别CNN |
| 训练/验证/校准 | seed42，65%/15%/20%，全部20类已知；不留伪未知类 |
| 训练 | ResNet18，primary_ce，最多100轮（保留默认早停），batch32，lr0.0001 |
| 分组与孤立突发门槛 | 保持原配置，包括singleton_min_confidence=0.5 |
| 校准与推理 | 同一个0.02检测阈值；cover，50ms步长，max_slices=0 |
| 拒识 | 原Maha++ primary，known_recheck关闭 |
| 未知聚类 | spectral，近邻10，seed42，任务一联合数量约束 |
| 测试 | 新模型对1-1至2-2全部18220个文件重新推理，不拼接旧预测 |

降低的是YOLO候选框入口，不能保证新框都会被事件分组接纳。孤立突发的0.5门槛保持不变，这是为了使本轮主要变量是检测入口阈值。更多框也可能增加干扰或任务二超类，因此必须比较检测/分组诊断、验证集表现及最终提交校验。

类别24仅用于导出时占位，不加入CNN的20类训练标签。严格通过时的submission.zip仅包含工作簿，报告保存在外侧；占位版ZIP内包含醒目的报告。

### 先检查参数（不扫描数据，不加载模型，不占用GPU）

```bash
WORK="$PWD/experiment/lowconf_002_20261110"

python scripts/low_confidence_retrain.py \
  --work-dir "$WORK" \
  --data-root "$DATA" \
  --train-root "$DATA/train_dataset" \
  --confidence 0.02 \
  --dry-run
```

配置源文件为 `config/competition_lowconf_train.yaml`。训练端和校准/测试端均显式写入检测阈值，不使用旧bbox缓存、checkpoint或旧profile。

### 启动完整流程

```bash
python scripts/low_confidence_retrain.py \
  --work-dir "$WORK" \
  --data-root "$DATA" \
  --train-root "$DATA/train_dataset" \
  --confidence 0.02 \
  --wait-for-gpu --gpu-id 0 --gpu-min-free-mib 16384
```

默认持续等待所选GPU达到空闲显存要求。16384为MiB，可按GPU容量修改；无需等待时使用 `--no-wait-for-gpu`。资源准入检查不能阻止非本项目程序在之后占用显存。

如需先只生成配置和检查原始数据，可在同一条命令增加 `--prepare-only`；它生成 `$WORK/training.yaml` 与 `workflow.json` 后停止。准备完成后，去掉该参数，用同样的其他参数正式运行。

更激进的0.01对照应使用另一个目录，例如 `lowconf_001_20261110` 并传入 `--confidence 0.01`。不要在同一个WORK中更改阈值后继续沿用已有训练。

### 自动执行的阶段

1. 检查真实20类训练目录、六组DAT数量/长度、权重，并记录配置、依赖版本、代码和输入元数据签名。
2. 新建all_known_training运行；从头训练识别CNN，随后使用本次checkpoint重新提取已知参考/校准特征，建立新profile和匹配的inference.yaml。无需再运行refresh_competition_profile。
3. 验证profile的检测阈值确为0.02（或指定值）、cover及20类映射，然后运行六项全量推理。按运行指针记录确切目录，不猜测时间戳。
4. 联合选择任务一N1/N2，重新聚类完整未知候选池，保留任务二逐段突发。
5. 先写严格检查与可用的正式导出；若严格检查未通过，另外尝试将任务一no_signal统一为24的占位导出。仍有频率、突发冲突等格式错误时不会写出假装有效的Excel。

训练/提取真实失败时立即停止，不会继续启动后续阶段。推理已经完成但含no_signal的needs_review结果会保留并进入明确的导出策略，不会反复占用GPU重新计算。

### 续跑与结果位置

重复同一条启动命令即可续跑：

- 已完成的训练/profile和六项结果通过散列核验后跳过。
- 父进程退出后，仍活跃的工作进程持有工作流锁，避免重复启动。
- 输入元数据、权重、源码、依赖版本或配置改变时要求新WORK，不静默沿用旧结果。
- 中断在训练/profile内部时不承诺自动接续优化器。先检查记录的训练目录；明确接受另起一次训练时添加 `--retry-training`，原运行保留。
- 仅重新生成导出时添加 `--retry-export`，会创建新的results_attempt目录，已完成GPU阶段不重复。

`$WORK/workflow.json`记录实际训练目录、批次清单、阶段日志和最终导出路径。日志如 `training.attempt1.log`、`inference.attempt1.log`，同时在终端显示进度。

```text
WORK/
  training.yaml
  workflow.json
  training.attempt1.pointer.json
  inference.attempt1.pointer.json
  results_attempt1/
    reclustered/
    final_predictions.json
    strict/submission_report.json
    strict/submission/                       # 只有严格检查通过才存在
    shared_no_signal/submission_no_signal_extra/  # 严格检查不通过且占位导出成功时存在
```

工作流状态 `complete` 表示严格导出ready=true；`complete_with_nonstandard_export` 表示实验流程完成，但导出仍含占位或其他竞赛约束问题。后者不是官方合格提交。无真实测试标签时，不以簇数量或更多检测框代替准确率提升结论。

## 验证

```bash
python -m unittest tests.test_lowconf_retrain tests.test_legacy_submission_recovery tests.test_competition_mainline tests.test_run_context
```

包括统一24编码、编码碰撞拒绝、未知未分配标签不被占位策略掩盖、三阶段一致阈值、dry-run不启动GPU、失败停止、完整推理结果复用和阶段续跑。真实重训仍由用户执行上述启动命令。
