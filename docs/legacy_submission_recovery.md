# 旧批次导出与全时长补推理

适用于 `competition_batch_20261107_124532`，训练源为 `all_known_training_20261107_113635`。
两个脚本都只向新目录写入，不修改原推理结果、不训练模型。旧批次已有缓存足以离线重新聚类，不能为没有特征的 `no_signal` 恢复真实标签。

代码在GitHub的temp分支。原目录若仍有实验运行，请等其结束后再切换分支/更新代码，或在独立工作目录使用temp；不要在运行中切换其源码。

## 一、将任务一 no_signal 作为额外类别导出

在仓库根目录、原实验环境中运行：

```bash
REPO=/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code
cd "$REPO"
BATCH="$REPO/experiment/competition_batch_20261107_124532/batch_manifest.json"
TRAIN="$REPO/experiment/all_known_training_20261107_113635"
DATA=/media/kaneki/05E4235537EC7AC2/dataset_wjq

python scripts/export_legacy_submission.py \
  --batch "$BATCH" \
  --data-root "$DATA" \
  --output "$REPO/experiment/legacy_extra_class_submission_20261109"
```

输出目录必须是新目录；若已存在，换一个名称。脚本先对所有未知候选执行谱聚类（邻居10、seed42、n_init10），联合选择任务一类别数，再导出。原已知预测不变；未知类别会统一重新编号。

当前数据任务一得到 `N1=3,N2=4`。随后 `no_signal` 单独占用一个编号：

| 项目 | 真实特征产生的未知簇编号 | no_signal 额外编号 | no_signal 数量 |
| --- | --- | --- | ---: |
| 1-1 | 20–22 | 23 | 741 |
| 1-2 | 20–23 | 24 | 957 |
| 1-3 | 20–22 | 23 | 590 |
| 1-4 | 20–23 | 24 | 790 |

编号按每个数据集的实际聚类结果生成，不跨数据集证明同一未知设备身份。任务二空场景继续使用官方示例的 `[-1,0,0]`，不会凭空添加有频带的信号。

输出包含：

- `submission_no_signal_extra/`：4个任务一工作簿和任务二2×1750个工作簿，字段沿用官方示例。
- `submission_no_signal_extra.zip`：上述文件与校验报告。
- `submission_no_signal_extra_report.json`、`export_summary.json`：实际额外类别编码、覆盖检查和超类清单。
- `source_manifest.json`、`reclustered/`、`no_signal_policy_csv/`：来源散列和处理过程。

这是一版按明确要求生成的非标准导出。额外类别没有信号特征依据，会使任务一未知类数变成4/5，超出公开范围；任务二仍有106/261个文件超过三类。脚本保留这些预测和所有突发行，不通过删除信号、截断类别或改写 `submit.sh` 来掩盖问题。`ready=false,preview=true` 是预期结果。格式无错误时返回0表示文件生成成功，不表示官方要求已满足。

本地快照也可以运行，将 `--batch` 指向下载的清单，使用 `--run-root /path/to/source_snapshot` 重定位各推理目录。没有真实DAT时省略 `--data-root`；报告会明确显示未完成本地原始数据覆盖校验。

## 二、全时长补推理并生成另一版导出

### 1. 准备匹配的 cover profile

`tail_policy=cover,max_slices=0` 才是全时长推理。`fast=true` 仅减少结果文件和图片，不缩短IQ时长；不要把 `--no-fast` 或 `infer.sh --full` 当成时长开关。

旧profile的 `drop` 设置和旧代码已进入特征签名，不能只把推理YAML改为cover后继续使用旧profile。使用同一个20类CNN和同一个YOLO重新提取已知参考/校准特征，不重新训练：

```bash
REFRESH="$REPO/experiment/profile_cover_recovery_20261109"

python scripts/refresh_competition_profile.py "$TRAIN" \
  --output "$REFRESH" \
  --tail-policy cover \
  --wait-for-gpu --gpu-id 0 --gpu-min-free-mib 16384
```

如果已有完成的同训练源cover刷新目录，直接把 `REFRESH` 指向它即可复用。该目录必须包含 `refresh_manifest.json` 且 `status=complete`；不能选择原 `open_set/` 目录。推理脚本会检查原CNN/YOLO散列、当前代码/依赖、profile与配置及全时长覆盖。

### 2. 生成计划（不占用GPU）

```bash
PLAN="$REPO/experiment/full_duration_supplement_20261109"

python scripts/supplement_full_duration.py prepare \
  --batch "$BATCH" \
  --profile-refresh "$REFRESH" \
  --data-root "$DATA" \
  --scope pending \
  --output "$PLAN"
```

`pending` 选中所有 `no_signal` 文件、所有原未知候选所在文件（含已经分簇的候选）、所有超过三类的多信号文件，以及有提取错误/不确定状态的文件。多信号始终重新处理完整文件，不单独替换某条轨迹。

| 项目 | 默认补推理文件数 | 原文件数 |
| --- | ---: | ---: |
| 1-1 | 1117 | 3520 |
| 1-2 | 1454 | 3840 |
| 1-3 | 1115 | 3520 |
| 1-4 | 1450 | 3840 |
| 2-1 | 1070 | 1750 |
| 2-2 | 1135 | 1750 |
| 合计 | 7341 | 18220 |

不能只推3078个任务一无信号文件后与旧未知特征直接拼接：旧特征使用drop，新特征使用cover，签名不同。默认计划同时重提取全部原未知文件，保证最终聚类只有一个新特征签名。未选中的旧已知预测会保留，因此这是一版有逐文件来源记录的混合推理结果。

如希望六项所有决定都使用新profile与全时长配置，把 `--scope pending` 改为 `--scope all`，处理18220个文件。它不重新训练，但耗时更长。

### 3. 执行和续跑

```bash
python scripts/supplement_full_duration.py run \
  --plan "$PLAN/supplement_plan.json" \
  --wait-for-gpu --gpu-id 0 --gpu-min-free-mib 16384
```

默认持续等待GPU可用；无需资源等待时可用 `--no-wait-for-gpu`。该等待是资源准入检查，其他非本项目进程仍可能在之后占用显存。若中断，重新运行同一条命令，已完成且散列一致的项目会跳过；源码、权重、profile、配置、原CSV/缓存或选中文件改变时拒绝续跑。

如果父进程退出后子进程已完成，续跑会恢复其结果；若前次子进程仍报告running，则拒绝重复启动，请先检查该进程。`pending`还要求原始DAT目录的文件名、大小和时间戳与旧批次一致；数据已经更换或重新生成时使用 `--scope all`，避免沿用不对应的旧已知预测。

每项的运行目录、日志、返回指标和散列记录在计划中。原始提取错误和任务二突发标签冲突会终止流程。子集上的未知类别数量暂时不满足范围时可以先保存特征，最终在合并后的完整候选池上重新聚类。

### 4. 合并、重新聚类、严格导出

```bash
python scripts/supplement_full_duration.py export \
  --plan "$PLAN/supplement_plan.json" \
  --output "$REPO/experiment/full_duration_submission_20261109"
```

先按原文件ID整体替换选中文件的所有轨迹与突发，再仅用新的cover候选特征对六项重新聚类。旧未知缓存不会参与拼接；遗漏原未知文件、重复轨迹、缺失文件、签名或散列不一致会报错。

默认严格检查已知映射、任务一类别范围及关系、任务二每文件最多三类、逐段突发、频率和实际DAT覆盖。通过后生成 `submission/` 和 `submission.zip`；若仍有任务一no_signal、超类或其他问题，返回2，保留 `submission_report.json` 和合并结果，不宣称已通过、不输出冒充合格的Excel。

如希望补推理后仍按第一套策略把剩余no_signal当额外类别导出，明确添加 `--no-signal-extra-class`，并换一个新输出目录：

```bash
python scripts/supplement_full_duration.py export \
  --plan "$PLAN/supplement_plan.json" \
  --no-signal-extra-class \
  --output "$REPO/experiment/full_duration_extra_class_submission_20261109"
```

仅校验可在export命令增加 `--check-only`。两种导出均保留多段突发，不把同频多段信号压成一个包络。

## 可预期的限制

原数据只有32个短于100ms的文件存在drop尾段遗漏，不能把3078个任务一no_signal全部归因为尾段处理。新提取也包含当前版本的完整检测/分组逻辑，但不能保证消除未检出或错误类别。严格导出会如实保留这些失败项。

未知到已知的额外重判在本方案中保持关闭；只有单独验证已知恢复和未知误接收率后才应启用新的接受规则。这里没有假定未知簇必须等大，也没有为了三类上限强制删除或归并信号。

## 验证

```bash
python -m unittest tests.test_legacy_submission_recovery tests.test_spectral_inference tests.test_competition_mainline
```

包括额外类别独立编码、数值型Excel读回、任务二空场景、严格导出失败、整文件替换、旧未知遗漏拒绝、重复/缺失记录及缓存变化检查。真实服务器补推理需要用户运行上述GPU命令，本地测试不替代其实际结果。
