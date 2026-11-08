# 任务一、二开集优化：代码与实验方法

本版基于 main `a06c449d9858672ff2300c4992579e9ec52b7f41`。竞赛部署仍为20个已知型号，保持原拒识A、关闭 known_recheck，并沿用范围约束发现。开发实验将部分型号留出，验证表征和拒识的泛化；这与任务三的增量更新实验分开。

## 本版实现

- `classifier_input_mode=mask` 保留原单通道；`mask_amplitude` 使用 `[二值掩膜, 掩膜内标准化幅度谱]`。两通道按完全相同的频率平移居中、最近邻缩放，幅度通道在掩膜外为零。幅度标准化复用现有前端的背景、裁剪和参考带规则。只有原始DAT或科学MAT提供幅度，渲染PNG不能代替。
- 训练、普通推理、fast、特征提取/profile共用输入构建。输入模式与通道数必须匹配 checkpoint；权重/profile签名绑定模式、有效配置及源码。旧单通道权重可继续加载，升级源码后须重建匹配的profile。双通道需要重新训练。
- SupCon第二视图携带原始幅度，不再用空白谱制造双通道视图。可增加 `primary_supcon` 配方；配方覆盖显式训练字段时给出提示。默认主线仍为 `primary_ce`。
- 每轮普通训练写入 `training/result/train_metrics/epoch_N.json`：损失分量、真实优化步数、有效实例数、记录匹配数、耗时。双通道分类输入可视化会额外保存幅度通道。
- 普通及fast推理都写入 `failure_diagnostics.json`：检测框、事件分组、投票、轨迹及最终结果计数。没有事件真值时不把这些计数称为检测率；空场景可以合理地无检测。
- `audit_open_set_predictions.py` 从预测CSV审计逐突发输出并生成本地评价JSONL。任务一把无检测/未分配保留为null弃权；任务二只要有未决标签、缺失突发、上游错误或几何冲突，就不发布部分事件文件。
- `evaluate_competition.py` 支持任务一记录级和任务二事件级评价，显式传入活跃已知ID，输出已知微平均、逐类与宏平均、两种宏F1、未知指标、ARI样本数/覆盖率。频率IoU默认0.5、一对一最大匹配数；时间IoU是单独的可选诊断。

这里的 `feedback_style_score` 只采用反馈中的权重及本地声明的匹配/ARI口径，不宣称复现未公开的官方评估器。任务一使用逐类F1平均；任务二同时报告逐类F1平均与宏P/R调和平均，后者用于反馈风格分数。ARI不足两个有效未知样本、或检测分母未定义时，分数保持null。

## 1. 在服务器准备独立目录

保留正在使用的仓库和实验，在新的工作目录运行。已有同名目录时换一个名称。

```bash
cd /media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code
git fetch origin temp
git worktree add ../code_task12_v3 origin/temp
cd ../code_task12_v3
git submodule update --init --recursive
conda activate myconda

export DATA=/media/kaneki/05E4235537EC7AC2/dataset_wjq/train_dataset
export YOLO=/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code/yolov5/runs/train/exp10/weights/best.pt
```

本版沿用仓库现有依赖，无须为双通道额外安装包。YOLO子模块必须为仓库固定版本。

## 2. 先运行四组开发实验

下例固定留出四种型号，基础模型完全不训练这些类别；不要加载已学习20类的分类器作为该开发实验的起点。所有case保持同一原始记录划分、检测器、主组件策略、批量设置和最大训练轮数。四组为 `mask_ce`、`mask_supcon`、`dual_ce`、`dual_supcon`。

```bash
export SUITE="$PWD/experiment/task12_dev_s42"
python scripts/task12_suite.py prepare \
  --suite "$SUITE" --inputs "$DATA" --yolo-weights "$YOLO" \
  --protocol open_set --seed 42 --epochs 100 \
  --unknown-classes 'DJI_Inspire_2' 'DJI_MATRICE_200' 'DJMiniSE' 'FRSKY X14'

python scripts/task12_suite.py check --suite "$SUITE"
python -u scripts/task12_suite.py run --suite "$SUITE"
python scripts/task12_suite.py compare --suite "$SUITE"
```

`prepare`只写配置，`check`只做配置预检，不运行YOLO或训练，也不保证全部真实数据都合格。需要先检查实际数据划分时，可对生成的任一case执行：

```bash
python main.py experiment --config "$SUITE/mask_ce.yaml" --split_only
```

这一步索引真实文件、计算原始记录哈希并验证分区，会读取原始文件字节，但不训练。查看新实验目录中的 `split_manifest.json`，确认留出型号不在训练/参考/校准集合。

每个case的完整实验地址记录在 `$SUITE/suite.json`；统一对照在 `$SUITE/comparison.json`。已完成case只有在产物摘要仍一致时才跳过；失败case重跑会重新启动该case的实验，不承诺从失败的profile阶段自动续接。也可单独运行：

```bash
python -u scripts/task12_suite.py run --suite "$SUITE" --case dual_ce
```

准备后不要修改生成的YAML、检测器或代码；如需改变条件，创建新suite。比较器核对真实分区、内容摘要、类别集合后才输出结果。固定100轮上限时关闭提前停止，仍保留验证选优；实际优化步数可能受检测失败等因素影响，必须查看 `compute` 中的步数、实例数与训练耗时。原训练器仍按固定中心50ms验证选checkpoint，比较阶段按完整记录推理；不要把它写成已实现逐轮100ms开集选模。

首次用seed42筛选，再在新suite目录用43、44复验；还应轮换留出类别，包含相近型号与较远型号。不要根据官方测试汇总反复挑选全部参数。当前数据已多次用于开发，最终需新记录或明确未参与选型的验证方案。

## 3. 锁定配方后重训20类部署模型

例如开发实验支持 `dual_supcon` 时，以下命令训练全部20类并构建匹配profile；如果其他配方更好，替换case名。

```bash
export FINAL="$PWD/experiment/task12_all20_s42"
python scripts/task12_suite.py prepare \
  --suite "$FINAL" --inputs "$DATA" --yolo-weights "$YOLO" \
  --protocol all_known --seed 42 --epochs 100
python scripts/task12_suite.py check --suite "$FINAL"
python -u scripts/task12_suite.py run --suite "$FINAL" --case dual_supcon
```

从 `$FINAL/suite.json` 中对应case的 `run_dir` 取出完整训练地址。all_known保留原26训练/6验证/8校准的比例，没有内部测试集；不输出虚构测试成绩。部署前确认已知类为20种。

```bash
export TRAIN=/absolute/path/to/all_known_training_时间戳
export TEST_ROOT=/media/kaneki/05E4235537EC7AC2/dataset_wjq
bash infer.sh "$TRAIN" "$TEST_ROOT"
```

使用日志打印的新批次清单地址，先严格检查，再正式导出：

```bash
export BATCH=/absolute/path/to/competition_batch_时间戳/batch_manifest.json
bash submit.sh "$BATCH" --check
bash submit.sh "$BATCH"
```

若清单文件名与示例不同，以批次输出为准。`ready=false`、未分配、同频事件标签冲突或类别预算错误都需要先复核，不能改占位符编号来绕过。

## 4. 审计预测输出

```bash
python scripts/audit_open_set_predictions.py \
  --predictions /absolute/path/to/open_set_inference_时间戳/predictions.csv \
  --task task2 --test-item 2-1 \
  --output experiment/audit_task2_2_1
```

审计输出 `audit.json`、输入文件清单和完整可解析时的 `events.jsonl`。脚本不运行模型；旧轨迹级CSV缺少 `bursts_json` 时会失败，不能凭包络恢复突发。一次审计一个测试项，避免混用批次局部匿名编号。

任务一：

```bash
python scripts/audit_open_set_predictions.py \
  --predictions /absolute/path/to/inference_1_1/predictions.csv \
  --task task1 --test-item 1-1 --output experiment/audit_task1_1_1
```

有未决记录时返回码为2，但任务一仍保存包含null弃权的 `records.jsonl`，用于保留完整分母的开发评估。任务二有任何不完整事件时不保存部分 `events.jsonl`。数据集内文件名重复时使用 `--file-id path`，并使真值的 `file_id` 使用相同完整路径。

## 5. 本地带真值评价

任务一真值JSONL一行一个原始记录：

```json
{"file_id":"sample_0000","class_id":2}
{"file_id":"sample_0001","class_id":3}
```

预测同样一行一个记录，`class_id:null`表示弃权。已知ID遵循官方映射；匿名未知ID从20起。开发留出型号的真值ID可以仍在0至19中，因此必须用 `--known-classes` 列明本轮活跃已知集合。例如仅0、1、2为已知的玩具例子：

```bash
python scripts/evaluate_competition.py --task task1 \
  --truth /absolute/path/to/truth_records.jsonl \
  --predictions /absolute/path/to/records.jsonl \
  --known-classes 0 1 2 --output experiment/local_record_metrics.json
```

任务二真值每行一个物理突发；重复频率的不同时间事件仍需分别标注：

```json
{"file_id":"2-1_0000","class_id":2,"frequency_low_hz":1000000,"frequency_high_hz":2000000,"start_ms":1.0,"end_ms":2.0}
{"file_id":"2-1_0000","class_id":2,"frequency_low_hz":1000000,"frequency_high_hz":2000000,"start_ms":4.0,"end_ms":5.0}
```

```bash
python scripts/evaluate_competition.py --task task2 \
  --truth /absolute/path/to/truth_events.jsonl \
  --predictions experiment/audit_task2_2_1/events.jsonl \
  --output experiment/local_event_metrics.json
```

默认只按频率IoU匹配；额外加 `--time-iou 0.5` 的结果应单独命名，作为时频诊断，不替代官方频率口径。每类输出TP/FP/FN、Precision/Recall/F1，整体输出CDR/FAR及匹配记录。

复杂场景应从已校验的基础IQ事件标注出发，按原始记录先划分，再做时移、频移、幅度缩放、噪声和干扰叠加。覆盖0至3种型号、同频重复突发、跨窗边界与重叠。混合信号必须有逐事件来源与型号，不能给全部框继承一个父目录标签；现有YOLO输出不能直接冒充事件真值。附加噪声参数与真实载噪比估计分别记录。

## 6. 测试与后续实验边界

```bash
python -m unittest discover -s tests -v
UAV_TEST_REAL_YOLO=1 python -m unittest discover -s tests -p test_yolo_native_contract.py -v
```

回归覆盖双通道支持区域与平移一致性、SupCon幅度视图/反向传播、checkpoint兼容性、普通与fast统计一致、逐突发保留/跨窗去重、未决标签阻断、留出ID评价及实验摘要校验。

本版先落地观测/评估与表征对照。检测器复检、可靠性加权汇总、OOD策略解耦、聚类降维等仍需在这些基线上逐项实现或比较，不能把本版回归通过当作它们已取得真实收益。新输入/权重/前端都应重新提取参考、校准和测试特征。
