# 六项竞赛流程

## 数据与配置

训练根目录按20个官方型号命名，每类目录存放原始`.dat`；测试根目录包含`1-1/`、`1-2/`、`1-3/`、`1-4/`、`2-1/`、`2-2/`。已知型号到提交编码的映射在导出器中独立维护，不能使用checkpoint的类别索引代替官方0–19编码。

默认`config/competition_train.yaml`：随机种子42、验证15%、校准20%、余下训练；划分按原始记录身份/内容分组，禁止同一记录的不同导出跨集合。训练使用主分组CE，固定YOLO权重，置信阈值0.02；检测阈值同时应用于训练、参考库、校准与推理。拒识为Maha++收缩协方差、主轨迹校准；未知发现使用谱聚类和公开数量范围，关闭known_recheck。

## 一次完成

```bash
python scripts/competition_workflow.py \
  --work-dir "$PWD/experiment/competition_v3_s42" \
  --train-root /path/to/train_dataset \
  --data-root /path/to/competition_data \
  --yolo-weights /path/to/detector.pt \
  --confidence 0.02 --wait-for-gpu --gpu-min-free-mib 16384
```

`--prepare-only`只生成工作流和配置；`--dry-run`检查配置，不读取原始数据或启动模型。资源不足时等待GPU；查询失败、驱动错误或要求显存超过总量会直接报告，避免无意义等待。

工作流打印实际训练目录和批次manifest，不能按“最新时间戳”猜测来源。重复同一命令只复用本版已完成且校验通过的阶段；未完成训练需要`--retry-training`明确重训，`--retry-export`会在新目录重新导出。这种阶段复用不恢复训练优化器，也不迁移旧版实验。源码、检测器或输入清单发生变化时使用新work目录。

## 分步执行

训练模板路径改好后：

```bash
bash run.sh experiment --config config/competition_train.yaml
bash infer.sh /path/to/new_all_known_training /path/to/competition_data \
  --cluster-method spectral --wait-for-gpu --gpu-min-free-mib 16384
bash submit.sh --batch /path/to/new_batch/batch_manifest.json \
  --output /path/to/new_export
```

训练生成的`inference.yaml`是实际有效配置。`config/competition_infer.yaml`只是单项输入模板；不要用手工覆盖切片/分组设置来绕过已校准profile。任务二保留同频不同时间的每个物理突发及其频率范围。全时长为`tail_policy=cover`、`max_slices=0`，包括实际略短于100ms的对齐IQ文件。

批次`needs_review`可能表示前向推理完成但含未分配/no_signal行；不应只因状态不是complete而反复推理。上游文件/切片错误和突发冲突则必须先处理。批次缺项、仍在运行或原始文件/输出哈希变化时，导出器会拒绝。

## 严格提交与人工占位

默认严格导出。报告中`ready=true`才代表内置格式、覆盖和数量约束检查通过；这不代表官方识别率已达标。

如果明确需要将任务一no_signal统一编码为24：

```bash
bash submit.sh --batch /path/to/new_batch/batch_manifest.json \
  --output /path/to/new_preview --no-signal-code 24
```

这是一份非标准预览，`ready=false`，随包附报告。24只是人工占位编号，没有信号证据，不能据此声称发现了新型号。任务二无信号保留官方`(-1,0,0)`空场景表示。严格任务二输出是每个样本一个Excel，所有突发逐行输出；已知类别索引和未知匿名编号分别处理。

已完成的本版联合聚类结果可直接重新导出，不再推理或重新聚类：

```bash
bash submit.sh --source-results /path/to/results_attempt1 \
  --output /path/to/reexport
```

目录必须有本版`final_predictions.json`及匹配的CSV校验值。单个CSV到XLSX的低层入口为`python prepare_submission.py --help`；日常六项提交流程统一使用`submit.sh`。
