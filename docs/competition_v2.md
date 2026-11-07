# 竞赛版本2.0：使用与迁移

本文保留 v2.0 的历史迁移和底层手动入口说明。当前 temp 已集成后续修复，完整训练、六项推理、联合选类和提交请以 [当前主线说明](competition_mainline.md) 为准，不再应用本文的旧补丁。当前任务一事件诊断不再直接阻止主记录运行，任务二仍严格检查事件冲突。

基线：main `47ab9d5f54f2f70d3d082798a4536e062cc14d23`。修改位于独立的`codex/competition-v2`本地分支，未修改或推送远端main。本版替代上一轮独立转换器：prepare_submission.py是唯一维护实现，export_submission.py使用相同参数调用它。

## 完成的修改

- 显式竞赛协议：task1_n1的K=2..3、task1_n2的K=3..4、task2的K≤5；保留unconstrained通用开集模式。
- 在冻结特征上比较受限K候选，记录独立支持、分离程度和记录级重采样稳定性；不压缩编号、不使用未知真值挑K，证据不足明确失败。
- 完整infer、fast、离线重聚类与提交共享突发归一：同频分离事件逐条保留，重复观测去重，符合窗口边界证据的连续片段拼接，冲突标签保留诊断。
- 检测器保留置信度，physical模式提供独立单突发接受路径、可配置时间上下文分割和可选轨迹特征门槛。
- source频带策略在分类/聚类之前限制DAT真实可观测带宽，保留122.88MHz重采样网格，不盲目缩放导出频率。
- 完整开集infer新增与fast一致的predictions.csv；fast可保留轻量未知特征缓存，离线聚类核对并更新原CSV。
- 提交严格检查格式、类别预算/关系、每文件类别数、时间、频率、覆盖率、占位和上游错误；XLSX读回校验后发布。仅显式预览允许不ready输出。
- 新增事件级一对一匹配和CDR/FAR、已知/未知F分数、带覆盖率的ARI诊断；修复CI过时入口、依赖和触发路径。

## 安装

在原仓库创建新分支，应用补丁：

```bash
git switch -c codex/competition-v2 47ab9d5f54f2f70d3d082798a4536e062cc14d23
git apply --check /path/to/UAV_detection_v2.patch
git apply /path/to/UAV_detection_v2.patch
git submodule update --init --recursive
```

若分支名已存在，使用另一个新分支名。完整源码ZIP也可解压使用，内含固定YOLO运行源码与许可证，不含模型权重、数据集或第三方图片/二进制演示资产。YOLO固定为17cfa633999107857a337d51c9c8271fef97edc7；补丁保持原子模块版本。

推荐Python3.11，在自己的环境安装requirements。转换脚本仅使用标准库；模型、聚类、评估使用仓库依赖。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

设置UAV_TEST_REAL_YOLO=1可启用需要完整YOLO依赖的实际后端张量契约测试。常规测试用可重复模型/检测器替身验证链路，不代表真实检测性能。

## 模型与profile迁移

特征源码与设置签名已有变化，旧profile会被拒绝。必须在独立已知参考/校准集上重建profile，不可绕过签名检查。physical策略改变了分类输入，建议重新训练官方20个已知类并建立匹配profile。

修改config/competition_v2_train.yaml中的数据/YOLO路径后运行：

```bash
bash run.sh experiment --config config/competition_v2_train.yaml
```

将同一新实验的checkpoint、profile、相同YOLO路径填入config/competition_v2_infer.yaml。不要把16已知+4伪未知开发模型当作最终20类模型。若先用legacy策略过渡，也需为v2源码重建profile；该路径不会获得physical的短突发处理变化。

单突发置信度0.5、对比门槛1.0、上下文间隔5ms、轨迹特征门槛0.8和分离门槛0.15是可配置的开发起点，未用新射频实测证明最优。请在伪未知开发划分及逐事件真值集上验证。

## 六个测试项

每次输入一个完整测试项，不混合独立批次的匿名簇：

| 测试项 | competition_profile | unknown_input_mode | test_item |
|---|---|---|---|
| 1-1 | task1_n1 | single | 1-1 |
| 1-2 | task1_n2 | single | 1-2 |
| 1-3 | task1_n1 | single | 1-3 |
| 1-4 | task1_n2 | single | 1-4 |
| 2-1 | task2 | multi | 2-1 |
| 2-2 | task2 | multi | 2-2 |

```bash
bash run.sh infer --config config/competition_v2_infer.yaml \
  --inputs /data/1-1 --competition_profile task1_n1 --unknown_input_mode single --test_item 1-1
bash run.sh infer --config config/competition_v2_infer.yaml \
  --inputs /data/2-1 --competition_profile task2 --unknown_input_mode multi --test_item 2-1
# 完整模式同样输出canonical predictions.csv，并保留详情
bash run.sh infer --config config/competition_v2_infer.yaml \
  --inputs /data/2-2 --competition_profile task2 --unknown_input_mode multi --test_item 2-2 --no-fast
```

本配置启用save_discovery_cache，因此fast会额外保存unknown_features.npz。完整模式还保留bursts.jsonl、results.json、slices.jsonl等。候选不足、无受支持分区或出现事件标签冲突时，退出码2且运行状态为completed_with_errors，诊断仍保存。

查看candidate_K_scores、selected_K、constraint_satisfied、detection_decisions、burst_diagnostics。no_signal不进入未知聚类池。任务二允许空场景，不能将所有空文件算作漏检；单文件0–3种型号不限制突发条数。

## 离线联合选择N1/N2

以下命令使用各次推理匹配的缓存和canonical CSV，不重跑YOLO/CNN：

```bash
python scripts/recluster_competition_batch.py \
  --item 1-1 /runs/a/unknown_features.npz /runs/a/predictions.csv \
  --item 1-2 /runs/b/unknown_features.npz /runs/b/predictions.csv \
  --item 1-3 /runs/c/unknown_features.npz /runs/c/predictions.csv \
  --item 1-4 /runs/d/unknown_features.npz /runs/d/predictions.csv \
  --output /runs/new_joint_clustering
```

joint_selection.json列出最终CSV路径。只比较(2,3)、(2,4)、(3,4)三种公开合法计数组合，不跨项目强行对齐未知型号身份。也支持提供部分项目，最终提交器检查可检查的项目关系。

单项任务二重聚类：

```bash
bash run.sh cluster --features /runs/x/unknown_features.npz \
  --predictions_csv /runs/x/predictions.csv \
  --competition_profile task2 --unknown_input_mode multi --test_item 2-1
```

CSV未知轨迹/主记录集合须与缓存一致；报告保存输入摘要。只有旧CSV而没有特征无法恢复嵌入。多信号旧CSV缺少bursts_json时，还需要对应slices.jsonl或重跑。完整模式新生成的predictions.csv可直接用于上述命令；单信号原results.csv也支持更新主决策。

## 提交

官方20类映射内置，也支持--label-map指定官方XLSX。未知簇在每个测试项从20编号，簇名不证明跨批次型号相同。以下示例都要求新输出目录：

```bash
python prepare_submission.py --predictions /runs/1-1/predictions.csv /runs/1-2/predictions.csv \
  /runs/1-3/predictions.csv /runs/1-4/predictions.csv \
  --input-root /data/task1 --output /submissions/new_task1
bash test2.sh /runs/2-1/predictions.csv /submissions/new_2-1 /data/2-1
python prepare_submission.py --predictions /runs/2-2/predictions.csv \
  --input-root /data/2-2 --output /submissions/check_2-2 --check
```

任务一通过完整DAT根目录或已核实的--expected-count确认覆盖；仅从0000连续不能证明末尾齐全。任务二各测试项严格要求1750份，不能用expected-count把小样本声明成正式结果。

报告放在输出目录旁边的“输出目录名_report.json”，不混入提交文件夹。ready要求格式、突发、覆盖率、类别预算均通过，且无占位、无导出裁剪、无上游错误。失败退出2；显式--preview可退出0但ready仍false。只有检查占位格式时才显式增加--unresolved shared/singleton；轨迹包络和带外裁剪也只能预览。

## 事件级评估

truth/predictions均为JSONL，每行一个带真实实例标注的事件；不能把文件标签当作多设备逐事件真值：

```json
{"file_id":"2-1_0000","class_id":20,"frequency_low_hz":1000000,"frequency_high_hz":2000000,"start_ms":10,"end_ms":12}
```

```bash
python scripts/evaluate_competition.py --truth truth.jsonl --predictions predictions.jsonl --output new_metrics.json
python scripts/evaluate_competition.py --truth truth.jsonl --predictions predictions.jsonl --time-iou 0.5 --output new_time_metrics.json
```

默认频率IoU0.5、一对一最大匹配，时间门槛为额外诊断。一个预测不会匹配多个同频真值。ARI同时报告未知覆盖率和显式缺失单例策略。PDF的FAR符号/总分方向有歧义，因此只报告原始计数及各项指标，不猜测主办方总分。

## 验证边界

新增测试覆盖K关系、输入顺序、重复组件支持、不可分特征、单突发、源频带、时间上下文/轨迹特征门槛、重复/边界事件、严格提交、事件匹配和完整infer→fast→缓存→重聚类→提交链路。测试结果见发行包TEST_RESULTS.json。

需在具备数据和权重的训练机上完成真实训练、校准和性能评测。本版没有伪造新权重，不把满足类别预算等同于型号分类正确。任务三自动入库、部署监控和回滚属于独立系统工作；本版完成本次任务一/二的推理、发现、提交、评估与CI改动。
