# 主线 A：完整训练、六项推理和提交

主线固定为原记录平衡 Maha++ 拒识、主轨迹校准、公开类别数量范围下的 density_merge。known_recheck 和 profile_recheck 默认关闭，保留为研究对照入口。任务一按每个原始记录的主轨迹决定类别；任务二保留多轨迹和逐突发输出。

本轮检查报告见 [主线审核记录](mainline_audit_20261107.md)。新的源码签名包含本轮检测边界修复，应从当前版本重新训练/提取参考和校准，不绕过旧 profile 签名检查。

## 获取代码

代码发布到 temp，main 不变。先保存当前工作目录未提交的改动。如果原目录仍有实验或未提交代码，建议另建工作目录而不切换正在使用的代码：

```bash
git fetch origin temp
git worktree add --detach ../code_mainline origin/temp
cd ../code_mainline
git submodule update --init --recursive
```

继续使用已有环境，必要时按 requirements.txt 补齐依赖。YOLO 权重和数据不包含在 Git 中；config/competition_v2_train.yaml 默认指向服务器原数据目录和原 exp10/best.pt，可按实际路径修改。

## 1. 完整 20 类训练

```bash
bash run.sh experiment --config config/competition_v2_train.yaml --dry_run
bash run.sh experiment --config config/competition_v2_train.yaml
```

这是 all_known：自动发现全部 20 类并按原始记录划分训练/验证/校准，默认比例 65%/15%/20%，没有伪未知类别留出。每类 40 个独立记录时为 26/6/8。生成最佳模型、已知 profile 和 inference.yaml。

有同一采集的多个不同派生文件时，在 experiment.recording_ids_json 提供原始采集 ID 映射；不同内容哈希不能自动识别所有派生关系。训练、校准和验证不得共享原始记录。

dry_run 只检查配置，不检验真实权重、数据内容或训练性能。记录本次实际训练目录，不能使用旧的固定时间戳目录：

```bash
TRAIN="/absolute/path/to/code_mainline/experiment/all_known_training_实际时间"
DATA="/media/kaneki/05E4235537EC7AC2/dataset_wjq"
```

必须等待 experiment.json 为 complete。16 类伪未知开发模型不用于正式提交。

## 2. 六项推理

```bash
bash infer.sh "$TRAIN" "$DATA" --dry-run
bash infer.sh "$TRAIN" "$DATA"
```

infer.sh 现在要求显式 TRAIN 和 DATA，不再固定旧模型、注释部分项目或手动猜测输出时间戳。它默认使用本次训练生成的 inference.yaml，确保特征设置一致，并强制 known_recheck=false、density_merge、auto/hdbscan、保存未知缓存；快模式为默认输出。需要详细结果时用 --full。

预检确认完成的 all_known 模型、官方 20 类名称、权重哈希、特征源码及依赖版本匹配，随后检查输入文件名、连续编号和完整 int16 I/Q 对（每对 4 字节）。标称 100-ms 输入允许 99..100 ms，即 39,600,000..40,000,000 字节，兼容已观察到的 39,845,888 字节/99.61472-ms 转换文件；这是程序的输入兼容范围，不是官方定义的容差。未对齐、明显过短及超出 100 ms 的文件仍拒绝，并打印实际字节数。任务二每项严格要求 1750 份。不会读取全部 IQ 做预检或启动新训练。

非整 100-ms 文件的数量、实际时长和切片覆盖写入 dry-run 输出及批次清单。原有 tail_policy=drop 在 99.61472-ms 文件上只取前 50 ms，剩余 49.61472 ms 不参与推理；预检会明确警告。新补丁保留当前 profile 签名对应的切片设置，不会自动改变它。要完整覆盖尾段，需用 tail_policy=cover、max_slices=0 和不大于 50 ms 的步长重新提取参考/校准并建立匹配 profile；分类 CNN 权重可复用，但不能单改推理 YAML 绕过签名检查。仅修复本次长度预检无需重新训练或重建 profile。

| 项目 | 类别数量约束 | 记录模式 |
|---|---|---|
| 1-1、1-3 | N1=2..3 | single |
| 1-2、1-4 | N2=3..4 | single |
| 2-1、2-2 | N=0..5 | multi |

六项都使用 signal_mode=multi 的检测/分类入口。single/multi 上表指 unknown_input_mode。任务二不排除任何未指明的已知型号；网络仍包含训练集中全部 20 类，测试场景只有 19 类是数据属性。

每项使用新的 open_set_inference 目录。父批次打印并保存：

    experiment/competition_batch_时间戳/batch_manifest.json

清单记录确切模型、配置、输入元数据、缓存摘要、子运行目录和日志。每项失败后继续后续项，最后返回非零状态，不会因最后一项成功掩盖前面失败。needs_review 表示 CSV 尚有未解决预测，如任务一 no_signal/Unknown_unassigned；程序处理完成不等于可提交。

可以选择部分项目运行：

```bash
bash infer.sh "$TRAIN" "$DATA" --items 1-1 1-2
```

未齐六项的清单为 incomplete。续跑同一清单可补齐或重试失败项目，完整项自动跳过：

```bash
BATCH="/absolute/path/to/experiment/competition_batch_实际时间/batch_manifest.json"
bash infer.sh "$TRAIN" "$DATA" --resume "$BATCH"
```

配置、权重、流水线代码、已完成项的文件名/大小/修改时间或缓存内容变化时拒绝静默续跑。输入元数据签名不等于完整 IQ 内容哈希；替换测试数据后应新建批次。外层可以重定向日志，子任务日志也逐项保存在批次目录。一次只允许一个进程使用同一清单。

## 3. 联合选类与提交

```bash
bash submit.sh "$BATCH"
```

自动完成四项任务一的 N1/N2 联合选择，继续使用 density_merge 及同一支持数、密度和分离门槛。比较公开合法组合 (2,3)、(2,4)、(3,4)，输出的类别身份不靠测试真值确定。导出使用 joint_selection.json 指向的最终 CSV；任务二使用对应完整批次 CSV。

默认写入新的 submission/competition_时间戳，可用 --output 指定新目录。只做检查时：

```bash
bash submit.sh "$BATCH" --check
```

最终目录仅包含：

    1-1.xlsx
    1-2.xlsx
    1-3.xlsx
    1-4.xlsx
    2-1/2-1_0000.xlsx ... 2-1_1749.xlsx
    2-2/2-2_0000.xlsx ... 2-2_1749.xlsx

每份仅用 Sheet1。任务一两列：序号、无人机类别。任务二三列：无人机类别、起始频率(Hz)、终止频率(Hz)。空场景按官方示例写一行 -1,0,0。已知 ID 按官方映射为 0..19；未知从 20 编码。相对频率严格在 -50,000,000..50,000,000 Hz，同频但时间分离的突发保留多行。

检查报告在提交目录旁的“目录名_report.json”，必须 ready=true。覆盖不足、类别预算不符、未解决预测、频率异常、上游错误或事件冲突均不通过正式提交。不要用预览、人工占位类、频率裁剪绕过检查。示例表中的注释行和彩色演示数据不复制到提交文件。

task1 的次要轨迹事件矛盾仍作为诊断保留，不再使主记录运行退出；task2 的事件冲突仍为错误。几何重叠的不同轨迹不会自动合并或视为同一观测，共享观测证据或同轨迹的冲突仍检查。此规则不保证物理目标区分正确，仍需真实事件标注验证。

## 评估与性能边界

类别数合法不等于型号正确；density_merge 只合并有支持的核心簇，不凭空补齐类别，也不会把未分配噪声自动归入最近簇。初始核心不足时明确失败。密度计算仍有二次方内存开销，合并诊断有 32 核心/2000 评分样本保护，不能据此宣称实时部署已验证。

当前每类 8 个校准记录不足以稳定证明 1% 部署误拒率。已知微平均精确率/召回率/F1、未知误收率、覆盖率和带缺失单例处理的 ARI 用于诊断；不是主办方最终总分的替代。旧开发划分的准确率不能保证重新训练后的性能，仍需独立类别留出验证和逐事件真值评估。

任务三的自动入库、在线更新调度、监控与回滚没有完整闭环实现，当前增量训练选项和设计报告不能视为已完成该系统。
