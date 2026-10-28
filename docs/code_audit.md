# 自动实验目录与代码检查

本轮以 temp 的 d6aae97 为基准检查仓库自身代码：入口、YAML、数据适配与划分、前端、检测适配、分组、mask、CNN/表示学习、训练与 LwF、checkpoint、切片/投票/轨迹、拒识、发现、评分、图表/缓存和测试。YOLOv5 使用固定子模块，真实权重效果和 GPU 环境仍需实际数据验证。

## 统一目录

全部公共命令固定写入项目根目录的 experiment，与当前工作路径无关；采用 Asia/Shanghai 时间。命名形如 train_20261028_173000，同秒并发使用 _001、_002 等后缀，以原子目录创建防止覆盖。

| 运行内容 | 目录前缀 |
| --- | --- |
| 普通训练 | train |
| 基础类别训练 | base_train |
| 增量/LwF 训练 | incremental_train |
| 旧 GT 验证或有标签普通预测 | validation |
| 无标签普通预测 | inference |
| 完整开集实验 | open_set_experiment |
| 仅检查开集划分 | open_set_split |
| 无标签开集推理 | open_set_inference |
| 有标签开集预测 | open_set_validation |
| 已知 profile / 归档校准 | known_profile |
| 校准与测试发现 | open_set_discovery |
| 离线未知重聚类 | unknown_clustering |

每条运行都有 run.json，记录类型、开始/结束时间、时区、耗时、配置文件、实际参数和状态。异常/中断保留 error.log；推理发生部分文件或切片错误时状态为 completed_with_errors，统一入口返回非零退出码 2。dry_run 只校验参数，不分配目录。

完整开集实验的 training、open_set、test 都是该实验内部阶段，保留在同一个时间目录下。内部 Python 组件由编排器传入这些路径；命令行和 YAML 不再暴露自选输出位置。

迁移时删除 YAML 的 output_dir、test_output_dir、experiment_dir，以及相应命令行选项；独立归档校准脚本也移除了 --output。运行方式保持 `bash run.sh TASK --config FILE`。已有 experiments/ 下的权重仍可作为输入；新结果写入 experiment/。profile 与聚类输入需要填写真实工件路径，完整 experiment 会自动导出匹配的 inference.yaml。

普通训练可用 config/train_iq.yaml；增量训练示例为 config/train_incremental.yaml；带标签 predict 自动归入 validation/open_set_validation；无标签 infer 为 open_set_inference。普通自动划分也支持 train.recording_ids_json，以识别命名无法表示的同源增强数据。

## 已修复的问题

| 问题 | 修复 |
| --- | --- |
| 不同入口各自分配目录，训练目录可能在同秒冲突，直接脚本行为不一致 | util/run_context.py 统一分配、命名、状态与错误记录；Config 和所有任务复用 |
| 普通 MAT/PNG 自动划分在只有一个原始记录时会退回随机切片划分 | 删除此回退，记录不足明确报错；按类别和原始记录划分 |
| 普通 IQ 自动划分按文件进行，且随机种子写死 42 | 与 MAT/PNG 复用 util/recordings.py，合并同源/重复内容，使用 config.seed，训练/验证均覆盖类别 |
| 开集类别排除后，仍可能校验被排除文件的训练格式 | 元数据校验只使用实际参与实验的文件；被排除记录仍在清单中标识 |
| 普通 checkpoint strict=False 可静默留下随机参数；优化器/调度器恢复异常仅打印警告 | 默认严格检查权重结构，恢复状态失败报错；显式 skip_mismatch 仍保留为内部能力 |
| 恢复训练继承旧最佳分数，新目录可能始终没有 best.pth；目标 epoch 已跑完时静默结束 | 继续恢复权重、状态和 epoch，但在新运行内重新选择最佳模型，旧最佳分数记录到日志；无剩余 epoch 明确报错 |
| LwF 教师类别 ID 可能与学生旧类前缀不一致，任意维度不匹配被跳过 | 先校验教师 config.classes，严格加载教师，学生只允许分类头大小差异；预检失败不改学生，配置冻结状态在异常后恢复 |
| 冻结骨干只关闭梯度，train() 仍更新 BatchNorm 运行统计 | 冻结骨干保持 eval，只让分类头处于训练态；权重键不变 |
| 推理部分文件失败仍被顶层当作完全成功 | 保留有效结果，记录 completed_with_errors 并返回退出码 2；实验中断也同步写入状态 |
| 无效保存间隔、学习率等配置可能到训练中途才失败 | 普通训练和开集编排复用训练前数值校验 |
| 加载完整 checkpoint 或 LwF 教师前仍可能无用地下载 ImageNet 初始化权重 | 这些路径直接从存档初始化骨干，预训练下载只用于需要它的从头训练 |
| 存在没有实际消费的配置项 | 移除 yolo_half、yolo_warmup、merge_w_log_thresh、match_size_penalty，避免参数看似生效 |

## 当前结构

- main.py / run.sh：入口与退出码。
- util/cli.py：配置解析和任务分派；util/run_context.py：目录及运行记录。
- util/open_set_experiment.py：完整实验编排；util/recordings.py：文件身份和公共划分；util/experiment_split.py：已知/未知协议划分。
- data/：只负责各格式适配和 loader；model/：CNN、特征和 SupCon。
- util/train_entry.py / trainer.py / lwf_trainer.py：训练装配、循环和增量策略；checkpoint.py：存档语义与状态恢复。
- util/eval_common.py / group_predictor.py：共享推理配置、输入、投票和逐组预测；multi_signal_eval.py：物理位置与轨迹。
- util/open_set.py / unknown_discovery.py / dcg_discovery.py / open_set_metrics.py：校准、拒识、发现与事后评分；测试真值不参与预测。
- scripts/：三个薄任务入口；数值前端、分组、mask、报告和缓存按职责保留。

此次没有更换频谱、分组、mask 或未知发现的科学算法。源码绑定签名发生变化，旧 profile/特征归档应重新生成，不绕过检查。普通完整 checkpoint 和 CNN 参数键保持兼容；LwF 教师现在必须包含正确的类别映射。

## 验证范围与后续边界

回归覆盖自动目录、同秒并发、无副作用 dry_run、嵌套实验、状态与退出码、同源划分、随机种子、严格 checkpoint、LwF 类别错序及冻结骨干统计，同时保留前端数值、坐标、投票、表示和发现测试。完整实验编排使用受控训练产物和特征，真实 YOLO 后端属于可选集成检查。

仍需保留的边界：

- 单类别记录的主轨迹准确率不等于多实例检测准确率；混合文件需要逐信号时间/频率真值。
- 开集阈值和未知 K 的可靠性需要独立真实数据验证，匿名簇不会自动成为真实协议名称。
- 恢复训练读取模型、优化器、调度器和 epoch，但没有保存全部 RNG/loader 状态，因此不能宣称逐位确定性恢复。
- Trainer 仍包含训练验证与旧 GT 实例评价，文件较长。后续可在保持行为和回归覆盖的前提下拆分匹配、可视化和循环；当前保留这些语义不同的功能。
