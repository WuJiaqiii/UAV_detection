# 两种开集运行方式

统一入口为 `bash run.sh TASK --config FILE`，也可用 `python main.py TASK --config FILE`。

temp 分支已补回与 main 相同的 YOLOv5 子模块指针（17cfa633999107857a337d51c9c8271fef97edc7）。更新代码后先运行 `git submodule update --init --recursive`，再准备本地检测权重。

## 训练、校准与带标签评估

复制并修改 `config/experiment_iq.yaml`，设置真实数据路径、已知类目录名、YOLO 权重和采集率：

```bash
# 只校验配置，不读取数据、不加载模型。
bash run.sh experiment --config config/experiment_iq.yaml --dry_run
# 扫描数据并保存划分与阶段配置，不训练；不要求权重文件已存在。
bash run.sh experiment --config config/experiment_iq.yaml --split_only \
  --output_dir experiments/fold01_split_review
# 完整实验，使用另一个新输出目录。
bash run.sh experiment --config config/experiment_iq.yaml
```

`experiment` 段的关键参数：

| 参数 | 含义 |
| --- | --- |
| inputs / recursive | 数据根目录、多个文件/目录或通配符；默认递归 |
| input_type | iq_dat、mat 或 png；本轮只使用对应格式，避免混合格式重复训练 |
| known_classes / unknown_classes | 至少指定一项，另一项为剩余已发现类别；两项都指定时必须不重叠并覆盖未排除类别 |
| exclude_classes | 从整个实验中排除的实际类别，例如 Background；不会变成未知类 |
| val_ratio / calibration_ratio / test_ratio | 每个已知类按独立记录分配的比例，默认 0.15 / 0.20 / 0.20；余下用于训练 |
| seed | 划分和训练种子，默认 42 |
| recording_ids_json | 文件到原始采集 ID 的映射；无法从命名判断的同源数据须指定 |
| output_dir | 新目录或空目录；不覆盖旧实验 |

DAT 标签来自文件直接父目录，不需要 `train.class_names`。MAT/PNG 使用现有 `Protocol-[至少4字段]-SNR-...-Figure-N` 文件名及元数据解析，实验输入必须是每文件一个类别的单信号训练数据。其配置可从 IQ 示例修改 `experiment.input_type: mat` 或 `png`；PNG/MAT 的训练前端选项仍在 train 段设置。

同内容哈希、原始采集 ID、同一 `-Figure-N` 前缀的数据通过传递关系合并为一条记录，再按类别划分；同一记录绝不跨集合。同源记录若有冲突类别会报错，实验者仍需为命名无法识别的导出提供真实 recording ID。

每个已知类的 train、val、calibration、test_known 都非空，按比例取整后为每个留出集合至少分配一条记录。未知类全部进入 test_unknown，不进入 CNN、模型选择、参考库或阈值校准。因此测试未知比例由实际数据量决定，不强制为 10%。

train 段填写 epochs、batch_size、lr、supcon_weight、前端和分组参数等。数据路径、类别空间、增量模式和 checkpoint 路径由实验统一生成，不能在 train 段重复设置。此入口从头进行正常已知类训练；LwF 和恢复训练继续使用原 train 任务。

evaluation 段控制所有特征提取、切片、投票、轨迹关联和测试设置；profile 段设置参考/校准计数和目标误拒率；discovery 段设置 HDBSCAN/DCG 和预览参数。实验会自动接续最佳 checkpoint，并用训练集作为参考库，独立 calibration 集确定阈值。IQ 采集率和字节序从 train 段传给所有阶段，显式不一致会报错。

默认每类至少需要 5 条独立参考记录和 20 条独立校准记录；划分后计数不足会在训练前报错。减少 min_calibration_per_class 适合检查流程，但不能证明尾部阈值稳定。实际检测/分组若产生的有效轨迹不足，profile 阶段仍会明确失败，不会伪造校准结果。

## 无标签推理

完整实验成功后可直接使用自动导出的配置：

```bash
bash run.sh infer --config experiments/open_set_fold01/inference.yaml \
  --inputs /data/uav/unlabeled_test --recursive \
  --output_dir experiments/unlabeled_run02
```

该配置保存本轮相同的模型、YOLO、profile 和有效推理参数，不带测试标签。路径为当前实验的绝对路径；迁移机器或移动实验后需修改路径。更改影响特征或轨迹的参数会触发 profile 签名校验，须重新构建 profile。

已有模型和 profile 可修改 `config/infer_open_set.yaml` 后运行：

```bash
bash run.sh infer --config config/infer_open_set.yaml
```

infer 必须提供 open_set_profile，只接受 multi 模式，不允许 labels_json、父目录/文件名真值或特征导出。测试目录名不作为分类标签。已知/未知路由、跨切片轨迹和全批次未知聚类复用原推理实现。Unknown_001 是本批次匿名类别；样本不足时可为 Unknown_unassigned，估计 K 也可以为 0。

仅有 CNN 权重时，先用独立的已知参考/校准数据执行原 profile 任务；无须为了构建 profile 重新训练 CNN：

```bash
bash run.sh profile --config config/open_set.yaml --label_mode parent
```

profile 必须与模型和有效设置匹配。未知真实类别名称不能从无标签聚类中直接恢复，匿名簇也不会自动注册为永久类别。

本次提取公共参数校验更新了签名绑定的 eval_common.py，旧 profile 需要使用同一已知参考/校准集重新生成；已有 CNN 权重仍可使用。

## 输出和评分定义

```text
output_dir/
  split_manifest.json        类别、种子、原始记录、文件划分和计数
  recording_ids.json         本轮统一原始记录映射
  *_labels.json             三个阶段各自的真值；测试真值不进入校准
  effective_train.json       解析后的训练参数
  effective_discover.json    解析后的校准/测试参数
  experiment.json           当前阶段、完成/失败状态、指标和错误
  training/models/best.pth   按已知验证准确率选择的存档
  open_set/known_profile.npz 已知参考库与阈值
  test/results.json          完整测试报告
  test/open_set_metrics.json 记录评分、主轨迹选择及跳过原因
  test/tracks.csv            每条轨迹的类别、状态、时间和频率
  test/unknown_clusters.json 匿名簇和估计未知类别数
  inference.yaml            可用于无标签推理的配置
```

失败时保留已生成文件并在 experiment.json 记录阶段；重试使用新输出目录。split_only 也保留清单供检查；它不自动恢复为完整实验。

`results.json.metrics.open_set_evaluation` 中的新增指标：

- known_accuracy：已知记录分类准确率，拒识、不确定或漏检均计为错误。
- unknown_accuracy：未知记录的端到端准确率；仅在评分时对匿名簇与真实未知类别做一对一匈牙利匹配。未拒识、漏检和未归簇均留在分母中。
- all_accuracy：以上已知与未知正确数之和除以全部可评分记录数；已知名称固定，不参与未知簇的重命名。
- unknown_detection_precision / recall / f1、known_false_reject_rate；known_abstention_rate 额外包含不确定和无有效信号。
- unknown_cluster_coverage、clustered_unknown_accuracy / ari / nmi：成功归簇覆盖率及其条件质量；ARI/NMI 至少需两条有效记录。必须与端到端准确率一起查看。
- true_unknown_class_count、estimated_unknown_class_count、类别数有符号及绝对误差。只有完整单类别真值时比较全批次 K；部分真值只报告已观察到的标签数。
- ood_auroc 仅对有有效距离分数且同时含已知/未知的记录计算，并同时报告 ood_scored_recordings。

评分单位为原始记录；未提供 recording ID 的普通 predict/discover 输入以文件为单位。同一记录如有多条预测轨迹，按有效票数、观测数、路径/轨迹 ID 确定一个主轨迹，选择不看真值。DAT 切片和同源导出不会增加评分记录数。具体选中的轨迹在 open_set_metrics.json 可查。

这是一套单类别记录的端到端评分定义。多信号文件的文件级类别列表无法给每条轨迹提供真值，因此这些记录会在新增准确率中明确跳过，保留原文件级已知/未知存在性指标。完整多实例定位/分类评估需要时间、频率和类别标注；不能将当前主轨迹准确率解释成多实例检测准确率。

真值只在推理、拒识和聚类结束后用于评分；不参与特征、阈值、聚类参数或 K 估计。无标签 infer 的准确率与真值类别数为空，仍输出预测数量、轨迹和匿名簇。
