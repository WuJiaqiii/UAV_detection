# 一个根目录：自动划分全部已知类训练数据

`experiment --protocol all_known` 从带标签的数据集根目录自动建立索引，发现所有未排除类别，以原始记录为单位划分训练、验证和独立校准集合，完成训练和profile构建后导出配套的inference.yaml。原始文件不复制、不移动；DataLoader读取生成的文件列表，IQ仍按需切片。

## 输入与命令

DAT类名来自文件直接父目录，不预设协议名称或类别数：

```text
dataset/
  类别A/*.dat
  类别B/*.dat
  ...
```

在仓库根目录运行：

```bash
bash run.sh experiment --config config/experiment_all_known_iq.yaml \
  --inputs /path/to/dataset \
  --yolo_weights /path/to/best.pt
```

Windows同样可以使用Python入口，路径包含空格时加引号：

```powershell
python main.py experiment --config config/experiment_all_known_iq.yaml --inputs "D:/signal/dataset" --yolo_weights "D:/signal/best.pt"
```

命令行覆盖config中的根目录和检测权重路径，无需提供train/val/calibration三个目录，无需填写known_classes/unknown_classes。训练使用主分组CE和现有YOLO，完成后自动使用最佳checkpoint提取参考/校准特征并生成Maha++ profile。

## 划分规则

配置中的experiment段：

```yaml
experiment:
  protocol: all_known
  inputs: [/path/to/dataset]
  input_type: iq_dat
  exclude_classes: []
  val_ratio: 0.15
  calibration_ratio: 0.20
  test_ratio: 0.0
  seed: 42
```

每类40条独立记录时，train/val/calibration为26/6/8。测试比例默认0，不从当前数据留出模拟未知类；所有未排除类别都进入已知分类头。若显式设置test_ratio>0，可额外留出已知测试记录，但不会产生模拟未知类别。

同内容哈希、同原始采集ID、同Figure导出前缀通过已有公共记录逻辑归并；整个原始记录的所有切片/导出随同一集合。无法从命名和内容确定同源关系时，提供`--recording_ids_json /path/to/ids.json`。划分按类别分层、固定种子，清单保存于split_manifest.json。

划分比例按独立记录数计算，不按切片数。DataLoader建立前先做索引和哈希检查，大文件首次索引会有磁盘读取，但不将完整IQ数组载入内存。各类至少留下非空训练、验证和校准集；参考/校准支持不足会在训练前报错，实际有效特征不足仍会在profile阶段明确失败。

只检查配置：

```bash
bash run.sh experiment --config config/experiment_all_known_iq.yaml --dry_run
```

只检查真实划分，不训练：

```bash
bash run.sh experiment --config config/experiment_all_known_iq.yaml \
  --inputs /path/to/dataset --yolo_weights /path/to/best.pt --split_only
```

split_only创建独立的all_known_split记录；之后运行完整命令会创建新的all_known_training记录并按相同种子重新索引划分，不是恢复split_only中的训练。

## 自动输出与外部无标签推理

```text
experiment/all_known_training_<北京时间>/
  run.json
  experiment.json
  split_manifest.json
  recording_ids.json
  reference_labels.json
  calibration_labels.json
  effective_train.json
  effective_profile.json         # test_ratio=0时
  training/models/best.pth
  open_set/known_profile.npz
  open_set/reference/features.npz
  open_set/calibration/features.npz
  inference.yaml
```

默认流程在profile完成后结束，test_ratio=0时不创建test/或test_labels.json，也不访问外部测试数据。experiment.json报告已知类别数量、各集合记录数及profile提取错误数，不能将这些数量视为测试准确率。

随后对独立无标签数据运行：

```bash
bash run.sh infer \
  --config experiment/all_known_training_实际时间/inference.yaml \
  --inputs /path/to/external_unlabeled --recursive \
  --unknown_input_mode multi
```

确定每条原始记录只有一个设备/类别时，改为`--unknown_input_mode single`。已保存配置包含同一checkpoint、YOLO、profile和有效参数，无需手工填写两份存档路径。导出配置不携带训练标签或训练录波ID映射；外部数据有同源导出时，可在infer命令另传该批数据的recording_ids_json。

无标签输入的真实目录名不作为真值，准确率和ARI/NMI需要之后取得真实标签才能计算。信号频率、类别、未知候选与匿名簇仍正常输出。单/多信号策略见[未知发现模式](unknown_discovery_modes.md)。

## 与原有任务的关系

默认protocol仍为open_set，原experiment_iq.yaml继续用于内部留出未知类别的对照。all_known禁止known_classes和unknown_classes选择器，防止沿用旧配置而漏掉部分类别；可显式用exclude_classes排除背景等实际目录。

原train任务及自动train/val划分仍可用；需要独立校准与成套部署工件时使用本页的all_known流程。PNG/MAT也可用同一协议，改input_type并沿用其现有Protocol文件名标签规则。

## 本机检查

以D:/signal/dataset实际800条DAT执行split_only，发现20类，每类26/6/8，总计520/120/160，无内部测试记录；记录未跨集合、原始路径保持存在。另用小型可控数据完成自动划分、训练回调、真实profile流程和外部无标签infer的端到端检查；该检查不代表完成了全量模型训练或性能测试。
