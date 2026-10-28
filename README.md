# UAV_detection

无人机信号的已知类训练、多信号分类、开集拒识与未知类别发现。主线为：MAT/PNG/DAT → 统一频谱 → YOLO 框 → 信号分组 → 二值 mask CNN → DAT 轨迹投票 → 已知 profile 拒识 → 未知候选聚类。输出类别、时间和频率位置。

## 安装

使用 Python 3.11，先安装适合机器的 PyTorch 2.6.0 / torchvision 0.21.0，然后：

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
```

检测仍需要本地 `yolov5/` 子模块和已有 `.pt` 权重。核心仓库不再包含背景/合成数据、标签 remap、标注导出或 YOLO 训练工具；不需要 TensorFlow、TensorRT 等模型导出运行时。

## 统一入口

根目录只保留一个 Bash 包装，所有任务也可以直接使用 `python main.py TASK`：

```bash
bash run.sh --help
bash run.sh train --config config/train.yaml --dry_run
bash run.sh train --config config/train.yaml
bash run.sh predict --config config/predict.yaml
bash run.sh profile --config config/open_set.yaml
bash run.sh predict --config config/open_set.yaml
bash run.sh cluster --config config/cluster.yaml
# 新输出目录上，一次完成参考提取、独立校准与混合测试发现：
bash run.sh discover --config config/open_set.yaml
```

`--dry_run` 校验并显示参数，不读取数据、构建模型或开始训练。示例中的 `/data`、`/models` 是占位路径，运行前修改。YAML 内相对路径以 YAML 所在目录为基准；直接命令行相对路径以当前工作目录为基准。

| 任务 | 工作 |
| --- | --- |
| train | 已知类 CE / CE+SupCon，保留可选 LwF |
| predict | 逐组分类和 DAT 轨迹投票；有 open_set_profile 时启用拒识与未知发现 |
| profile | 从 reference_inputs / calibration_inputs 提取特征并建立已知 profile |
| cluster | 复用 unknown_features.npz 重新聚类，不加载 YOLO/CNN |
| discover | 顺序运行 profile 与 predict，保持相同特征设置 |

## 实验配置

`config/` 中提供 PNG 训练、IQ 训练、普通预测、开集流程、离线重聚类五个示例。支持 `train / evaluation / profile / predict / discovery / cluster` 配置段：

- train：训练参数、类别顺序和排除项。
- evaluation：profile 提取和 predict 共用的权重、前端、分组、切片和投票参数。
- profile：参考/校准输入、真值来源和阈值设置。
- discovery：discover 专用的混合测试输入、标签和后端；不会进入参考/校准标签。
- predict / cluster：相应任务参数，覆盖 evaluation 的同名设置。

命令行显式值覆盖 YAML：

```bash
# CE 基线；原配置仍为 SupCon。
bash run.sh train --config config/train.yaml --supcon_weight 0
# 不保存图片；保留 HTML、CSV、JSON。
bash run.sh predict --config config/predict.yaml --no-save_visuals
# 只换后端，不重新检测和提取特征。
bash run.sh cluster --config config/cluster.yaml --dcg_cosine_threshold 0.9
```

未识别的配置段/参数、错误列表/布尔类型会报错。`settings` 可用映射表达底层推理覆盖，例如 `settings: {min_group_len: 1}`，等价于脚本的 `--set min_group_len=1`。YAML 中枚举字符串如 `bbox_cache_mode: "off"` 应加引号，避免 YAML 将 off 解析为布尔。

## 训练和数据

PNG/MAT 训练读取文件名 `Protocol-[至少4字段]-SNR-...-Figure-N`，解析类别、中心频率及带宽；PNG/MAT 目录只读取第一层，可传多个目录。输入保持原生 512×750，不自行拉伸到 CNN 尺寸。训练只接受单信号样本。

DAT 训练的类别直接取自 `.dat` 文件的父目录，不需要预设协议名称。训练随机取 50 ms 片段，验证取中心片段。例如：

```text
known_train/                 known_val/
  Device-A/001.dat             Device-A/101.dat
  Device-Z/002.dat             Device-Z/102.dat
```

修改 `config/train_iq.yaml` 中的训练/验证路径、采集率和 YOLO 权重后运行：

```bash
bash run.sh train --config config/train_iq.yaml
```

IQ 会从实际使用的训练、验证路径共同发现类别，按名称排序建立统一 ID（上例 `Device-A: 0, Device-Z: 1`），不会自动添加 Background。显式划分需要同时提供 `train_dataset_path` 和 `val_dataset_path`，此时忽略 `dataset_path`；自动划分则只设置 `dataset_path`。输入可为数据根目录、类别目录或文件列表，目录扫描覆盖当前层及下一层，类别名始终是文件的直接父目录名。

IQ 通常应省略 `class_names`；若为恢复旧 checkpoint 指定顺序，列表必须包含发现的全部类别，不能用它隐式过滤数据。需要留出模拟未知类时，用实际目录名设置 `exclude_classes`；排除后重新生成连续 ID。训练和验证共享同一映射，数据中未映射且未明确过滤的类别会报错。PNG/MAT 的 `class_names` 仍可自定义类别空间，不提供时保留原协议默认映射。

最终映射会写入实验 `config.yaml` 和 checkpoint 的 `config.classes`。恢复训练时必须保持类别名称与 ID 一致；预测从 checkpoint 读取类别映射，不根据待测目录重新编号。采集率为 `iq_sample_rate`，与输出频谱的 122.88 MHz 网格区别见 [前端契约](docs/spectrogram_frontend.md)。

`config/train.yaml` 示例将 FPV/Skylink1 留作模拟未知、排除 Background。`supcon_weight=0` 保留原 CE；正值使用有限框丢弃和平移构建第二视图。不要冻结骨干后开启 SupCon。训练存档仍在 `experiments/UAV_时间/models/best.pth`；按已知验证准确率选模型，同时查看匹配召回。

## 开集与输出

参考/校准都应为干净的单已知类别输入，且按原始采集记录隔离。默认每类至少 20 条独立校准记录；同一 DAT 的切片不能充当多条记录。unknown 只在最后的混合输入出现，不参与网络训练或阈值校准，也不预设未知比例/K。

有 profile 时，最终结论读 `tracks.csv` 的 `display_label / recognition_status`；逐组 `class_name` 仍为已知分类头的原始候选。`unknown_clusters.json` 保存估计 K、支持记录和原型；`unknown_features.npz` 可重聚类，`unknown_prototypes.npz` 保存同空间簇中心。图示/HTML 保留原始框的时间和频率；给出 RF 中心后可输出绝对 MHz。

详见 [开集发现说明](docs/open_set_discovery.md) 与 [第一轮清理迁移说明](docs/cleanup_migration.md)。模型权重结构和类别语义保持；本次移动源码后需要重建 profile。

## 目录与兼容

```text
main.py / run.sh       统一任务入口
config/               实验 YAML
data/                 MAT/PNG 和原始 IQ 数据适配
model/                CNN 和实际使用的 SupCon
util/cli.py           配置、任务分派与开集编排
util/train_entry.py   原训练入口（保留旧 GT 评价和 LwF）
util/eval_common.py   输入、配置恢复、切片和投票
util/group_predictor.py  共用检测、分组、mask 和分类
util/                 前端、分组、checkpoint、缓存、轨迹、开集和报告
scripts/              仅保留评估、profile 校准和重聚类三个 Python 入口
tests/fixtures.py     共用夹具，不再由测试文件互相导入
yolov5/               固定外部检测后端
```

既有 `python main.py --run_mode train/infer ...` 暂时转发到原实现，三个核心 Python 脚本仍可直接使用。新的普通/开集推理建议走 `predict`；旧 GT 实例级评价保留在原路径，未用文件级指标替代。

## 检查

```bash
python -m compileall -q main.py model util data scripts tests
python -m unittest discover -s tests -v
bash -n run.sh
# 可选：实际本地 YOLO 后端的无权重输入契约检查
UAV_TEST_REAL_YOLO=1 python -m unittest tests.test_yolo_native_contract -v
```

当前只看到框几何的 CNN 不能保证将几何相似协议分离；YOLO 完全漏检的信号也无法进入类别发现。文件级未知出现指标与实例级 GCD ACC 不同；没有逐轨迹真值时不报告实例级发现准确率。
