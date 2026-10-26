# 已知类训练、开集拒识与未知类别发现

此实现基于 UAV_detection main `2ef9d9b931b5afdd5d14513052e52d187d5af297`。适用条件是：**训练阶段只有已知类；未知类只在最终测试出现；每个输入可以包含多个信号，DAT 可以产生多个切片。**

流程为：已知类 CE + 可选 SupCon 训练 → 冻结分类器 → YOLO 检测与分组 → DAT 轨迹关联及投票 → 已知参考 profile 拒识 → 跨文件汇总未知候选 → HDBSCAN 或 DCG-Louvain → 独立记录支持过滤 → 匿名类别、类别数及时间/频率位置。未知比例不参与决策，不要求预先知道未知类别数，也不新增一个监督的 Unknown 类。

## 从 AMR_NCD 迁移了什么

参考了 AMR_NCD main `f250f5d5ee56692162424e0562b3f2adbec989ed` 和 onestage/v3 `42fe6fba90e3e97dd0ddd8617e170c590cc3e822`：

| 可用思想 | UAV 实现 | 适用边界 |
| --- | --- | --- |
| 已知类分类与特征学习结合 | 在现有 mask 编码器上加入 CE + SupCon，两种损失共享骨干 | 只使用已知类标签；不需要真实未知训练数据 |
| 分类头与发现表示分离 | 保留已知分类头，导出最后分类层前的特征 | 不对已知 softmax 向量聚类 |
| 动态图发现未知类别 | 冻结特征构建稀疏余弦邻接图，以 Louvain 社区发现估计 K | 本实现明确命名为 DCG-Louvain；不是 v3 原算法的完整复现 |
| 原型汇总 | 导出归一化簇中心、支持记录与代表轨迹 | 匿名簇名仅在当前批次有效；相似度不等于类别概率 |

没有移植需要“训练时含真实未知”的联合训练器、预设总 K 的新类头、测试真值身份掩码、IQ 网络或其权重。图和原型始终使用同一组 L2 标准化特征，不在测试时额外白化后又与原空间查询混用。后端失败会报错；无边图不会被强行当作一个新类别。

## 安装与兼容

保留仓库原环境和 YOLOv5 子模块，额外安装：

```bash
pip install -r requirements-open-set.txt
```

新增依赖是 NetworkX >= 3.0、< 4。HDBSCAN 使用 `sklearn.cluster.HDBSCAN`，需要 scikit-learn >= 1.3；原 requirements 已满足。不要替换成另一个同名第三方包。

`MaskImageClassifier.forward(x)` 的原接口不变；`forward(x, return_features=True)` 返回 `(logits, embedding)`。ResNet18/34 特征为 512 维，MobileNetV3-small 为 1024 维。没有新增模型参数或修改 checkpoint 权重键，旧 checkpoint 可严格加载。`--supcon_weight 0` 和不提供 `--open_set_profile` 时保留原训练、评估行为。

可先使用旧模型建立 profile 作为基线，再训练 SupCon 模型对比。**换模型后必须重新提取参考/校准特征并重建 profile。**

## 仅用已知类训练特征

在原来的 main.py 训练命令上增加以下参数即可，原数据、类别和分组参数继续使用：

```bash
python main.py \
  --supcon_weight 0.1 \
  --supcon_temperature 0.2 \
  --supcon_drop_prob 0.1 \
  --supcon_max_drop_fraction 0.25 \
  --supcon_jitter_px 1
```

这里是追加参数的示例；请同时保留实际训练所需的路径和原有选项。0.1 是起始实验值，不是 UAV 数据上已验证的最优值。

启用后，每个有效信号组生成两张二值 mask：原 mask，以及模拟少量漏检/框坐标偏差的 mask。增强只平移矩形，不缩放矩形的大小形状；最多丢弃 25% 的框，至少保留两个框，原本只有一个框时保留该框。极细框经缩放消失时回退到原 mask。两个视图都计算 CE，同时用已知类标签计算 SupCon。相同样本的两个视图及同类样本构成正对。

训练目标为 `CE + supcon_weight * SupCon`，TensorBoard 单独记录 CE、SupCon 和总损失。普通训练和 LwF 均接入该目标，LwF 的旧类蒸馏继续生效。开启 SupCon 时不能冻结骨干。每个实例需要两个前向视图，显存和训练时间会增加；可相应调小原 batch size。

## 数据划分与记录身份

准备互不混用的集合：

1. **known_train / reference**：只包含已知训练类，用于训练和建立参考库。
2. **known_calibration**：同一组已知类的独立原始采集记录。建议不参与模型拟合或模型选择，仅用于校准拒识阈值；可从已知验证记录另划一份。
3. **test_mixed**：最终已知/未知混合输入，只做推理和离线发现。测试标签及真实 K 不进入训练、profile 或聚类。

参考和校准输入应为**干净的单已知类别文件**。同一信号被拆成多条轨迹是允许的，但文件内所有有效轨迹必须确实来自标注协议。一个含不同协议的文件不能仅凭父目录标签给所有轨迹赋真值。

支持三种显式真值来源，类别名称必须与 checkpoint 中的名称一致：

- `--label_from_filename`：读取项目已有 `Protocol-[...]-SNR-...-Figure-...` 文件名中的协议字段，适用于项目导出的 PNG/MAT。
- `--label_from_parent`：用父目录名称作为类别，适用于按协议整理的 DAT/PNG/MAT。
- `--labels_json labels.json`：输入路径到类别名的映射。相对路径以 JSON 所在目录为基准；可覆盖无法从名称识别的文件。

profile 默认要求分类头中每个类别均有参考和校准样本。当前 main 的 `Background` 往往没有有效信号组，可在建 profile 时显式 `--ignore_classes Background`。被忽略类不进入参考库，其预测路由到 `uncertain`，不会自动成为未知候选。若训练时已经排除了 Background，则不要再传这个忽略项。不能忽略全部类别。

参考库每类至少有 k 条轨迹；校准默认每类至少 **20 条独立原始记录**。可显式降低 `--min_calibration_per_class` 进行开发，但少量记录不足以可靠估计 1% 的尾部风险。

相邻 DAT 切片、同一采集的 MAT/PNG 导出必须划入同一集合。项目 `...-Figure-N` 命名会自动按 Figure 之前的名称合并为原始记录；其他输入默认按源文件 SHA256 识别。任意裁剪、重命名及跨格式导出不能总能自动辨认，请提供 `--recording_ids_json`：

```json
{
  "known_train/ProtocolA/capture01_slice000.png": "session01/capture01",
  "known_train/ProtocolA/capture01_slice001.png": "session01/capture01",
  "known_calibration/ProtocolA/capture02_slice000.png": "session02/capture02"
}
```

显式映射优先；为不同采集分配不同 ID。参考与校准若有相同内容哈希或 recording_id 会被阻止。这个 ID 也决定未知簇的独立支持数，避免一个 DAT 的大量切片或同文件副本充当多条独立记录。

## 四步评估

以下在仓库根目录执行，路径为 Linux 示例。输入可为目录、多个文件、通配符，MAT/PNG/DAT 可混合；目录用 `--recursive` 递归。每次 output_dir 必须不存在或为空。

第一步，提取已知参考特征：

```bash
python scripts/evaluate_single_signal.py \
  --inputs /data/known_train --recursive \
  --checkpoint_path /models/classifier_best.pth \
  --yolo_weights /models/signal_detector.pt \
  --output_dir /results/open_set/reference \
  --signal_mode multi --export_features --label_from_filename \
  --no-save_visuals
```

第二步，提取独立已知校准特征：

```bash
python scripts/evaluate_single_signal.py \
  --inputs /data/known_calibration --recursive \
  --checkpoint_path /models/classifier_best.pth \
  --yolo_weights /models/signal_detector.pt \
  --output_dir /results/open_set/calibration \
  --signal_mode multi --export_features --label_from_filename \
  --no-save_visuals
```

DAT 若按类别目录整理，将两处 `--label_from_filename` 改为 `--label_from_parent`。必要时在两步均加原始记录映射。

第三步，生成冻结 profile：

```bash
python scripts/build_open_set_profile.py \
  --reference_features /results/open_set/reference/features.npz \
  --calibration_features /results/open_set/calibration/features.npz \
  --output /results/open_set/known_profile.npz \
  --knn_k 5 --known_false_reject_rate 0.01 \
  --ignore_classes Background
```

0.01 是**已知校准轨迹上的目标误拒比例**，不是未知比例或未知召回率。阈值取来自独立校准记录的轨迹距离的保守上分位点，距离等于阈值时保留为已知。输出含实际校准误拒率、每类误拒及样本量提示，不保证部署数据仍达到同一比例；同记录内轨迹不视为独立风险保证。每类参考特征默认最多保留 2000 条，在校准前确定；用 `--max_reference_per_class` 调整。

第四步，对整批最终输入评估和发现，默认 HDBSCAN：

```bash
python scripts/evaluate_single_signal.py \
  --inputs /data/test_mixed --recursive \
  --checkpoint_path /models/classifier_best.pth \
  --yolo_weights /models/signal_detector.pt \
  --output_dir /results/open_set/discovery_run01 \
  --signal_mode multi \
  --open_set_profile /results/open_set/known_profile.npz \
  --unknown_cluster_backend hdbscan \
  --unknown_min_cluster_size 5 --unknown_min_samples 3 \
  --unknown_min_recordings 2
```

切换到图发现时，将后端改为 `--unknown_cluster_backend dcg`，可追加 `--dcg_topk 15 --dcg_cosine_threshold 0.85 --dcg_resolution 1.0`。`--unknown_min_samples` 和 `--unknown_allow_single_cluster` 仅用于 HDBSCAN。

三个特征/推理步骤必须保持相同的权重、前端、分组、mask、切片、投票及关联设置。若需要 `--grouping_mode hopping --hop_association shape` 或调整 YOLO 阈值，三步一起改。profile 绑定有效配置、分类器/YOLO 权重哈希、特征版本、核心源码及相关库版本；设备、输出目录、真值、绘图和 RF 中心不影响特征签名。改变已绑定项后需重新提取并校准。

DAT 默认采样率为 100 MHz；122.88 MHz 数据应在三步均传 `--iq_sample_rate 122880000`。MAT/PNG 沿用原生 512×750 前端契约，见 [频谱前端说明](spectrogram_frontend.md)。测试若知道 RF 中心，可加 `--rf_center_mhz 2400` 或 `--rf_centers_json /data/centers.json`，得到绝对 MHz；否则保留频偏。原始频率坐标不作为协议聚类特征。

也可以使用辅助脚本顺序执行上述四步：

```bash
export CLASSIFIER_CHECKPOINT=/models/classifier_best.pth
export YOLO_WEIGHTS=/models/signal_detector.pt
export KNOWN_REFERENCE_INPUT=/data/known_train
export KNOWN_CALIBRATION_INPUT=/data/known_calibration
export TEST_INPUT=/data/test_mixed
export OUTPUT_ROOT=/results/open_set/run01
export LABEL_MODE=filename  # DAT 类别目录用 parent；自定义映射用 json
export UNKNOWN_BACKEND=dcg
bash run_open_set.sh
```

辅助脚本使用已经训练好的 checkpoint，不启动训练。OUTPUT_ROOT 必须不存在。默认忽略 Background；分类头不含它时先 `export IGNORE_PROFILE_CLASS=""`。json 模式还需 REFERENCE_LABELS_JSON / CALIBRATION_LABELS_JSON；可设置 RECORDING_IDS_JSON、IQ_SAMPLE_RATE、IQ_STRIDE_MS、MAX_SLICES、MIN_VOTES、GROUPING_MODE、HOP_ASSOCIATION、RF_CENTER_MHZ、RF_CENTERS_JSON。更复杂的设置请直接使用四步命令。

## 决策、聚类与参数

有效观测先各自 L2 标准化，在轨迹内求均值后再次标准化，每条轨迹产生一个代表特征。`ood_score` 是代表特征到已知库第 k 近邻的欧氏距离，不是新类别概率。超过 profile 阈值，且至少默认 60% 的有效观测也超过阈值，才进入候选池。单个 PNG/MAT 的单个观测也可以成为候选。

| 状态 | 含义 | 进入未知聚类 |
| --- | --- | --- |
| `known` | 有效轨迹处于已知参考分布范围；保留原投票类别 | 否 |
| `novel_candidate` | 距离拒识且观测一致性达标 | 是 |
| `uncertain` | 投票不足、不稳定、拒识不一致或命中忽略类 | 否 |
| `no_signal`（文件） | 无检测框或无有效分组，另有 reason 区分 | 否 |
| `error`（文件/切片） | 数据或模型流程错误 | 否 |

两种发现后端只处理 novel_candidate，共享最小轨迹数和独立记录数检查。`Unknown_001` 等为当前批次内匿名簇；`Unknown_unassigned` 表示噪声或支持不足，不额外算一个类别。`estimated_unknown_class_count` 可为 0 或 1，没有预设 K。

DCG 先合并完全相同的特征构图，再将结果展开回全部轨迹，防止重复 mask 的近邻并列导致随机漏分。邻接边采用真实余弦相似度门限，默认要求互为近邻。没有连边的单个特征保持未归簇；多个完全相同的特征可构成候选社区，但仍须满足独立支持要求。Louvain 的分辨率控制社区粒度，不等于已知类别数。

| 参数 | 默认 | 调整含义 |
| --- | --- | --- |
| profile `--known_false_reject_rate` | 0.01 | 增大通常降低距离阈值，同时增加未知召回与已知误拒；需重建 profile |
| profile `--knn_k` | 5 | 已知库邻域尺度；需重建 profile |
| `--min_novel_observation_share` | 0.6 | 切片拒识一致性；会改变最终误拒/召回，不重新拟合距离阈值 |
| `--unknown_min_cluster_size` | 5 | 两种后端的最小候选轨迹数；越小越容易保留稀有类和伪簇 |
| `--unknown_min_recordings` | 2 | 两种后端的最少独立采集记录；同一 DAT 多切片不增加记录数 |
| `--unknown_min_samples` | 3 | HDBSCAN 的密度保守程度，增大通常增加噪声 |
| `--no-unknown_allow_single_cluster` | 默认允许单簇 | HDBSCAN 参数敏感性实验 |
| `--dcg_topk` | 15 | 图近邻数量；增大通常增加连通性 |
| `--dcg_cosine_threshold` | 0.85 | 真实余弦门限；降低通常更容易合并，也增加错并风险 |
| `--dcg_resolution` | 1.0 | Louvain 社区分辨率；增大通常得到更细社区 |
| `--no-dcg_mutual` | 默认互为近邻 | 放宽为单向近邻边，提高连通性也可能混合不同协议 |
| `--dcg_seed` | 42 | Louvain 随机种子；重采样/换种子可检查稳定性 |

只评估一条 DAT 时可以拒识并保留频率位置，但默认不能确认具有两条独立记录支持的新类。显式 `--unknown_min_recordings 1` 可以探索该文件内的簇；此时 K 证据较弱。只有一个候选且不满足最小簇大小时，保留未归簇。

## 固定候选池比较后端

首次推理后保存的 `unknown_features.npz` 可直接复用，无需重新运行 YOLO/CNN，也不改变已知阈值：

```bash
python scripts/cluster_unknown_features.py \
  --features /results/open_set/discovery_run01/unknown_features.npz \
  --output_dir /results/open_set/recluster_hdbscan \
  --unknown_cluster_backend hdbscan

python scripts/cluster_unknown_features.py \
  --features /results/open_set/discovery_run01/unknown_features.npz \
  --output_dir /results/open_set/recluster_dcg \
  --unknown_cluster_backend dcg \
  --dcg_topk 15 --dcg_cosine_threshold 0.85
```

输出 `unknown_assignments.csv`、更新后的候选 NPZ、簇 JSON 和原型 NPZ，保留原时间/频率和 recording_id。两次结果记录输入候选归档 SHA256，用于确认比较的是同一批特征。不会修改原评估目录的 HTML/图片；那里仍展示首次评估的最终决策。

## 输出与阅读方法

- `results.json`、`results.csv`：整批及每文件结果、profile 信息和未知发现摘要。
- `tracks.csv`：最终 `display_label`、`recognition_status`、`known_candidate_class`、`ood_score`、阈值、一致性、匿名簇号，以及时间和频率。
- `signals.csv`、`slices.jsonl`：逐组原始 `class_name`/probabilities 与最终轨迹标签。**class_name 仍是已知分类头输出；开集结论读 display_label / recognition_status。**
- `unknown_clusters.json`：后端、参数、估计 K、支持记录、代表轨迹、归一化原型及图诊断。
- `unknown_features.npz`：候选特征及路径、记录 ID、轨迹、时间/频率、拒识分数；没有候选时也保存空矩阵。
- `unknown_prototypes.npz`：同一特征空间中的归一化簇中心，可用于后续人工核验和同签名特征的相似度比较；本版没有自动将其作为永久新类注册。
- `report.html`、`visualizations/*.png`：发现完成后按最终标签绘图，并保留原始检测框的时间/频率；不是居中 mask 的坐标。
- 提取时的 `features.npz`：已知参考/校准用逐轨迹特征、单类别真值和来源。

`cluster_similarity` 是到簇中心的余弦相似度；`cluster_membership` 仅来自 HDBSCAN，DCG 为 null。图中 `known p` 始终是已知分类头概率，这些数值都不能当作匿名新类的后验概率。

开集测试也可显式提供真值标签，允许包含未见类别名；它们只用于报告。没有逐信号位置真值时，只提供文件级是否含未知及已知类出现指标，不宣称实例级未知召回、ARI/NMI 或真实新类准确率。已知/未知分类器、参考库及阈值先冻结，再汇总本批候选，是离线发现，不等同于独立输入的在线永久类别识别。

## 训练与验证建议

已有 checkpoint 可以直接尝试，SupCon 训练可作为特征改进实验；它不能保证真实未知一定可分。建议轮换留出完整已知类别进行开发：重新训练不含留出类的模型，只用剩余类校准，最后用留出类检查拒识与聚类。只隐藏一个已经训练过的类别的标签不算未知实验。最终模型纳入全部真实已知类后，重新建库和校准。真实最终测试类始终隔离。

约 10% 未知时，少量已知误拒也会污染候选池，因此应同时看已知分类准确率、已知误拒、未知召回、聚类覆盖率/噪声比例，以及换记录重采样后的 K 稳定性。不能通过强制拒识 10% 或取测试真 K 选结果。

当前编码器只看到二值检测框 mask：若两个协议的形状、排列相同，SupCon 和图聚类无法补回已经丢掉的频谱纹理/IQ 信息。YOLO 完全漏掉的未知信号也无法由本流程发现。若类别留出实验仍无法分离，再考虑频谱纹理/IQ 分支，并重新训练和校准。

自动检查：

```bash
python -m compileall -q main.py model util scripts tests
python -m unittest discover -s tests -v
bash -n run_open_set.sh
```

新增测试覆盖损失稳定性、真实 ResNet/LwF 优化步骤、旧 checkpoint、独立校准隔离、两种后端、空池/无边图/重复特征、支持过滤、固定候选池重聚类，以及 DAT 切片到带时间/频率的最终报告。训练步骤和 DAT 流程测试使用合成夹具，不代表真实 RF 数据训练或未知识别精度验证；本地未提供真实 YOLO 权重及数据，原真实 YOLO 集成检查需要配置子模块和 `UAV_TEST_REAL_YOLO=1` 后执行。
