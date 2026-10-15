# 统一频谱前端与 YOLO 数据集迁移

## 数据边界

两套数据集继续独立组织、解析标签：
- `data/data_loader.py`：MAT/PNG，标签取自原有文件名，支持单/多目标。
- `data/iq_dat_loader.py`：类别目录中的交错 int16 I/Q DAT，训练随机裁剪、验证中心裁剪，类别取目录名；没有真实频率标签。

公共科学表示为 [频率, 时间] 的未截断对数功率矩阵。行 0 是最低频率，不做上下翻转。
IQ 使用 Hann 窗、无隐式补零、无中心 padding、按窗能量归一化的周期图；FFT 分批计算。
不再减整图最大值，不再截断到 [-100,0]，不再重复做分位数或 log1p 增强。
PNG 已经是显示域，只保留像素，不宣称可以从 PNG 恢复功率或 SNR。

MAT 必须声明单位：`--mat_spec_domain power|amplitude|db|image`。
默认 power 不能用于负值 dB 矩阵。未知来源的数据必须先确认单位；不要靠值域猜测。
已有渲染灰度存于 MAT 时使用 image；hardnegative 图片域 MAT 也使用 image。
旧 PNG 已经过有损动态范围映射，更新算法需要从原始 MAT/IQ 重新导出。

## 背景标准化

`util/spectrogram_frontend.py` 是唯一的科学处理与栅格生成实现。

- global（默认）：每个检测块减中位数，除以 max(1.4826×MAD, 1 dB)。
- frequency（实验选项）：先按频率取时间 20% 分位背景并沿频率做 9-bin 中值平滑，再执行稳健标准化。
- none：不做背景标准化；使用固定 dB 显示范围。必须显式设置例如
  `--frontend_z_min -100 --frontend_z_max 0`，默认 [-2,8] 是标准化尺度，不是 dB 范围。

最终固定 [-2,8] 标准化范围映射 uint8。该统计量不是经标定的 SNR 或 CFAR 虚警概率。
默认 global 不依赖每个频率一定有静默时间，但高信号占用仍会污染估计。
frequency 可以补偿有色背景，也可能减掉持续宽带信号。必须检查持续信号和高占用场景。
固定数值下限、int16 量化和接收机饱和限制了极端增益不变性；归一化不能恢复低 SNR 丢失的信息。

旧 `--yolo_input_norm`、`--yolo_input_p_low/high`、`--yolo_db_min/max/eps` 已从主入口移除。
兼容 Python 转换函数仅转调公共实现，分位数/log_gain 参数会明确报错。
旧 YOLO 权重不应直接作为新前端的精度结论；需要重新微调与验证。

## 分块和坐标

默认时间块宽 768 列、重叠 192 列；频率轴完整保留。每块在自身有效区域计算背景统计，
之后等比例缩放并以 114 补边到 640×640，三个通道内容相同。所有最终输入尺寸是 32 的倍数。
导出 PNG 就是这个最终 RGB 栅格；运行时调用同一个 render_tiles()，不再从头增强它。

每块保存精确的横纵缩放比例（考虑整数 resize 尺寸）、补边和原始起始列。
检测框先映射回完整频谱，裁去 padding，再按类别做全局 IoU NMS。
后续归组和分类仍使用完整频谱坐标与完整观察时长。

跨边界帧依赖重叠区域获得完整框；长于重叠区/检测块的帧可能仍被切断。
当前不盲目拼接相邻框，以免合并不同帧；应按最长目标帧校准块宽/重叠。
`--detector_tile_width 0` 可禁用分块，但极宽谱图整图缩放会损失细节。

IQ 默认 hop 从 1024 改为 256（FFT 512），消除原来窗之间的盲区。
100 MHz、100 ms 对应约 512×39061，增加计算与内存；训练脚本 batch 从 128 降至 4。
分块改善检测分辨率，不会减少完整频谱本身的存储开销。按内存继续调整 batch/workers。
IQ 频率坐标按真实 FFT bin 计算；旧 MAT/PNG 默认保留原来的端点映射，若原数据确为 FFT 网格可选 `--frequency_axis fft`。

## 从 IQ 导出用于人工标注的图片

先导出少量独立录音，检查信号和背景，再批量标注。输出目录必须为空且在源目录之外。
下面的采样率、字节序必须与采集设备一致。

```bash
python scripts/export_iq_yolo_dataset.py \
  --input_dir /data/iq \
  --output_dir /data/yolo_frontend \
  --input_type iq_dat \
  --iq_sample_rate 100000000 \
  --iq_segment_ms 100 \
  --iq_n_fft 512 \
  --iq_hop_length 256 \
  --iq_byte_order '<' \
  --segments_per_file 1 \
  --frontend_mode global
```

默认按源文件分组后划分 train/val，再裁剪、切块。1 段时取中心，多段时在文件内均匀取样。
MAT 的 Figure 切片、文件名 SNR 变体尽可能归同组。对同一次采集拆成多个文件的情况，
使用 `--groups_json` 指定相同采集会话 ID，避免数据泄漏：
`{"DeviceA/a.dat": "session-01", "DeviceA/b.dat": "session-01", "DeviceB/c.dat": "session-02"}`。
该划分不承诺类别分层；小数据集需检查每类是否有足够独立录音。

输出：
- images/train、images/val：无坐标轴、无 colormap、无 JPEG 损失的最终检测输入。
- labels/train、labels/val：留给人工标注，不生成假的空标签。
- metadata.jsonl：源路径、采样位置、组 ID、坐标变换、像素 SHA256。
- frontend_config.json：冻结前端、STFT 和源数据格式配置。

标注工具应导出 YOLO txt，单类 0 = signal_frame。显式确认的纯背景图使用空 txt；
未审阅图片不要通过创建空 txt 批量标成背景。
所有图片标注完成后执行：
```bash
python scripts/export_iq_yolo_dataset.py \
  --output_dir /data/yolo_frontend --finalize
```
该操作验证所有标签存在、类别为 0、框为合法归一化坐标后才创建 data.yaml。

## MAT/PNG 数据集

```bash
python scripts/export_iq_yolo_dataset.py \
  --input_dir /data/mat \
  --output_dir /data/yolo_mat \
  --input_type mat --mat_spec_domain power --mat_key summed_submatrices
```

`scripts/plot_mat_to_png.py` 现为相同导出流程的 MAT 默认入口。
PNG 源使用 `--input_type png`；只进行必要的分块/几何变换，不再增强像素。
不同采样带宽的 MAT 和 IQ 不能假设拥有相同的频率物理尺度。

## 一致性检查

```bash
python scripts/check_yolo_input_equivalence.py \
  --dataset /data/yolo_frontend --source_root /data/iq --limit 20
```

该命令从元数据读取完全相同的 IQ 片段/原始矩阵，重新渲染，核对 PNG 像素、SHA256、
CHW/255 转换及坐标往返。默认不加载模型，会明确报告实际 YOLO 张量检查未运行。

有真实权重和完整 YOLO 子模块时：
```bash
python scripts/check_yolo_input_equivalence.py \
  --dataset /data/yolo_frontend --source_root /data/iq --limit 20 \
  --yolo_weights /models/best.pt --device cpu
```

它调用实际 YOLO 训练加载器（augment=False、rect=False）核对 PNG 的 CHW/255 张量，
并在真实 AutoShape 后端入口挂 hook，核对矩阵路径与 PNG 路径的输入张量完全一致，再比较预测结果。该检查验证预处理一致性，不代表检测精度。
基础输入不应二次缩放；训练基线关闭 multi-scale、mosaic、mixup、HSV、几何增强、
翻转及 Albumentations，再检查训练加载器的最终 CHW/255 张量与同一 PNG 对齐。
检查必须在你实际使用的 YOLO 子模块和依赖环境中运行；无法加载该环境时会报错，不会假装通过。

## 训练接入

IQ 分类训练使用导出的同类型配置，避免 STFT 或增强参数漂移：
```bash
UAV_IQ_DATASET_PATH=/data/iq \
UAV_YOLO_WEIGHTS=/models/new_best.pt \
UAV_FRONTEND_CONFIG=/data/yolo_frontend/frontend_config.json \
bash run_train_iq.sh
```

MAT/PNG 的原有入口与标签解析保持不变；主入口可用同源数据导出的
`--frontend_config /data/yolo_mat/frontend_config.json`。
配置拒绝把 IQ manifest 用到 PNG 数据集，防止把科学域和已渲染域混用。
现有 IQ infer 入口仍拒绝无频率真值的端到端评估；直接 detector.detect() 可检测，
但本次不将占位频率伪装成真值，也未新增部署服务。

缓存签名包含前端版本、完整渲染参数和数据域；随机 IQ 裁剪继续禁用文件级缓存。
切换前端后也需重新评估/训练后端分类器，因为检测框几何分布发生变化。

## 验证与实验范围

```bash
python -m unittest discover -s tests -v
python scripts/evaluate_frontend.py --output_dir work/frontend-diagnostics
```

诊断覆盖同一合成场景不同整体增益、不同 active-sample SNR、持续信号案例。
报告与对照图不等同于真实 IQ 的 mAP、召回率或误检率。
实际录音、人工框、训练权重未包含在仓库，最终参数应依据留出的真实标注数据选择。
