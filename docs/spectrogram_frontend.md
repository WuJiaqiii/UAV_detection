# V2：MAT / IQ 共用 512×750 幅度谱与原生 PNG

## 固定契约

- 输出频谱：512 行频率 × 750 列时间，50 ms，目标频谱采样率 122.88 MHz。
- 读取器输出：float32 相对幅度 [0,1]，按 MATLAB 的最大值归一化和 uint16 量化规则统一。
- PNG：**原生 512×750** 三通道等值灰度，不加坐标轴、颜色条、不 resize、不补边、不切块。
- 网络输入：由公共函数单独按 --imgsz 缩放/补边；该图不用于人工标注。
- 行 0 是最低频率，不上下翻转。频率 bin 为 -61.44 到 +61.20 MHz，间隔 0.24 MHz。
- 时间列间隔 8192/122880000 秒，约 66.67 微秒。不能将 FFT hop 简单设为 8192 替代聚合。

## 两套数据集

MAT/PNG 继续使用 data/data_loader.py，保留原文件名类别、频率、带宽、SNR 解析。
IQ 继续使用 data/iq_dat_loader.py，按目录取类别，不用伪造频率真值进行端到端评估。

已有 MATLAB MAT 的 summed_submatrices 是归一化后的幅度、uint16 0..65535，默认：
--mat_spec_domain amplitude。power / db 仍支持，但必须按真实含义声明。
读取 MAT 不重做 STFT，不重采样，不再直接转换为 dB；先进入公共幅度表示。
不是 512×750 的科学矩阵会报错，不通过图片 resize 掩盖算法不一致。
已有 0..255 PNG 可按 image 接入，只有几何验证，不再做背景增强；无法恢复其原始功率。
MAT 导出的 PNG 保留原文件名，只将扩展名改为 .png；同时保留相对于输入目录的子目录，避免不同目录下同名文件互相覆盖。
对应 YOLO 标签保留相同相对路径，扩展名为 .txt。若源文件仅扩展名或大小写不同并导致目标路径冲突，会在写入前报错。
IQ 和 PNG 输入继续使用原有哈希命名规则。导出图的处理参数与来源以 frontend_config.json 和 metadata.jsonl 为准。

IQ 的原始采样率默认 **100 MHz**，与输出网格采样率分开保存：
1. 从交错 int16 I/Q DAT 截取 50 ms，即 5,000,000 个复采样点。
2. scipy.signal.resample_poly 按 768/625、Kaiser(5)、constant padding 重采样到 6,144,000 点。
3. 周期 Hamming 窗，512 点 FFT、hop=512，不重叠、不留盲区，fftshift。
4. 每 16 个连续复数 FFT 求和，再取 abs，得到 512×750。
5. 最大幅度归一化、16 位量化，再转 float32 相对幅度。
这是所提供 MATLAB 的核心表示，不是功率平均或普通 STFT 后缩放。
另一支持的真实源采样率 122.88 MHz 会跳过重采样，仍必须读取恰好 50 ms。
--iq_segment_ms=50、--iq_n_fft=512、--iq_hop_length=512 是固定兼容参数，不接受旧 10/100 ms、hop=256。

不同 FFT 库可能有浮点/量化末位差异，不承诺与 MATLAB 跨语言逐位相等；
Python 导出与主干则调用同一实现，要求逐像素相等。
此处不复现 MATLAB 的随机搬频、混合和加噪，已有 MAT 中这些改变本来就属于不同信号内容。

## 一次公共归一化

公共幅度 -> 20log10 -> 稳健背景标准化 -> 固定范围 uint8 映射。
对数幅度下限统一为 1/65535（约 -96.33 dB 相对幅度），不再用不同单位的 epsilon。
不再有双遍 percentile/log1p，也不在 detector 或导出器再次增强。

默认 global：用参考区中位数和 max(1.4826×MAD,1 dB)。
默认参考区为中央 ±50 MHz，**对 MAT 和 IQ 使用同一个掩码**。
100 MHz IQ 的重采样不会创造 ±50 MHz 之外的真实接收带宽，所以不把插值带外低值用于背景估计。
掩码只选背景统计，不裁掉或抹掉 MAT 外侧频率行；不同采集带宽的物理差异仍然存在。
frequency 为实验选项，可能削弱持续宽带信号；none 不做背景标准化，需显式设置 dB 范围
（例如 --frontend_z_min -96.33 --frontend_z_max 0），默认 [-2,8] 是标准化尺度。
背景标准化不是标定 SNR，也不能恢复量化或相干相消丢失的信息。

## MAT：一文件一张图

在项目根目录执行，输出目录必须为空、位于源目录之外：
```bash
python scripts/export_iq_yolo_dataset.py \
  --input_dir ./data/test_signal \
  --output_dir ./data/mat_png_v2 \
  --input_type mat \
  --mat_spec_domain amplitude \
  --frontend_mode global \
  --split_mode none
```
每个 MAT 生成一张 512×750 PNG，位于 images/unassigned，文件名保留不变（仅更换扩展名）。
例如输入 DeviceA/example.mat，输出 images/unassigned/DeviceA/example.png，标签对应 labels/unassigned/DeviceA/example.txt。
正式划分时将 unassigned 替换为 train 或 val，其余命名规则相同。MAT 输入不使用 --iq_* 参数。
这适合先核对效果或人工标注，不生成训练配置。
scripts/plot_mat_to_png.py 是同一导出流程的 MAT 默认入口。

## IQ：裁剪、重采样、一片段一张图

```bash
python scripts/export_iq_yolo_dataset.py \
  --input_dir /data/iq \
  --output_dir /data/iq_png_v2 \
  --input_type iq_dat \
  --iq_sample_rate 100000000 \
  --iq_segment_ms 50 \
  --iq_n_fft 512 --iq_hop_length 512 \
  --segments_per_file 1 \
  --frontend_mode global \
  --split_mode none
```
默认一份录音取一个中心片段；--segments_per_file N 在录音内均匀选 N 段，每段一张图。
所有切片按源录音分组，不把同一录音的相邻切片分到训练与验证两边。

## 正式划分与标注

正式数据用 --split_mode group --val_ratio 0.2，至少两个真正独立来源组。
--groups_json 可传“相对源路径 -> 采集会话 ID”的 JSON，已有 Figure/SNR 名称会保守分组。
来源信息不充分时不要仅凭仿真 Task/Sample 编号假定独立原始采集。
单组预览模式不能通过 --finalize 伪装成正式训练集。

输出保留 metadata.jsonl 和 frontend_config.json：
前者记录源路径、组 ID、IQ 起始采样点/长度、原生形状和 RGB 像素 SHA256；
后者记录源采样率、完整频谱算法、背景参数和模型尺寸。
人工标注在 **512×750 PNG 坐标**上进行，单类 0=signal_frame，导出对应 YOLO txt；
无信号图片需人工确认后保存空 txt，不默认把未标注图当背景。
全部完成后：
```bash
python scripts/export_iq_yolo_dataset.py --output_dir /data/yolo_v2 --finalize
```
生成 data.yaml。旧 v1 的方形图片、标注、manifest 不能混用，需重新导出并转换/复核标注。

## 像素与模型张量一致性

```bash
python scripts/check_yolo_input_equivalence.py \
  --dataset ./data/mat_png_v2 --source_root ./data/test_signal --limit 0
```
从原始 MAT 或同一段 IQ 重新读取，调用实际适配器，核对：
- native PNG 512×750 像素、哈希与主干渲染相等；
- 网络 resize/letterbox 结果相等；
- 模型坐标到原生坐标的往返。

实际本地 YOLO 子模块和权重具备时：
```bash
python scripts/check_yolo_input_equivalence.py \
  --dataset /data/yolo_v2 --source_root /data/mat \
  --yolo_weights /models/new_best.pt --device cpu --limit 20
```
会安装与训练入口相同的内存适配器（不写入 YOLO 子模块），比较真实训练加载器张量与
框架实际 AutoShape 后端张量，再比较矩阵/PNG 路径预测。缺少权重时明确报告未运行。
直接调用未经适配的 AutoShape(native_png) 或普通 YOLO train.py 不保证逐位一致：
它们的 AREA/LINEAR、ceil/round、矩形补边可能不同。

## YOLO 训练使用公共入口

```bash
python scripts/train_yolo_frontend.py \
  --dataset /data/yolo_v2 \
  --weights /models/yolov5s.pt \
  --epochs 100 --batch_size 16
```
此基线禁用随机增强、multi-scale 和 rect batching，并让实际训练加载器使用公共 resize。
--imgsz 从 manifest 读取，默认 640；输入网络时才变为 640×640。
后续有意增加增强时，要区分基础预处理一致性与随机训练增强，重新验证基础契约。

## 框架训练

MAT 数据：
main.py --input_type mat --mat_spec_domain amplitude --frontend_config /data/yolo_v2/frontend_config.json ...
IQ 数据：
```bash
UAV_IQ_DATASET_PATH=/data/iq \
UAV_YOLO_WEIGHTS=/models/new_best.pt \
UAV_FRONTEND_CONFIG=/data/yolo_v2/frontend_config.json \
bash run_train_iq.sh
```
MAT manifest 可用于 IQ，因为公共频谱/归一化契约相同；不会把 MAT 导出的无关源采样率覆盖到 IQ。
--sampling_rate 恒指输出网格 122.88 MHz；--iq_sample_rate 是真实采集速率。
缓存包含 v2 指纹；随机 IQ 裁剪仍禁用文件级框缓存。旧权重需要重新微调/评估。

## 验证范围

python -m unittest discover -s tests -v
python scripts/evaluate_frontend.py --output_dir work/frontend-diagnostics
测试覆盖完整 50 ms 重采样、与独立 MATLAB reshape 公式对照、MAT/IQ 适配器、
一 MAT 一原生 PNG、像素重建、模型几何及分组标注检查。
诊断图使用合成频谱单元，不代表 IQ 实测 SNR、真实检测精度或 mAP。
真实录音、人工框和 YOLO 权重不在仓库内时，不声称实际训练精度或带权重检查已通过。

CI 另用公开 YOLO 快照 88af13e3e25602344aab68d44f9d0b666dacf695 验证真实训练加载器与真实 AutoShape 的输入张量；
网络为随机初始化，仅检查预处理。项目原固定子模块不被改写，仍需用你的本地版本和真实权重复核。
