# 运行时频谱与坐标契约

所有 MAT、PNG、DAT 都进入同一检测/分组/mask 链。科学计算只有 util/spectrogram_frontend.py 一套实现；数据适配、YOLO、训练和评估均复用它。

## 输入

- 原生频谱固定 512×750。形状错误时报告错误，不自动拉伸为科学数据。
- MAT 默认读取 summed_submatrices 幅度谱，mat_spec_domain 可明确为 amplitude / power / db / image。整数与归一化浮点遵守相同幅度规范；不凭 dtype 猜测 dB。
- PNG 是已经渲染的 uint8 图像域；运行时灰度读取，绕过幅度/dB增强，保留原生大小。拒绝 16 位或错误尺寸 PNG。
- DAT 是交错 int16 I/Q，默认小端、100 MHz，必须提供正确采集率和字节序。单个复数样本 4 字节，不足 50 ms 或文件长度不整齐时明确报错。

## IQ 与前端

真实采集率使用 iq_sample_rate（100000000 或 122880000）。100 MHz 片段先重采样到 122.88 MHz 输出网格；122.88 MHz 直接使用。50 ms 目标片段的 FFT / 聚合保持原 MATLAB 列优先约定。频率轴来自目标输出网格，不能把原采集率直接用于频谱 y 坐标。

sampling_rate 描述 122.88 MHz 输出坐标，不应改成 100 MHz 来表示采集率。幅度按固定单位归一化后进入 dB、背景统计与灰度映射。默认 frontend_mode=global，可显式选择既有 manifest/设置；PNG 不重复处理。

## YOLO 与 mask

检测在统一 resize/letterbox 图像上进行，之后还原到原生 512×750 框。detector.detect 与 detect_rgb 共用相同准备路径。裁剪、取整与边界规则在前端和 detector 中统一。

分组后，mask_input 将整组沿频率方向平移居中，填充二值框，以最近邻缩放到 CNN 输入。时间位置和组内相对结构保留。训练、普通预测、特征提取及 SupCon 视图都使用同一个 mask 构造函数。

## 时间、频率和 DAT 切片

原始框的 x / 750 乘以片段时长并加片段起点，得到文件时间。y 轴按 FFT 网格换算频偏；绝对频率还需给出采集 RF 中心。跳频组保留离散频带和带间空隙，不将包络内全部频率当成占用。

DAT 训练随机截取，训练验证取中心片段。预测计划多个 50 ms 切片，按 stride_ms 生成窗口；tail_policy=drop / cover 控制尾部，max_slices 正数均匀抽取、0 用全部可用窗口。每条频带轨迹内部投票，不因已知预测类别相同而合并不同载频的独立信号。

## 一致性检查

tests/test_spectrogram_frontend.py 保留幅度单位、无重复增强、重采样、FFT、MATLAB 聚合、采样坐标、原生检测坐标和数据集读取的运行时测试。可选的 tests/test_yolo_native_contract.py 在实际本地 YOLO 上核对真实输入张量，不需要训练权重。

旧的标注导出、生成和检测器训练工具已退出核心。它们的专用测试随功能移除；上述运行时契约继续维护。
