# UAV_detection 项目整体逻辑说明 (ChatGPT)

## 1. 项目目标

这个项目的目标是：对无人机信号的时频谱图进行自动处理，先利用 YOLOv5 在频谱图中检测信号对应的矩形框，再对这些检测结果做后处理，最后把后处理后的框信息转成一张二值 mask 图，并交给图像分类网络完成协议类别识别。当前 `main` 分支的训练入口实际实例化的是 `MaskImageClassifier`、`YoloV5Detector`、`SignalPreprocessor` 和 `Trainer`；同时命令行参数里已经加入了 `backbone`、`mask_img_size`、`mask_source`、`mask_fill_value` 等 mask+CNN 相关配置。([GitHub][1])

需要注意的是，当前 `main.py` 的 argparse 描述字符串和 README 里仍残留了 “Transformer classification” 的表述，但实际代码主线已经切到了 `mask + CNN`。因此，理解这个分支时应以 `main.py + util/trainer.py + util/boxmask.py` 的实际行为为准。([GitHub][1])

---

## 2. 项目整体处理逻辑

从整体上看，项目的数据通路可以概括为：

**原始频谱图（`.mat` 或 `.png`）
→ 数据集索引与读取
→ YOLOv5 检测候选框
→ `SignalPreprocessor` 对检测框做筛选与聚类
→ 将最终框渲染成白框黑底的 mask 图
→ 用 ResNet18 / ResNet34 / MobileNetV3-Small 做分类
→ 输出训练指标、混淆矩阵、检测可视化与模型权重。** ([GitHub][1])

这个设计的核心思想是：**不直接把原始频谱图送入分类器，而是先把“信号几何结构”从背景中提取出来**。YOLO 负责找框，preprocess 负责把噪声框尽量过滤掉，`boxmask` 负责把这些框变成结构清晰的二值图，分类器只学习这些框的空间布局和形状模式。([GitHub][1])

---

## 3. 各个部分的主要功能

## 3.1 `main.py`：项目总入口

`main.py` 是整个工程的调度中心，负责完成以下工作：

先解析命令行参数，包括 STFT 映射参数、预处理参数、数据集与 DataLoader 参数、YOLO 推理参数、优化器参数、以及 mask+CNN 相关参数。然后创建 `Config` 对象，按 `exclude_classes` 重建当前实验的类别映射与 `num_classes`，再创建日志目录、模型目录和结果目录。接着，`main.py` 会依次实例化 `UAVDataset`、`get_dataloader()`、`MaskImageClassifier`、`YoloV5Detector`、`SignalPreprocessor` 和 `Trainer`，最后调用 `trainer.train()` 进入训练主循环。([GitHub][1])

从职责上看，`main.py` 并不负责算法细节，它只负责把各个模块连起来，并把命令行配置注入到这些模块中。也就是说，它定义的是**系统装配方式**，不是具体的数据处理逻辑。([GitHub][1])

---

## 3.2 `util/config.py`：实验配置与目录管理

`Config` 会把 argparse 解析得到的参数全部转成属性，然后补充一组工程内部固定配置：包括 8 个协议类别映射、`mat_key="summed_submatrices"`、默认运行设备，以及实验目录结构。它会自动在 `experiments/` 下创建以时间戳命名的实验目录，并进一步划分为 `models/`、`log/`、`cache/`、`result/` 等子目录。([GitHub][2])

因此，`Config` 的作用可以理解为：**把一次训练所需的“运行参数 + 项目固定配置 + 实验目录结构”统一封装起来**。后续模块大多只与 `config` 交互，而不直接依赖命令行解析器。([GitHub][2])

---

## 3.3 `data/data_loader.py`：数据索引、文件解析与 train/val 划分

`UAVDataset` 负责从 `dataset_path` 指定的目录中索引数据文件。它支持两种输入格式：

* `input_type="png"`：按灰度图读取频谱图；
* `input_type="mat"`：从 `.mat` 中读取 `config.mat_key` 对应的二维矩阵。([GitHub][3])

文件名中携带了协议名、括号字段和可选 SNR。`UAVDataset` 会用正则从文件名中解析出：

* `protocol`
* `freq`
* `bw`
* `snr`

再根据 `config.classes` 把协议名映射成数值标签。它的 `__getitem__()` 最终返回的是：

**`x, protocol_label, freq, bw, snr, fp`**。([GitHub][3])

此外，数据集还支持 `whitelist_csv` 白名单过滤；在构造 DataLoader 时，代码会用 `train_test_split` 按 `val_ratio` 把整个数据集切成 train / val 两部分，并在类别数大于 1 时使用 `stratify=labels_np` 做分层划分。非 DDP 情况下还会通过 `sample_ratio` 做随机采样。([GitHub][3])

从系统职责上看，`data_loader.py` 负责的是**把原始文件系统中的频谱图样本整理成可供训练循环直接消费的张量与标签流**。([GitHub][3])

---

## 3.4 `util/detector.py`：YOLOv5 检测器封装

`YoloV5Detector` 在 `main.py` 中被实例化后传入 `Trainer`，后续由训练器按样本调用 `detector.detect(spec)` 获取检测框。它的职责不是训练检测器，而是作为一个**已训练 YOLOv5 权重的推理包装器**：对输入频谱图进行前处理和推理，输出检测到的矩形框列表。([GitHub][1])

从项目角度看，这一层的作用是把“原始频谱图”转成“候选框集合”，也就是从稠密像素空间切换到更稀疏、更结构化的中间表示。后面所有的 preprocess 和 mask 生成，都是围绕这些候选框进行的。([GitHub][1])

---

## 3.5 `util/preprocess.py`：检测框后处理与主信号选择

`SignalPreprocessor` 是当前工程里最关键的中间模块之一。它的类注释里明确写出了 pipeline：

1. 基础尺寸/几何过滤
2. 可选的局部能量过滤
3. 基于频率中心的 DBSCAN 聚类
4. 对 cluster 进行评分，偏向更大、更强、更连续的主簇
5. 再做 NMS 和时间排序。([GitHub][4])

从代码设计上看，`SignalPreprocessor` 不是简单删框，而是在做一件更具体的事：**从 YOLO 给出的全部候选框里，尽量挑出“最像真实主信号”的那一组框**。因此它内部既有几何阈值（如最小面积、最小宽高），也有能量/对比度相关阈值，还有 DBSCAN 聚类参数和 cluster 评分权重。([GitHub][4])

在当前 `main` 分支的主流程里，训练器会调用 `self.preprocessor.select_main_boxes(yolo_boxes, spectrogram=spec)`，也就是让 preprocess 在每张图上输出一组“最终框”。这说明，当前版本默认仍然假设**每张图有一个最主要的目标信号结构**，而不是一次性保留所有检测簇。([GitHub][5])

---

## 3.6 `util/boxmask.py`：把框渲染成二值几何图

`boxmask.py` 的职责非常单纯但非常关键：把一组矩形框变成一张单通道 mask 图。

`boxes_to_white_mask()` 会创建一张黑色背景图，然后把每个框对应的区域填成白色；`mask_to_tensor()` 再把 `uint8` mask 转成 `[0,1]` 的 `float32`，并按 `mask_img_size` 做 resize。当前默认模式是 `fill`，也就是**框内填白、框外全黑**。([GitHub][6])

这一层的意义在于：它把前面 detector + preprocess 得到的“框集合”，转成了一张可以被常规图像分类器直接读取的二维图像。换句话说，它完成了从**框级结构表示**到**图像分类输入表示**的桥接。([GitHub][6])

---

## 3.7 `model/resnet.py`：mask 图像分类器

`MaskImageClassifier` 是当前 `main` 分支的分类主干。它支持三种 backbone：

* `resnet18`
* `resnet34`
* `mobilenet_v3_small`。([GitHub][1])

这个类还会根据 `mask_in_chans` 自适应修改第一层卷积，使其适配单通道输入；如果启用 `freeze_backbone`，则只保留最后分类头为可训练参数。默认情况下，`main.py` 里使用的是 `resnet18`，输入通道数为 `1`，说明当前设计本质上是一个**单通道几何 mask 图分类任务**。([GitHub][1])

因此，分类器学习到的不是频谱图细节纹理，而是由框构成的几何模式：框的位置、大小、数量、排列与整体结构。([GitHub][6])

---

## 3.8 `util/trainer.py`：训练主循环与真实数据通路

`Trainer` 是整个项目的运行核心。它的真实逻辑已经明确是 mask+CNN，而不是 token+Transformer。训练器会先从 `config` 读取 `mask_img_size`、`mask_source`、`mask_fill_value`、可视化开关等参数，然后初始化优化器、调度器、AMP、BBox 缓存和早停器。([GitHub][5])

在每个 batch 上，`Trainer._batch_to_mask_images()` 会做下面几步：

1. 取出当前样本的原始频谱图 `spec`；
2. 调 `_get_boxes_for_sample()` 获取 YOLO 框；
3. 如果 `mask_source="final"`，就再用 `self.preprocessor.select_main_boxes(...)` 筛出最终框；
4. 调 `_build_mask_from_boxes()`，内部通过 `boxes_to_white_mask()` 和 `mask_to_tensor()` 生成 `[1,H,W]` 的单通道 mask；
5. 最后把整个 batch 堆成 `[B,1,H,W]`，送给分类器前向。([GitHub][5])

`train_one_epoch()` 和 `validate()` 都是在这个 mask 图基础上做普通图像分类训练与验证。损失函数是交叉熵，训练和验证都会统计准确率；验证结束后还会生成混淆矩阵。由此可以看出：**当前工程的真正训练对象不是原始频谱图，也不是序列 token，而是由检测框渲染得到的二值 mask 图像。** ([GitHub][5])

---

## 4. 一次样本从输入到输出究竟经历了什么

以一张样本图为例，当前项目的处理逻辑可以概括成下面这条链：

首先，`UAVDataset` 从磁盘读取一张 `.mat` 或 `.png` 频谱图，并同时返回标签、频率、带宽、SNR 和文件路径。然后，训练器调用 YOLO 检测器在这张图上找候选框；如果开启了 bbox cache，会优先从缓存中读取框结果。接下来，`SignalPreprocessor` 根据面积、宽高、局部能量、频率聚类和 cluster 打分，从所有候选框里选出最主要的一组框。之后，`boxmask` 把这组框绘制成一张白框黑底图，并 resize 到 `mask_img_size`。最后，这张 mask 被送入 `MaskImageClassifier`，输出各类别 logits，经交叉熵计算损失并参与反向传播。([GitHub][3])

从信息流角度理解，这个过程其实是在连续做三次“信息压缩”：

* 第一次：原始图像 → YOLO 候选框
* 第二次：全部候选框 → preprocess 主框集合
* 第三次：主框集合 → 二值 mask 图

最后，分类器只针对这份被压缩后的几何信息做判断。([GitHub][1])

---

## 5. 训练阶段会产生哪些结果

训练期间，项目会在 `experiments/时间戳/` 下自动生成实验目录，并进一步保存：

* 配置文件 `config.yaml`
* 训练日志
* 模型权重
* bbox 缓存
* 检测可视化结果
* 混淆矩阵结果。([GitHub][2])

验证阶段默认会保存检测可视化，也会把整个验证集的预测结果汇总成混淆矩阵图。这些输出的作用分别是：

* **日志**：看 loss/acc 曲线；
* **可视化**：看 YOLO 框和 preprocess 最终框是否合理；
* **混淆矩阵**：看哪些协议之间最容易混淆；
* **checkpoint**：保留当前最优或定期模型。([GitHub][5])

---

## 6. 当前 main 分支最需要注意的一点

当前 `main` 分支在“入口描述”和“实际训练实现”之间还存在一处明显不一致：

* `main.py` 的 argparse 描述字符串仍然写着 “Yolov5 detection + preprocessing -> Transformer classification training”；
* README 也仍然保留了旧的 Transformer 表述；
* 但入口代码已经实例化 `MaskImageClassifier`，训练器内部也已经明确走 `_batch_to_mask_images()` 这条 mask 图分类链路。([GitHub][1])

因此，给这个分支写文档时，最准确的说法应当是：

**这是一个“YOLO 检测 + 框后处理 + 几何 mask 图分类”的工程，而不是当前仍在使用 Transformer 的工程。** ([GitHub][1])

---

## 7. 一句话总结

这个项目当前 `main` 分支的本质，是一个**先检测、再抽取框级结构、最后用 CNN 识别几何模式**的两阶段系统：
YOLO 负责“找哪里有信号”，preprocess 负责“从候选框里找主信号”，boxmask 负责“把框转成几何图”，ResNet 负责“根据这些几何结构完成协议分类”。([GitHub][1])

