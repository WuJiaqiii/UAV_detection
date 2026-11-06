# 快速开集推理

在原有 infer 命令后加 `--fast`。检测、分组、网络、轨迹投票、profile 拒识和未知聚类仍使用原有实现，绘图和详细结果输出关闭。

```bash
bash run.sh infer \
  --config config/infer_open_set.yaml \
  --inputs /media/kaneki/05E4235537EC7AC2/dataset_wjq/1-1 \
  --recursive \
  --unknown_input_mode single \
  --fast
```

也可以在部署 YAML 的 infer 段写 `fast: true`。需要把运行日志重定向到文件时，可加 `--no-progress`；进度条默认写 stderr。`--no-fast` 恢复完整输出。

## 进度和输出

加载期间提示 Loading model/profile；之后显示整体 Files 进度，以及当前文件已处理的 slice 数量、速度和预计剩余时间。切片读取失败也计入已处理数量，结果表保留错误信息。Recording IDs 进度表示录制身份解析/必要的文件哈希；Clustering 提示表示正在对整个批次的未知候选聚类。这一阶段不伪造聚类百分比。

输出在启动时打印的 experiment/open_set_inference_北京时间 目录中：

- **predictions.csv**：最终结果表，UTF-8 BOM，可直接用常见表格软件打开。
- **run.json**：参数、时间、完成状态等小体积运行记录。发生致命异常时还会写 error.log。

快速模式不保存预览图、PCA/t-SNE、HTML、逐切片 JSONL、详细 CSV/JSON、未知特征或原型 NPZ。已有切片详情在每个文件完成拒识后释放，只保留后续聚类需要的轨迹和特征。未知聚类需要整个输入批次的候选，最终 CSV 在聚类结束后写入；可以用进度条查看中途进展。需要离线重聚类、补图或详细诊断时，应使用完整模式重新运行。

## single：每个文件一行

| 字段 | 含义 |
| --- | --- |
| path | 输入文件完整路径 |
| label | 已知类别名、Unknown_001 等簇编号，或未确定/错误状态 |
| status | known / novel_candidate / uncertain / no_signal / error |
| error | 文件或切片错误摘要，正常时为空 |

它与完整模式的 single_signal_prediction 一致：先在同一原始录制的全部轨迹中选主轨迹，再使用其最终决策。次轨迹不会因为被拒识而顶替已知主轨迹；同一录制的其他文件可以继承该录制决策。

Unknown_unassigned 表示已被拒识、但未满足成簇支持条件；uncertain 表示有效证据不足，两者不会伪装成已知类别。完全读取失败的文件始终标为 error。部分切片失败时保留其余有效信号，同时在 error 列提示；公共命令会返回非零错误码。

## multi：每条频率轨迹一行

```bash
bash run.sh infer \
  --config config/infer_open_set.yaml \
  --inputs /path/to/mixed_signals \
  --recursive --unknown_input_mode multi --fast
```

列为 path、track_id、label、frequency_low_offset_mhz、frequency_high_offset_mhz、frequency_low_absolute_mhz、frequency_high_absolute_mhz、status、error。同一文件中的多个信号保留为多行，track_id 是文件内轨迹编号。

offset 两列是相对采集中心的 MHz 偏移。提供 `--rf_center_mhz 2400` 或逐文件 rf_centers_json 后，absolute 两列才有真实 RF 频率；未声明中心时留空。跳频轨迹的上下界是其全部观测频段的外包范围，中间可能包含未占用频段。轨迹表示频率链路候选，不能直接当作物理设备 ID。

无有效信号的文件仍保留一行 no_signal，频率和 track_id 为空；读取失败保留一行 error，避免文件悄悄消失。

## 已有 profile 的兼容性

本次没有修改被 inference_signature 绑定的特征前端、检测器、分组、mask、网络、投票和 profile 算法源码；新增的 fast / progress 开关也不进入特征签名。同环境、同权重和同特征设置下，原 profile 可以直接使用。若原 profile 本来就与当前 main 或环境不匹配，仍需按原有规则重新生成。

只优化绘图、输出、切片详情存储，以及不必要的输入哈希。显式录制 ID 或 Figure 来源 ID 已能确定身份时，不再为了丢弃其结果而扫描整个大文件；其余输入仍计算内容哈希，保持相同内容和同录制的聚类语义。

## 其他加速方法

1. **确认 GPU 路径。** 默认 auto 会选可用 CUDA；可显式指定 `--device cuda:0`。它同时供分类器和 YOLO 使用。先确认实际安装的是匹配 GPU 的 PyTorch 环境。
2. **把 DAT 放在本机 SSD。** 大 DAT 的频谱计算和身份哈希都会访问磁盘。若已有可信的原始录制 ID，可传 recording_ids_json，避免身份解析时再次通读文件；不要为重复裁片随意生成互不相关的 ID。
3. **调分类批量。** 可尝试 `--classifier_batch_size 64` 或 128，观察显存和速度。它只合并同一切片里的多个分组，不会把多个 YOLO 切片合批；每片只有一两个组时收益有限。
4. **限制 CPU 过量线程。** 对 CPU FFT、NumPy/BLAS 与 GPU 混合执行，可比较 OMP_NUM_THREADS / MKL_NUM_THREADS 为 2、4、8 时的总时长。最佳值依机器和负载决定，不默认使用最大核数。
5. **下一步代码优化：有界预取和跨切片批量。** 当前 infer 的 IQ 读取/频谱与 YOLO 主要按片串行。可以增加有限长度的 CPU 预取队列、YOLO 切片合批、分类组跨片合批，再按原顺序关联轨迹。训练数据集已有的预处理线程选项不会自动加速这条 infer 路径。
6. **FP16、AMP、TensorRT 或编译。** 适合在确认模型计算是瓶颈后单独评估；可能改变检测框或嵌入数值，应验证精度与拒识阈值，并按变化重新生成匹配的参考/校准 profile。torch.compile 有首次编译成本，短任务不一定划算。本次没有打开这些数值相关优化。
7. **减少切片是有损加速。** 增大 iq_stride_ms 或限制 max_slices 可能漏掉短时信号、改变轨迹和聚类。本项目把这些选项绑定在 profile 签名中，不能直接改命令后继续配旧 profile；需以相同采样策略重新提取参考与校准。

模型已使用 eval 和 torch.inference_mode；再次添加 no_grad 不会解决串行频谱计算、磁盘或逐片同步问题。HDBSCAN auto 仍需 N×N 距离矩阵，关闭图片不会消除未知候选过多时的内存与聚类开销。更换聚类后端或拆批可能改变簇结果，需独立验证。

上述模型优化的依据见 [PyTorch 性能调优](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide) 和 [torch.compile 的启动成本](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.reducing_compile_time.html)。这些属于后续优化方向，不是本次实测加速倍数。
