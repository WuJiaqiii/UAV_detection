# UAV 已知类训练与开集发现实现

基准：WuJiaqiii/UAV_detection main `2ef9d9b931b5afdd5d14513052e52d187d5af297`。21 个变更文件。此目录是增量覆盖包，不含未修改的原仓库文件、YOLO 子模块、模型权重或数据。

## 合入

在该 main 版本的干净仓库根目录执行（补丁文件位于本目录的上一级）：

```bash
git apply --check /path/to/uav_openworld_from_main.patch
git apply /path/to/uav_openworld_from_main.patch
pip install -r requirements-open-set.txt
```

也可将本目录中的源码子目录和文件按相对路径覆盖到仓库。DELIVERY.md、BASE_MAIN.json、VALIDATION.json 及两个测试日志仅为交付记录，不需要覆盖原仓库。原 YOLO 子模块及权重继续使用，未推送 GitHub。

## 已完成的流程

已知训练 CE/SupCon → 冻结 → 多信号检测分组与 DAT 投票 → 已知 profile 拒识 → 候选池 HDBSCAN / DCG-Louvain → 支持过滤 → 匿名 K、类别 ID、时间和频率。没有真实未知训练，也没有预设未知比例或总类别数。

| 文件 | 修改内容 |
| --- | --- |
| `model/representation.py; main.py; util/trainer.py; util/lwf_trainer.py` | 已知类 CE + 可选 SupCon；普通训练和 LwF |
| `model/resnet.py` | 不改变参数键的分类头前特征接口 |
| `util/open_set.py; scripts/build_open_set_profile.py` | 独立已知校准、冻结 kNN profile 与拒识 |
| `util/unknown_discovery.py; util/dcg_discovery.py` | HDBSCAN / DCG-Louvain、记录支持过滤、匿名簇及原型 |
| `scripts/evaluate_single_signal.py; util/multi_signal_eval.py; util/signal_eval_report.py` | 多信号/DAT 轨迹开集输出、最终标签与原时间/频率 |
| `scripts/cluster_unknown_features.py` | 同一候选特征池离线比较后端 |
| `run_open_set.sh; requirements-open-set.txt; docs/open_set_discovery.md` | 安装、四步评估、参数与使用限制 |
| `tests/test_open_set_eval.py; tests/test_openworld_discovery.py; tests/test_representation_training.py` | 新增开集、图发现与训练集成测试 |

## 开始使用

先阅读 [完整使用说明](docs/open_set_discovery.md)。可直接用已有 checkpoint 建立基线；在原训练命令追加 `--supcon_weight 0.1` 可训练改进特征，之后重新建库。`--supcon_weight 0` 与不提供 profile 时保留原行为。

分别准备已知参考、独立已知校准、最终混合测试三份数据。`run_open_set.sh` 依次完成特征提取、profile 校准和发现；它使用已经训练好的 checkpoint。设置 CLASSIFIER_CHECKPOINT、YOLO_WEIGHTS、KNOWN_REFERENCE_INPUT、KNOWN_CALIBRATION_INPUT、TEST_INPUT、OUTPUT_ROOT 后运行。项目文件名标签用 LABEL_MODE=filename；DAT 类别目录用 LABEL_MODE=parent。UNKNOWN_BACKEND 选择 hdbscan 或 dcg。分类头不含 Background 时设置 IGNORE_PROFILE_CLASS=""。

结果优先读取 tracks.csv 的 display_label / recognition_status 和时间/频率字段；逐组 class_name 仍保留已知分类头的原始结果。unknown_features.npz 可以用 cluster_unknown_features.py 换后端重新聚类，unknown_prototypes.npz 保存匿名簇原型。

## 验证结果与边界

- 已验证 56 个远端基准文件的 Git blob SHA；补丁在该基准副本上通过 git apply 检查、应用和逐文件字节比对。
- 在实际应用补丁后的代码上运行 109 项测试：106 通过、2 个已有 MAT/PNG 导出测试失败、1 个真实 YOLO 集成测试因缺少实际子模块/依赖而跳过。两个失败在未修改 main 的 20 项前端检查中复现。
- 84 项本次相关检查全部通过：单信号 23、多信号 17、开集 18、表示训练 12、图发现/集成 14。真实 ResNet 和 LwF 进行了合成夹具上的优化步骤，含骨干梯度与旧 checkpoint 往返检查。
- Python 编译、三个 CLI 帮助入口、Bash 辅助脚本语法通过。真实 100 MHz DAT 加载、两切片轨迹/投票、最终频率标注及固定候选池复用已检查；分类 logits/特征使用测试夹具。
- 合成报告示例在旁边的 uav_openworld_example/ 中；8 图、8 已知轨迹、16 候选，两种后端均输出 K=2。该结果只展示流程与格式，不能用作真实模型精度结论。

没有在用户真实 RF 数据上训练完整模型或评估未知召回/K 准确率。二值框 mask 无法表达相同几何下的协议纹理差异；YOLO 完全漏检的信号不会进入后续发现。Unknown_001 等是批次内匿名簇，不能自动命名为真实协议。

详细机器记录见 [VALIDATION.json](VALIDATION.json)，完整检查输出见 [TEST_RESULTS.txt](TEST_RESULTS.txt)。
