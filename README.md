# UAV_detection v3

无人机射频开集识别：原始IQ → 时频谱 → YOLO检测 → 信号分组 → mask分类器 → 已知拒识 → 未知聚类 → 竞赛提交。训练、推理、导出和增量实验使用独立入口，共用实际算法实现。

本版按全新训练设计：删除旧结果恢复和兼容入口，不支持旧训练续跑或优化器恢复。请使用新目录运行实验；原有实验资料无需删除。YOLO检测器权重继续作为外部输入。

## 安装

Python 3.11，PyTorch 2.6 / torchvision 0.21。先安装适合本机CPU/CUDA的PyTorch，再安装其余依赖：

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
```

## 从头训练并完成六项推理

```bash
python scripts/competition_workflow.py \
  --work-dir "$PWD/experiment/competition_v3_s42" \
  --train-root /path/to/train_dataset \
  --data-root /path/to/competition_data \
  --yolo-weights /path/to/detector.pt \
  --confidence 0.02 --wait-for-gpu --gpu-min-free-mib 16384
```

训练数据按真实类别文件夹组织；测试目录包含`1-1`到`2-2`。流程使用全部20个已知类别，独立划分训练/验证/校准，执行完整时长推理、未知类联合选择和严格导出。`no_signal`仍存在时，严格导出会返回待检查状态；人工占位结果需明确加`--no-signal-code 24`，且不会标为正式可提交。

## 入口

| 入口 | 用途 |
| --- | --- |
| `run.sh train/experiment/profile/predict/infer/cluster/visualize/ablate` | 单个训练、评价或分析任务 |
| `scripts/competition_workflow.py` | 从头完成全已知训练、六项推理和导出 |
| `infer.sh TRAIN DATA` | 已完成的新训练结果执行六项推理 |
| `submit.sh --batch MANIFEST --output OUTPUT` | 核验批次、联合聚类和导出 |
| `lwf.sh prepare/run/check/...` | 从原始数据构建增量实验与方法对比 |
| `scripts/benchmark_unknown_clustering.py` | 冻结特征上的聚类方法/参数矩阵 |

YAML相对路径以配置文件所在目录为基准；CLI相对路径以当前工作目录为基准。单任务输出自动进入`experiment/`，完整工作流的目录通过`--work-dir`指定。`--dry_run`用于单任务参数检查；完整工作流使用`--dry-run`。

## 当前文档

- [六项竞赛流程与提交](docs/competition.md)
- [系统链路与算法实现](docs/pipeline.md)
- [增量识别与LwF对比](docs/incremental.md)
- [开发集实验和聚类矩阵](docs/experiments.md)
- [结果目录与存储策略](docs/outputs.md)
- [开发、测试与v3改动](docs/development.md)
