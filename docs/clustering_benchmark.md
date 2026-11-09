# 固定特征的聚类算法与参数结果矩阵

本工具复用 `open_set_experiment_*/test/unknown_features.npz` 和完整真实标签报告，只执行CPU聚类。不会加载YOLO/CNN、重新训练或改变拒识阈值。每个输入必须来自官方20类的类别留出实验，例如16类已知＋4类伪未知；不能使用已学过20类的分类器再事后将其中4类声明为未知。

## 已核查的开发基准

| 实验 | 伪未知型号 | 测试已知/未知记录 | 使用范围 |
|---|---|---|---|
| open_set_experiment_20261104_074313 | DEVENTION DEVO、DJI AVATA2、DJI MAVIC3 PRO、FRSKY X14 | 96 / 160 | 主开发基准，无文件/切片错误 |
| open_set_experiment_20261107_085954 | DJI AVATA2、DJI MAVIC3 PRO、FRSKY X20R、HubsanZino2 | 128 / 160 | 第二组记录级开发对照；原结果存在12处突发标签冲突 |

程序检查类别集合、测试报告完整性、缓存轨迹集合、原始记录ID、内容摘要分区及真值一致性。伪未知不能进入训练/验证/校准集合。源文件路径仅作为身份，不需要在当前机器找到原始DAT。

两组历史数据均已参与开发，排名应称为“当前开发条件下最好”。文件哈希隔离不能证明采集场次独立，也不能据此宣称官方测试或真实同场多设备性能。分类器留出不代表固定YOLO检测器也未接触这些型号。任务二事件完整性须另外验证。

## 算法与搜索条件

默认 `config/clustering_benchmark.yaml` 包含10类方法：记录密度HDBSCAN（auto）、现有density_merge、K-means、球面K-means、平均/完全/Ward层次聚类、对角/球形GMM、DBSCAN、OPTICS、谱聚类和DCG-Louvain。

- `standard`：原始特征、PCA32、PCA64；随机算法用42/43/44三个种子，确定性算法只运行一次。按默认配置，每组数据1107次拟合，两组共2214次拟合，另生成180条不依赖标签的选K结果。
- `quick`：两个特征空间、单个种子、缩减参数列表；两组共352次拟合，适合先检查环境。
- `full`：原始特征及PCA16/32/64/128，五个种子，K候选1–10。PCA维度超过样本/特征秩限制时明确标记skipped，不静默截断或丢样本。
- 自定义：复制YAML并增加参数取值，用 `--config` 指定；`--preset standard` 保留文件中的网格。`--algorithms` 可选择部分算法。

这里的多种子重复只检查同一缓存上的算法随机性，不会重新划分训练/测试记录，不能替代新的类别留出实验。

PCA在每份冻结候选的主轨迹特征上无标签拟合，再做L2归一化，不使用whitening；不改变原拒识特征/分数，也不是二维可视化。它是对当前批次的传导式后处理。

需要K的方法先扫描声明的候选范围，保存每个固定K的诊断行。主结果仅使用标签无关的选择行：在所有K个簇都通过公共支持门槛的候选中，最大化

`coverage * (cosine_silhouette + 0.1) - 0.01*(K-1)`。

K=1以silhouette=0作参考；无簇没有选择分数。这里的常数是冻结的开发规则，不是最优性保证。选K过程不读取真实型号、真实K或任何真实标签评分。density_merge保留既有记录子采样稳定性和合并策略，在同一声明的候选数量范围内搜索；HDBSCAN/DCG/DBSCAN/OPTICS自然估K，不强行压到该范围。输出保留 `k_information` 区分这些数量先验。

DBSCAN的eps使用当前无标签特征的第min_samples邻居余弦距离分位数；实际eps保存在每个case的diagnostics。球面K-means每轮归一化中心。GMM不收敛会记录 `converged=false` 并退出主排名。谱图不连通等警告保留在case JSON中。

所有算法使用相同的原始记录主轨迹池、原拒识和公共最小簇/独立记录支持过滤；已知误拒污染不能用真值提前删除。拟合前按原始记录ID的摘要统一排序，避免含类别名的目录顺序影响距离相同的邻居选择；输入归档重排不改变拟合数组。未通过支持的点保留为未分配。没有强制类别数量相等，没有把全部小簇无条件合入最近大簇。当前基准固定为单设备记录，不能将文件级真值套用到真实多设备文件的所有信号上。

## 服务器运行

更新到temp版本后，在仓库根目录、原myconda环境中执行：

```bash
ROOT="$PWD/experiment"
OUT="$ROOT/clustering_matrix_standard_20261109"

python scripts/benchmark_unknown_clustering.py \
  --source-experiment \
    "$ROOT/open_set_experiment_20261104_074313" \
    "$ROOT/open_set_experiment_20261107_085954" \
  --output "$OUT" --preset standard --threads 1 --plan-only

python -u scripts/benchmark_unknown_clustering.py \
  --source-experiment \
    "$ROOT/open_set_experiment_20261104_074313" \
    "$ROOT/open_set_experiment_20261107_085954" \
  --output "$OUT" --preset standard --threads 1
```

`--plan-only`检查输入并打印网格规模，不创建输出、不拟合算法。以上运行只占CPU和内存，默认限制数值库为单线程，不需要等待GPU。候选数量大时，HDBSCAN/距离矩阵/谱聚类仍可能需要二次方内存。

中断后保持原命令，追加 `--resume`。输入摘要、有效配置、算法源码、数值库版本和线程设置必须保持一致。已完成case按摘要验证后跳过；失败case默认保留，用 `--resume --retry-errors` 才重新尝试。`Ctrl+C`后已保存的case可继续使用；同一输出目录有运行锁。

更大网格应使用新的输出目录，例如 `--preset full --output "$ROOT/clustering_matrix_full_20261109"`。修改配置或升级相关算法代码后不要继续旧矩阵。不同实验模型的缓存不可拼成一个聚类池。

## 输出与解释

- `report.html`：可筛选数据集、搜索算法/参数、按列排序；默认隐藏固定K诊断。
- `results_long.csv`：每次拟合及自动选K结果，含参数、随机种子、K、准确率、ARI/NMI、覆盖率、耗时、错误状态。
- `aggregate_results.csv`：同一配方跨种子的均值、总体标准差和完整种子覆盖标志。缺少种子或不收敛的配方不进入热图和主排名。
- `*_matrix.csv`、`*_unknown_accuracy.png`：算法×特征空间矩阵。每个格子按开发未知准确率选择一组参数；ARI与覆盖率矩阵复用同一组参数，不各自挑最好的分数。
- `cross_dataset_development_ranking.csv`：同一配方在所有输入数据集上的等权平均，不按记录数加权，也不为每组单独挑参数后再求平均。
- `*_winner_unknown_confusion.png`：开发集最佳配方的第一个种子，包含错误接受为已知和未分配列，不挑最高种子作图。
- `benchmark_manifest.json`、`progress.json`、`cases/`：来源摘要、环境、逐case诊断、警告和主轨迹分配，可审计/续跑。

真实标签只在分区与几何分数确定后用于评分和开发集选型；未知簇用匈牙利匹配计算准确率。主未知准确率以全部真实未知记录为分母，未分配、未检出和错误接受为已知均不能删除。ARI/NMI是已分配真实未知子集上的条件指标，须结合覆盖率；另外保留缺失点视作单例的ARI诊断。

热图每格是从不同大小网格中选出的开发最佳值，因此存在多重比较与选择乐观偏差，不能作为独立最终测试。先锁定配方，再用新划分/新采集记录复验。自动选K行的耗时计入其全部K候选的拟合/几何计算；墙钟时间仍受当前主机负载影响，不是硬件性能评测。

纯聚类的未知准确率上限受原拒识召回限制：这两份缓存的上限分别为98.75%和75%。第二组已有40/160条未知记录被接受为已知，它们不在未知候选池中，换聚类算法无法恢复。`summary.json`会为每个输入计算这一上限。

参考：[scikit-learn聚类与评价](https://scikit-learn.org/stable/modules/clustering.html)、[PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)、[球面K-means](https://github.com/facebookresearch/faiss/wiki/K%E2%80%90means-clustering)。
