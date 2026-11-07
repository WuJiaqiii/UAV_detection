# 已知类别误拒复核：实现与两组对比实验

本文是可选研究对照的实现与历史实验说明。当前竞赛主线采用 A：原拒识＋范围策略，关闭 known_recheck；正式完整训练和提交按 [当前主线说明](competition_mainline.md) 执行，无需应用本文的旧补丁。

基于 main 的 3398f3af684b4519e6f972c66363f38c6c63a113。修改在独立源码目录中完成，没有修改远程 main，没有启动远程训练。

## 实现内容

保留原 Maha++ 拒识，新增默认关闭的 known_recheck。只处理已被第一阶段拒绝且拥有有效特征的主轨迹；原本接受的已知预测、未检测到信号的文件、无特征轨迹均不会被此功能改写。现阶段限定 unknown_input_mode=single；不将主轨迹校准规则直接用于任务二的多设备次要轨迹。

复核时要求分类头、原始局部距离最近类别、归一化局部距离最近类别三者一致，并通过局部半径与类别间隔门槛。候选小簇大小、竞赛测试标签和竞赛类别数量不参与拟合，也不强制将小簇合并。

参考库每个原始记录只保留一个主轨迹。对类别 c，局部距离为其最近 k 个独立参考记录的平均余弦距离：

    d_c(x) = mean_{r in NN_k(x, B_c)} [1 - cosine(x, r)]

每类的尺度 m_c 来自参考集内部的留一记录距离中位数。向各类中位数的中位数 m_pool 收缩：

    w_c = n_c / (n_c + scale_prior_records)
    scale_c = max(w_c*m_c + (1-w_c)*m_pool, 1e-6)
    s_c(x) = d_c(x) / scale_c

距离半径取独立已知校准集中“真实类别、分类头、两种最近邻类别均一致”的合格记录的归一化距离上分位数。间隔为 (s_second-s_first)/max(s_second,epsilon)，取合格校准记录的下分位数；零间隔不接受。没有合格校准记录的类别不执行恢复。

radius_frr=0.01 是该条件校准群体的经验距离尾部参数，不是全流程 1% 误拒保证，更不是未知类误收保证。当前每类样本有限，必须用伪未知对比实验判断有效性。数据规模增大后，可进一步评估类别独立尾部校准。

门槛与参考库均保存进 known_profile_recheck.npz。实时完整/fast 推理共用同一复核实现；fast 现在也保存 unknown_clusters.json，失败时可直接查看候选 K 与支持数原因。

## 安装

推荐将发布包中的补丁应用到新分支，保留现有数据与权重。以下命令由用户在服务器执行；补丁路径替换为实际下载位置：

```bash
REPO="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/UAV_signal_detection/code"
cd "$REPO"
git status --short
git switch -c codex/known-recheck 3398f3af684b4519e6f972c66363f38c6c63a113
git apply --check /path/to/UAV_known_recheck.patch
git apply /path/to/UAV_known_recheck.patch
```

如果当前存在未提交更改，先妥善保存；不要强制覆盖。若 main 又有新的代码修改，先检查补丁是否仍适用。继续使用原 myconda 环境，本次没有增加第三方依赖。完整源码包不包含数据或训练权重。

## 实验 A：原拒识方案

使用 config/known_recheck_a.yaml。默认在现有 20 类中留出以下 4 类作为伪未知：

- DJI AVATA2
- DJI MAVIC3 PRO
- FRSKY X20R
- HubsanZino2

DJMiniSE 保留为已知类别，以便评估它的误拒。可在运行前更换 unknown_classes，但同一对 A/B 必须保持一致。不要根据最终测试结果反复挑选类别或复核参数；调参应使用另一个开发划分，最终测试划分保持冻结。

该配置的 protocol=open_set，模型只训练其余 16 类。已知类别按原始记录划分：训练约 45%、验证 15%、校准 20%、测试 20%；伪未知类别仅进入 test_unknown，不进入模型训练、参考或校准。

如果同一原始采集被转换成多个不同文件，必须在 experiment.recording_ids_json 指定原始记录 ID 映射，防止不同转换结果跨集合泄漏。文件内容哈希只能自动识别完全相同的内容，不能识别所有派生关系。

检查 config/known_recheck_a.yaml 中 inputs 和 yolo_weights，默认已填服务器原路径。所有 A/B 参数保持冻结后运行：

```bash
cd "$REPO"
bash run.sh experiment --config config/known_recheck_a.yaml --dry_run
bash run.sh experiment --config config/known_recheck_a.yaml
```

也可以先运行 --split_only 检查独立记录划分，再用相同配置正式运行；正式运行会生成另一个目录。dry_run 不检查实际数据内容、文件存在性或权重。

终端打印新的 experiment/open_set_experiment_时间戳，记为 BASE，例如：

```bash
BASE="$REPO/experiment/open_set_experiment_替换为A组实际时间"
```

A 完成后应存在：

- split_manifest.json、effective_discover.json
- training/models/best.pth
- open_set/reference/features.npz、open_set/calibration/features.npz
- open_set/known_profile.npz
- test/results.json、test/unknown_features.npz、test/open_set_metrics.json

这组实验使用 unconstrained 聚类，A/B 的聚类参数完全相同。不用已知的伪未知类别数量强制指定 K。A 的 known_recheck=false。

固定的 YOLO 为 A/B 共用。如果它曾训练过这些留出信号，本实验衡量分类及拒识模块的类别留出表现，不能据此声称整套检测系统从未见过这些信号。

## 实验 B：同一权重与特征，增加已知类复核

无需第二次训练，无需重新运行检测器或分类网络。B 根据 A 保存的已知参考/校准特征生成增强 profile，然后仅重放 A 被拒绝的主轨迹，更新其已知/未知决定，最后重新对剩余未知候选聚类。

```bash
cd "$REPO"
python scripts/compare_known_recheck.py \
  --source_experiment "$BASE" \
  --config config/known_recheck_b.yaml
```

输出自动写入 experiment/known_recheck_comparison_时间戳。也可用 --output 指定新目录或空目录；不会覆盖 A。

本脚本的 YAML 是独立的 recheck 配置，不应传给 bash run.sh experiment。默认参数：k=3、scale_prior_records=5、radius_frr=0.01、margin_tail=0.05。

B 会检查：真实类别留出协议、已知模型类别空间、A 未启用复核、候选轨迹集合、参考/校准特征签名、源码版本、原 profile 数组与阈值保持完全一致。测试标签仅用于最后评分，代码测试验证了修改测试标签不会改变复核决定。

不要把旧的 all_known_training_20261107_033606 当作 BASE：该模型见过全部类别，没有本实验所需的真正类别留出。旧版本生成的 A 也应使用这版代码重新执行，以保持特征签名一致。

## 比较结果

B 输出：

- comparison.csv：A、B、B-A 的主要指标。
- comparison.json：完整指标、输入摘要和三类恢复计数。
- known_recheck_decisions.json：每个复核记录的类别一致性、距离、间隔及接受/拒绝原因。
- known_profile_recheck.npz：冻结的增强 profile。
- results.json、open_set_metrics.json、predictions.csv：B 的单记录结果。
- unknown_features.npz、unknown_clusters.json、unknown_prototypes.npz：剩余未知候选与聚类。

重点一起观察：

| 指标 | 希望的方向 |
|---|---|
| known_false_reject_rate | 下降；分母是全部已知测试记录，分子是被判为未知候选的已知记录 |
| known_abstention_rate | 下降；包括 no_signal、uncertain 等未接受状态 |
| known_accuracy | 上升；包括恢复到错误已知类的损失 |
| unknown_false_accept_rate | 尽量不增加；未知记录被接受为已知的比例 |
| unknown_detection_recall | 尽量不下降 |
| unknown_cluster_coverage、clustered_unknown_ari | 联合看覆盖与聚类质量 |
| unknown_class_count_absolute_error | 下降 |

comparison.json 另外分别列出 recovered_known_correct、recovered_known_wrong、newly_false_accepted_unknown。不能只报告“恢复多少条”，也不能只看已知准确率。保留的 ood_auroc 使用原始 Maha++ 分数，A/B 不应通过该分数的 AUROC 评价复核效果。

B 的 results.json 是主记录离线重放结果，移除了过时的事件输出，不用于任务二突发定位。A 中的 no_signal 不会因重放而恢复。comparison_has_upstream_errors=true 或退出码 2 表示原 A 有上游错误，应检查后再解释性能。

诊断更新：comparison.json 和终端现在列出 source_error_breakdown。如果错误仅为 burst_label_conflicts，主记录对比仍已完成；默认继续返回 2，保留严格行为。明确只做主记录分析时，可在 B 命令添加 --allow-event-conflicts，此时仅事件冲突告警允许返回 0，报告仍保留 comparison_has_upstream_errors=true、冲突文件清单及 event_export_valid=false。文件/切片读取、聚类或绘图错误不会被这个选项忽略。此选项不修改分类结果、复核门槛或事件提交检查；不需要重训 A。

如做多个随机种子，可分别用 --seed 42、43、44 训练独立 A，然后分别运行对应 B；每一对内共享权重、划分和特征。不要把不同 A 的缓存拼接。

## 验证后用于实时任务一推理

只有在伪未知实验确认收益可接受后，才考虑部署。增强 profile 与其源 checkpoint 必须一致；本次 16 类实验的增强 profile 不能配给正式 20 类模型。

对匹配的增强 profile，实时使用：

```bash
bash run.sh infer --config config/competition_v2_infer.yaml \
  --checkpoint_path "$BASE/training/models/best.pth" \
  --open_set_profile /path/to/known_profile_recheck.npz \
  --inputs /path/to/your/test_item \
  --competition_profile task1_n1 --test_item 1-1 \
  --unknown_input_mode single --known_recheck
```

这里仅展示接口，不表示应把 16 类模型用于正式提交。完整/fast 均支持复核；原 profile 可继续在 --no-known_recheck 下使用，但必须属于本代码版本的匹配签名。默认关闭使新版本中的 A 保留原决策规则。

## 验证范围

新增测试覆盖独立记录校准、参考尺度隔离、主次轨迹隔离、门槛和类别一致性、缺失特征保护、profile 持久化与基线数组不变、完整/fast 一致、配对缓存重放、真实类别留出检查、测试标签仅影响评分。

本地测试使用确定性模型/检测器替身和少量真实缓存的数值检查，不代表真实射频性能。实际 A/B 训练和性能测试由用户在服务器执行；本次没有启动它们。
