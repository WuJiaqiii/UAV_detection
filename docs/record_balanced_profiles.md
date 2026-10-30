# 按记录平衡的已知类参考库与多链路输出

训练、参考、校准数据的标签范围是单无人机类别：一条记录可以同时包含该无人机的跳频遥控链路和定频图传链路，两者共享无人机类别标签。单记录标签不代表只有一组矩形框。最终推理可以输入多个无人机、多个信号，不从文件名或目录名推断类别。

本次保持检测、框分组、FrequencyTracker、CNN 权重结构和原有单目标训练循环不变。参考与校准阶段内部提取各链路视图，但只接受每条轨迹具有单一已知类别真值的归档。原有训练仍选择主组，不会因本次变更自动变为多组件联合训练；多原型参考库可以将同一类别不同链路的现有编码器特征关联到同一个已知类别。若这些特征本身不可分，后续仍可能需要使用多组件视图微调编码器。

## 默认记录平衡方法

`profile_score_method: record_class_knn` 是新的默认值。每类保留多个组件参考原型，不将遥控和图传特征直接平均为单一中心。对于查询特征，先取它与一条原始参考记录所有组件原型的最小距离，再在该类别不同原始记录之间取第 k 个近邻距离。同一个 DAT 的多条轨迹、切片或同源导出不能占据全部 k 个邻居。

记录身份同时合并 recording_id 和内容哈希，支持传递别名；同一原始记录具有冲突类别会报错。重复的相同组件特征去重，保留不同模式。`max_reference_per_class` 在新方法下限制每类原始记录数量，记录内的不同组件原型继续保留；旧 `global_knn` 下仍限制轨迹数量。

每个类别的距离尺度从参考集单独估计：排除查询所属的整个原始记录，计算类内距离；每条记录取最难组件的距离，再对记录取中位数。类别尺度向共享尺度收缩，收缩强度由 `profile_scale_prior_records` 控制（默认 5）。尺度不使用校准集或测试集拟合。参考记录只有 k 条时，留一记录尺度使用 k-1 个邻居；无法留一估计的类别使用共享尺度并产生警告。

最终异常分数是各已知类别“距离 / 类别尺度”的最小值，已知标签采用对应参考类别，原分类头候选保留在 known_candidate_class。不只检查分类头第一名；原型库选出的类别记录在 profile_class_name。

## 单无人机校准

参考与校准必须按原始记录隔离。每条校准记录只贡献一次：取该记录各有效组件异常分数的最大值，再在独立记录之间计算上分位阈值。这样校准目标为一条单无人机记录的有效组件均被接受为已知，轨迹很多的记录不会重复计权。最大值对异常组件敏感；错误分组或背景轨迹仍可能抬高阈值，应检查参考/校准质量，不能据此承诺提高未知召回。

每类最低独立校准数量检查继续保留。小样本下 1% 只是经验目标，不是有限样本部署保证。忽略类仍成为 uncertain；无检测或无有效轨迹不会被当成新类别。

`known_profile.npz` 增加 reference_recording_ids、reference_labels、class_scales、calibration_recording_ids、逐记录 calibration_scores 和诊断用 calibration_track_scores；metadata 保存类别尺度、参考记录计数、组件来源和校准单位。类别原型是参考实例库，不是每类单一平均向量。

## 两切片拒识

默认 `novelty_policy: calibrated_track` 使用与校准一致的轨迹特征异常分数直接路由。逐观测 novel_observation_share 仍然输出，但不再作为额外否决条件，因此两次观测不会被默认 60% 门槛变成全票要求。有效投票数量和原有投票质量检查继续保留。

要比较旧的额外一致性门槛，设置 `novelty_policy: observation_guard`，再配置 `min_novel_observation_share`（新默认 0.5）。这是一项更保守的消融，不能消除真实观测分歧。

## 推理中的多链路

普通 multi 预测和开集推理均保留每条轨迹的时间、频带和分类，新增 `class_evidence.csv`，JSON 中增加 class_evidence，HTML 增加类别组件列表。例如同一已知类别 DJFPV 可同时列出 hopping 与 frequency 组件，各自保留频率位置。

类别证据汇总不修改分组、不重写轨迹、不把同频/同文件作为设备身份。`identity_status: category_evidence_only` 表示类别一致：同一类别可以有多台无人机。hopping/frequency 是分组形态，不是经过人工确认的遥控/图传角色。

未知候选仍保持独立轨迹并由现有后端聚类。仅凭遥控与图传同时出现，无法确认两个未知簇属于同一未知型号或同一设备；本次不会强制合并。解决这类未知跨链路语义对应需要额外的设备配对信息、物理证据或专门的跨链路表示学习。

## 运行与迁移

```yaml
profile:
  profile_score_method: record_class_knn
  profile_scale_prior_records: 5.0
  knn_k: 5
  min_calibration_per_class: 5
  known_false_reject_rate: 0.01
evaluation:
  signal_mode: multi
  novelty_policy: calibrated_track
```

入口不变：`bash run.sh experiment --config config/experiment_iq.yaml` 完整训练/校准/测试；已有 CNN 可通过 `bash run.sh profile --config config/open_set.yaml` 重建参考库，再使用 predict 或 infer。DAT profile 的 label_mode 需为 parent，或提供显式 JSON 标签。完整实验自动生成标签和无标签 inference.yaml。

旧原始距离方法用 `profile_score_method: global_knn` 保留，用于消融。不能直接比较两种方法阈值的数值大小，因为分数单位不同。新代码的源码签名发生变化，部署须使用新代码重新提取参考、校准特征并建立 profile；不需要仅为构建新 profile 重新训练 CNN。仅对旧缓存特征拟合的方法诊断不等于可部署的新 profile，也不等于重新完成最终测试。

建议比较同一类别划分、相同权重下的旧方法、新记录平衡方法与可选 observation_guard；同时报告已知准确率/误拒率、未知召回/覆盖率、未知端到端准确率和簇数量。当前代码不使用未知真实类别或真实 K 拟合参考库、尺度和阈值。
