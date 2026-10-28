# 第一轮代码清理迁移

基于 main a3089d27f169bd66e1e9939693d6466757176595。核心分组阈值、频谱算法、CNN 参数结构、DAT 轨迹与新颖性决策未改动。

## 入口

| 原入口 | 当前入口 |
| --- | --- |
| run_train.sh / run_train_base.sh | run.sh train --config config/train.yaml；类别与留出项放配置 |
| run_train_iq.sh | run.sh train --config config/train_iq.yaml |
| run_train_incremental.sh | run.sh train，配置 trainer_type=lwf、incremental_split_mode=incremental 和教师路径 |
| run_evaluate.sh / run_inference.sh | 新预测用 run.sh predict；旧 main GT 实例级评价仍可用 Python 原参数 |
| run_open_set.sh | run.sh discover --config config/open_set.yaml |
| 固定候选池重聚类 | run.sh cluster --config config/cluster.yaml |

root main.py 是轻量分派；原训练/GT 评价主体移动到 util/train_entry.py。旧 `python main.py --run_mode ...` 参数仍转发，三个核心 scripts Python 入口仍可直接使用。没有删除 LwF 或旧实例级指标，也没有将它们伪装为新文件级指标。

## 公共组件

- util/eval_common.py：混合输入发现、checkpoint 推理设置、DAT 窗口、原生频谱读取、投票、路径映射。
- util/group_predictor.py：共用检测/分组与 mask 分类。单/多信号策略只负责选择组和组织输出；多信号不再继承单信号策略类。
- util/checkpoint.py：统一去 module 前缀，供 LwF 和普通加载复用。
- tests/fixtures.py：公共特征归档和轨迹；测试不再导入其他测试文件。

single_signal_eval.py 暂时转出公共函数的导入别名，保持旧 Python 调用兼容，但算法实现只有一份。训练器的旧推理方法本轮保留，避免初步清理时改变实例级评估语义；后续可在专门迁移其指标后继续精简。

## 删除范围

scripts 仅保留 evaluate_single_signal.py、build_open_set_profile.py、cluster_unknown_features.py。删除背景/合成/细信号数据生成、remap、标注导出、YOLO 训练及对应诊断工具。model/losses.py 的旧损失未被主线使用，删除；实际 SupCon 保留在 representation.py。YOLOv5 子模块及推理适配继续使用，原训练 loader 补丁退出。

删除两个根目录历史测试日志、补丁交付说明、environment.yaml/spec.txt 环境快照、绑定该快照的 Dockerfile。requirements-open-set.txt 合并进 requirements.txt；保留固定 YOLO PyTorch 后端实际依赖，删除 TensorFlow/TensorRT 等导出运行时。

测试只删除对应已退出功能的 remap 4 条和标注导出 4 条。原前端数值、数据适配、坐标、投票、表示、拒识、发现和训练测试保留；真实 YOLO 检查改为推理输入契约。新增 YAML/CLI 覆盖、懒加载与完整 discover 编排测试。

## 现有存档和 profile

已有 checkpoint 的权重键和类别映射不变；新增 class_names / seed 是可选训练参数。IQ/DAT 默认从实际训练/验证路径中的文件父目录发现类别，train_iq.yaml 不预设协议名称；IQ 的 class_names 仅用于显式排列全部已发现类别，排除类使用 exclude_classes。继续训练时应使用同样类别顺序和原 checkpoint 路径。YAML 不默默覆盖 checkpoint 中的表示设置；predict 使用原解析器的显式覆盖规则。

旧 profile 因源码签名变化应重建；旧的参考/校准 features.npz 也有旧签名，需要重新提取。不修改已保存阈值的签名，不关闭检查。新 profile 可继续用于同一模型和有效设置的预测，聚类换后端不需重建。

## 覆盖包与删除

如果使用补丁，git apply 会包含删除。完整源码包应解压到新目录；单纯把新文件覆盖旧目录不会删除旧工具、环境快照和 bash。保持原有 yolov5 子模块与权重，或在新 checkout 重新初始化该子模块。
