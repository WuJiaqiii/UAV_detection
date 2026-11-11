# 开发与验证

## 结构

- 根目录保留`main.py`、`prepare_submission.py`及四个薄Shell入口。
- `scripts/`是实际任务工具，不保存本机一次性运行命令。
- `util/`是共用推理、统计、训练与编排组件；`data/`和`model/`保存数据适配及网络。
- `docs/`仅保留当前维护的6份手册；已有参赛技术报告作为外部交付资料保留。
- `yolov5/`固定为Git子模块，初始化时读取Git记录的提交，不将权重提交进源码。

## v3整理范围

删除旧结果补推理/恢复、旧profile刷新与配对拒识试验入口、根目录旧CLI别名和硬编码启动脚本。两个历史LwF编排器由一个从原始数据开始的增量实验入口取代。训练器中旧GT实例推理引擎退出，所有预测使用当前共用链路。旧的测试结果快照不再放在根目录冒充当前验证。

拒识评分、分组阈值、谱聚类与物理突发逻辑保持现状。新训练产物精简存储、统一普通配置序列化，取消优化器恢复。旧模型不作为本版兼容目标；YOLO检测器仍需要有效权重。

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q main.py prepare_submission.py util data model scripts tests
bash -n run.sh infer.sh submit.sh lwf.sh
UAV_TEST_REAL_YOLO=1 python -m unittest tests.test_yolo_native_contract -v
```

常规测试在CPU上覆盖前端、数据划分、分组/轨迹、拒识与聚类、提交、LwF梯度及新编排。真实YOLO输入契约测试需要初始化子模块；普通测试中该项会跳过。回归通过证明接口和这些性质一致，不等于已经重训并确认比赛识别率。

改变特征生产代码后必须使用新profile。完整工作流及增量suite记录源码、权重和输入指纹，防止同一实验中途更换实现；不要修改历史指纹绕过检查。检查源码删除时同时核对动态导入、命令行子进程、测试和文档引用。
