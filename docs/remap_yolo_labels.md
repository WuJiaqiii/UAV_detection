# 批量修改 YOLOv5 检测标签类别

在项目根目录执行：

```bash
python scripts/remap_yolo_labels.py --dataset_dir /data/my_yolo_dataset
```

支持标准 images/、labels/ 组织形式，递归处理 labels/train、labels/val、labels/test 和更深子目录中的所有 .txt。
也可直接将 --dataset_dir 指向 labels 目录。脚本仅依赖 Python 标准库。

每个非空标签行应为检测格式：class xc yc width height。
保留类别 0、1、2；其他非负整数类别全部改为 0。
仅替换需要修改的行的第一个字段，保留坐标字符串、空白、换行和 UTF-8 BOM。
空标签文件保持为空；图片、data.yaml 和原文件名不变。

可先检查与统计而不写入：

```bash
python scripts/remap_yolo_labels.py --dataset_dir /data/my_yolo_dataset --dry_run
```

正式运行会先检查全部标签，格式错误时报告文件和行号，在修改之前停止。
修改前自动在 labels 同级创建 labels_backup_时间戳_随机后缀，包含：
- labels/：被修改文件的完整原始内容，保留相对于原 labels/ 的目录；
- caches/：被失效的 YOLO 标签缓存原件；
- manifest.json：标签来源与缓存备份路径映射。

类别修改可能不改变文件字节数，而 YOLOv5 的标签缓存哈希使用文件路径和大小；
因此脚本会备份并删除相关 .cache，让下次训练重新扫描标签。
JSON 输出提供修改前后类别数量、扫描文件数、修改框数、备份路径。
重复运行时，若没有需要修改的类别，不会写入标签或新建备份。

若训练按三类进行，需自行确认 data.yaml 中的 nc/names 对应 0、1、2；
本工具不猜测类别名称。当前频谱导出器 --finalize 仍检查单类 signal_frame，
三类数据不应使用该单类 finalize 流程。
