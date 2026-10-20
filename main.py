import os
import argparse
import logging
from pathlib import Path

from data.data_loader import UAVDataset, get_dataloader, create_dataloader
from data.iq_dat_loader import RawIQDataset, create_iq_train_val_loaders, discover_iq_classes
from model.resnet import MaskImageClassifier
from util.trainer import Trainer
from util.lwf_trainer import LwFTrainer
from util.preprocess import SignalPreprocessor
from util.detector import YoloV5Detector
from util.utils import create_logger, set_seed, _path_is_set
from util.config import Config
from util.bboxcache import BBoxCache, detector_signature_from_config
from util.spectrogram_frontend import add_frontend_arguments, FrontendConfig, apply_frontend_manifest, TARGET_SAMPLE_RATE, validate_iq_settings

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Single-signal training + multi-signal inference: STFT spectrogram -> YOLO detection -> preprocess -> CNN classification"
    )

    add_frontend_arguments(parser)

    # Runtime / Mode
    g_run = parser.add_argument_group("Runtime")
    g_run.add_argument("--run_mode", type=str, default="train", choices=["train", "infer"])
    g_run.add_argument("--train_signal_mode", type=str, default="single", choices=["single", "multi"], help="Only used in train mode. Recommended: single")
    g_run.add_argument("--use_data_parallel", action=argparse.BooleanOptionalAction, default=False)
    g_run.add_argument("--use_amp_autocast", action=argparse.BooleanOptionalAction, default=False)

    # Spectrogram -> Physical mapping
    g_stft = parser.add_argument_group("STFT / Frequency Mapping")
    g_stft.add_argument("--sampling_rate", type=float, default=122.88e6)
    g_stft.add_argument("--n_fft", type=int, default=512)
    g_stft.add_argument("--hop_length", type=int, default=int(122.88e6 * 0.05 / 750))
    g_stft.add_argument("--mat_key", type=str, default="summed_submatrices")

    # YOLOv5 Detector
    g_yolo = parser.add_argument_group("YOLO Detector")
    g_yolo.add_argument("--yolo_weights", type=str, required=True)
    g_yolo.add_argument("--yolo_device", type=str, default="")
    g_yolo.add_argument("--yolo_imgsz_h", type=int, default=640)
    g_yolo.add_argument("--yolo_imgsz_w", type=int, default=640)
    g_yolo.add_argument("--yolo_conf_thres", type=float, default=0.05)
    g_yolo.add_argument("--yolo_iou_thres", type=float, default=0.30)
    g_yolo.add_argument("--yolo_max_det", type=int, default=1000)
    g_yolo.add_argument("--yolo_classes", type=int, nargs="*", default=None)
    g_yolo.add_argument("--yolo_half", action=argparse.BooleanOptionalAction, default=False)
    g_yolo.add_argument("--yolo_warmup", action=argparse.BooleanOptionalAction, default=True)



    # Preprocessor
    g_pre = parser.add_argument_group("Preprocess")
    g_pre.add_argument("--grouping_mode", choices=["frequency", "hopping"], default="frequency", help="frequency preserves legacy clustering; hopping associates compatible burst trains across carriers before final filtering.")
    g_pre.add_argument("--hop_association", choices=["shape", "periodic"], default="shape", help="shape tolerates missing pulses and uses cadence only as auxiliary evidence; periodic preserves the strict cadence policy.")
    g_pre.add_argument("--hop_shape_log_tol", type=float, default=0.35)
    g_pre.add_argument("--hop_use_contrast_gate", action=argparse.BooleanOptionalAction, default=False, help="Optional contrast hard gate in shape mode; periodic mode always applies it.")
    g_pre.add_argument("--hop_contrast_delta", type=float, default=3.0)
    g_pre.add_argument("--hop_period_rel_tol", type=float, default=0.15, help="Relative timing tolerance, measured against one pulse period, not the entire missing-pulse gap.")
    g_pre.add_argument("--hop_max_missing_pulses", type=int, default=2)
    g_pre.add_argument("--hop_max_overlap_ratio", type=float, default=0.2, help="Substantial time overlap as a fraction of the shorter pulse; periodic mode rejects any, shape mode limits the total affected fraction.")
    g_pre.add_argument("--hop_max_conflict_fraction", type=float, default=0.2, help="Shape mode: maximum fraction of pulses involved in substantial time overlaps. Set 0 for strict separation.")
    g_pre.add_argument("--hop_min_timing_consistency", type=float, default=0.85)
    g_pre.add_argument("--hop_min_pulses", type=int, default=3, help="Minimum combined detected pulse count before considering a cross-frequency association.")
    g_pre.add_argument("--hop_min_time_extent_ratio", type=float, default=0.15)
    g_pre.add_argument("--hop_min_time_coverage_ratio", type=float, default=0.02)
    g_pre.add_argument("--min_area", type=int, default=4)
    g_pre.add_argument("--min_ratio", type=float, default=0.0)
    g_pre.add_argument("--min_width", type=int, default=2)
    g_pre.add_argument("--min_height", type=int, default=2)
    g_pre.add_argument("--max_width", type=int, default=0)
    g_pre.add_argument("--max_height", type=int, default=0)
    g_pre.add_argument("--ring_margin", type=int, default=5)
    g_pre.add_argument("--min_contrast_z", type=float, default=-2.0)
    g_pre.add_argument("--freq_eps", type=float, default=5.0)
    g_pre.add_argument("--freq_min_samples", type=int, default=1)
    g_pre.add_argument("--merge_freq_thresh", type=float, default=10.0)
    g_pre.add_argument("--merge_w_log_thresh", type=float, default=0.35)
    g_pre.add_argument("--merge_h_log_thresh", type=float, default=0.35)
    g_pre.add_argument("--merge_energy_thresh", type=float, default=1.0)
    g_pre.add_argument("--min_group_len", type=int, default=2)
    g_pre.add_argument("--min_group_time_span_ratio", type=float, default=0.15, help="Legacy time COVERAGE threshold, used only in frequency mode. Hopping mode uses separate extent/coverage thresholds.")
    g_pre.add_argument("--score_n_boxes_weight", type=float, default=0.50)
    g_pre.add_argument("--score_time_span_weight", type=float, default=2.00)
    g_pre.add_argument("--score_contrast_weight", type=float, default=1.0)
    g_pre.add_argument("--score_w_std_weight", type=float, default=1.0)
    g_pre.add_argument("--score_h_std_weight", type=float, default=10.0)
    g_pre.add_argument("--score_contrast_std_weight", type=float, default=3.0)
    g_pre.add_argument("--nms_thresh", type=float, default=0.2)

    # Matching (used in multi-signal inference)
    g_match = parser.add_argument_group("Target Matching")
    g_match.add_argument("--match_freq_thresh", type=float, default=30.0)
    g_match.add_argument("--match_bw_thresh", type=float, default=20.0)
    g_match.add_argument("--match_bandwidth_weight", type=float, default=1.0)
    g_match.add_argument("--match_size_penalty", type=float, default=1.0)
    g_match.add_argument("--skip_unmatched", action=argparse.BooleanOptionalAction, default=True)
    g_match.add_argument("--match_use_bandwidth", action=argparse.BooleanOptionalAction, default=False)

    # Dataset / DataLoader
    g_data = parser.add_argument_group("Data")
    g_data.add_argument("--dataset_path", type=str, nargs="+", default=None, help="One or more dataset paths. Used by auto split mode or infer mode.")
    g_data.add_argument("--train_dataset_path", type=str, nargs="+", default=None, help="Optional explicit training dataset path(s).")
    g_data.add_argument("--val_dataset_path", type=str, nargs="+", default=None, help="Optional explicit validation dataset path(s).")
    g_data.add_argument("--input_type", type=str, default="mat", choices=["mat", "png", "iq_dat"])
    g_data.add_argument("--iq_sample_rate", type=int, default=100_000_000, help="Complex samples per second in each .dat recording.")
    g_data.add_argument("--iq_segment_ms", type=float, default=50.0, help="Length of each IQ crop converted to a spectrogram.")
    g_data.add_argument("--iq_byte_order", choices=["<", ">"], default="<")
    g_data.add_argument("--iq_n_fft", type=int, default=512)
    g_data.add_argument("--iq_hop_length", type=int, default=512)
    g_data.add_argument("--val_ratio", type=float, default=0.2)
    g_data.add_argument("--batch_size", type=int, default=32)
    g_data.add_argument("--num_workers", type=int, default=4)
    g_data.add_argument("--sample_ratio", type=float, default=1.0)

    g_data.add_argument("--exclude_classes", type=str, nargs="*", default=[])
    g_data.add_argument("--eval_exclude_classes", type=str, nargs="*", default=[])
    g_data.add_argument(
        "--incremental_split_mode",
        type=str,
        default="normal",
        choices=["normal", "base", "incremental"],
    )

    g_data.add_argument(
        "--incremental_new_classes",
        type=str,
        nargs="*",
        default=[],
    )

    # Classifier Model
    g_model = parser.add_argument_group("CNN Classifier")
    g_model.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "mobilenet_v3_small"])
    g_model.add_argument("--mask_img_size", type=int, default=224)
    g_model.add_argument("--mask_in_chans", type=int, default=1)
    g_model.add_argument("--mask_pretrained", action=argparse.BooleanOptionalAction, default=True)
    g_model.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=False)
    g_model.add_argument("--cnn_dropout", type=float, default=0.0)
    g_model.add_argument("--mask_fill_value", type=int, default=255)

    # Training / IO
    g_train = parser.add_argument_group("Training")
    g_train.add_argument("--epochs", type=int, default=50)
    g_train.add_argument("--lr", type=float, default=1e-4)
    g_train.add_argument("--weight_decay", type=float, default=1e-2)
    g_train.add_argument("--early_stop_patience", type=int, default=20)
    g_train.add_argument("--cosine_annealing_T0", type=int, default=50)
    g_train.add_argument("--cosine_annealing_mult", type=int, default=2)

    g_train.add_argument("--trainer_type", type=str, default="normal", choices=["normal", "lwf"])
    g_train.add_argument("--lwf_teacher_checkpoint", type=str, default="")
    g_train.add_argument("--lwf_old_num_classes", type=int, default=0)
    g_train.add_argument("--lwf_temperature", type=float, default=4.0)
    g_train.add_argument("--lwf_lambda_kd", type=float, default=10.0)
    g_train.add_argument("--lwf_freeze_backbone", action=argparse.BooleanOptionalAction, default=False)

    g_io = parser.add_argument_group("Checkpoint / Cache")
    g_io.add_argument("--checkpoint_path", type=str, default=None)
    g_io.add_argument("--save_interval", type=int, default=5)

    g_cache = parser.add_argument_group("BBox Cache")
    g_cache.add_argument("--bbox_cache_mode", type=str, default="off",
        choices=["off", "read", "write", "refresh", "readwrite"], help="BBox cache mode.")
    g_cache.add_argument("--bbox_cache_path", type=str, default="", help="Directory to save/load YOLO bbox cache.")
    g_cache.add_argument("--bbox_cache_mem_max", type=int, default=0, help="Max number of bbox entries kept in memory. 0 means no memory cache.")
    g_cache.add_argument("--bbox_cache_dataset_root", type=str, default="", help="Optional dataset root used to build cache metadata.")

    g_vis = parser.add_argument_group("Visualization")
    g_vis.add_argument("--save_val_detect_vis", action=argparse.BooleanOptionalAction, default=True, help="Save class-balanced validation visualizations in separate epoch directories.")
    g_vis.add_argument("--val_detect_vis_ratio", type=float, default=0.01, help="Fraction sampled within each validation class every epoch. Set 0 to disable.")
    g_vis.add_argument("--val_detect_vis_min_per_class", type=int, default=1, help="Minimum visualized samples per present validation class, capped by available samples; ratio=0 still disables saving.")
    g_vis.add_argument("--infer_detect_vis_ratio", type=float, default=1)
    g_vis.add_argument("--clear_val_detect_vis_each_epoch", action=argparse.BooleanOptionalAction, default=False, help="Legacy option, ignored: epoch-specific validation directories are retained.")
    g_vis.add_argument("--save_infer_detect_vis", action=argparse.BooleanOptionalAction, default=True, help="Whether to save infer yolo/groups visualization images.")
    g_vis.add_argument("--save_infer_classified_vis", action=argparse.BooleanOptionalAction, default=True, help="Whether to save infer classified group visualization images.")

    return parser.parse_args()

def setup_incremental_class_split(config, logger=None):
    all_classes = dict(config.classes)
    config.classes_all = dict(all_classes)

    mode = str(getattr(config, "incremental_split_mode", "normal")).lower()

    exclude = set(getattr(config, "exclude_classes", []) or [])
    new_classes = list(getattr(config, "incremental_new_classes", []) or [])
    if len(new_classes) != len(set(new_classes)):
        raise ValueError("incremental_new_classes contains duplicate class names")

    for name in exclude:
        if name not in all_classes:
            raise ValueError(f"exclude class not found in config.classes: {name}")

    for name in new_classes:
        if name not in all_classes:
            raise ValueError(f"incremental new class not found in config.classes: {name}")
        if name in exclude:
            raise ValueError(
                f"class {name} is both in exclude_classes and incremental_new_classes"
            )

    # 1. 全局排除 exclude_classes
    remaining = [
        name
        for name, _ in sorted(all_classes.items(), key=lambda x: x[1])
        if name not in exclude
    ]

    # 2. 按模式决定模型类别空间和数据筛选策略
    if mode == "normal":
        ordered = remaining

        # 数据只排除 exclude_classes
        data_exclude = list(exclude)
        data_include = []

    elif mode == "base":
        if not new_classes:
            raise ValueError(
                "--incremental_new_classes is required when incremental_split_mode=base"
            )

        new_set = set(new_classes)

        # base 类别 = remaining 中除去 new_classes
        ordered = [name for name in remaining if name not in new_set]

        # 数据排除 = 全局 exclude + 增量新类
        data_exclude = list(exclude | new_set)
        data_include = []

    elif mode == "incremental":
        if not new_classes:
            raise ValueError(
                "--incremental_new_classes is required when incremental_split_mode=incremental"
            )

        new_set = set(new_classes)

        base_classes = [name for name in remaining if name not in new_set]

        # student 类别顺序必须 old 在前，new 在后
        ordered = base_classes + new_classes

        train_data_exclude = list(exclude)
        train_data_include = list(new_classes)

        val_data_exclude = list(exclude)
        val_data_include = []

        if int(getattr(config, "lwf_old_num_classes", 0)) <= 0:
            config.lwf_old_num_classes = len(base_classes)

    else:
        raise ValueError(f"Unsupported incremental_split_mode={mode}")

    if len(ordered) == 0:
        raise ValueError(
            f"No model classes left after split. "
            f"mode={mode}, exclude_classes={list(exclude)}, "
            f"incremental_new_classes={new_classes}"
        )

    # 3. 标签重映射
    config.classes = {name: i for i, name in enumerate(ordered)}
    config.num_classes = len(config.classes)

    if mode == "normal":
        config.train_data_exclude_classes = data_exclude
        config.train_data_include_classes = data_include
        config.val_data_exclude_classes = data_exclude
        config.val_data_include_classes = data_include

    elif mode == "base":
        config.train_data_exclude_classes = data_exclude
        config.train_data_include_classes = data_include
        config.val_data_exclude_classes = data_exclude
        config.val_data_include_classes = data_include

    elif mode == "incremental":
        config.train_data_exclude_classes = train_data_exclude
        config.train_data_include_classes = train_data_include
        config.val_data_exclude_classes = val_data_exclude
        config.val_data_include_classes = val_data_include

    if logger is not None:
        logger.info(f"[ClassSplit] mode={mode}")
        logger.info(f"[ClassSplit] original_classes={all_classes}")
        logger.info(f"[ClassSplit] exclude_classes={list(exclude)}")
        logger.info(f"[ClassSplit] incremental_new_classes={new_classes}")
        logger.info(f"[ClassSplit] model_classes={config.classes}")
        logger.info(f"[ClassSplit] num_classes={config.num_classes}")
        logger.info(f"[ClassSplit] data_exclude_classes={data_exclude}")
        logger.info(f"[ClassSplit] data_include_classes={data_include}")
        logger.info(
            f"[ClassSplit] lwf_old_num_classes="
            f"{getattr(config, 'lwf_old_num_classes', None)}"
        )

def main(args):
    set_seed(seed=42)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("Distributed training is not implemented for the detection/classification pipeline")
    if args.use_data_parallel or args.use_amp_autocast:
        raise ValueError("--use_data_parallel and --use_amp_autocast are not implemented")
    if args.run_mode == "train" and args.train_signal_mode != "single":
        raise ValueError("Training currently supports only --train_signal_mode single")
    if args.input_type == "iq_dat" and args.run_mode == "infer":
        raise ValueError("IQ .dat files have no target frequency labels; the current inference evaluator requires them")
    if args.input_type == "iq_dat" and args.bbox_cache_mode != "off":
        raise ValueError("IQ training uses random time crops; file-level bbox caching would reuse boxes from the wrong crop")
    if not (0.0 < float(args.sample_ratio) <= 1.0):
        raise ValueError("--sample_ratio must be in (0, 1]")
    if not (0.0 < float(args.val_ratio) < 1.0):
        raise ValueError("--val_ratio must be in (0, 1)")
    if args.trainer_type == "lwf" and args.incremental_split_mode != "incremental":
        raise ValueError("LwF requires --incremental_split_mode incremental")
    if args.mask_in_chans != 1:
        raise ValueError("The current mask builder produces one channel; --mask_in_chans must be 1")
    if args.mask_img_size < 1 or not (1 <= args.mask_fill_value <= 255):
        raise ValueError("mask_img_size must be positive and mask_fill_value must be in [1, 255]")
    if not (Path(__file__).resolve().parent / "yolov5" / "hubconf.py").is_file():
        raise FileNotFoundError("The yolov5 submodule is empty; initialize it before training")
    if not Path(args.yolo_weights).is_file():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo_weights}")
    if args.checkpoint_path and not Path(args.checkpoint_path).is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {args.checkpoint_path}")

    rank, local_rank, world_size = 0, 0, 1

    apply_frontend_manifest(args)
    config = Config(args)
    FrontendConfig.from_config(config)  # validate before initializing datasets/models
    if config.sampling_rate != TARGET_SAMPLE_RATE:
        raise ValueError("--sampling_rate describes the 122.88 MHz output grid; use --iq_sample_rate for capture rate")
    config.frequency_axis = "fft"
    config.detector_spec_domain = ("image" if config.input_type == "png" or
                                  (config.input_type == "mat" and config.mat_spec_domain == "image")
                                  else "amplitude")
    if config.input_type == "iq_dat":
        validate_iq_settings(config.iq_sample_rate, config.iq_segment_ms,
                             config.iq_n_fft, config.iq_hop_length)
        iq_paths = (config.dataset_path or []) + (config.train_dataset_path or []) + (config.val_dataset_path or [])
        config.classes = discover_iq_classes(iq_paths)
    config.rank, config.world_size, config.local_rank = rank, world_size, local_rank

    setup_incremental_class_split(config)
    if config.run_mode == "infer" and "Background" not in config.classes:
        raise ValueError("End-to-end inference evaluation requires a Background class")

    use_explicit_train_val = _path_is_set(config.train_dataset_path) or _path_is_set(config.val_dataset_path)
    if config.run_mode == "train" and config.incremental_split_mode == "incremental" and not use_explicit_train_val:
        raise ValueError("Incremental mode requires explicit train and validation dataset paths")
    if rank == 0:
        config.freeze()
        config.make_dir()
        config.save_config()
        logger = create_logger(os.path.join(config.log_dir, "train_log.log" if config.run_mode == "train" else "infer_log.log"))
        logger.init_exp(config)
        logger.info(f"DDP initialized: world_size={world_size}")
        logger.info(f"run_mode={config.run_mode}, train_signal_mode={getattr(config, 'train_signal_mode', 'single')}")
        if str(config.run_mode).lower() == "train" and use_explicit_train_val:
            logger.info(f"Use explicit train/val datasets.")
            logger.info(f"train_dataset_path={config.train_dataset_path}")
            logger.info(f"val_dataset_path={config.val_dataset_path}")
    else:
        logger = logging.getLogger("ddp_logger")
        logger.addHandler(logging.NullHandler())

    if str(config.run_mode).lower() == "train":
        if use_explicit_train_val and not (_path_is_set(config.train_dataset_path) and _path_is_set(config.val_dataset_path)):
            raise ValueError("Provide both --train_dataset_path and --val_dataset_path")
        if use_explicit_train_val:
            dataset_cls = RawIQDataset if config.input_type == "iq_dat" else UAVDataset
            train_dataset = dataset_cls(config, logger, dataset_path=config.train_dataset_path, split_name="train")
            val_dataset = dataset_cls(config, logger, dataset_path=config.val_dataset_path, split_name="val")
            train_loader = create_dataloader(train_dataset, config, shuffle=True, sample_ratio=float(config.sample_ratio))
            val_loader = create_dataloader(val_dataset, config, shuffle=False, sample_ratio=1.0)
        elif config.input_type == "iq_dat":
            train_loader, val_loader = create_iq_train_val_loaders(config, logger)
        else:
            dataset = UAVDataset(config, logger)
            train_loader, val_loader = get_dataloader(dataset, config, mode="train")
    else:
        dataset_cls = RawIQDataset if config.input_type == "iq_dat" else UAVDataset
        dataset = dataset_cls(config, logger, split_name="infer")
        infer_loader = get_dataloader(dataset, config, mode="infer")
        train_loader, val_loader = None, None

    classifier = MaskImageClassifier(
        backbone=config.backbone,
        num_classes=config.num_classes,
        in_chans=config.mask_in_chans,
        pretrained=config.mask_pretrained,
        dropout=config.cnn_dropout,
        freeze_backbone=config.freeze_backbone,
    )

    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    logger.info(f"Num of trainable params in backbone: {trainable_params:,}")

    yolo_device = config.yolo_device

    detector = YoloV5Detector(config, yolo_device)
    preprocessor = SignalPreprocessor(config, logger)

    bbox_cache = None
    if config.bbox_cache_mode != "off":
        if not str(config.bbox_cache_path).strip():
            raise ValueError("--bbox_cache_path must be provided when bbox_cache_mode != off")

        dataset_root = str(config.bbox_cache_dataset_root).strip()
        if not dataset_root:
            dataset_root = None

        bbox_cache = BBoxCache(
            base_dir=config.bbox_cache_path,
            dataset_root=dataset_root,
            mode=config.bbox_cache_mode,
            mem_max=config.bbox_cache_mem_max,
            detector_signature=detector_signature_from_config(config),
            logger=logger,
        )

        logger.info(
            f"[BBoxCache] enabled: mode={config.bbox_cache_mode}, "
            f"path={config.bbox_cache_path}, "
            f"mem_max={config.bbox_cache_mem_max}, "
            f"dataset_root={dataset_root}"
        )
    else:
        logger.info("[BBoxCache] disabled")

    trainer_cls = LwFTrainer if str(config.trainer_type).lower() == "lwf" else Trainer

    trainer = trainer_cls(
        config=config,
        dataloaders=(train_loader, val_loader),
        logger=logger,
        detector=detector,
        preprocessor=preprocessor,
        classifier=classifier,
        bbox_cache=bbox_cache,
    )

    if str(config.run_mode).lower() == "train":
        trainer.train()
    else:
        trainer.infer(infer_loader)

if __name__ == "__main__":
    args = get_parser()
    main(args)
