import os
import argparse
import torch
import logging
import torch.distributed

from data.data_loader import UAVDataset, get_dataloader, create_dataloader
from model.resnet import MaskImageClassifier
from util.trainer import Trainer
from util.preprocess import SignalPreprocessor
from util.detector import YoloV5Detector
from util.utils import create_logger, set_seed, _path_is_set
from util.config import Config
from util.bboxcache import BBoxCache

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Single-signal training + multi-signal inference: STFT spectrogram -> YOLO detection -> preprocess -> CNN classification"
    )

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
    g_yolo.add_argument("--yolo_iou_thres", type=float, default=0.10)
    g_yolo.add_argument("--yolo_max_det", type=int, default=1000)
    g_yolo.add_argument("--yolo_classes", type=int, nargs="*", default=None)
    g_yolo.add_argument("--yolo_half", action=argparse.BooleanOptionalAction, default=False)
    g_yolo.add_argument("--yolo_warmup", action=argparse.BooleanOptionalAction, default=True)

    # Preprocessor
    g_pre = parser.add_argument_group("Preprocess")
    g_pre.add_argument("--min_area", type=int, default=20)
    g_pre.add_argument("--min_ratio", type=float, default=0.0)
    g_pre.add_argument("--min_width", type=int, default=2)
    g_pre.add_argument("--min_height", type=int, default=2)
    g_pre.add_argument("--max_width", type=int, default=0)
    g_pre.add_argument("--max_height", type=int, default=0)
    g_pre.add_argument("--ring_margin", type=int, default=5)
    g_pre.add_argument("--min_contrast_z", type=float, default=0.5)
    g_pre.add_argument("--freq_eps", type=float, default=5.0)
    g_pre.add_argument("--freq_min_samples", type=int, default=1)
    g_pre.add_argument("--merge_freq_thresh", type=float, default=10.0)
    g_pre.add_argument("--merge_w_log_thresh", type=float, default=0.35)
    g_pre.add_argument("--merge_h_log_thresh", type=float, default=0.35)
    g_pre.add_argument("--merge_energy_thresh", type=float, default=1.0)
    g_pre.add_argument("--min_group_len", type=int, default=2)
    g_pre.add_argument("--min_group_time_span_ratio", type=float, default=0.15)
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
    g_data.add_argument("--input_type", type=str, default="mat", choices=["mat", "png"])
    g_data.add_argument("--val_ratio", type=float, default=0.2)
    g_data.add_argument("--batch_size", type=int, default=32)
    g_data.add_argument("--num_workers", type=int, default=4)
    g_data.add_argument("--sample_ratio", type=float, default=1.0)
    g_data.add_argument("--exclude_classes", type=str, nargs="*", default=[])
    g_data.add_argument("--eval_exclude_classes", type=str, nargs="*", default=[])

    # Classifier Model
    g_model = parser.add_argument_group("CNN Classifier")
    g_model.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "mobilenet_v3_small"])
    g_model.add_argument("--mask_img_size", type=int, default=224)
    g_model.add_argument("--mask_in_chans", type=int, default=1)
    g_model.add_argument("--mask_pretrained", action=argparse.BooleanOptionalAction, default=True)
    g_model.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=False)
    g_model.add_argument("--cnn_dropout", type=float, default=0.0)
    g_model.add_argument("--cnn_input_mode", type=str, default="mask", choices=["mask", "raw", "raw_with_boxes", "raw_in_boxes"])
    g_model.add_argument("--box_draw_thickness", type=int, default=2)
    g_model.add_argument("--box_draw_value", type=int, default=255)
    g_model.add_argument("--mask_fill_value", type=int, default=255)

    # Training / IO
    g_train = parser.add_argument_group("Training")
    g_train.add_argument("--epochs", type=int, default=50)
    g_train.add_argument("--lr", type=float, default=1e-4)
    g_train.add_argument("--weight_decay", type=float, default=1e-2)
    g_train.add_argument("--early_stop_patience", type=int, default=20)
    g_train.add_argument("--cosine_annealing_T0", type=int, default=50)
    g_train.add_argument("--cosine_annealing_mult", type=int, default=2)

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
    g_vis.add_argument("--save_val_detect_vis", action=argparse.BooleanOptionalAction, default=True, help="Whether to save random validation visualizations every epoch.")
    g_vis.add_argument("--val_detect_vis_ratio", type=float, default=0.01, help="Ratio of validation samples to randomly save every epoch. Set 0 to disable.")
    g_vis.add_argument("--infer_detect_vis_ratio", type=float, default=1)
    g_vis.add_argument("--clear_val_detect_vis_each_epoch", action=argparse.BooleanOptionalAction, default=True)
    g_vis.add_argument("--save_infer_detect_vis", action=argparse.BooleanOptionalAction, default=True, help="Whether to save infer yolo/groups visualization images.")
    g_vis.add_argument("--save_infer_classified_vis", action=argparse.BooleanOptionalAction, default=True, help="Whether to save infer classified group visualization images.")

    return parser.parse_args()



def main(args):
    set_seed(seed=42)

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank, local_rank, world_size = int(os.environ['RANK']), int(os.environ.get('LOCAL_RANK', 0)), int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        rank, local_rank, world_size = 0, 0, 1

    config = Config(args)
    if world_size > 1:
        config.device = torch.device(f'cuda:{local_rank}')
    config.rank, config.world_size, config.local_rank = rank, world_size, local_rank

    # exclude classes -> remap labels compactly
    exclude_set = set(getattr(config, "exclude_classes", []) or [])
    config.classes_all = dict(config.classes)
    kept_items = [(name, old_idx) for name, old_idx in sorted(config.classes.items(), key=lambda x: x[1]) if name not in exclude_set]
    config.classes = {name: new_idx for new_idx, (name, _) in enumerate(kept_items)}
    config.num_classes = len(config.classes)
    
    use_explicit_train_val = _path_is_set(config.train_dataset_path) or _path_is_set(config.val_dataset_path)
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
        if use_explicit_train_val:
            train_dataset = UAVDataset(config, logger, dataset_path=config.train_dataset_path)
            val_dataset = UAVDataset(config, logger, dataset_path=config.val_dataset_path)
            train_loader = create_dataloader(train_dataset, config, shuffle=True, sample_ratio=float(config.sample_ratio))
            val_loader = create_dataloader(val_dataset, config, shuffle=False, sample_ratio=1.0)
        else:
            dataset = UAVDataset(config, logger)
            train_loader, val_loader = get_dataloader(dataset, config, mode="train")
    else:
        dataset = UAVDataset(config, logger)
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

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        yolo_device = str(local_rank)
    else:
        yolo_device = config.yolo_device if hasattr(config, "yolo_device") else ""

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
    
    trainer = Trainer(config, (train_loader, val_loader), logger, detector, preprocessor, classifier, bbox_cache)

    if str(config.run_mode).lower() == "train":
        trainer.train()
    else:
        trainer.infer(infer_loader)

if __name__ == "__main__":
    args = get_parser()
    main(args)
