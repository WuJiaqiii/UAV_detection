import os
import argparse
import torch
import logging
import torch.distributed

from data.data_loader import UAVDataset, get_dataloader

from models.transformer import SignalTransformerClassifier

from util.trainer import Trainer

from util.preprocess import SignalPreprocessor
from util.detector import YoloV5Detector

from util.checkpoint import load_checkpoint
from util.utils import create_logger, set_seed
from util.config import Config

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
    torch.autograd.set_detect_anomaly(True)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*find_unused_parameters=True was specified in DDP constructor.*", category=UserWarning)

import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="STFT spectrogram -> (Yolov5 detection + preprocessing) -> Transformer classification training"
    )
    
    # Spectrogram -> Physical mapping
    g_stft = parser.add_argument_group("STFT / Physical Mapping")
    g_stft.add_argument(
        "--sampling_rate", type=float, default=122.88e6,
        help="Sampling rate (Hz) used to generate STFT."
    )
    g_stft.add_argument(
        "--n_fft", type=int, default=512,
        help="FFT size used in STFT (also equals spectrogram frequency bins in your setting)."
    )
    g_stft.add_argument(
        "--hop_length", type=int, default=int(122.88e6 * 0.05 / 750),
        help="Hop length in samples. If total duration is 50ms with 750 columns, hop ≈ sr*0.05/750."
    )
    
    # Preprocessor (box post-processing)
    g_pre = parser.add_argument_group("Preprocessor (Box post-processing)")
    g_pre.add_argument(
        "--pre_min_area", type=int, default=20,
        help="Minimum bbox area (pixels) used in preprocess.py filtering."
    )
    g_pre.add_argument(
        "--pre_min_ratio", type=float, default=0,
        help="Minimum width/height ratio for bboxes in preprocess.py. Larger -> prefer horizontal boxes."
    )
    g_pre.add_argument(
        "--pre_freq_eps", type=int, default=5,
        help="DBSCAN eps on frequency-center (pixels). Larger -> merge clusters more easily."
    )
    g_pre.add_argument(
        "--pre_freq_min_samples", type=int, default=2,
        help="DBSCAN min_samples. 1 means every box can form a cluster; >1 suppresses isolated boxes."
    )
    g_pre.add_argument(
        "--pre_nms_iou_thresh", type=float, default=0.5,
        help="NMS IoU threshold in preprocess.py for removing overlapped bboxes."
    )

    # Dataset / DataLoader
    g_data = parser.add_argument_group("Dataset & DataLoader")
    g_data.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to dataset file. You should modify this to your own dataset index/annotation file."
    )
    g_data.add_argument(
        "--val_ratio", type=float, default=0.2,
        help="Validation split ratio (0~1). Used when you split train/val from one dataset."
    )
    g_data.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of DataLoader workers."
    )
    g_data.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size for training/validation."
    )
    g_data.add_argument(
        "--sample_ratio", type=float, default=1.0,
        help="The ratio of sampling in the dataset every epoch."
    )
    
    # YOLOv5 Detector
    g_yolo = parser.add_argument_group("YOLOv5 Detector")
    g_yolo.add_argument(
        "--yolo_weights", type=str, default="",
        help="Path to YOLOv5 weights (.pt / .onnx / etc.)."
    )
    g_yolo.add_argument(
        "--yolo_device", type=str, default="",
        help='Device for YOLO inference. "" for auto, "cpu" for CPU, or GPU id like "0". '
    )
    g_yolo.add_argument(
        "--yolo_imgsz_h", type=int, default=640,
        help="YOLO inference image height after letterbox resize."
    )
    g_yolo.add_argument(
        "--yolo_imgsz_w", type=int, default=640,
        help="YOLO inference image width after letterbox resize."
    )
    g_yolo.add_argument(
        "--yolo_conf_thres", type=float, default=0.25,
        help="Confidence threshold for YOLO detections."
    )
    g_yolo.add_argument(
        "--yolo_iou_thres", type=float, default=0.45,
        help="IoU threshold for YOLO NMS."
    )
    g_yolo.add_argument(
        "--yolo_max_det", type=int, default=100,
        help="Maximum number of detections per image after NMS."
    )
    g_yolo.add_argument(
        "--yolo_classes", type=int, nargs="*", default=None,
        help="Optional class filter for YOLO detections, e.g. --yolo_classes 0 or --yolo_classes 0 1"
    )
    g_yolo.add_argument(
        "--yolo_half", action=argparse.BooleanOptionalAction, default=False,
        help="Use FP16 half-precision inference for YOLO (CUDA only)."
    )
    g_yolo.add_argument(
        "--yolo_warmup", action=argparse.BooleanOptionalAction, default=True,
        help="Run model warmup once after loading YOLO."
    )

    # Experiment / IO
    g_exp = parser.add_argument_group("Experiment & Checkpoints")
    g_exp.add_argument(
        "--epochs", type=int, default=50,
        help="Total training epochs."
    )
    g_exp.add_argument(
        "--save_interval", type=int, default=1,
        help="Save checkpoint every N epochs."
    )
    g_exp.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Optional path to resume training or load pretrained transformer weights."
    )

    # Optimizer / Scheduler
    g_opt = parser.add_argument_group("Optimizer & LR Scheduler")
    g_opt.add_argument(
        "--lr", type=float, default=1e-4,
        help="Base learning rate (e.g., 1e-4~1e-6 depending on your setup)."
    )
    g_opt.add_argument(
        "--weight_decay", type=float, default=1e-2,
        help="Weight decay for AdamW."
    )
    g_opt.add_argument(
        "--early_stop_patience", type=int, default=20,
        help="Early stopping patience (epochs). Stop if val metric does not improve."
    )
    g_opt.add_argument(
        "--cosine_annealing_T0", type=int, default=50,
        help="CosineAnnealingWarmRestarts: T0 (initial restart period in epochs)."
    )
    g_opt.add_argument(
        "--cosine_annealing_mult", type=int, default=2,
        help="CosineAnnealingWarmRestarts: T_mult (period multiplier at each restart)."
    )

    # Transformer Model
    g_model = parser.add_argument_group("Transformer Model")
    g_model.add_argument(
        "--feature_dim", type=int, default=8,
        help="Feature size extracted by preprocess network."
    )
    g_model.add_argument(
        "--d_model", type=int, default=128,
        help="Transformer hidden size (token embedding dimension)."
    )
    g_model.add_argument(
        "--nhead", type=int, default=4,
        help="Number of attention heads."
    )
    g_model.add_argument(
        "--num_layers", type=int, default=4,
        help="Number of Transformer encoder layers."
    )
    g_model.add_argument(
        "--num_classes", type=int, default=8,
        help="Number of protocol classes (multi-class classification)."
    )
    g_model.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout probability used in Transformer."
    )
    g_model.add_argument(
        "--max_tokens", type=int, default=32,
        help="Max number of tokens per sample (truncate if too many boxes survive)."
    )

    # Misc
    g_misc = parser.add_argument_group("Miscellaneous")
    g_misc.add_argument(
        "--use_data_parallel", action=argparse.BooleanOptionalAction, default=False,
        help="Use DataParallel for multi-GPU training (simpler but not as efficient as DDP)."
    )
    g_misc.add_argument(
        "--use_amp_autocast", action=argparse.BooleanOptionalAction, default=False,
        help="Enable AMP autocast (CUDA only). Recommended for faster training if stable."
    )

    args = parser.parse_args()
    return args

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

    if rank == 0:
        config.freeze()
        config.make_dir()
        config.save_config()
        logger = create_logger(os.path.join(config.log_dir, "train_log.log"))
        logger.init_exp(config)
        logger.info(f"DDP initialized: world_size={world_size}")
    else:
        logger = logging.getLogger("ddp_logger")
        logger.addHandler(logging.NullHandler())
    
    ## dataset 
    dataset = UAVDataset(config, logger)
    train_loader, val_loader = get_dataloader(dataset, config)

    ## model
    classifier = SignalTransformerClassifier(
        feature_dim=config.feature_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        pooling='attn')

    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    logger.info(f"Num of trainable params in backbone: {trainable_params:,}")
    
    if config.checkpoint_path and os.path.isfile(config.checkpoint_path):
        checkpoint_path = config.checkpoint_path
        load_checkpoint({"model": classifier}, path=checkpoint_path, device='cpu', logger=logger)
    else:
        checkpoint_path = None
        logger.warning(f'Checkpoint "{config.checkpoint_path}" not found')
    
    ## yolo 矩形框检测    
    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        yolo_device = str(local_rank)
    else:
        yolo_device = ""
        
    detector = YoloV5Detector(config, yolo_device)
    
    preprocessor = SignalPreprocessor(
        sampling_rate=config.sampling_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        min_area=config.pre_min_area,
        min_ratio=config.pre_min_ratio,
        freq_eps=config.pre_freq_eps,
        freq_min_samples=config.pre_freq_min_samples,
        nms_iou_thresh=config.pre_nms_iou_thresh,
    )
    
    trainer = Trainer(config, (train_loader, val_loader), logger, detector, preprocessor, classifier)

    trainer.train()


if __name__ == "__main__":

    args = get_parser()
    main(args)
