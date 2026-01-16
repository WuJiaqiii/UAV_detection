import os
import argparse
import torch
import logging
import torch.distributed

from data.data_loader import UAVDataset, get_dataloader

from models.transformer import SignalTransformerClassifier

from util.trainer import Trainer

from util.preprocess import SignalPreprocessor
from sam2.sam2.build_sam import build_sam2
from sam2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

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
        description="STFT spectrogram -> (SAM2 detection + preprocessing) -> Transformer classification training"
    )

    # Dataset / DataLoader
    g_data = parser.add_argument_group("Dataset & DataLoader")
    g_data.add_argument(
        "--dataset_path", type=str, default="data/RML2016.10a_dict.pkl",
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
    g_data.add_argument(
        "--feature_size", type=int, default=128,
        help="(Legacy/Optional) Feature size extracted by some network. Not used in token-transformer path unless you wire it."
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
        "--lr", type=float, default=1e-6,
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
        "--lr_scheduler_patience", type=int, default=10,
        help="ReduceLROnPlateau patience (epochs). Only used if you choose plateau scheduler."
    )
    g_opt.add_argument(
        "--lr_reduce_factor", type=float, default=0.1,
        help="ReduceLROnPlateau factor: new_lr = lr * factor."
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
        help="Input token feature dimension produced by preprocess.py (typically 8)."
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
        "--num_classes", type=int, default=10,
        help="Number of protocol classes (multi-class classification)."
    )
    g_model.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout probability used in Transformer."
    )
    g_model.add_argument(
        "--max_tokens", type=int, default=128,
        help="Max number of tokens per sample (truncate if too many boxes survive)."
    )

    # SAM2 Detector (Inference-only)
    g_sam2 = parser.add_argument_group("SAM2 Detector (Inference-only)")
    g_sam2.add_argument(
        "--sam2_checkpoint", type=str, default="sam2/checkpoints/sam2.1_hiera_large.pt",
        help="Path to SAM2 checkpoint (.pt). Used for box generation only (no training)."
    )
    g_sam2.add_argument(
        "--model_cfg", type=str, default="sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to SAM2 model config (.yaml)."
    )
    g_sam2.add_argument(
        "--sam2_points_per_side", type=int, default=32,
        help="SAM2 mask generator sampling density. Lower -> faster & fewer masks; Higher -> more recall but slower."
    )
    g_sam2.add_argument(
        "--sam2_points_per_batch", type=int, default=128,
        help="SAM2 points processed per batch. Increase may speed up but uses more VRAM."
    )
    g_sam2.add_argument(
        "--sam2_crop_n_layers", type=int, default=0,
        help="SAM2 cropping layers. 0 = no multi-scale crops (fast). 1+ improves small-object recall but increases masks & runtime."
    )
    g_sam2.add_argument(
        "--sam2_pred_iou_thresh", type=float, default=0.85,
        help="SAM2 predicted IoU threshold for keeping masks. Higher -> fewer masks; lower -> more masks."
    )
    g_sam2.add_argument(
        "--sam2_stability_score_thresh", type=float, default=0.85,
        help="SAM2 stability score threshold. Higher -> fewer masks; lower -> more masks."
    )
    g_sam2.add_argument(
        "--sam2_box_nms_thresh", type=float, default=0.6,
        help="NMS threshold to suppress similar masks by their bbox overlap."
    )
    g_sam2.add_argument(
        "--sam2_crop_nms_thresh", type=float, default=0.6,
        help="NMS threshold between crops (only meaningful when crop_n_layers>0)."
    )
    g_sam2.add_argument(
        "--sam2_min_mask_region_area", type=int, default=120,
        help="Minimum connected mask region area (pixels). Helps remove tiny fragments."
    )

    g_sam2.add_argument(
        "--min_rectangularity", type=float, default=0.75,
        help="Rectangularity filter for masks: rectangularity = mask_area / bbox_area. Higher -> keep more rectangle-like masks."
    )
    g_sam2.add_argument(
        "--max_bbox_area_ratio", type=float, default=0.20,
        help="Upper bound on bbox area ratio: bbox_area / (H*W). Used to remove too-large boxes (e.g., <=0.20 means <=20%% image area)."
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
        "--pre_min_ratio", type=float, default=2.0,
        help="Minimum width/height ratio for bboxes in preprocess.py. Larger -> prefer horizontal boxes."
    )
    g_pre.add_argument(
        "--pre_freq_eps", type=int, default=5,
        help="DBSCAN eps on frequency-center (pixels). Larger -> merge clusters more easily."
    )
    g_pre.add_argument(
        "--pre_freq_min_samples", type=int, default=1,
        help="DBSCAN min_samples. 1 means every box can form a cluster; >1 suppresses isolated boxes."
    )
    g_pre.add_argument(
        "--pre_nms_iou_thresh", type=float, default=0.5,
        help="NMS IoU threshold in preprocess.py for removing overlapped bboxes."
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
    model = SignalTransformerClassifier(
        feature_dim=config.feature_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        pooling='attn')

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Num of trainable params in backbone: {trainable_params:,}")
    
    if config.checkpoint_path and os.path.isfile(config.checkpoint_path):
        checkpoint_path = config.checkpoint_path
        load_checkpoint({"model": model}, path=checkpoint_path, device='cpu', logger=logger)
    else:
        checkpoint_path = None
        logger.warning(f'Checkpoint "{config.checkpoint_path}" not found')
    
    ## SAM2 矩形框检测    
    sam2_model = build_sam2(config.model_cfg, config.sam2_checkpoint, device=config.device)
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=config.sam2_points_per_side,
        points_per_batch=config.sam2_points_per_batch,
        crop_n_layers=config.sam2_crop_n_layers,
        pred_iou_thresh=config.sam2_pred_iou_thresh,
        stability_score_thresh=config.sam2_stability_score_thresh,
        box_nms_thresh=config.sam2_box_nms_thresh,
        crop_nms_thresh=config.sam2_crop_nms_thresh,
        min_mask_region_area=config.sam2_min_mask_region_area,
    )
    
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
    
    trainer = Trainer(config, (train_loader, val_loader), logger, model, mask_generator, preprocessor)
    trainer.train()


if __name__ == "__main__":

    args = get_parser()
    main(args)
