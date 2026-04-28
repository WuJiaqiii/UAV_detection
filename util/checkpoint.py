import os
from pathlib import Path

import torch


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict

    first_key = next(iter(state_dict.keys()))
    if not str(first_key).startswith("module."):
        return state_dict

    return {
        str(k).replace("module.", "", 1): v
        for k, v in state_dict.items()
    }


def save_training_checkpoint(model, optimizer, scheduler, epoch: int, best_val_acc: float, model_dir, config=None, logger=None, is_best: bool = False):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    save_obj = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": float(best_val_acc),
        "config": vars(config) if hasattr(config, "__dict__") else {},
    }

    ckpt_path = model_dir / f"epoch_{epoch + 1}.pth"
    tmp_path = model_dir / f"epoch_{epoch + 1}.pth.tmp"

    torch.save(save_obj, tmp_path)
    os.replace(tmp_path, ckpt_path)

    if logger is not None:
        logger.info(f"[ckpt] saved: {ckpt_path}")

    if is_best:
        best_path = model_dir / "best.pth"
        best_tmp = model_dir / "best.pth.tmp"

        torch.save(save_obj, best_tmp)
        os.replace(best_tmp, best_path)

        if logger is not None:
            logger.info(f"[ckpt] saved best: {best_path}")

    return ckpt_path


def load_training_checkpoint(model, checkpoint_path: str, device, optimizer=None, scheduler=None, logger=None, load_optimizer: bool = False):
    checkpoint_path = str(checkpoint_path or "").strip()
    if not checkpoint_path:
        return None

    if not os.path.isfile(checkpoint_path):
        if logger is not None:
            logger.warning(f'Checkpoint "{checkpoint_path}" not found')
        return None

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    if "model_state_dict" not in ckpt:
        raise RuntimeError(
            f"No model_state_dict found in checkpoint: {checkpoint_path}"
        )

    state_dict = _strip_module_prefix(ckpt["model_state_dict"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if logger is not None:
        logger.info(f"[ckpt] loaded model from {checkpoint_path}")
        if len(missing) > 0:
            logger.warning(f"[ckpt] missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if len(unexpected) > 0:
            logger.warning(f"[ckpt] unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

    if load_optimizer:
        if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if logger is not None:
                    logger.info("[ckpt] optimizer loaded")
            except Exception as e:
                if logger is not None:
                    logger.warning(f"[ckpt] failed to load optimizer: {e}")

        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if logger is not None:
                    logger.info("[ckpt] scheduler loaded")
            except Exception as e:
                if logger is not None:
                    logger.warning(f"[ckpt] failed to load scheduler: {e}")

    return ckpt