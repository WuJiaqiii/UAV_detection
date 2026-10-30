import os
import warnings
from collections.abc import Mapping
from pathlib import Path

import torch


def load_checkpoint_object(path, map_location="cpu"):
    """Load a trusted project checkpoint, including foreign-OS path metadata."""
    from util import checkpoint_pickle
    return torch.load(path, map_location=map_location, weights_only=False,
                      pickle_module=checkpoint_pickle)


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


def _validate_checkpoint_classes(ckpt, expected_classes, logger=None):
    """Check label semantics before changing model or optimizer state.

    Dictionary insertion order is irrelevant; each name must retain its ID.
    Older checkpoints without class metadata remain loadable with a warning.
    """
    if expected_classes is None:
        return
    if not isinstance(expected_classes, Mapping):
        raise ValueError("expected_classes must be a mapping of class names to IDs")

    config = ckpt.get("config")
    saved_classes = config.get("classes") if isinstance(config, Mapping) else None
    if saved_classes is None:
        message = (
            "[ckpt] checkpoint has no config.classes metadata; class names and IDs "
            "cannot be verified. Ensure the current classes match the original training."
        )
        if logger is not None:
            logger.warning(message)
        else:
            warnings.warn(message, UserWarning, stacklevel=3)
        return
    if not isinstance(saved_classes, Mapping):
        raise ValueError("Checkpoint config.classes must be a mapping of class names to IDs")
    if dict(saved_classes) != dict(expected_classes):
        raise ValueError(
            "Checkpoint class mapping mismatch: "
            f"saved={dict(saved_classes)}, expected={dict(expected_classes)}. "
            "Use the same class names and label IDs as the original training, "
            "including exclude_classes and incremental_split_mode settings. "
            "To add new classes, use the dedicated LwF teacher initialization path."
        )


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


def load_training_checkpoint(
    model,
    checkpoint_path: str,
    device,
    optimizer=None,
    scheduler=None,
    logger=None,
    load_optimizer: bool = False,
    skip_mismatch: bool = False,
    expected_classes=None,
):
    """Load classifier state, optionally restoring training state and checking labels.

    ``expected_classes`` is a name-to-ID mapping, as used by ``Config.classes``.
    ``skip_mismatch`` only controls tensor shapes, not class mapping validation.
    """
    checkpoint_path = str(checkpoint_path or "").strip()
    if not checkpoint_path:
        return None

    if not os.path.isfile(checkpoint_path):
        if logger is not None:
            logger.warning(f'Checkpoint "{checkpoint_path}" not found')
        return None

    ckpt = load_checkpoint_object(checkpoint_path, map_location=device)

    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

    if "model_state_dict" not in ckpt:
        raise RuntimeError(
            f"No model_state_dict found in checkpoint: {checkpoint_path}"
        )

    _validate_checkpoint_classes(ckpt, expected_classes, logger=logger)
    state_dict = _strip_module_prefix(ckpt["model_state_dict"])

    if skip_mismatch:
        model_state = model.state_dict()
        filtered = {}
        skipped = []

        for k, v in state_dict.items():
            if k in model_state and tuple(v.shape) == tuple(model_state[k].shape):
                filtered[k] = v
            else:
                skipped.append(k)

        state_dict = filtered

        if logger is not None and skipped:
            logger.warning(
                f"[ckpt] skipped mismatch keys: {skipped[:20]}"
                f"{' ...' if len(skipped) > 20 else ''}"
            )

    missing, unexpected = model.load_state_dict(state_dict, strict=not skip_mismatch)

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
                raise RuntimeError("Checkpoint optimizer state cannot be restored") from e

        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if logger is not None:
                    logger.info("[ckpt] scheduler loaded")
            except Exception as e:
                raise RuntimeError("Checkpoint scheduler state cannot be restored") from e

    return ckpt
