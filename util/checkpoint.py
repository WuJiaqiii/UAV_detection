"""Compact classifier artifacts for fresh training, inference and LwF teachers."""
import os
import re
from collections.abc import Mapping
from pathlib import Path

import torch


def load_checkpoint_object(path, map_location='cpu'):
    """Read a trusted checkpoint. Foreign-OS legacy pickle shims are not supported."""
    return torch.load(path, map_location=map_location, weights_only=False)


def _strip_module_prefix(state_dict):
    return {key.removeprefix('module.'): value for key, value in state_dict.items()}


def _validate_checkpoint_classes(checkpoint, expected_classes, logger=None):
    if expected_classes is None:
        return
    if not isinstance(expected_classes, Mapping):
        raise ValueError('expected_classes must be a mapping of class names to IDs')
    saved = checkpoint.get('config', {}).get('classes')
    if not isinstance(saved, Mapping):
        raise ValueError('Checkpoint config.classes is required')
    if dict(saved) != dict(expected_classes):
        raise ValueError(f'Checkpoint class mapping mismatch: saved={dict(saved)}, expected={dict(expected_classes)}')


def load_classifier_checkpoint(model, path, device='cpu', *, expected_classes=None):
    checkpoint = load_checkpoint_object(path, device)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise RuntimeError(f'No model_state_dict found in checkpoint: {path}')
    _validate_checkpoint_classes(checkpoint, expected_classes)
    model.load_state_dict(_strip_module_prefix(checkpoint['model_state_dict']), strict=True)
    return checkpoint


def metadata(value):
    """Store plain values; new artifacts never pickle platform-specific Path objects."""
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, Mapping):
        return {key: metadata(item) for key, item in value.items() if not str(key).startswith('_')}
    if isinstance(value, (tuple, list)):
        return [metadata(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported checkpoint metadata: {type(value).__name__}')


def prune_epoch_checkpoints(model_dir, keep_last, keep_epochs):
    """Only prune this run's epoch_N.pth files; best and explicit selected epochs survive."""
    if keep_last < 1 or any(n < 1 for n in keep_epochs):
        raise ValueError('checkpoint_keep_last>=1 and positive checkpoint_keep_epochs required')
    files = []
    for path in Path(model_dir).glob('epoch_*.pth'):
        match = re.fullmatch(r'epoch_(\d+)\.pth', path.name)
        if match and path.is_file() and not path.is_symlink():
            files.append((int(match[1]), path))
    keep = {n for n, _ in sorted(files, reverse=True)[:keep_last]} | set(keep_epochs)
    for number, path in files:
        if number not in keep:
            path.unlink()


def save_training_checkpoint(model, epoch, best_val_acc, model_dir, config=None, logger=None,
                             is_best=False, selection_metric='acc', selection_score=None,
                             validation_metrics=None):
    model_dir = Path(model_dir)
    # Validate retention before creating or replacing any file.
    keep_last = int(getattr(config, 'checkpoint_keep_last', 2))
    keep_epochs = list(getattr(config, 'checkpoint_keep_epochs', []))
    if keep_last < 1 or any(n < 1 for n in keep_epochs):
        raise ValueError('checkpoint_keep_last>=1 and positive checkpoint_keep_epochs required')
    model_dir.mkdir(parents=True, exist_ok=True)
    saved = dict(format_version=3, epoch=int(epoch), model_state_dict=model.state_dict(),
                 best_val_acc=float(best_val_acc), selection_metric=selection_metric,
                 selection_score=float(best_val_acc if selection_score is None else selection_score),
                 validation_metrics=validation_metrics or {},
                 config=metadata(vars(config)) if config is not None else {})
    path = model_dir/f'epoch_{epoch+1}.pth'
    targets = [path, model_dir/'best.pth'] if is_best else [path]
    for target in targets:
        temporary = target.with_suffix('.pth.tmp')
        torch.save(saved, temporary)
        os.replace(temporary, target)
    prune_epoch_checkpoints(model_dir, keep_last, keep_epochs)
    if logger:
        logger.info(f'[ckpt] saved {path}; best={is_best}; retained latest {keep_last} plus {keep_epochs}')
    return path
