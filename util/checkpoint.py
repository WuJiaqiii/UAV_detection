import logging, torch, collections
import torch.nn as nn

_LOG = logging.getLogger("ckpt")

def _strip_module_prefix(state_dict):
    """允许单卡/DP 混用：把 'module.' 前缀去掉"""
    if not next(iter(state_dict)).startswith("module."):
        return state_dict
    return collections.OrderedDict((k.replace("module.", "", 1), v)
                                   for k, v in state_dict.items())

def _get_state_dict(m: nn.Module):
    if isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return m.module.state_dict()
    return m.state_dict()

def _is_state_dict_compatible(module: nn.Module, sd: dict) -> bool:
    """判断 checkpoint 中的 state_dict 与当前模块是否完全兼容"""
    cur_sd = module.state_dict()
    if set(cur_sd.keys()) != set(sd.keys()):
        return False
    for k in cur_sd:
        if cur_sd[k].shape != sd[k].shape:
            return False
    return True

def save_checkpoint(models, optimizer, scheduler, epoch, path, cfg=None, logger=None, concept_prototypes=None, concept_map=None):
    ckpt = dict(
        version = 1, epoch = epoch,
        models  = {n: _get_state_dict(m) for n, m in models.items()},
        optimizer = optimizer.state_dict() if optimizer else None,
        scheduler = scheduler.state_dict() if scheduler else None,
        cfg = cfg,
    )
    if concept_prototypes is not None:
        ckpt["concept_prototypes"] = concept_prototypes
    if concept_map is not None:
        ckpt["concept_map"] = concept_map
    torch.save(ckpt, path)
    (logger or _LOG).info(f"[ckpt] saved to {path}")

def load_checkpoint(models: dict, path: str,
                    optimizer=None, scheduler=None,
                    strict=False, device="cpu", logger=None):
    
    try:
        from util.config import Config
        torch.serialization.add_safe_globals([Config])
    except Exception:
        pass

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    ckpt_models = ckpt.get("models", {})

    for name, module in models.items():
        if name not in ckpt_models:
            (logger or _LOG).warning(f"[ckpt] <{name}> not found, keep init params.")
            continue
        sd = _strip_module_prefix(ckpt_models[name])
        if not _is_state_dict_compatible(module, sd):
            (logger or _LOG).warning(
                f"[ckpt] <{name}> param mismatch, skip loading for this model."
            )
            continue

        module.load_state_dict(sd)
        (logger or _LOG).info(f"[ckpt] <{name}> loaded OK.")

    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    (logger or _LOG).info(f"[ckpt] loaded from {path}, epoch={ckpt.get('epoch')}, "
                f"version={ckpt.get('version')}")
    return ckpt
