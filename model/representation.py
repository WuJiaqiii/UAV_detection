"""Known-class representation learning on the existing binary signal masks."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from util.mask_input import build_mask_input


def validate_representation_options(config):
    weight = float(getattr(config, "supcon_weight", 0.))
    temperature = float(getattr(config, "supcon_temperature", .2))
    probability = float(getattr(config, "supcon_drop_prob", .1))
    fraction = float(getattr(config, "supcon_max_drop_fraction", .25))
    jitter = getattr(config, "supcon_jitter_px", 1)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("supcon_weight must be finite and >= 0")
    if not math.isfinite(temperature) or temperature < 1e-6:
        raise ValueError("supcon_temperature must be finite and >= 1e-6")
    if not 0 <= probability <= 1 or not 0 <= fraction <= .5:
        raise ValueError("Need supcon_drop_prob in [0,1] and supcon_max_drop_fraction in [0,0.5]")
    if isinstance(jitter, bool) or not isinstance(jitter, int) or not 0 <= jitter <= 8:
        raise ValueError("supcon_jitter_px must be an integer in [0,8] on the native raster")
    if weight and (getattr(config, "freeze_backbone", False) or getattr(config, "lwf_freeze_backbone", False)):
        raise ValueError("SupCon requires feature training; disable freeze_backbone/lwf_freeze_backbone")
    return weight


class SupervisedContrastiveLoss(nn.Module):
    """Stable SupCon on [samples, views, features]; labels are known truth only.

    Positives include views of the same sample and other samples of that class.
    A batch without a positive pair contributes a differentiable zero.
    """
    def __init__(self, temperature=.2):
        super().__init__()
        if not math.isfinite(temperature) or temperature < 1e-6:
            raise ValueError("temperature must be finite and >= 1e-6")
        self.temperature = temperature

    def forward(self, features, labels):
        if features.ndim != 3 or features.shape[-1] < 1 or features.shape[1] < 1:
            raise ValueError("SupCon features must be [B,V,D] with V,D > 0")
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("SupCon labels must be [B]")
        if not torch.isfinite(features).all():
            raise ValueError("SupCon features must be finite")
        if not len(features):
            return features.sum() * 0.
        values = F.normalize(features.float().reshape(-1, features.shape[-1]), dim=1)
        ids = labels.to(values.device).repeat_interleave(features.shape[1])
        diagonal = torch.eye(len(values), dtype=torch.bool, device=values.device)
        positives = ids[:, None].eq(ids[None, :]) & ~diagonal
        counts = positives.sum(1)
        valid = counts > 0
        if not valid.any():
            return features.sum() * 0.
        logits = values @ values.T / self.temperature
        denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
        log_prob = logits - denominator[:, None]
        positive_log_prob = torch.where(positives, log_prob, torch.zeros_like(log_prob)).sum(1)
        return -(positive_log_prob[valid] / counts[valid]).mean()


def augment_group_boxes(boxes, shape, config, rng=None):
    """Simulate a few missing/jittered detections without rescaling pulse shape.

    Keep at least two pulses (or the original singleton), cap the fraction dropped,
    and translate each rectangle by a few native pixels while preserving its size.
    """
    rng = np.random if rng is None else rng
    height, width = map(int, shape)
    original = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
    if not len(original) or ((original[:, 2:] - original[:, :2]) <= 0).any():
        raise ValueError("Contrastive views require nonempty valid group boxes")
    if ((original[:, [0, 2]] < 0).any() or (original[:, [0, 2]] > width).any()
            or (original[:, [1, 3]] < 0).any() or (original[:, [1, 3]] > height).any()):
        raise ValueError("Contrastive group boxes must lie within the native raster")
    probability = float(getattr(config, "supcon_drop_prob", .1))
    fraction = float(getattr(config, "supcon_max_drop_fraction", .25))
    limit = min(int(np.floor(len(original)*fraction)), max(0, len(original)-2))
    candidates = np.flatnonzero(rng.random(len(original)) < probability)
    dropped = rng.choice(candidates, min(limit, len(candidates)), replace=False) if limit and len(candidates) else []
    keep = np.ones(len(original), dtype=bool)
    keep[dropped] = False
    result = original[keep].copy()
    jitter = int(getattr(config, "supcon_jitter_px", 1))
    if jitter:
        randint = rng.integers if hasattr(rng, "integers") else rng.randint
        shifts = randint(-jitter, jitter+1, size=(len(result), 2))
        for box, (dx, dy) in zip(result, shifts):
            dx = int(np.clip(dx, -box[0], width-box[2]))
            dy = int(np.clip(dy, -box[1], height-box[3]))
            box += np.array([dx, dy, dx, dy])
    return result


def contrastive_mask_batch(metas, config):
    tensors = []
    for meta in metas:
        if "boxes" not in meta or "spec_shape" not in meta:
            raise ValueError("SupCon requires selected native boxes and spec_shape in training metadata")
        shape = tuple(meta["spec_shape"])
        boxes = augment_group_boxes(meta["boxes"], shape, config)
        tensor, _ = build_mask_input(np.zeros(shape, dtype=np.uint8), boxes, config)
        if not torch.any(tensor):
            # A one-pixel translation can erase a very thin pulse at resize time.
            tensor, _ = build_mask_input(np.zeros(shape, dtype=np.uint8), meta["boxes"], config)
        tensors.append(tensor)
    if not tensors:
        raise ValueError("Cannot build a contrastive view for an empty instance batch")
    return torch.stack(tensors)
