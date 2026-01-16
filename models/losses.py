import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function

class NormalizedCenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(NormalizedCenterLoss, self).__init__()
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(self.device))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim

    def forward(self, x, label):

        feat = x.to(self.device)
        label = label.to(self.device)

        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)

        feat = F.normalize(feat, p=2, dim=1)
        norm_centers = F.normalize(self.centers, p=2, dim=1)
        # self.centers = F.normalize(self.centers, p=2, dim=1)

        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size).to(self.device)
        loss = self.centerlossfunc(feat, label, norm_centers, batch_size_tensor)
        return loss

# class CenterlossFunc(Function):
#     @staticmethod
#     def forward(ctx, feature, label, centers, batch_size):
#         ctx.save_for_backward(feature, label, centers, batch_size)
#         centers_batch = centers.index_select(0, label.long())
#         return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

#     @staticmethod
#     def backward(ctx, grad_output):
#         feature, label, centers, batch_size = ctx.saved_tensors
#         centers_batch = centers.index_select(0, label.long())
#         diff = centers_batch - feature
        
#         counts = centers.new_ones(centers.size(0)).to(feature.device)
#         ones = centers.new_ones(label.size(0)).to(feature.device)
#         grad_centers = centers.new_zeros(centers.size()).to(feature.device)

#         counts = counts.scatter_add_(0, label.long(), ones)
#         grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
#         grad_centers = grad_centers/counts.view(-1, 1)
#         return - grad_output * diff / batch_size, None, grad_centers / batch_size, None

class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feat, label, centers, batch_size):
        ctx.save_for_backward(feat, label, centers, batch_size)
        centers_batch = centers.index_select(0, label)
        return (feat - centers_batch).pow(2).sum() / (2.0 * batch_size)

    @staticmethod
    def backward(ctx, grad_output):
        feat, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label)

        diff = centers_batch - feat                        # (B, D)
        grad_centers = torch.zeros_like(centers)           # NEW tensor, leaf
        grad_centers = grad_centers.index_add(0, label, diff) / batch_size

        grad_feat = -diff / batch_size

        return grad_output * grad_feat, None, grad_centers, None

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
import numpy as np


# class SupConLoss(torch.nn.Module):
#     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#     It also supports the unsupervised contrastive loss in SimCLR
#     From: https://github.com/HobbitLong/SupContrast"""
#     def __init__(self, temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.base_temperature = base_temperature

#     def forward(self, features, labels=None, mask=None):
#         """Compute loss for model. If both `labels` and `mask` are None,
#         it degenerates to SimCLR unsupervised loss:
#         https://arxiv.org/pdf/2002.05709.pdf
#         Args:
#             features: hidden vector of shape [bsz, n_views, ...].
#             labels: ground truth of shape [bsz].
#             mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
#                 has the same class as sample i. Can be asymmetric.
#         Returns:
#             A loss scalar.
#         """

#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))

#         if len(features.shape) < 3:
#             raise ValueError('`features` needs to be [bsz, n_views, ...],'
#                              'at least 3 dimensions are required')
#         if len(features.shape) > 3:
#             features = features.view(features.shape[0], features.shape[1], -1)

#         batch_size = features.shape[0]
#         if labels is not None and mask is not None:
#             raise ValueError('Cannot define both `labels` and `mask`')
#         elif labels is None and mask is None:
#             mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#         elif labels is not None:
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError('Num of labels does not match num of features')
#             mask = torch.eq(labels, labels.T).float().to(device)
#         else:
#             mask = mask.float().to(device)

#         contrast_count = features.shape[1]
#         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
#         if self.contrast_mode == 'one':
#             anchor_feature = features[:, 0]
#             anchor_count = 1
#         elif self.contrast_mode == 'all':
#             anchor_feature = contrast_feature
#             anchor_count = contrast_count
#         else:
#             raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(anchor_feature, contrast_feature.T),
#             self.temperature)

#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()

#         # tile mask
#         mask = mask.repeat(anchor_count, contrast_count)
#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#             0
#         )
#         mask = mask * logits_mask

#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

#         # compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss = loss.view(anchor_count, batch_size).mean()

#         return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (robust version)
    expected input:
        features: [B, n_views, D]  (or [B, D] for single view fallback)
        labels: [B] (long)
    returns:
        scalar loss (averaged over valid anchor samples)
    """
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels):
        """
        features: torch.Tensor, shape (B, n_views, D) or (B, D)
        labels: torch.LongTensor shape (B,)
        """
        device = features.device
        if features.dim() == 2:
            # fallback: single-view features
            features = features.unsqueeze(1)

        B = features.shape[0]
        n_views = features.shape[1]
        # reshape features -> (B * n_views, D)
        feat = features.view(B * n_views, -1)
        feat = F.normalize(feat, p=2, dim=1)

        # compute similarity matrix
        sim_matrix = torch.matmul(feat, feat.T) / self.temperature  # (B*n_views, B*n_views)

        # mask for positives: same label AND not same view index for same sample
        labels = labels.contiguous().view(-1, 1)  # (B, 1)
        if labels.dtype == torch.long or labels.dtype == torch.int:
            # Build label mask across expanded views
            labels_exp = labels.repeat(1, n_views).view(-1, 1)  # (B*n_views, 1)
            mask = torch.eq(labels_exp, labels_exp.T).float().to(device)  # (B*n_views, B*n_views)
        else:
            # if labels are -1 or something else, fallback to instance-level (no positives)
            mask = torch.eye(B * n_views, device=device)

        # remove self-comparisons by zeroing diagonal
        logits_mask = (~torch.eye(B * n_views, device=device).bool()).float()
        # exponential of similarity excluding diagonal
        exp_sim = torch.exp(sim_matrix) * logits_mask  # (BN, BN)

        # For each anchor (row) compute log_prob = sim - log(sum(exp_sim_row))
        denom = exp_sim.sum(dim=1, keepdim=True)  # (BN, 1)
        # numerical stability
        denom = denom + 1e-12
        log_prob = sim_matrix - torch.log(denom)

        # Positive mask (we should count positives across views)
        # But ensure we exclude self pairs (same sample same view) by zeros on diagonal
        positive_mask = mask * logits_mask  # (BN, BN)

        # sum over positives
        numerator = (positive_mask * log_prob).sum(dim=1)  # (BN,)
        pos_count = positive_mask.sum(dim=1)  # (BN,)

        # valid anchors: those that have at least one positive
        valid = (pos_count > 0).float()  # (BN,)
        valid_count = int(valid.sum().item())

        if valid_count == 0:
            # no valid anchors in batch => return zero loss (safe)
            return torch.tensor(0.0, device=device, requires_grad=True)

        # compute mean log prob for positives for valid anchors
        # avoid division by zero by replacing zeros with ones, numerator will be 0 so result 0
        pos_count_safe = torch.where(pos_count == 0, torch.ones_like(pos_count), pos_count)
        mean_log_prob_pos = numerator / pos_count_safe  # (BN,)

        # only consider valid anchors in final loss
        loss = - (mean_log_prob_pos * valid).sum() / valid_count
        return loss

class NTXentLoss(nn.Module):
        def __init__(self, temperature=0.07):
            super().__init__()
            self.temperature = temperature
        def forward(self, features, labels):
            # features: (B, D), labels: (B,)
            device = features.device
            f = nn.functional.normalize(features, dim=1)
            sim = torch.matmul(f, f.t()) / self.temperature  # (B,B)
            labels = labels.contiguous().view(-1,1)
            mask = torch.eq(labels, labels.t()).float().to(device)
            diag = torch.eye(mask.size(0), device=device)
            mask = mask - diag
            exp_sim = torch.exp(sim) * (1 - diag)
            log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
            mask_sum = mask.sum(dim=1)
            mask_sum = torch.where(mask_sum==0, torch.ones_like(mask_sum), mask_sum)
            mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum
            loss = - mean_log_prob_pos
            return loss.mean()