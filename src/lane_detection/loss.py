from helpers import get_device

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

device = get_device()


class FocalLoss(nn.Module):
    """
    Only consider two class now: foreground, background.
    """

    def __init__(self, gamma=2, alpha=[0.25, 0.75], n_class=2, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.n_class = n_class
        self.device = device

    def forward(self, input, target):
        pt = F.softmax(input, dim=1)
        pt = pt.clamp(min=0.000001, max=0.999999)
        target_onehot = torch.zeros(
            (target.size(0), self.n_class, target.size(1), target.size(2))
        ).to(self.device)
        loss = 0
        for i in range(self.n_class):
            target_onehot[:, i, ...][target == i] = 1
        for i in range(self.n_class):
            loss -= (
                self.alpha[i]
                * (1 - pt[:, i, ...]) ** self.gamma
                * target_onehot[:, i, ...]
                * torch.log(pt[:, i, ...])
            )

        if self.reduction == "mean":
            loss = torch.mean(loss)
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss


class DiscriminativeLoss(_Loss):
    def __init__(
        self, delta_var=0.5, delta_dist=1.5, norm=1.0, alpha=1.0, beta=1.0, gamma=0.001
    ):
        super().__init__(reduction="mean")

        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, _input, target):
        return self.compute_loss(_input, target)

    def compute_loss(self, _input, target):
        batch_size, embed_dim, H, W = _input.shape

        _input = _input.reshape(batch_size, embed_dim, H * W)
        target = target.reshape(batch_size, H * W)
        var_loss = torch.tensor(0, dtype=_input.dtype, device=_input.device)
        dist_loss = torch.tensor(0, dtype=_input.dtype, device=_input.device)

        for b in range(batch_size):
            # embed_dim, h*w
            batch_input = _input[b]
            # w*h
            batch_target = target[b]

            labels, indexs = torch.unique(batch_target, return_inverse=True)
            num_lanes = len(labels)
            if num_lanes == 0:
                _nonsense = batch_input.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = batch_target == lane_idx

                if seg_mask_i.all():
                    continue
                embedding_i = batch_input * seg_mask_i
                mean_i = torch.sum(embedding_i, dim=1) / torch.sum(seg_mask_i)
                centroid_mean.append(mean_i)
                var_loss = (
                    var_loss
                    + torch.sum(
                        F.relu(
                            torch.norm(
                                embedding_i[:, seg_mask_i]
                                - mean_i.reshape(embed_dim, 1),
                                dim=0,
                            )
                            - self.delta_var
                        )
                        ** 2
                    )
                    / torch.sum(seg_mask_i)
                    / num_lanes
                )
            if centroid_mean:
                centroid_mean = torch.stack(centroid_mean)
                if num_lanes > 1:
                    centroid_mean1 = centroid_mean.reshape(-1, 1, embed_dim)
                    centroid_mean2 = centroid_mean.reshape(1, -1, embed_dim)

                    # shape (num_lanes, num_lanes)
                    dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2)
                    dist = (
                        dist
                        + torch.eye(num_lanes, dtype=dist.dtype, device=dist.device)
                        * self.delta_dist
                    )

                    dist_loss = (
                        dist_loss
                        + torch.sum(F.relu(-dist + self.delta_dist) ** 2)
                        / (num_lanes * (num_lanes - 1))
                        / 2
                    )
        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        return var_loss, dist_loss
