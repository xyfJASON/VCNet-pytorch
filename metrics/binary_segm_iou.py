import torch
import torch.nn as nn
from torch import Tensor


def binary_segm_iou(pred: Tensor, target: Tensor, threshold: float = 0.5):
    pred = torch.where(torch.lt(pred, threshold), 0, 1).bool().flatten(start_dim=1)
    target = torch.where(torch.lt(target, threshold), 0, 1).bool().flatten(start_dim=1)
    intersection = torch.sum(pred & target, dim=-1).float()
    union = torch.sum(pred | target, dim=-1).float()
    iou = intersection / union
    return iou


class BinarySegmIoU(nn.Module):
    def __init__(self, threshold: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction should be 'none', 'mean' or 'sum', get '{reduction}'")
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor):
        assert (0. <= pred.float()).all() and (pred.float() <= 1).all()
        assert pred.shape == target.shape
        assert pred.shape[1] == target.shape[1] == 1

        iou = binary_segm_iou(pred, target, threshold=self.threshold)

        if self.reduction == 'mean':
            return iou.mean()
        elif self.reduction == 'sum':
            return iou.sum()
        elif self.reduction == 'none':
            return iou
        else:
            raise ValueError(f'Invalid reduction: {self.reduction}')
