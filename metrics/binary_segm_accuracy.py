import torch
import torch.nn as nn
from torch import Tensor


def binary_segm_accuracy(pred: Tensor, target: Tensor, threshold: float = 0.5):
    pred = torch.where(torch.lt(pred, threshold), 0, 1).bool().flatten(start_dim=1)
    target = torch.where(torch.lt(target, threshold), 0, 1).bool().flatten(start_dim=1)
    correct = torch.sum(torch.eq(pred, target), dim=-1).float()
    total = pred.numel() // pred.shape[0]
    acc = correct / total
    return acc


class BinarySegmAccuracy(nn.Module):
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

        acc = binary_segm_accuracy(pred, target, threshold=self.threshold)

        if self.reduction == 'mean':
            return acc.mean()
        elif self.reduction == 'sum':
            return acc.sum()
        elif self.reduction == 'none':
            return acc
        else:
            raise ValueError(f'Invalid reduction: {self.reduction}')
