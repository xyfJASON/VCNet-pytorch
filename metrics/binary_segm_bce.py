import torch
import torch.nn as nn
from torch import Tensor


class BinarySegmBCE(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction should be 'none', 'mean' or 'sum', get '{reduction}'")
        self.bce = nn.BCELoss(reduction='none')
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor):
        assert (0. <= pred.float()).all() and (pred.float() <= 1).all()
        assert pred.shape == target.shape
        assert pred.shape[1] == target.shape[1] == 1
        pred = pred.flatten(start_dim=1)
        target = target.flatten(start_dim=1)
        bce = torch.mean(self.bce(pred, target), dim=-1)
        if self.reduction == 'mean':
            return bce.mean()
        elif self.reduction == 'sum':
            return bce.sum()
        elif self.reduction == 'none':
            return bce
        else:
            raise ValueError(f'Invalid reduction: {self.reduction}')
