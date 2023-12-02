import torch
import torch.nn as nn
from torch import Tensor


def binary_segm_precision_recall_f1(pred: Tensor, target: Tensor, threshold: float = 0.5):
    pred = torch.where(torch.lt(pred, threshold), 0, 1).bool().flatten(start_dim=1)
    target = torch.where(torch.lt(target, threshold), 0, 1).bool().flatten(start_dim=1)
    tp = torch.sum(pred & target, dim=-1).float()
    fp = torch.sum(pred & ~target, dim=-1).float()
    fn = torch.sum(~pred & target, dim=-1).float()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


class BinarySegmPrecisionRecallF1(nn.Module):
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

        precision, recall, f1 = binary_segm_precision_recall_f1(pred, target, threshold=self.threshold)

        if self.reduction == 'mean':
            return dict(precision=precision.mean(), recall=recall.mean(), f1=f1.mean())
        elif self.reduction == 'sum':
            return dict(precision=precision.sum(), recall=recall.sum(), f1=f1.sum())
        elif self.reduction == 'none':
            return dict(precision=precision, recall=recall, f1=f1)
        else:
            raise ValueError(f'Invalid reduction: {self.reduction}')
