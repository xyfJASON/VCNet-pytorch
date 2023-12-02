"""
PCN and PCBlock is modified from https://github.com/birdortyedi/vcnet-blind-image-inpainting
"""

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


def init_weights(init_type=None, gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__

        if classname.find('BatchNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=gain)
            elif init_type is None:
                m.reset_parameters()
            else:
                raise ValueError(f'invalid initialization method: {init_type}.')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_func


def get_activation_layer(act: str = None, *args, **kwargs):
    if act == 'relu':
        return nn.ReLU(*args, **kwargs)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(*args, **kwargs)
    elif act == 'elu':
        return nn.ELU(*args, **kwargs)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act is None:
        return None
    else:
        raise ValueError(f'invalid activation: {act}.')


def get_normalization_layer(norm: str = None, *args, **kwargs):
    if norm == 'bn':
        return nn.BatchNorm2d(*args, **kwargs)
    elif norm == 'ln':
        return nn.LayerNorm(*args, **kwargs)
    elif norm is None:
        return None
    else:
        raise ValueError(f'invalid normalization: {norm}.')


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 padding: int = 0, dilation: int = 1, norm: str = None, activation: str = None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm = get_normalization_layer(norm)
        self.act = get_activation_layer(activation)

    def forward(self, X: Tensor):
        out = self.conv(X)
        if self.norm:
            out = self.norm(out)
        if self.act:
            out = self.act(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, norm: str = None, activation: str = None):
        super().__init__()

        if in_channels == out_channels and stride == 1:
            self.projection = None
        else:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm1 = get_normalization_layer(norm)
        self.act1 = get_activation_layer(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
        self.norm2 = get_normalization_layer(norm)
        self.act2 = get_activation_layer(activation)

    def forward(self, X: torch.Tensor):
        residual = self.projection(X) if self.projection else X
        out = self.conv1(X)
        if self.norm1:
            out = self.norm1(out)
        if self.act1:
            out = self.act1(out)
        out = self.conv2(out)
        if self.norm2:
            out = self.norm2(out)
        out = out + residual
        if self.act2:
            out = self.act2(out)
        return out


class SEBlock(nn.Module):
    def __init__(self, n_features: int, reduction: int = 16):
        super().__init__()
        assert n_features % reduction == 0

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(n_features, n_features // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(n_features // reduction, n_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, X: Tensor):
        weights = self.excitation(self.squeeze(X))
        return weights


class PCN(nn.Module):
    def __init__(self, n_features: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.se = SEBlock(n_features)

    def forward(self, X: Tensor, mask: Tensor):
        """
        Note that here in mask, 1 denotes valid/known pixels and 0 denotes invalid pixels (holes),
        which is contrary to the original paper.
        """
        _T = self._compute_T(X, mask)
        beta = self.se(X)
        context_feat = beta * _T * (1. - mask) + (1. - beta) * X * (1. - mask)
        preserved_feat = X * mask
        return context_feat + preserved_feat

    def _compute_T(self, X: Tensor, mask: Tensor):
        """ Note that `H` in the paper is actually 1 - mask here """
        X_p = X * (1. - mask)
        X_q = X * mask
        X_p_mean = self._compute_weighted_mean(X_p, 1. - mask)
        X_p_std = self._compute_weighted_std(X_p, 1. - mask)
        X_q_mean = self._compute_weighted_mean(X_q, mask)
        X_q_std = self._compute_weighted_std(X_q, mask)
        return ((X_p - X_p_mean) / X_p_std) * X_q_std + X_q_mean

    def _compute_weighted_mean(self, Y: Tensor, T: Tensor):
        return torch.sum(Y * T, dim=(2, 3), keepdim=True) / (torch.sum(T, dim=(2, 3), keepdim=True) + self.epsilon)

    def _compute_weighted_std(self, Y: Tensor, T: Tensor):
        _mean = self._compute_weighted_mean(Y, T)
        return torch.sqrt((torch.sum(torch.pow(Y * T - _mean, 2), dim=(2, 3), keepdim=True) /
                          (torch.sum(T, dim=(2, 3), keepdim=True) + self.epsilon)) + self.epsilon)


class PCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 1, dilation: int = 1, activation: str = None):
        super().__init__()

        if in_channels == out_channels and stride == 1:
            self.projection = None
        else:
            self.projection = nn.Conv2d(in_channels, out_channels, 1, stride=stride, dilation=1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.act1 = get_activation_layer(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation)
        self.pcn = PCN(out_channels)
        self.act2 = get_activation_layer(activation)

    def forward(self, X: Tensor, mask: Tensor):
        residual = self.projection(X) if self.projection else X
        out = self.conv1(X)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.pcn(out, F.interpolate(mask, out.shape[-2:], mode='nearest'))
        out = out + residual
        out = self.act2(out)
        return out
