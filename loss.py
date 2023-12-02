from typing import Sequence
from functools import reduce

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch.autograd as autograd


class AdaptiveBCELoss(nn.Module):
    def __init__(self, epsilon=1e-2, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, out, target):
        out = out.view(out.shape[0], -1)
        target = target.view(target.shape[0], -1)
        out = out.clamp(self.epsilon, 1 - self.epsilon)
        weight = torch.sum(target == 0, dim=1, keepdim=True) / target.shape[1]
        loss = weight * (target * torch.log(out)) + (1 - weight) * ((1 - target) * torch.log(1 - out))
        loss = torch.neg(loss)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f'invalid reduction method: {self.reduction}')


class SemanticConsistencyLoss(nn.Module):
    def __init__(self,
                 feature_extractor: nn.Module,
                 use_feature: Sequence[str] = ('relu3_2', ),
                 weights: Sequence[float] = (1.0, )):
        super().__init__()
        self.use_feature = use_feature
        self.weights = weights
        self.feature_extractor = feature_extractor
        self.L1 = nn.L1Loss()

    def forward(self, fake_img: Tensor, real_img: Tensor):
        fake_feature = self.feature_extractor(fake_img)
        real_feature = self.feature_extractor(real_img)
        loss = sum([w * self.L1(fake_feature[f], real_feature[f])
                    for f, w in zip(self.use_feature, self.weights)])
        return loss


class AdversarialLoss(nn.Module):
    def __init__(self, D: nn.Module, lambda_gp: float = 10.):
        super().__init__()
        self.D = D
        self.lambda_gp = lambda_gp

    def gradient_penalty(self, real_img: Tensor, fake_img: Tensor, mask: Tensor = None):
        alpha = torch.rand(1, device=real_img.device)
        interX = alpha * real_img + (1 - alpha) * fake_img
        interX.requires_grad_()
        d_interX = self.D(interX) if mask is None else self.D(interX, mask)
        gradients = autograd.grad(outputs=d_interX, inputs=interX,
                                  grad_outputs=torch.ones_like(d_interX),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.flatten(start_dim=1) + 1e-16
        return torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

    def forward_D(self, real_img: Tensor, fake_img: Tensor, mask: Tensor = None):
        """ min E[D(G(z))] - E[D(x)] + lambda * gp """
        d_fake = self.D(fake_img) if mask is None else self.D(fake_img, mask)
        d_real = self.D(real_img) if mask is None else self.D(real_img, mask)
        lossD = torch.mean(d_fake) - torch.mean(d_real)
        lossD = lossD + self.lambda_gp * self.gradient_penalty(real_img, fake_img, mask)
        return lossD

    def forward_G(self, fake_img: Tensor, mask: Tensor = None):
        """ max E[D(G(z))] <=> min E[-D(G(z))] """
        d_fake = self.D(fake_img) if mask is None else self.D(fake_img, mask)
        lossG = -torch.mean(d_fake)
        return lossG


class IDMRFLoss(nn.Module):
    """ Copied and modified from https://github.com/shepnerd/inpainting_gmcnn/blob/master/pytorch/model/loss.py """
    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.featlayer = feature_extractor
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    @staticmethod
    def sum_normalize(featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    @staticmethod
    def patch_extraction(featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = patches_OIHW.size()
        patches_OIHW = patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return patches_OIHW

    @staticmethod
    def compute_relative_distances(cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    # Note: original implementation (commented below) may get inf in torch.exp().
    #       using softmax is safer.
    # def exp_norm_relative_dist(self, relative_dist):
    #     scaled_dist = relative_dist
    #     dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
    #     cs_NCHW = self.sum_normalize(dist_before_norm)
    #     return cs_NCHW

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        cs_NCHW = F.softmax((self.bias - scaled_dist) / self.nn_stretch_sigma, dim=1)
        return cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i+1, :, :, :]
            gen_feat_i = gen_normalized[i:i+1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2

        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf + 1e-5)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)

        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_style_layers]
        style_loss = reduce(lambda x, y: x+y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_content_layers]
        content_loss = reduce(lambda x, y: x+y, content_loss_list) * self.lambda_content

        return style_loss + content_loss
