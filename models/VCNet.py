import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import spectral_norm

from models.modules import ResBlock, ConvNormAct, PCBlock, init_weights


class MPN(nn.Module):
    def __init__(self, base_n_channels: int = 64, neck_n_channels: int = 128):
        super().__init__()
        assert base_n_channels >= 4, "Base num channels should be at least 4"
        assert neck_n_channels >= 16, "Neck num channels should be at least 16"
        self.rb1 = ResBlock(3, base_n_channels, kernel_size=5, stride=2, padding=2, activation='elu')
        self.rb2 = ResBlock(base_n_channels, base_n_channels * 2, kernel_size=3, stride=2, padding=1, activation='elu')
        self.rb3 = ResBlock(base_n_channels * 2, base_n_channels * 2, kernel_size=3, stride=1, padding=2, dilation=2, activation='elu')
        self.rb4 = ResBlock(base_n_channels * 2, neck_n_channels, kernel_size=3, stride=1, padding=4, dilation=4, activation='elu')

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)

        self.rb5 = ResBlock(base_n_channels * 2, base_n_channels * 2, kernel_size=3, stride=1, padding=1, activation='elu')
        self.rb6 = ResBlock(base_n_channels * 2, base_n_channels, kernel_size=3, stride=1, padding=1, activation='elu')
        self.rb7 = ResBlock(base_n_channels, base_n_channels // 2, kernel_size=3, stride=1, padding=1, activation='elu')

        self.conv1 = ConvNormAct(base_n_channels // 2, base_n_channels // 4, kernel_size=3, stride=1, padding=1, activation='elu')
        self.conv2 = ConvNormAct(base_n_channels // 4, 1, kernel_size=3, stride=1, padding=1, activation='sigmoid')

        self.apply(init_weights(init_type='normal', gain=0.02))

    def forward(self, X: Tensor):
        out = self.rb1(X)
        out = self.rb2(out)
        out = self.rb3(out)
        bottleneck = self.rb4(out)

        out = self.rb5(bottleneck)
        out = self.upsample(out)
        out = self.rb6(out)
        out = self.upsample(out)
        out = self.rb7(out)

        out = self.conv1(out)
        out = self.conv2(out)

        return out, bottleneck


class RIN(nn.Module):
    def __init__(self, base_n_channels: int = 32, neck_n_channels: int = 128):
        super(RIN, self).__init__()
        assert base_n_channels >= 8, "Base num channels should be at least 8"
        assert neck_n_channels >= 32, "Neck num channels should be at least 32"
        self.pcb1 = PCBlock(3, base_n_channels, kernel_size=5, stride=1, padding=2, activation='elu')
        self.pcb2 = PCBlock(base_n_channels, base_n_channels * 2, kernel_size=3, stride=2, padding=1, activation='elu')
        self.pcb3 = PCBlock(base_n_channels * 2, base_n_channels * 2, kernel_size=3, stride=1, padding=1, activation='elu')
        self.pcb4 = PCBlock(base_n_channels * 2, base_n_channels * 4, kernel_size=3, stride=2, padding=1, activation='elu')
        self.pcb5 = PCBlock(base_n_channels * 4, base_n_channels * 4, kernel_size=3, stride=1, padding=1, activation='elu')
        self.pcb6 = PCBlock(base_n_channels * 4, base_n_channels * 4, kernel_size=3, stride=1, padding=2, dilation=2, activation='elu')
        self.pcb7 = PCBlock(base_n_channels * 4, base_n_channels * 4, kernel_size=3, stride=1, padding=2, dilation=2, activation='elu')
        self.pcb8 = PCBlock(base_n_channels * 4, base_n_channels * 4, kernel_size=3, stride=1, padding=4, dilation=4, activation='elu')
        self.pcb9 = PCBlock(base_n_channels * 4, base_n_channels * 4, kernel_size=3, stride=1, padding=4, dilation=4, activation='elu')
        self.pcb10 = PCBlock(base_n_channels * 4, base_n_channels * 4, kernel_size=3, stride=1, padding=1, activation='elu')

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)

        self.pcb11 = PCBlock(base_n_channels * 4 + neck_n_channels, base_n_channels * 2, kernel_size=3, stride=1, padding=1, activation='elu')
        self.pcb12 = PCBlock(base_n_channels * 2, base_n_channels * 2, kernel_size=3, stride=1, padding=1, activation='elu')
        self.pcb13 = PCBlock(base_n_channels * 2, base_n_channels, kernel_size=3, stride=1, padding=1, activation='elu')
        self.pcb14 = PCBlock(base_n_channels, base_n_channels, kernel_size=3, stride=1, padding=1, activation='elu')

        self.conv1 = nn.Conv2d(base_n_channels, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        self.apply(init_weights(init_type='normal', gain=0.02))

    def forward(self, X: Tensor, mask: Tensor, bottleneck: Tensor):
        out = self.pcb1(X, mask)
        out = self.pcb2(out, mask)
        out = self.pcb3(out, mask)
        out = self.pcb4(out, mask)
        out = self.pcb5(out, mask)
        out = self.pcb6(out, mask)
        out = self.pcb7(out, mask)
        out = self.pcb8(out, mask)
        out = self.pcb9(out, mask)
        out = self.pcb10(out, mask)

        out = torch.cat([out, bottleneck], dim=1)
        out = self.upsample(out)
        out = self.pcb11(out, mask)
        out = self.pcb12(out, mask)
        out = self.upsample(out)
        out = self.pcb13(out, mask)
        out = self.pcb14(out, mask)
        out = self.conv1(out)

        return self.tanh(out)


class Discriminator(nn.Module):
    def __init__(self, base_n_channels: int = 64):
        super().__init__()
        self.disc = nn.Sequential(
            spectral_norm(nn.Conv2d(3, base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_n_channels, 2 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(2 * base_n_channels, 2 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(2 * base_n_channels, 4 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(4 * base_n_channels, 8 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Conv2d(8 * base_n_channels, 1, 8),
            nn.Flatten(),
        )

        self.apply(init_weights(init_type='normal', gain=0.02))

    def forward(self, X: Tensor):
        X = self.disc(X)
        X = self.fc(X)
        return X


class PatchDiscriminator(nn.Module):
    def __init__(self, base_n_channels: int = 64):
        super().__init__()
        self.disc = nn.Sequential(
            spectral_norm(nn.Conv2d(3, base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_n_channels, 2 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(2 * base_n_channels, 2 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(2 * base_n_channels, 4 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(4 * base_n_channels, 8 * base_n_channels, 5, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
        )

        self.apply(init_weights(init_type='normal', gain=0.02))

    def forward(self, X: Tensor, mask: Tensor = None):
        batch_size = X.shape[0]
        X = self.disc(X)
        mask = self.downsample(mask)
        X = X * mask
        X = X.view(batch_size, -1)
        mask = mask.view(batch_size, -1)
        X = torch.sum(X, dim=-1, keepdim=True) / torch.sum(mask, dim=-1, keepdim=True)
        return X


def _test():
    mpn = MPN()
    X = torch.randn(10, 3, 256, 256)
    mask = torch.rand((10, 1, 256, 256))
    pred_mask, bottleneck = mpn(X)
    print(pred_mask.shape, bottleneck.shape)
    print(sum(p.numel() for p in mpn.parameters()))

    rin = RIN()
    output = rin(X, pred_mask, bottleneck)
    print(output.shape)
    print(sum(p.numel() for p in rin.parameters()))

    disc = Discriminator()
    d = disc(output)
    print(d.shape)
    print(sum(p.numel() for p in disc.parameters()))

    patch_disc = PatchDiscriminator()
    patch_d = patch_disc(output, mask)
    print(patch_d.shape)
    print(sum(p.numel() for p in patch_disc.parameters()))


if __name__ == '__main__':
    _test()
