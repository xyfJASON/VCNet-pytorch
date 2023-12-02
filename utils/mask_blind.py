import math
import numpy as np
import scipy.stats as st
from typing import Tuple, Any, List, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T

from utils.mask import MaskGenerator


class DatasetWithMaskBlind(Dataset):
    def __init__(
            self,
            dataset: Dataset,

            mask_type: Union[str, List[str]] = 'center',
            dir_path: str = None,
            dir_invert_color: bool = False,
            center_length_ratio: Tuple[float, float] = (0.25, 0.25),
            rect_num: Tuple[int, int] = (1, 4),
            rect_length_ratio: Tuple[float, float] = (0.2, 0.8),
            brush_num: Tuple[int, int] = (1, 9),
            brush_n_vertex: Tuple[int, int] = (4, 18),
            brush_mean_angle: float = 2 * math.pi / 5,
            brush_angle_range: float = 2 * math.pi / 15,
            brush_width_ratio: Tuple[float, float] = (0.02, 0.1),

            noise_type: str = 'constant',
            constant_rgb: Tuple[float, float, float] = (0, 0, 0),
            real_dataset: Any = None,

            smooth_type: str = 'iterative_gaussian',
            smooth_kernel_size: int = 15,
            smooth_sigma: float = 1./40,
            smooth_iters: int = 4,

            is_train: bool = False,
    ):
        self.dataset = dataset
        self.mask_generator = MaskGenerator(
            mask_type=mask_type,
            dir_path=dir_path,
            dir_invert_color=dir_invert_color,
            center_length_ratio=center_length_ratio,
            rect_num=rect_num,
            rect_length_ratio=rect_length_ratio,
            brush_num=brush_num,
            brush_n_vertex=brush_n_vertex,
            brush_mean_angle=brush_mean_angle,
            brush_angle_range=brush_angle_range,
            brush_width_ratio=brush_width_ratio,
            is_train=is_train,
        )
        self.noise_generator = NoiseGenerator(
            noise_type=noise_type,
            constant_rgb=constant_rgb,
            real_dataset=real_dataset,
            is_train=is_train,
        )
        self.smoother = MaskSmoother(
            smooth_type=smooth_type,
            kernel_size=smooth_kernel_size,
            sigma=smooth_sigma,
            iters=smooth_iters,
        )

    def __len__(self):
        return len(self.dataset)  # type: ignore

    def __getitem__(self, item):
        image = self.dataset[item]
        image = image[0] if isinstance(image, (tuple, list)) else image
        C, H, W = image.shape
        mask = self.mask_generator.sample(int(H), int(W), item)
        noise = self.noise_generator.sample(int(H), int(W), item)
        smooth_mask = self.smoother.apply(mask.float())
        corrupted_image = smooth_mask * image + (1 - smooth_mask) * noise
        return corrupted_image, image, noise, mask


class NoiseGenerator:
    def __init__(
            self,
            noise_type: str = 'constant',
            constant_rgb: Tuple[float, float, float] = (0, 0, 0),
            real_dataset: Dataset = None,
            is_train: bool = False,
    ):
        """ Generates noise content to fill in the mask region in blind image inpainting setting.

        Args:
            noise_type: Type of noise.
                Options:
                 - 'constant': specified constant RGB value.
                 - 'random_noise': random gaussian noise.
                 - 'random_single_color': random RGB color.
                 - 'real': real images from a dataset, e.g. ImageNet.

            constant_rgb: Valid only when `noise_type` is 'constant'.
                The constant RGB value.

            real_dataset: Valid only when `noise_type` is 'real'.
                The dataset containing real images, e.g. ImageNet.

            is_train: Whether the noise is generated for training set or not.
                If False, the generation process will be seeded on the index `item` in `sample()`.

        """
        self.noise_type = noise_type
        self.constant_rgb = constant_rgb
        self.real_dataset = real_dataset
        self.is_train = is_train

    def sample(self, H: int, W: int, item: int = None):
        if isinstance(item, torch.Tensor):
            item = item.item()
        if self.is_train is False and item is not None:
            rndgn = torch.Generator()
            rndgn.manual_seed(item + 3407)
        else:
            rndgn = torch.default_generator

        if self.noise_type == 'constant':
            noise = torch.tensor(self.constant_rgb)
            noise = noise[:, None, None].expand((3, H, W))
        elif self.noise_type == 'random_noise':
            noise = torch.randn((3, H, W), generator=rndgn)
        elif self.noise_type == 'random_single_color':
            noise = torch.rand((3, ), generator=rndgn) * 2 - 1
            noise = noise[:, None, None].expand((3, H, W))
        elif self.noise_type == 'real':
            idx = torch.randint(0, len(self.real_dataset), (1, ), generator=rndgn).item()  # type: ignore
            noise = self.real_dataset[idx]
            if isinstance(noise, (tuple, list)):
                noise = noise[0]
        else:
            raise ValueError(f'noise type {self.noise_type} is not supported')
        return noise


def gauss_kernel(size: int = 21, sigma: float = 3):
    """ Based on https://github.com/shepnerd/blindinpainting_vcnet/blob/master/tensorflow_version/net/ops.py#L228 """
    interval = (2 * sigma + 1.0) / size
    x = np.linspace(-sigma-interval/2, sigma+interval/2, size+1)
    ker1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(ker1d, ker1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, size, size))
    return out_filter


class MaskSmoother:
    def __init__(
            self,
            smooth_type: str = 'iterative_gaussian',
            kernel_size: int = 15,
            sigma: Union[float, Tuple[float, float]] = 1./40,
            iters: int = 4,
    ):
        """ Smooth the binary mask on the boundary.

        Based on
         - https://github.com/shepnerd/blindinpainting_vcnet/blob/master/tensorflow_version/net/ops.py#L245
         - https://github.com/birdortyedi/vcnet-blind-image-inpainting/blob/main/utils/mask_utils.py#L103

        Args:
            smooth_type: Type of smoothing method.
                Options:
                 - 'iterative_gaussian': iteratively apply gaussian smoothing like VCNet, see links above.
                 - 'gaussian': directly apply gaussian smoothing once on the boundary, using torchvision.
            kernel_size: Size of gaussian kernel.
            sigma: Standard deviation of gaussian kernel.
                When `smooth_type` is 'iterative_gaussian', 'sigma' can only be a float.
                When `smooth_type` is 'gaussian', `sigma` can be a float or a tuple of floats.
                If float, sigma is fixed. If it is a tuple (min, max), sigma is chosen uniformly at random to lie
                in the given range.
            iters: Number of iterations. Valid only when `smooth_type` is 'iterative_gaussian'.

        """
        super().__init__()
        self.smooth_type = smooth_type
        self.kernel_size = kernel_size
        self.iters = iters
        if smooth_type == 'iterative_gaussian':
            assert isinstance(sigma, float)
            self.kernel = torch.from_numpy(gauss_kernel(kernel_size, sigma))
        elif smooth_type == 'gaussian':
            self.gaussian_blur = T.GaussianBlur(kernel_size, sigma)
        elif smooth_type is None:
            pass
        else:
            raise ValueError(f'smooth type {smooth_type} is not supported')

    def apply(self, mask: torch.Tensor):
        """ Apply gaussian soothing on mask.

        Note that 1 denotes valid pixels and 0 denotes invalid pixels.

        """
        if self.smooth_type == 'iterative_gaussian':
            return self.apply_iterative_gaussian(mask.float())
        elif self.smooth_type == 'gaussian':
            return self.apply_gaussian(mask.float())
        else:
            return mask.float()

    def apply_iterative_gaussian(self, mask: torch.Tensor):
        """ Note that 1 denotes valid pixels and 0 denotes invalid pixels """
        init = 1. - mask
        for i in range(self.iters):
            init = F.pad(init, pad=(self.kernel_size // 2, ) * 4, mode='replicate')
            mask_priority = F.conv2d(init, self.kernel, stride=1)
            mask_priority = mask_priority * mask
            init = mask_priority + (1. - mask)
        return 1. - init

    def apply_gaussian(self, mask: torch.Tensor):
        """ Note that 1 denotes valid pixels and 0 denotes invalid pixels """
        mask = self.gaussian_blur(mask)
        return mask


def _test():
    from torchvision.utils import make_grid

    mask = MaskGenerator('brush').sample(256, 256)

    smoother_iter = MaskSmoother(smooth_type='iterative_gaussian', kernel_size=15, sigma=1./40, iters=4)
    smooth_mask_iter = smoother_iter.apply(mask)

    smoother_gauss = MaskSmoother(smooth_type='gaussian', kernel_size=13, sigma=3)
    smooth_mask_gauss = smoother_gauss.apply(mask)

    from PIL import ImageFilter
    smooth_mask_pil = T.ToPILImage()(mask.float())
    smooth_mask_pil = smooth_mask_pil.filter(ImageFilter.GaussianBlur(radius=3))
    smooth_mask_pil = T.ToTensor()(smooth_mask_pil)

    img = make_grid([mask.float(), smooth_mask_iter, smooth_mask_gauss, smooth_mask_pil], nrow=4)
    T.ToPILImage()(img).show()


if __name__ == '__main__':
    _test()
