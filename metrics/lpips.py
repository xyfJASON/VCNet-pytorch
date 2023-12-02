import os
import lpips as lpips_package
from urllib.parse import urlparse

import torch.hub as hub


def download_weights(net: str = 'alex', version: str = 'v0.1'):
    hub_dir = hub.get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')
    repo = 'https://github.com/richzhang/PerceptualSimilarity'
    url = f'{repo}/raw/master/lpips/weights/{version}/{net}.pth'
    filename = os.path.basename(urlparse(url).path)
    weights = hub.load_state_dict_from_url(url)
    return os.path.join(model_dir, filename), weights


class LPIPS(lpips_package.LPIPS):
    """
    Inherited from the original implementation with the following modifications:
     1. Automatically download the model weights to pytorch default cache directory.
     2. Move the argument `normalize` from forward() to __init__().
     3. Set `verbose` to False by default.
     4. Set `spatial` to False all the time.
     5. Set `retPerLayer` to False all the time.
     6. Add `reduction` argument to reduce the returned tensor.
    """
    def __init__(
            self,
            normalize=True,
            reduction='mean',
            pretrained=True,
            net='alex',
            version='0.1',
            lpips=True,
            pnet_rand=False,
            pnet_tune=False,
            use_dropout=True,
            model_path=None,
            eval_mode=True,
            verbose=False,
    ):
        if model_path is None:
            model_path, _ = download_weights(net, version)
        super().__init__(
            pretrained=pretrained,
            net=net,
            version=version,
            lpips=lpips,
            spatial=False,
            pnet_rand=pnet_rand,
            pnet_tune=pnet_tune,
            use_dropout=use_dropout,
            model_path=model_path,
            eval_mode=eval_mode,
            verbose=verbose,
        )
        self.normalize = normalize
        self.reduction = reduction

    def forward(self, in0, in1, retPerLayer=False, normalize=False):
        # note that retPerLayer and normalize is useless here
        result = super().forward(in0, in1, False, self.normalize)
        assert result.shape[1:] == (1, 1, 1)
        result = result.flatten()
        if self.reduction == 'mean':
            return result.mean()
        elif self.reduction == 'sum':
            return result.sum()
        elif self.reduction == 'none':
            return result
        else:
            raise ValueError(f'Invalid reduction: {self.reduction}')
