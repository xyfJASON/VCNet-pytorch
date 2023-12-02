from yacs.config import CfgNode as CN

import torch
from torch.utils.data import Dataset, ConcatDataset

from utils.data import get_dataset
from utils.mask_blind import DatasetWithMaskBlind


def wrap_dataset_with_mask(cfg: CN, dataset: Dataset, split: str = 'train', is_train: bool = True):
    noise_dataset = None
    if cfg.mask.noise_type == 'real':
        noise_dataset = ConcatDataset([
            get_dataset(
                name=c['name'],
                dataroot=c['dataroot'],
                img_size=c['img_size'],
                split=split,
            ) for c in cfg.mask.real_dataset
        ])
    kwargs = cfg.mask.copy()
    kwargs['real_dataset'] = noise_dataset
    dataset = DatasetWithMaskBlind(
        dataset=dataset,
        is_train=is_train,
        **kwargs,
    )
    return dataset


def build_dataset(cfg: CN, is_train: bool = True):
    if is_train:
        train_set = get_dataset(
            name=cfg.data.name,
            dataroot=cfg.data.dataroot,
            img_size=cfg.data.img_size,
            split='train',
        )
        valid_set = get_dataset(
            name=cfg.data.name,
            dataroot=cfg.data.dataroot,
            img_size=cfg.data.img_size,
            split='valid',
            subset_ids=torch.arange(5000),
        )
        if hasattr(cfg, 'mask'):
            train_set = wrap_dataset_with_mask(
                cfg=cfg,
                dataset=train_set,
                split='train',
                is_train=True,
            )
            valid_set = wrap_dataset_with_mask(
                cfg=cfg,
                dataset=valid_set,
                split='valid',
                is_train=False,
            )
        return train_set, valid_set

    else:
        test_set = get_dataset(
            name=cfg.data.name,
            dataroot=cfg.data.dataroot,
            img_size=cfg.data.img_size,
            split='test',
        )
        if hasattr(cfg, 'mask'):
            test_set = wrap_dataset_with_mask(
                cfg=cfg,
                dataset=test_set,
                split='test',
                is_train=False,
            )
        return test_set
