import os
import tqdm
import argparse
from yacs.config import CfgNode as CN

import torch
from torchvision.utils import save_image
from torch.utils.data import Subset, DataLoader

import accelerate

from tools import build_dataset
from models.VCNet import MPN, RIN
from utils.logger import get_logger
from utils.misc import image_norm_to_float


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to pretrained model weights',
    )
    parser.add_argument(
        '--n_samples', type=int, default=100,
        help='Number of images to sample',
    )
    parser.add_argument(
        '--save_dir', type=str, required=True,
        help='Path to directory saving samples',
    )
    parser.add_argument(
        '--micro_batch', type=int, default=16,
        help='Batch size on each process.',
    )
    return parser


@torch.no_grad()
def sample():
    idx = 0
    for cor_img, gt_img, noise, mask in tqdm.tqdm(
            test_loader, desc='Sampling', disable=not accelerator.is_main_process,
    ):
        cor_img, gt_img, mask = cor_img.float(), gt_img.float(), mask.float()
        pred_mask, bottleneck = mpn(cor_img)
        rec_img = rin(cor_img, pred_mask, bottleneck.detach())

        cor_img, gt_img, noise, mask, pred_mask, rec_img = accelerator.gather_for_metrics(
            (cor_img, gt_img, noise, mask, pred_mask, rec_img),
        )
        cor_img = image_norm_to_float(cor_img)
        gt_img = image_norm_to_float(gt_img)
        noise = image_norm_to_float(noise)
        mask = mask.expand(gt_img.shape)
        pred_mask = pred_mask.expand(gt_img.shape)
        rec_img = image_norm_to_float(rec_img)

        if accelerator.is_main_process:
            for i in range(len(cor_img)):
                save_image(
                    tensor=[gt_img[i], noise[i], mask[i], cor_img[i], pred_mask[i], rec_img[i]],
                    fp=os.path.join(args.save_dir, f'{idx}.png'), nrow=6,
                )
                idx += 1


if __name__ == '__main__':
    args, unknown_args = get_parser().parse_known_args()
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.set_new_allowed(False)
    cfg.merge_from_list(unknown_args)
    cfg.freeze()

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()
    # INITIALIZE LOGGER
    logger = get_logger(
        use_tqdm_handler=True,
        is_main_process=accelerator.is_main_process,
    )
    # SET SEED
    accelerate.utils.set_seed(cfg.seed, device_specific=True)
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    test_set = build_dataset(cfg, is_train=False)
    test_set = Subset(test_set, torch.arange(min(args.n_samples, len(test_set))))
    test_loader = DataLoader(
        dataset=test_set,
        shuffle=False,
        drop_last=False,
        batch_size=args.micro_batch,
        pin_memory=cfg.dataloader.pin_memory,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    logger.info(f'Size of test set: {len(test_set)}')
    logger.info(f'Batch size per process: {args.micro_batch}')
    logger.info(f'Total batch size: {args.micro_batch * accelerator.num_processes}')

    # BUILD MODELS
    mpn = MPN(
        base_n_channels=cfg.model.mpn.base_n_channels,
        neck_n_channels=cfg.model.mpn.neck_n_channels,
    )
    rin = RIN(
        base_n_channels=cfg.model.rin.base_n_channels,
        neck_n_channels=cfg.model.rin.neck_n_channels,
    )
    # LOAD MODEL WEIGHTS
    ckpt_model = torch.load(args.model_path, map_location='cpu')
    mpn.load_state_dict(ckpt_model['mpn'])
    rin.load_state_dict(ckpt_model['rin'])
    logger.info(f'Successfully load mpn and rin from {args.model_path}')
    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    mpn, rin, test_loader = accelerator.prepare(mpn, rin, test_loader)  # type: ignore
    mpn.eval(); rin.eval()

    accelerator.wait_for_everyone()

    # START SAMPLING
    logger.info('Start sampling...')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    sample()
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')
