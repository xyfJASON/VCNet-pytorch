import os
import tqdm
import argparse
from contextlib import nullcontext
from yacs.config import CfgNode as CN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import accelerate

import metrics
from tools import build_dataset
from models.VCNet import MPN, RIN
from models.vgg import VGG19FeatureExtractor
from loss import AdaptiveBCELoss, SemanticConsistencyLoss, IDMRFLoss
from utils.data import get_data_generator
from utils.logger import StatusTracker, get_logger
from utils.misc import get_time_str, create_exp_dir, check_freq, find_resume_checkpoint, image_norm_to_float


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    parser.add_argument(
        '-e', '--exp_dir', type=str,
        help='Path to the experiment directory. Default to be ./runs/exp-{current time}/',
    )
    parser.add_argument(
        '-ni', '--no_interaction', action='store_true', default=False,
        help='Do not interact with the user (always choose yes when interacting)',
    )
    return parser


def train(args, cfg):
    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()
    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir,
            cfg_dump=cfg.dump(sort_keys=False),
            exist_ok=cfg.train.resume is not None,
            time_str=args.time_str,
            no_interaction=args.no_interaction,
        )
    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True,
        is_main_process=accelerator.is_main_process,
    )
    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger,
        exp_dir=exp_dir,
        print_freq=cfg.train.print_freq,
        is_main_process=accelerator.is_main_process,
    )
    # SET SEED
    accelerate.utils.set_seed(cfg.seed, device_specific=True)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    assert cfg.train.batch_size % accelerator.num_processes == 0
    batch_size_per_process = cfg.train.batch_size // accelerator.num_processes
    micro_batch = cfg.dataloader.micro_batch or batch_size_per_process
    train_set, valid_set = build_dataset(cfg)
    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size_per_process,
        pin_memory=cfg.dataloader.pin_memory,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    valid_loader = DataLoader(
        dataset=valid_set,
        shuffle=False,
        drop_last=False,
        batch_size=micro_batch,
        pin_memory=cfg.dataloader.pin_memory,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Size of validation set: {len(valid_set)}')
    logger.info(f'Batch size per process: {batch_size_per_process}')
    logger.info(f'Total batch size: {cfg.train.batch_size}')

    # BUILD MODELS AND OPTIMIZERS
    mpn = MPN(
        base_n_channels=cfg.model.mpn.base_n_channels,
        neck_n_channels=cfg.model.mpn.neck_n_channels,
    )
    rin = RIN(
        base_n_channels=cfg.model.rin.base_n_channels,
        neck_n_channels=cfg.model.rin.neck_n_channels,
    )
    vgg = VGG19FeatureExtractor().to(device)
    optimizer_mpn = optim.Adam(mpn.parameters(), lr=cfg.train.optim_mpn.lr, betas=cfg.train.optim_mpn.betas)
    optimizer_rin = optim.Adam(rin.parameters(), lr=cfg.train.optim_rin.lr, betas=cfg.train.optim_rin.betas)
    step = 0

    def load_ckpt(ckpt_path: str):
        nonlocal step
        # load models
        ckpt_model = torch.load(os.path.join(ckpt_path, 'model.pt'), map_location='cpu')
        mpn.load_state_dict(ckpt_model['mpn'])
        rin.load_state_dict(ckpt_model['rin'])
        logger.info(f'Successfully load mpn and rin from {ckpt_path}')
        # load optimizers
        ckpt_optimizer = torch.load(os.path.join(ckpt_path, 'optimizer.pt'), map_location='cpu')
        optimizer_mpn.load_state_dict(ckpt_optimizer['optimizer_mpn'])
        optimizer_rin.load_state_dict(ckpt_optimizer['optimizer_rin'])
        logger.info(f'Successfully load optimizers from {ckpt_path}')
        # load meta information
        ckpt_meta = torch.load(os.path.join(ckpt_path, 'meta.pt'), map_location='cpu')
        step = ckpt_meta['step'] + 1

    @accelerator.on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save models
        unwrapped_mpn = accelerator.unwrap_model(mpn)
        unwrapped_rin = accelerator.unwrap_model(rin)
        model_state_dicts = dict(
            mpn=unwrapped_mpn.state_dict(),
            rin=unwrapped_rin.state_dict(),
        )
        accelerator.save(model_state_dicts, os.path.join(save_path, 'model.pt'))
        # save optimizers
        optimizer_state_dicts = dict(
            optimizer_mpn=optimizer_mpn.state_dict(),
            optimizer_rin=optimizer_rin.state_dict(),
        )
        accelerator.save(optimizer_state_dicts, os.path.join(save_path, 'optimizer.pt'))
        # save meta information
        meta_state_dicts = dict(step=step)
        accelerator.save(meta_state_dicts, os.path.join(save_path, 'meta.pt'))

    # RESUME TRAINING
    if cfg.train.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, cfg.train.resume)
        logger.info(f'Resume from {resume_path}')
        load_ckpt(resume_path)
        logger.info(f'Restart training at step {step}')

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    mpn, rin, optimizer_mpn, optimizer_rin, train_loader, valid_loader = accelerator.prepare(
        mpn, rin, optimizer_mpn, optimizer_rin, train_loader, valid_loader,  # type: ignore
    )

    # DEFINE LOSSES
    adaptive_bce = AdaptiveBCELoss()
    l1 = nn.L1Loss()
    semantic = SemanticConsistencyLoss(feature_extractor=vgg)
    idmrf = IDMRFLoss(feature_extractor=vgg)

    # EVALUATION METRICS
    metric_bce = metrics.BinarySegmBCE(reduction='none').to(device)
    metric_psnr = metrics.PSNR(reduction='none', data_range=1.).to(device)
    metric_ssim = metrics.SSIM(size_average=False, data_range=1.).to(device)
    metric_lpips = metrics.LPIPS(reduction='none', normalize=True).to(device)

    accelerator.wait_for_everyone()

    def run_step(_batch):
        optimizer_mpn.zero_grad()
        optimizer_rin.zero_grad()
        batch_cor_img, batch_gt_img, batch_noise, batch_mask = _batch
        batch_size = batch_cor_img.shape[0]
        loss_meter = metrics.KeyValueAverageMeter(
            keys=['loss_mpn', 'loss_rec', 'loss_semantic', 'loss_idmrf'],
        )
        for i in range(0, batch_size, micro_batch):
            cor_img = batch_cor_img[i:i+micro_batch].float()
            gt_img = batch_gt_img[i:i+micro_batch].float()
            mask = batch_mask[i:i+micro_batch].float()
            loss_scale = cor_img.shape[0] / batch_size
            no_sync = (i + micro_batch) < batch_size
            cm1 = accelerator.no_sync(mpn) if no_sync else nullcontext()
            cm2 = accelerator.no_sync(rin) if no_sync else nullcontext()
            with cm1, cm2:
                # train mpn
                pred_mask, bottleneck = mpn(cor_img)
                loss_mpn = adaptive_bce(pred_mask, mask)
                accelerator.backward(loss_mpn * loss_scale)
                # train rin: use ground-truth mask in separate training
                rec_img = rin(cor_img, mask, bottleneck.detach())
                loss_rec = l1(rec_img, gt_img)
                loss_semantic = semantic(rec_img, gt_img)
                loss_idmrf = idmrf(rec_img, gt_img)
                loss_rin = (cfg.train.coef_rec * loss_rec +
                            cfg.train.coef_semantic * loss_semantic +
                            cfg.train.coef_idmrf * loss_idmrf)
                accelerator.backward(loss_rin * loss_scale)
            loss_meter.update(dict(
                loss_mpn=loss_mpn.detach(),
                loss_rec=loss_rec.detach(),
                loss_semantic=loss_semantic.detach(),
                loss_idmrf=loss_idmrf.detach(),
            ), cor_img.shape[0])
        optimizer_mpn.step()
        optimizer_rin.step()
        return dict(
            **loss_meter.avg,
            lr_mpn=optimizer_mpn.param_groups[0]['lr'],
            lr_rin=optimizer_rin.param_groups[0]['lr'],
        )

    @torch.no_grad()
    def evaluate(dataloader):
        metric_meter = metrics.KeyValueAverageMeter(keys=['bce', 'psnr', 'ssim', 'lpips'])
        for cor_img, gt_img, noise, mask in tqdm.tqdm(
                dataloader, desc='Evaluating', leave=False,
                disable=not accelerator.is_main_process,
        ):
            cor_img, gt_img, mask = cor_img.float(), gt_img.float(), mask.float()
            pred_mask, bottleneck = mpn(cor_img)
            # use ground-truth mask in separate training
            rec_img = rin(cor_img, mask, bottleneck.detach())

            rec_img = image_norm_to_float(rec_img)
            gt_img = image_norm_to_float(gt_img)
            bce = metric_bce(pred_mask, mask)
            psnr = metric_psnr(rec_img, gt_img)
            ssim = metric_ssim(rec_img, gt_img)
            lpips = metric_lpips(rec_img, gt_img)
            bce, psnr, ssim, lpips = accelerator.gather_for_metrics((bce, psnr, ssim, lpips))
            metric_meter.update(dict(
                bce=bce.mean(),
                psnr=psnr.mean(),
                ssim=ssim.mean(),
                lpips=lpips.mean(),
            ), bce.shape[0])
        return metric_meter.avg

    @accelerator.on_main_process
    @torch.no_grad()
    def sample(savepath: str):
        unwrapped_mpn = accelerator.unwrap_model(mpn)
        unwrapped_rin = accelerator.unwrap_model(rin)

        cor_img = torch.stack([valid_set[i][0] for i in range(16)], dim=0).float().to(device)
        gt_img = torch.stack([valid_set[i][1] for i in range(16)], dim=0).float().to(device)
        mask = torch.stack([valid_set[i][3] for i in range(16)], dim=0).float().to(device)

        pred_mask, bottleneck = unwrapped_mpn(cor_img)
        # use ground-truth mask in separate training
        rec_img = unwrapped_rin(cor_img, mask, bottleneck.detach())

        show = []
        img_shape = (3, cfg.data.img_size, cfg.data.img_size)
        for i in tqdm.tqdm(range(16), desc='Sampling', leave=False,
                           disable=not accelerator.is_main_process):
            show.extend([
                image_norm_to_float(gt_img[i]).cpu(),
                mask[i].expand(*img_shape).cpu(),
                image_norm_to_float(cor_img[i]).cpu(),
                pred_mask[i].expand(*img_shape).cpu(),
                image_norm_to_float(rec_img[i]).cpu(),
            ])
        save_image(show, savepath, nrow=10)

    # START TRAINING
    logger.info('Start training...')
    train_data_generator = get_data_generator(
        dataloader=train_loader,
        is_main_process=accelerator.is_main_process,
        with_tqdm=True,
    )
    while step < cfg.train.n_steps:
        # get a batch of data
        batch = next(train_data_generator)
        # run a step
        mpn.train(); rin.train()
        train_status = run_step(batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()

        mpn.eval(); rin.eval()
        # evaluate
        if check_freq(cfg.train.eval_freq, step):
            eval_status = evaluate(valid_loader)
            status_tracker.track_status('Eval', eval_status, step)
            accelerator.wait_for_everyone()
        # save checkpoint
        if check_freq(cfg.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>6d}'))
            accelerator.wait_for_everyone()
        # sample from current model
        if check_freq(cfg.train.sample_freq, step):
            sample(os.path.join(exp_dir, 'samples', f'step{step:0>6d}.png'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(cfg.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>6d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    logger.info('End of training')


def main():
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.set_new_allowed(False)
    cfg.merge_from_list(unknown_args)
    cfg.freeze()

    train(args, cfg)


if __name__ == '__main__':
    main()
