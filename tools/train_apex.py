from __future__ import division

import argparse
import copy
import logging
import os
from collections import OrderedDict

import torch
from apex import amp
from apex.parallel import DistributedDataParallel, convert_syncbn_model
from mmcv import Config
from mmcv.parallel import scatter_kwargs
from mmcv.runner import DistSamplerSeedHook, Runner, OptimizerHook

from mmdet import __version__
from mmcv.runner import init_dist
from mmdet.apis import (get_root_logger, set_random_seed)
from mmdet.apis.train import build_optimizer, batch_processor
from mmdet.core import CocoDistEvalmAPHook, wrap_fp16_model
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector


class ApexRunner(Runner):
    def __init__(self, model, batch_processor, amp,
                 optimizer=None,
                 work_dir=None,
                 log_level=logging.INFO,
                 logger=None):
        super().__init__(model, batch_processor, optimizer, work_dir, log_level, logger)
        self._amp = amp

    def save_checkpoint(self,
                        out_dir,
                        c='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None):
        if meta is None:
            meta = {}
        meta['amp'] = self._amp.state_dict()
        super().save_checkpoint(out_dir, c, save_optimizer, meta)

    def resume(self, checkpoint, resume_optimizer=True, map_location='default'):
        self._amp.load_state_dict(checkpoint['meta']['amp'])
        super().resume(checkpoint, resume_optimizer, map_location)


class ApexDistributedDataParallel(DistributedDataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=0)

    def forward(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, [torch.cuda.current_device()])
        return super().forward(*inputs[0], **kwargs[0])


class ApexOptimizerHook(OptimizerHook):
    def __init__(self, opt_level: str, amp, grad_clip=None):
        self.grad_clip = grad_clip
        self.opt_level = opt_level
        self.amp = amp

    def after_train_iter(self, runner):
        runner.model.zero_grad()
        runner.optimizer.zero_grad()
        with amp.scale_loss(runner.outputs['loss'], runner.optimizer) as scaled_loss:
            scaled_loss.backward()
        if self.grad_clip is not None:
            self.clip_grads(self.amp.master_params(runner.optimizer))
        runner.optimizer.step()


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    opt_level = 'O1'

    args = parse_args()

    cfg = Config.fromfile(args.config)
    torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('MMDetection Version: {}'.format(__version__))
    logger.info('Config: {}'.format(cfg.text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()
    model = convert_syncbn_model(model)

    dataset = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        dataset.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=dataset[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = dataset[0].CLASSES

    # prepare data loaders
    data_loaders = [
        build_dataloader(
            ds, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, dist=True)
        for ds in dataset
    ]
    # After amp.initialize, wrap the model with apex.parallel.DistributedDataParallel.
    optimizer = build_optimizer(model, cfg.optimizer)
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    model = ApexDistributedDataParallel(model, message_size=1e8, delay_allreduce=True)

    # build runner
    runner = ApexRunner(model, batch_processor, amp, optimizer, cfg.work_dir, cfg.log_level)
    optimizer_config = ApexOptimizerHook(opt_level=opt_level, amp=amp, **cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if args.validate:
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(CocoDistEvalmAPHook(val_dataset_cfg, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


if __name__ == '__main__':
    main()
