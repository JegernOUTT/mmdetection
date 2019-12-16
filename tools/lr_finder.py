from __future__ import division

import argparse

import torch
from detector_utils.pytorch.utils import MMDataParallel
from mmcv import Config
from mmcv.runner import Runner

from mmdet import __version__
from mmdet.apis import (get_root_logger, set_random_seed)
from mmdet.apis.train import build_optimizer, batch_processor
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--max_lr', type=float, default=1., help='maximum lr for lr finder')
    parser.add_argument('--num_iter', type=int, default=200, help='lr finder iterations count')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--validate', action='store_true', help='whether to evaluate the checkpoint during training')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    logger = get_root_logger(cfg.log_level)
    logger.info('MMDetection Version: {}'.format(__version__))
    logger.info('Config: {}'.format(cfg.text))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    dataset = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            1,
            dist=False) for ds in dataset
    ]
    # put model on gpu
    model = MMDataParallel(model, device_ids=[0]).cuda()
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir, cfg.log_level)
    cfg.log_config['interval'] = 1
    lr_config = {'policy': 'lrfinder', 'end_lr': args.max_lr, 'num_iter': args.num_iter}
    runner.register_training_hooks(lr_config, cfg.optimizer_config, cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


if __name__ == '__main__':
    main()
