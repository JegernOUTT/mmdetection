from pathlib import Path
# model settings
model = dict(
    type='CenterNet',
    pretrained='torchvision://resnet18',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=None,
    bbox_head=dict(
        type='CenternetDetectionHead',
        require_upsampling=True,
        inplanes=(64, 128, 256, 512),
        planes=(256, 128, 64),
        base_down_ratio=32,
        hm_head_conv=128,
        hm_offset_heads_conv=128,
        wh_heads_conv=128,
        with_deformable=False,
        hm_head_conv_num=2,
        hm_offset_head_conv_num=2,
        wh_head_conv_num=2,
        num_classes=2,
        shortcut_kernel=3,
        norm_cfg=dict(type='BN'),
        shortcut_cfg=(1, 2, 3),
        num_stacks=1,  # It can be > 1 in backbones such as hourglass
        ellipse_gaussian=True,
        exp_wh=True,
        hm_weight=0.5,
        hm_offset_weight=1.,
        wh_weight=1.,
        max_objs=128))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    max_per_img=100)
# dataset settings
albu_train_transforms = [
    dict(type='HorizontalFlip'),
    dict(
        type='ShiftScaleRotate',
        shift_limit=[-0.2, 0.2],
        scale_limit=[-0.7, 0.7],
        rotate_limit=[-15, 15],
        border_mode=0,
        value=[128, 128, 128],
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.5],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=1.0)
        ],
        p=0.2),
    dict(type='JpegCompression', quality_lower=50, quality_upper=100, p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
        ],
        p=0.2),
    dict(
        type='CLAHE',
        p=0.3),
    dict(
        type='ToGray',
        p=0.2),
    dict(
        type='Cutout',
        num_holes=5,
        max_h_size=10,
        max_w_size=10,
        fill_value=[128, 128, 128])
]
width, height = 320, 128
albu_center_crop_pad = [
    dict(type='PadIfNeeded', min_height=max(width, height),
         min_width=max(width, height), border_mode=0, value=[128, 128, 128]),
    dict(type='CenterCrop', height=height, width=width),

]
dataset_type = 'DsslDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(width, height), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.01,
            filter_lost_elements=True),
        update_pad_shape=True,
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(
        type='Albu',
        transforms=albu_center_crop_pad,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.01,
            filter_lost_elements=True),
        update_pad_shape=True,
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(width, height),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Pad', size_divisor=32, pad_val=128),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file='./dssl_bags_loader.py',
        load_and_dump_config_name='load_and_dump_train_config',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='./dssl_bags_loader.py',
        load_and_dump_config_name='load_and_dump_test_config',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='./dssl_bags_loader.py',
        load_and_dump_config_name='load_and_dump_test_config',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='RAdam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=1.0 / 3,
    step=[60, 65])
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 70
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/centernet_bags'
load_from = None
resume_from = None
workflow = [('train', 1)]
# Need to import for inject hack registering
from detector_utils.pytorch.utils.mmdet_wandb_hook import WandbLoggerHook
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', project='Centernet bags detector',
             config_filename=Path.absolute(Path(__file__)))
    ])
