# model settings
from pathlib import Path
# Need to import for inject hack registering
from detector_utils.pytorch.utils import inject_all_hooks
inject_all_hooks()
model = dict(
    type='TTFNet',
    pretrained=None,
    backbone=dict(
        type='ConvnetLprVehicle',
        out_indices=(1, 2, 3, 4)),
    neck=None,
    bbox_head=dict(
        type='TTFHead',
        inplanes=(32, 64, 128, 256),
        head_conv=128,
        wh_conv=64,
        hm_head_conv_num=2,
        wh_head_conv_num=1,
        num_classes=6,
        wh_offset_base=16,
        wh_agnostic=True,
        wh_gaussian=True,
        shortcut_cfg=(1, 2, 3),
        norm_cfg=dict(type='BN'),
        alpha=0.54,
        hm_weight=1.,
        wh_weight=5.,
        max_objs=256,
        receptive_field_layer='rfb'))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    max_per_img=256)
# dataset settings
albu_train_transforms = [
    dict(type='HorizontalFlip'),
    dict(
        type='ShiftScaleRotate',
        shift_limit=[-0.05, 0.05],
        scale_limit=[-0.05, 0.05],
        rotate_limit=[-10, 10],
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
        max_h_size=5,
        max_w_size=5,
        fill_value=[128, 128, 128])
]
width, height = 192, 160
albu_center_crop_pad = [
    dict(type='PadIfNeeded', min_height=max(width, height),
         min_width=max(width, height), border_mode=0, value=[128, 128, 128]),
    dict(type='CenterCrop', height=height, width=width),
]
dataset_type = 'DsslDataset'
img_norm_cfg = dict(
    mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=True)
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
width, height = 192, 160
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(width, height),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file='./loader.py',
        load_and_dump_config_name='load_and_dump_train_config',
        composer_config_name='train_composer_config',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='./loader.py',
        load_and_dump_config_name='load_and_dump_test_config',
        composer_config_name='test_composer_config',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='./loader.py',
        load_and_dump_config_name='load_and_dump_test_config',
        composer_config_name='test_composer_config',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0004,
                 paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='torch',
    torch_scheduler='OneCycleLR',
    max_lr=0.015)
checkpoint_config = dict(interval=4)
bbox_head_hist_config = dict(
    model_type=['ConvModule', 'DeformConvPack'],
    sub_modules=['bbox_head'],
    save_every_n_steps=500)
# yapf:enable
# runtime settings
total_epochs = 200
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ttfnet18_1x'
load_from = "./work_dirs/ttfnet18_1x/coco_pretrained.pth"
# load_from = None
resume_from = None
workflow = [('train', 1)]
log_config = dict(
    interval=30,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', project='Debug test', config_filename=Path.absolute(Path(__file__)))
    ])
