from pathlib import Path
from detector_utils.pytorch.utils import inject_all_hooks
inject_all_hooks()
# model settings
model = dict(
    type='TTFNet',
    pretrained='torchvision://resnet34',
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_eval=False,
        style='pytorch'),
    neck=None,
    bbox_head=dict(
        type='TTFHead',
        inplanes=(64, 128, 256, 512),
        head_conv=256,
        wh_conv=256,
        hm_head_conv_num=2,
        wh_head_conv_num=2,
        num_classes=2,
        wh_offset_base=16,
        wh_agnostic=True,
        wh_gaussian=True,
        shortcut_cfg=(1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        alpha=0.54,
        beta=0.54,
        hm_weight=1.,
        wh_weight=5.,
        max_objs=1024,
        receptive_field_layer='rfb'))
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    max_per_img=1024)
# dataset settings
albu_train_transforms = [
    dict(type='HorizontalFlip'),
    dict(type='ShiftScaleRotate',
         shift_limit=[-0.2, 0.2],
         scale_limit=[-0.7, 0.7],
         rotate_limit=[-15, 15],
         border_mode=0,
         value=[128, 128, 128],
         p=0.5),
    dict(type='RandomBrightnessContrast',
         brightness_limit=[0.1, 0.3],
         contrast_limit=[0.1, 0.5],
         p=0.2),
    dict(type='OneOf',
         transforms=[
             dict(type='RGBShift',
                  r_shift_limit=20,
                  g_shift_limit=20,
                  b_shift_limit=20,
                  p=1.0),
             dict(type='HueSaturationValue',
                  hue_shift_limit=10,
                  sat_shift_limit=15,
                  val_shift_limit=10,
                  p=1.0)
         ], p=0.2),
    dict(type='JpegCompression', quality_lower=50, quality_upper=100, p=0.2),
    dict(type='OneOf',
         transforms=[
             dict(type='Blur', blur_limit=3, p=1.0),
             dict(type='MedianBlur', blur_limit=3, p=1.0),
         ], p=0.2),
    dict(type='CLAHE', p=0.3),
    dict(type='OpticalDistortion', distort_limit=[0., 0.5], shift_limit=[-0.3, 0.3],
         p=1, border_mode=0, value=[128, 128, 128]),
]
width, height = 1024, 768
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
    dict(type='Albu',
         transforms=albu_train_transforms,
         bbox_params=dict(
             type='BboxParams',
             format='pascal_voc',
             label_fields=['gt_labels'],
             min_visibility=0.2,
             filter_lost_elements=True),
         update_pad_shape=True,
         keymap={
             'img': 'image',
             'gt_bboxes': 'bboxes'
         }),
    dict(type='Albu',
         transforms=albu_center_crop_pad,
         bbox_params=dict(
             type='BboxParams',
             format='pascal_voc',
             label_fields=['gt_labels'],
             min_visibility=0.2,
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
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        ann_file='./shelves/loader.py',
        load_and_dump_config_name='load_and_dump_train_config',
        composer_config_name='composer_train_config',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='./shelves/loader.py',
        load_and_dump_config_name='load_and_dump_test_config',
        composer_config_name='composer_test_config',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='./shelves/loader.py',
        load_and_dump_config_name='load_and_dump_test_config',
        composer_config_name='composer_test_config',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0004,
                 paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='torch',
    torch_scheduler='OneCycleLR',
    max_lr=0.007)
checkpoint_config = dict(interval=2)
bbox_head_hist_config = dict(
    model_type=['ConvModule', 'DeformConvPack'],
    sub_modules=['bbox_head'],
    save_every_n_steps=500)
# yapf:enable
# runtime settings
total_epochs = 50
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ttfnet_shelves'
load_from = './shelves/pretrained.pth'
# load_from = None
resume_from = None
workflow = [('train', 1)]
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', project='Shelves', config_filename=Path.absolute(Path(__file__)))
    ])
