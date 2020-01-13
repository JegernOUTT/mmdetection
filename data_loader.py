#!/usr/bin/python3

from pathlib import Path

from detector_utils import *

__all__ = ['train_load_config', 'test_load_config', 'train_composer_config', 'test_composer_config']

categories = {0: 'car', 1: 'motorcycle', 2: 'bus', 3: 'truck', 4: 'van'}
_base_path = Path('/mnt/nfs/Data/lpr/lpr_5/')
_base_output_path = Path('.')
plain_data_loader_info = PlainImagesLoadInformation(dataset_path=_base_path / 'new_lpr_data' / 'plates_2_lpr')
train_load_config = {
    'categories': categories,
    'data_loaders': [
        SuperviselyLoadInformation(dataset_path=_base_path / 'lpr5_supervisely' / 'part_1'),
        SuperviselyLoadInformation(dataset_path=_base_path / 'lpr5_supervisely' / 'part_2'),
        LegacyPickleLoadInformation(
            data_paths=[
                DataPathInformation(path=_base_path / 'lpr_cam'),
                DataPathInformation(path=_base_path / 'platesmania')
            ])
    ],
    'dump': DumpConfig(
        clone_images=True,
        annotations_dump_filename=Path(_base_output_path / f'train/annotations/annotations'),
        images_clone_path=Path(_base_output_path / f'train/images'))
}

test_load_config = {
    'categories': categories,
    'data_loader': LegacyPickleLoadInformation(data_paths=[DataPathInformation(path=_base_path / 'lpr5_test')]),
    'dump': DumpConfig(
        clone_images=True,
        annotations_dump_filename=Path(f'test/annotations/annotations'),
        images_clone_path=Path(f'test/images'))
}

train_composer_config = {
    'filters': [
        {'type': 'ImageValidityFilter'},
        {'type': 'BboxAbsoluteSizeFilter',
         'min_size': Size2DF(width=0.05, height=0.05),
         'max_size': Size2DF(width=1., height=1.)},
        {'type': 'ImageSizeFilter',
         'min_size': Size2D(width=32, height=32),
         'max_size': Size2D(width=10000, height=10000)},
        {'type': 'IgnoreMaskImagesRendererFilter', 'render_if_exists': False}
    ],

    'sampler': {'type': 'RandomSampler'}
}

test_composer_config = {
    'filters': [{'type': 'ImageValidityFilter'}],
    'sampler': {'type': 'SimpleSampler'}
}

if __name__ == '__main__':
    # plain_images = load(categories=train_load_config['categories'], load_information=plain_data_loader_info)
    # for img_ann in plain_images:
    #     img_ann.image_info.meta = 'plain'
    train_data = load_many(categories=train_load_config['categories'],
                           load_informations=train_load_config['data_loaders'])
    # train_data += plain_images
    dump(images_annotations=train_data, dump_config=train_load_config['dump'])

    load_and_dump(test_load_config)
    create_composer_and_debug(load_and_dump_configs=train_load_config, composer_config=train_composer_config)
