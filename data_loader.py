#!/usr/bin/python3

from pathlib import Path

from detector_utils import *

__all__ = ['train_load_configs', 'test_load_config', 'composer_train_config', 'composer_test_config']

categories = {0: 'car', 1: 'motorcycle', 2: 'bus', 3: 'truck', 4: 'van'}

_supervisely_train_load_config = {
    'categories': categories,
    'data_loader': SuperviselyLoadInformation(
        dataset_path=Path('/Users/sergejvahreev/Pictures/lpr5_supervisely')),
    'dump': DumpConfig(
        clone_images=True,
        annotations_dump_filename=Path(f'train/supervisely_annotations/annotations'),
        images_clone_path=Path(f'train/supervisely_images'))
}

_pickle_train_load_config = {
    'categories': categories,
    'data_loader': LegacyPickleLoadInformation(
        data_paths=[DataPathInformation(path=Path('/Users/sergejvahreev/Pictures/lpr_cam'))]
    ),
    'dump': DumpConfig(
        clone_images=True,
        annotations_dump_filename=Path(f'train/pickle_annotations/annotations'),
        images_clone_path=Path(f'train/pickle_images'))
}

train_load_configs = [_supervisely_train_load_config, _pickle_train_load_config]

test_load_config = {
    'categories': categories,
    'data_loader': LegacyPickleLoadInformation(
        data_paths=[DataPathInformation(path=Path('/Users/sergejvahreev/Pictures/lpr_cam'))]
    ),
    'dump': DumpConfig(
        clone_images=True,
        annotations_dump_filename=Path(f'test/annotations/annotations'),
        images_clone_path=Path(f'test/images'))
}

composer_train_config = {
    'filters': [
        {'type': 'ImageValidityFilter'},
        {'type': 'ImageSizeFilter',
         'min_size': Size2D(width=32, height=32),
         'max_size': Size2D(width=10000, height=10000)},
        {'type': 'IgnoreMaskImagesRendererFilter', 'render_if_exists': False}
    ],

    'sampler': {'type': 'SimpleSampler'}
}

composer_test_config = {
    'filters': [{'type': 'ImageValidityFilter'}],
    'sampler': {'type': 'SimpleSampler'}
}


if __name__ == '__main__':
    for config in train_load_configs:
        load_and_dump(load_and_dump_config=config)
    create_composer_and_debug(load_and_dump_configs=train_load_configs, composer_config=composer_config)
