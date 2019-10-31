#!/usr/bin/python3

from pathlib import Path

from detector_utils import *

__all__ = ['load_and_dump_train_config', 'load_and_dump_test_config', 'composer_config']

categories = {0: 'person', 1: 'car'}
train_paths = [Path('/mnt/nfs/Data/detector_markup/person_head_car_2_wheel_train/train')]
test_paths = [Path('/mnt/nfs/Data/detector_markup/person_head_car_2_wheel_train/test')]


def _create_load_dump_config(dataset_paths, prefix):
    return {
        'categories': categories,

        'data_loader': LegacyPickleLoadInformation(
            data_paths=[DataPathInformation(path=p) for p in dataset_paths]
        ),

        'dump': DumpConfig(
            clone_images=True,
            annotations_dump_filename=Path(f'data_{prefix}/annotations/annotation'),
            images_clone_path=Path(f'data_{prefix}/images'))
    }


load_and_dump_train_config = _create_load_dump_config(dataset_paths=train_paths, prefix='train')
load_and_dump_test_config = _create_load_dump_config(dataset_paths=test_paths, prefix='test')

composer_config = {
    'filters': [
        {'type': 'ImageValidityFilter'},
        {'type': 'ImagesWithoutAnnotationsFilter'},
        {'type': 'ImageSizeFilter',
         'min_size': Size2D(width=32, height=32),
         'max_size': Size2D(width=10000, height=10000)},
        {'type': 'BboxPixSizeFilter',
         'min_size': Size2D(width=5, height=5),
         'max_size': Size2D(width=10000, height=10000)},
    ],

    'sampler': {'type': 'RandomSampler'}
}

if __name__ == '__main__':
    load_and_dump(load_and_dump_config=load_and_dump_train_config)
    load_and_dump(load_and_dump_config=load_and_dump_test_config)
