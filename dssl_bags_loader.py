#!/usr/bin/python3

from pathlib import Path

from detector_utils import *

__all__ = ['load_and_dump_train_config', 'load_and_dump_test_config', 'composer_config']

categories = {0: 'item'}
train_paths = [Path('/mnt/nfs/input/saved_temporary_data/bag_counter/bags_edge_v_8/release/train'),
               Path('/mnt/nfs/Data/empty_shelves/shelves_crops')]
test_paths = [Path('/mnt/nfs/input/saved_temporary_data/bag_counter/bags_edge_v_8/release/test')]


def _create_load_dump_config(dataset_paths, prefix, clone_images=True):
    return {
        'categories': categories,

        'data_loader': LegacyPickleLoadInformation(
            data_paths=[DataPathInformation(path=p) for p in dataset_paths]
        ),

        'dump': DumpConfig(
            clone_images=clone_images,
            annotations_dump_filename=Path(f'data_{prefix}/annotations/annotation'),
            images_clone_path=Path(f'data_{prefix}/images'))
    }


load_and_dump_train_config = _create_load_dump_config(dataset_paths=train_paths, prefix='train', clone_images=True)
load_and_dump_test_config = _create_load_dump_config(dataset_paths=test_paths, prefix='test', clone_images=True)

composer_config = {
    'filters': [
        {'type': 'ImageValidityFilter'},
        {'type': 'ImageSizeFilter',
         'min_size': Size2D(width=32, height=32),
         'max_size': Size2D(width=10000, height=10000)},
        {'type': 'BboxAbsoluteSizeFilter',
         'min_size': Size2DF(width=0.06, height=0.06),
         'max_size': Size2DF(width=1., height=1.)},
    ],

    'sampler': {'type': 'RandomSampler'}
}

if __name__ == '__main__':
    load_and_dump(load_and_dump_config=load_and_dump_train_config)
    load_and_dump(load_and_dump_config=load_and_dump_test_config)
    # create_composer_and_debug(load_and_dump_configs=load_and_dump_train_config, composer_config=composer_config)