#!/usr/bin/python3

from pathlib import Path

from detector_utils import *

__all__ = ['load_and_dump_train_config', 'load_and_dump_test_config',
           'composer_train_config', 'composer_test_config']

categories = {0: 'item'}
train_paths = [Path('/mnt/nfs/Data/empty_shelves/SKU110K/train'),
               Path('/mnt/nfs/Data/empty_shelves/SKU110K/val'),
               Path('/mnt/nfs/Data/empty_shelves/magnit')]
test_paths = [Path('/mnt/nfs/Data/empty_shelves/SKU110K/test_split/0_0_200')]


def _create_load_dump_config(dataset_paths, prefix, clone_images=True):
    return {
        'categories': categories,

        'data_loader': LegacyPickleLoadInformation(
            data_paths=[DataPathInformation(path=p) for p in dataset_paths],
        ),

        'dump': DumpConfig(
            clone_images=clone_images,
            annotations_dump_filename=Path(f'shelves/data_{prefix}/annotations/annotation'),
            images_clone_path=Path(f'shelves/data_{prefix}/images'))
    }


load_and_dump_train_config = _create_load_dump_config(dataset_paths=train_paths, prefix='train')
load_and_dump_test_config = _create_load_dump_config(dataset_paths=test_paths, prefix='test')

composer_train_config = {
    'filters': [
        {'type': 'ImageValidityFilter'},
        {'type': 'IgnoreMaskImagesRendererFilter', 'render_if_exists': False, 'with_respect_gt_bboxes': True}
    ],

    'sampler': {'type': 'RandomSampler'}
}

composer_test_config = {
    'filters': [
        {'type': 'ImageValidityFilter'},
    ],
    'sampler': {'type': 'SimpleSampler'}

}

if __name__ == '__main__':
    # load_and_dump(load_and_dump_config=load_and_dump_train_config)
    # load_and_dump(load_and_dump_config=load_and_dump_test_config)
    create_composer_and_debug(load_and_dump_configs=load_and_dump_test_config, composer_config=composer_test_config)