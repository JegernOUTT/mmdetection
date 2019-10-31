from logging import error

import numpy as np
from detector_utils import Composer as TrassirComposer
from detector_utils import create_composer
from detector_utils.utils.other import load_module

from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.registry import DATASETS


@DATASETS.register_module
class DsslDataset(CustomDataset):
    CLASSES = None

    def __init__(self, ann_file, pipeline, load_and_dump_config_name, test_mode=False):
        self._load_config_filename = ann_file
        self._test_mode = test_mode
        self._load_and_dump_config_name = load_and_dump_config_name
        self._pipeline = Compose(pipeline)
        self._trassir_composer: TrassirComposer = self.load_trassir_composer(self._load_config_filename)

        if not self._test_mode:
            self._set_group_flag()

    def __len__(self):
        return len(self._trassir_composer)

    def load_trassir_composer(self, load_config_filename):
        trassir_load_config = load_module(load_config_filename)
        load_and_dump_config = trassir_load_config.__getattribute__(self._load_and_dump_config_name)
        DsslDataset.CLASSES = tuple(load_and_dump_config['categories'].values())
        try:
            return create_composer(load_and_dump_configs=load_and_dump_config,
                                   composer_config=trassir_load_config.composer_config)
        except:
            error(f'Be sure that you had run "./tools/trassir_data_config.py" to create data dump')
            raise

    def _pre_pipeline(self, results):
        results['img_prefix'] = ''
        results['seg_prefix'] = ''
        results['proposal_file'] = ''
        results['bbox_fields'] = []
        results['mask_fields'] = []

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self._trassir_composer[i]
            if img_info.image_info.size.width / img_info.image_info.size.height > 1:
                self.flag[i] = 1

    def __getitem__(self, idx):
        if self._test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img_annotations = self._trassir_composer[idx]
        img_info = {
            'filename': str(img_annotations.image_info.filename),
            'width': img_annotations.image_info.size.width,
            'height': img_annotations.image_info.size.height,
        }
        bbox_count = len(img_annotations.objects)
        ann_info = {
            'bboxes': np.array([obj.bbox.xyxy(image_size=img_annotations.image_info.size)
                                for obj in img_annotations.objects],
                               dtype=np.float32).reshape((bbox_count, 4)),
            'labels': np.array([obj.category_id + 1 for obj in img_annotations.objects],
                               dtype=np.int64).reshape((bbox_count,)),
            'bboxes_ignore': np.zeros((0, 4), dtype=np.float32)
        }
        results = dict(img_info=img_info, ann_info=ann_info)
        self._pre_pipeline(results)
        return self._pipeline(results)

    def prepare_test_img(self, idx):
        img_annotations = self._trassir_composer[idx]
        img_info = {
            'filename': str(img_annotations.image_info.filename),
            'width': img_annotations.image_info.size.width,
            'height': img_annotations.image_info.size.height,
        }
        results = dict(img_info=img_info)
        self._pre_pipeline(results)
        return self._pipeline(results)
