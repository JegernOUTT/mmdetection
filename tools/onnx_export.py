import argparse

import mmcv
import numpy as np
import onnxruntime as rt
import torch
from mmcv.runner import load_checkpoint
from torch.onnx import export

from mmdet.models import build_detector


def similarity_test(pytorch_model, onnx_output_path, height_width):
    h, w = height_width
    input_data = np.random.random((1, 3, h, w)).astype(np.float32)
    with torch.no_grad():
        pytorch_out = pytorch_model(torch.from_numpy(input_data))
    pytorch_out = [o.cpu().numpy() for o in pytorch_out]

    sess = rt.InferenceSession(onnx_output_path)
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run([o.name for o in sess.get_outputs()], {input_name: input_data})

    assert np.all(np.isclose(pytorch_out[0], onnx_out[0], atol=0.01)), \
        "Your onnx and pytorch outputs is not same, check model conversion"
    assert not np.all(np.isclose(pytorch_out[0], np.zeros_like(pytorch_out[0]))), \
        "Your pytorch outputs is zeros, it can not possible to check model output validity"
    assert not np.any(np.isnan(pytorch_out[0])), \
        "Your pytorch outputs is nans, it can not possible to check model output validity"


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    output_h, output_w = 768, 1024
    for k, v in cfg.__dict__.items():
        if k.startswith('test'):
            for k_, v_ in v.__dict__.items():
                if k_.endswith('img_scale'):
                    output_w, output_h = v_

    assert 'forward_export' in model.__dir__()
    print(f'Width: {output_w}, height: {output_h}')
    model.forward = model.forward_export
    with torch.no_grad():
        export(model, torch.zeros((1, 3, output_h, output_w), dtype=torch.float32),
               args.out,
               opset_version=9,
               do_constant_folding=True)
    similarity_test(model, args.out, height_width=(output_h, output_w))


if __name__ == '__main__':
    main()
