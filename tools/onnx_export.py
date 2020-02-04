import argparse

import cv2
import mmcv
import numpy as np
import onnxruntime as rt
import torch
from mmcv.runner import load_checkpoint
from torch.onnx import export
from tqdm import tqdm

from mmdet.models import build_detector


def similarity_test(pytorch_model, onnx_output_path, height_width):
    h, w = height_width
    img = cv2.imread('/home/svakhreev/projects/mmdetection/test/images/0xs6qPmkzH_md0_day.avi-1755.jpg')
    img = (cv2.resize(img[..., ::-1], (w, h)) / 255.).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    input_data = np.concatenate([img for _ in range(32)], axis=0)
    with torch.no_grad():
        pytorch_out = pytorch_model(torch.from_numpy(input_data))
    pytorch_out = np.concatenate([o.cpu().numpy()[np.newaxis, ] for o in pytorch_out], axis=0)
    sess = rt.InferenceSession(onnx_output_path)
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run([o.name for o in sess.get_outputs()], {input_name: input_data})[0]

    assert np.all(np.isclose(pytorch_out, onnx_out, atol=0.01)), \
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
    output_path = './test.onnx'

    torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    import torch.onnx.symbolic_opset9 as onnx_symbolic
    def upsample_nearest2d(g, input, output_size):
        # Currently, TRT 5.1/6.0 ONNX Parser does not support all ONNX ops
        # needed to support dynamic upsampling ONNX forumlation
        # Here we hardcode scale=2 as a temporary workaround
        scales = g.op("Constant", value_t=torch.tensor([1., 1., 2., 2.]))
        return g.op("Upsample", input, scales, mode_s="nearest")

    onnx_symbolic.upsample_nearest2d = upsample_nearest2d

    output_h, output_w = 320, 416
    assert 'forward_export' in model.__dir__()
    model.forward = model.forward_export
    with torch.no_grad():
        export(model, torch.zeros((32, 3, output_h, output_w), dtype=torch.float32),
               output_path,
               do_constant_folding=True,
               opset_version=9)
    similarity_test(model, args.out, height_width=(output_h, output_w))


if __name__ == '__main__':
    main()
