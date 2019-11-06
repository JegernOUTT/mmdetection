import argparse
import os

import mmcv
import numpy as np
import onnx
import onnxruntime as rt
import torch
from mmcv.runner import load_checkpoint
from mmdet.core import wrap_fp16_model
from mmdet.models import build_detector
from onnx import optimizer
from torch.onnx import export


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
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    output_path = './test.onnx'
    output_h, output_w = 128, 160

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    assert 'forward_export' in model.__dir__()
    model.forward = model.forward_export
    with torch.no_grad():
        export(model, torch.zeros((1, 3, output_h, output_w), dtype=torch.float32), output_path,
               opset_version=9,
               do_constant_folding=True)

    onnx_model = onnx.load(output_path)
    onnx_model = optimizer.optimize(
        onnx_model,
        ['eliminate_identity', 'eliminate_nop_pad', 'eliminate_nop_transpose', 'eliminate_unused_initializer',
         'extract_constant_to_initializer',
         'fuse_bn_into_conv',
         'fuse_add_bias_into_conv', 'fuse_consecutive_squeezes',
         'fuse_consecutive_transposes', 'fuse_transpose_into_gemm',
         'lift_lexical_references', 'nop'])
    onnx.save(onnx_model, output_path)
    similarity_test(model, output_path, height_width=(output_h, output_w))


if __name__ == '__main__':
    main()
