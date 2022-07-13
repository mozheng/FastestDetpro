import argparse
import math
import os
import torch
from module.detector import Detector
from utils.tool import *

def parse_args():
    """ Parse arguments.
    Returns:
        args: args object.
    """
    parser = argparse.ArgumentParser(description='Train a detector.')
    parser.add_argument('--yaml', type=str, default="", help='.yaml config')
    parser.add_argument('--weight', type=str, default=None,
                        help='.weight config')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()
    opt.cfg = LoadYaml(opt.yaml)

    # 指定后端设备CUDA&CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Detector(opt.cfg.category_num, True).to(device)
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(opt.weight, map_location=torch.device(device)).items()})
    dummy_input = torch.rand(1, 3, 352, 352).to(device)

    torch.onnx.export(model, dummy_input, "model.onnx", verbose=True,
                  input_names=['image'], output_names=['output'], opset_version=11)
