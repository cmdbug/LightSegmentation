import os
import sys

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import transforms

from light.utils.visualize import get_color_pallete

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from light.model import get_segmentation_model
from train import parse_args


def demo_run(args):
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
        dtype = torch.cuda.FloatTensor
    else:
        args.distributed = False
        args.device = "cpu"
        dtype = torch.FloatTensor

    model = get_segmentation_model(args.model, dataset=args.dataset,
                                   aux=args.aux, norm_layer=nn.BatchNorm2d).to(args.device)

    model_path = '../weights/' + args.model + '_' + args.dataset + '_best_model.pth'
    checkpoint = torch.load(model_path, map_location=args.device)

    # 修改网络后保留原来的权重
    model_ckpt = model.state_dict()
    model_ckpt.update(checkpoint)
    model_ckptx = {k: v for k, v in model_ckpt.items()
                   if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model_ckpt = model.state_dict()
    model_ckpt.update(model_ckptx)

    model.load_state_dict(model_ckpt)
    model.eval()

    ONNX_EXPORT = False
    if ONNX_EXPORT:
        img = torch.zeros((1, 3, args.crop_size, args.crop_size))
        f = '../runs/' + args.model + '.onnx'
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['input'],
                          output_names=['output'])
        return

    input_transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    src = Image.open("../images/demo1.jpg").convert('RGB')
    image = input_transform(src)
    image = image[np.newaxis, :]
    image = torch.from_numpy(np.array(image))
    image = image.to(args.device)

    start_time = time.time()
    outputs = model(image)
    print("forward time:{}".format(time.time() - start_time))

    pred = torch.argmax(outputs[0], 1)
    pred = pred.cpu().data.numpy()
    predict = pred.squeeze(0)
    mask = get_color_pallete(predict, "citys")

    resize_src = transforms.Resize((args.crop_size, args.crop_size))(src)
    merge = Image.blend(resize_src.convert('RGBA'), mask.convert('RGBA'), 0.5)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.15, hspace=0.25, bottom=0.11, top=0.95)
    ax1.imshow(resize_src)
    ax2.imshow(mask)
    ax2.set_title(args.model)
    ax3.imshow(merge)
    plt.show()


if __name__ == '__main__':
    matplotlib.use("TKAgg")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.style.use(['fast'])
    plt.rcParams['figure.facecolor'] = 'gray'

    args = parse_args()
    args.dataset = 'citys'
    args.crop_size = 512
    args.model = "mobilenetv3_small"
    # args.model = "mobilenetv3_large"
    # args.model = "shufflenetv2"
    # args.model = "mobilenetv2"
    demo_run(args)
