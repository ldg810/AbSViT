import os
import sys
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import json
import math
from matplotlib import pyplot as plt 
import torch.autograd as autograd
import random

from pathlib import Path
import collections.abc as container_abcs

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

sys.path.append('../')
from datasets import build_dataset, build_transform
from utils import DistillationLoss, RASampler, unnormalize_image
import models.vit
import models.absvit
import utils
from config import config
from typing import Iterable, Optional
from einops import rearrange

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser('DeiT training and evaluation script')
    args = parser.parse_args(args=[])
    # args.attention = 'self-att'
    # args.model = 'absvit_tiny_patch8_224_gap'
    # args.resume = 'https://berkeley.box.com/shared/static/7415yz4d1l5z0ur6x32k35f8y99zgynq.pth'
    args.distributed = False
    return args

def get_attention(self, x, td=None):
    layer = len(self.blocks)-3
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    for i in range(layer):
        if td is None:
            x = self.blocks[i](x)
        else:
            x = self.blocks[i](x, td[i])
    attn = x.norm(dim=-1)[:, self.num_prefix_tokens:]
    return attn

def load_image(path):
    img = plt.imread(path)
    img = torch.Tensor(img).float().cuda().permute(2, 0, 1)
    img = img / 255
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(3, 1, 1)
    img = (img - mean) / std
    img = F.interpolate(img[None], size=224, mode='bicubic')[0]
    return img

def test_simple(args):
    device = torch.device("cuda")

    cudnn.benchmark = True

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_rate=0,
        drop_path_rate=0.1,
        drop_block_rate=None
    )
    model.to('cuda:0')

    model_without_ddp = model
    checkpoint = torch.utils.model_zoo.load_url(args.resume)
    model_without_ddp.load_state_dict(checkpoint['model'])

    _ = model.eval()

    subplot_row = 2
    subplot_col = 4

    plt.figure(figsize=(12, 5))

    y0 = 879 # umbrella
    y1 = 200 # dog
    x_bi = load_image('demo/multi_object_real_879_200.jpg')
    img = unnormalize_image(x_bi)
    plt.subplot(subplot_row, subplot_col, 1)
    plt.axis('off')
    plt.imshow(img)
    plt.title('Input image')
    x_bi = x_bi[None]

    input = x_bi
    x, _, __ = model.forward_features(input)
    cos_sim = F.normalize(x, dim=-1) @ F.normalize(model.prompt[None, ..., None], dim=1)  # B, N, 1
    mask = cos_sim.clamp(0, 1)
    x = x * mask
    td = model.feedback(x)

    att = get_attention(model, input, td)

    L = att.shape[-1]
    att = att[0].view(int(L**0.5), int(L**0.5))
    att = (att - (0.7*att.max() + 0.3*att.min())).clamp(0)
    plt.subplot(subplot_row, subplot_col, 2)
    plt.imshow(att.detach().cpu().numpy())
    plt.axis('off')
    plt.title('Bottom-up attention')

    input = x_bi
    x, _, __ = model.forward_features(input)
    cos_sim = F.normalize(x, dim=-1) @ F.normalize(model.head.weight[y0][None, ..., None], dim=1)  # B, N, 1
    mask = cos_sim.clamp(0, 1)*40
    plt.subplot(subplot_row, subplot_col, 3)
    plt.imshow(mask[0, 1:, 0].view(28, 28).detach().cpu().numpy())
    plt.axis('off')
    plt.title('prior mask of class 1')

    x = x * mask
    td = model.feedback(x)

    att = get_attention(model, input, td)

    L = att.shape[-1]
    att = att[0].view(int(L**0.5), int(L**0.5))
    att = (att - (0.7*att.max() + 0.3*att.min())).clamp(0)
    plt.subplot(subplot_row, subplot_col, 3+subplot_col)
    plt.imshow(att.detach().cpu().numpy())
    plt.axis('off')
    plt.title('top-down attention of class 1')

    input = x_bi

    x, _, __ = model.forward_features(input)
    cos_sim = F.normalize(x, dim=-1) @ F.normalize(model.head.weight[y1][None, ..., None], dim=1)  # B, N, 1
    mask = cos_sim.clamp(0, 1)*40
    plt.subplot(subplot_row, subplot_col, 4)
    plt.imshow(mask[0, 1:, 0].view(28, 28).detach().cpu().numpy())
    plt.axis('off')
    plt.title('prior mask of class 2')

    x = x * mask
    td = model.feedback(x)

    att = get_attention(model, input, td)

    L = att.shape[-1]
    att = att[0].view(int(L**0.5), int(L**0.5))
    att = (att - (0.7*att.max() + 0.3*att.min())).clamp(0)
    plt.subplot(subplot_row, subplot_col, 4+subplot_col)
    plt.imshow(att.detach().cpu().numpy())
    plt.axis('off')
    plt.title('top-down attention of class 2')

    plt.savefig('fig_attention_map.png')

if __name__ == '__main__':
    args = parse_args()
    test_simple(args)