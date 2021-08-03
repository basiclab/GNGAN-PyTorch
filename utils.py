import os
import random
from contextlib import contextmanager

import torch
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm


device = torch.device('cuda:0')


def save_images(images, output_dir, verbose=False):
    os.makedirs(output_dir, exist_ok=True)
    for i, image in enumerate(tqdm(images, dynamic_ncols=True, leave=False,
                                   disable=(not verbose), desc="save_images")):
        save_image(image, os.path.join(output_dir, '%d.png' % i))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


@contextmanager
def module_no_grad(m: torch.nn.Module):
    requires_grad_dict = dict()
    for name, param in m.named_parameters():
        requires_grad_dict[name] = param.requires_grad
        param.requires_grad_(False)
    yield m
    for name, param in m.named_parameters():
        param.requires_grad_(requires_grad_dict[name])
