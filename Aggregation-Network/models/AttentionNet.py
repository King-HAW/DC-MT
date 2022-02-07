import os
import logging
import torch.nn as nn
from .AttentionModel.url_maps import url_maps
from .AttentionModel.rnn import *
from torch.utils import model_zoo


class AttentionNet(nn.Module):
    """
    Build Attention Model
    """
    def __init__(self, model_arch, num_classes, ema=False):
        super(AttentionNet, self).__init__()
        self.arch = model_arch
        self.model = eval(model_arch)(num_classes=num_classes)
        if ema:
            for param in self.model.parameters():
                param.detach_()

    def forward(self, x):
        out = self.model(x)
        return out

    def load_pretrained_weights(self, load_fc=True):
        arch = self.arch
        pretrained_dict = model_zoo.load_url(url_maps[arch])
        if arch[:6] == 'resnet' or arch[:11] == 'wide_resnet' or arch == 'resnext50_32x4d' or arch == 'resnext101_32x8d' \
                or arch[:12] == 'efficientnet' or arch == 'inceptionv3' or arch[:8] == 'densenet':
            if load_fc:
                self.model.load_state_dict(pretrained_dict)
            else:
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
        else:
            if load_fc:
                self.model.load_state_dict(pretrained_dict)
            else:
                pretrained_dict.pop('last_linear.weight')
                pretrained_dict.pop('last_linear.bias')
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(pretrained_dict, strict=False)
        logging.info('Finish loading pretrained weights!')


def build_model(config, ema):
    # logging.info("Now supporting resnet, senet, densenet for Attention")
    arch = config['arch']
    num_classes = len(config['Data_CLASSES'])
    return AttentionNet(arch, num_classes, ema=ema)
