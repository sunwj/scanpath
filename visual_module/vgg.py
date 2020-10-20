import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from collections import OrderedDict


__all__ = ['VGG', 'vgg16', 'vgg19']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'D_2', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


class VGG(nn.Module):
    """
    VGG model with only feature layers
    """
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def make_layers(cfg):
    layers = []
    in_channels = 3
    padding = 1
    dilation = 1
    idx = 0
    for v in cfg:
        if v == 'M':
            layers.append((str(idx), nn.MaxPool2d(kernel_size=2, stride=2)))
            idx += 1
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=padding, dilation=dilation)
            layers.append((str(idx), conv2d))
            idx += 1
            layers.append((str(idx), nn.ReLU(inplace=True)))
            idx += 1
            in_channels = v

    return nn.Sequential(OrderedDict(layers))


def vgg16(pretrained=False):
    """
    dilated vgg16 model with 8 times downscale
    :param pretrained: load parameters pretrained on ImageNet
    :return: dilated vgg16 feature layers before fully connected layer
    """
    model = VGG(make_layers(cfg['D']))

    if pretrained:
        params = model_zoo.load_url(model_urls['vgg16'])
        params = OrderedDict([(key, item) for (key, item) in params.items() if 'features' in key])
        model.load_state_dict(params)

    return model


def vgg19(pretrained=False):
    """
    dilated vgg19 model with 8 times downscale
    :param pretrained: load parameters pretrained on ImageNet
    :return: dilated vgg16 feature layers before fully connected layer
    """
    model = VGG(make_layers(cfg['E']))

    if pretrained:
        params = model_zoo.load_url(model_urls['vgg19'])
        params = OrderedDict([(key, item) for (key, item) in params.items() if 'features' in key])
        model.load_state_dict(params)

    return model
