# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG


class VGGNet(VGG):
    def __init__(self, pretrained=True):
        super().__init__(make_layers(cfg['E']))

        if pretrained:
            super().load_state_dict(models.vgg19(pretrained=True).state_dict())

    def forward(self, x):
        output = {}
        ranges = ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))

        for idx in range(len(ranges)):
            for layer in range(ranges[idx][0], ranges[idx][1]):
                x = self.features[layer](x)
            print(x.size())
            output["x%d"%(idx+1)] = x

        return output


class FCN32s(VGG):

    def __init__(self, classes=32):
        super().__init__()
        self.classes = classes
        self.pretrained_net = VGGNet()

    def forward(self, x):
        pass


# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == "__main__":
    test_model = VGGNet()

    input = torch.autograd.Variable(torch.randn(16, 3, 224, 224))
    output = test_model(input)
    assert output['x5'].size() == torch.Size([16, 512, 7, 7])

