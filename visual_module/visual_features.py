import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg import vgg16, vgg19


class VGG(nn.Module):
    def __init__(self, model='vgg16', fine_tune=True):
        super(VGG, self).__init__()

        backend = vgg16(pretrained=False) if model == 'vgg16' else vgg19(pretrained=False)
        # backend.features[-1] = nn.Tanh()
        features = list(backend.features.children())

        if model == 'vgg16':
            self.front = nn.Sequential(*features[:24])
            self.back = nn.Sequential(*features[24:])
        else:
            self.front = nn.Sequential(*features[:28])
            self.back = nn.Sequential(*features[28:])

        self.fine_tune = fine_tune

        for p in self.front.parameters():
            p.requires_grad_(False)

        if not self.fine_tune:
            for p in self.back.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        with torch.no_grad():
            out = self.front(x)

        if self.fine_tune:
            out = self.back(out)
        else:
            with torch.no_grad():
                out = self.back(out)
        return out


class FeatureFusion(nn.Module):
    def __init__(self, sem_channels):
        super(FeatureFusion, self).__init__()

        self.sem_channels = sem_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(512 + sem_channels, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.Tanh()
        )

    def forward(self, img_feature, sem_info):
        if self.sem_channels == 0:
            return img_feature
        else:
            return self.fusion(torch.cat([img_feature, sem_info], dim=1))
