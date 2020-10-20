import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from bisect import bisect_left


class GlobalMeanPool(nn.Module):
    def __init__(self, keepdim=True):
        super(GlobalMeanPool, self).__init__()
        self.keepdim = keepdim

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.reshape(b, c, -1)
        x = torch.mean(x, dim=2)
        if self.keepdim:
            x = x.reshape(b, c, 1, 1)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, state_dim):
        super(ChannelAttention, self).__init__()
        self.pool = GlobalMeanPool()
        self.U = nn.Sequential(
            nn.Conv2d(channels + state_dim, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, roi_hidden, roi):
        u = self.U(torch.cat([x, roi_hidden], dim=1))
        u_roi = self.pool(u * roi)
        u_ctx = self.pool(u * (1 - roi))

        ca = 1 - u_roi * u_ctx
        b, c, h, w = ca.size()
        ca = ca.reshape(b, -1)
        ca = F.softmax(ca, dim=-1)
        ca = ca.reshape(b, c, h, w)

        return ca


class WeightedGlobalMeanPool(nn.Module):
    def __init__(self, keepdim=True):
        super(WeightedGlobalMeanPool, self).__init__()
        self.keepdim = keepdim

    def forward(self, x, weight):
        b, c, h, w = x.size()

        y = x * weight
        y = y.reshape(b, c, -1)
        y = torch.mean(y, dim=-1)
        if self.keepdim:
            y = y.reshape(b, c, 1, 1)

        return y


class Sampler2D:
    def __init__(self, pdf):
        self.conditional = np.cumsum(pdf, axis=0)
        self.marginal = np.cumsum(self.conditional[-1, :])

    def sample(self):
        v = np.random.rand()
        ind_v = bisect_left(self.marginal, v)

        conditional = self.conditional[:, ind_v].flatten()
        conditional = conditional / conditional[-1]

        u = np.random.rand()
        ind_u = bisect_left(conditional, u)

        return ind_v, ind_u  # x, y


class Sampler1D:
    def __init__(self, pdf, bin_size):
        self.cdf = np.cumsum(pdf)
        assert self.cdf[-1] <= 1
        self.bin_size = bin_size

    def sample(self):
        u = np.random.rand()
        ind_u = bisect_left(self.cdf, u)
        ind_u_right = ind_u + 1
        portion = (u - self.cdf[ind_u]) / (self.cdf[ind_u_right] - self.cdf[ind_u] + 1e-8)

        return self.bin_size * ind_u_right - self.bin_size * portion


class OculomotorBias:
    def __init__(self, ob_file, pixels_per_degree):
        data = loadmat(ob_file, squeeze_me=True, struct_as_record=False)
        self.ob = data['distributionSmooth']
        self.pixels_per_degree = pixels_per_degree
        self.last_x = None
        self.last_y = None

    def set_last_fixation(self, x, y):
        self.last_x = x
        self.last_y = y

    def prob(self, x, y, update=False):
        dx = x - self.last_x
        dy = y - self.last_y

        if dx == 0:
            if dy < 0:
                ang = 3 * np.pi / 2
            else:
                ang = np.pi / 2
        elif dx > 0:
            if dy >= 0:
                ang = np.arctan(dy / dx)
            else:
                ang = 2 * np.pi - np.arctan(-dy / dx)
        else:
            if dy < 0:
                ang = np.pi + np.arctan(-dy / dx)
            else:
                ang = np.pi - np.arctan(dy / dx)

        ang = int(ang / np.pi * 180)
        amp = int(np.sqrt(dx ** 2 + dy ** 2) / self.pixels_per_degree * 4)
        amp = np.clip(amp, 0, 79)

        if update:
            self.set_last_fixation(x, y)

        return self.ob[int(amp), int(ang)]


class ROIGenerator:
    def __init__(self, img_w, img_h, radius):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cy, cx = np.meshgrid(np.arange(img_h), np.arange(img_w))

        self.cx = torch.from_numpy(cx.T).float().to(device)
        self.cy = torch.from_numpy(cy.T).float().to(device)
        self.radius = torch.tensor(radius).float().to(device)

    def generate_roi(self, x, y):
        e2 = (self.cx - x) ** 2 + (self.cy - y) ** 2
        roi = torch.exp(-e2 / (2 * self.radius ** 2))
        roi[roi < 0.1] = 0
        return roi


def _gaussian_kernel(size, size_y=None):
    size = int(size)

    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)

    x, y = np.mgrid[-size: size + 1, -size_y: size_y + 1]
    g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size_y)))

    return g / g.sum()


def _LoG_kernel(size, sigma):
    x, y = np.mgrid[-size: size + 1, -size: size + 1]
    g = (x ** 2 + y ** 2 - 2 * sigma ** 2) / (4 * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return g


class GaussianFilter(nn.Module):
    def __init__(self, input_channels, gaussian_ksize=3):
        super(GaussianFilter, self).__init__()

        self.input_channels = input_channels
        fgk_size = gaussian_ksize * 2 + 1

        gaussian_kernel = _gaussian_kernel(gaussian_ksize)
        gaussian_kernel = np.broadcast_to(gaussian_kernel, (self.input_channels, fgk_size, fgk_size))
        gaussian_kernel = nn.Parameter(torch.from_numpy(gaussian_kernel).float().unsqueeze(1))

        self.conv_gaussian = nn.Sequential(
            nn.ReflectionPad2d(gaussian_ksize),
            nn.Conv2d(self.input_channels, self.input_channels, kernel_size=fgk_size, stride=1, bias=False,
                      groups=self.input_channels)
        )

        self.conv_gaussian[1].weight = gaussian_kernel

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        output = self.conv_gaussian(x)

        return output
