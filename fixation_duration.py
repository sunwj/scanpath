import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils import WeightedGlobalMeanPool


class FixationDuration(nn.Module):
    def __init__(self, feat_dim, roi_hidden_dim):
        super(FixationDuration, self).__init__()
        self.feat_dim = feat_dim
        self.roi_hidden_dim = roi_hidden_dim

        self.mu = nn.Sequential(
            nn.Linear(self.feat_dim + self.roi_hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.mu[-1].bias.data.copy_(torch.rand_like(self.mu[-1].bias))
        self.log_var = nn.Sequential(
            nn.Linear(self.feat_dim + self.roi_hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.pool = WeightedGlobalMeanPool(keepdim=False)

    def forward(self, feat, hidden, current_ROI):
        spatial_size = feat.size()[2:]
        with torch.no_grad():
            current_roi = F.interpolate(current_ROI, size=spatial_size, mode='bilinear')

        feat_vec = self.pool(feat, current_roi)
        hidden_vec = self.pool(hidden, current_roi)

        gaussian_input = torch.cat([feat_vec, hidden_vec], dim=-1)
        mu = self.mu(gaussian_input)
        mu = torch.clamp(mu, 0, 4)
        sigma_squared = torch.exp(self.log_var(gaussian_input))
        sigma = torch.sqrt(sigma_squared)
        sigma = sigma.clamp(0.05, 0.1)

        normal_distribution = Normal(mu, sigma)
        sample = normal_distribution.rsample()
        sample = sample.clamp(0, 4)

        return sample
