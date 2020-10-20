import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical


class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_gaussians):
        super(MDN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gaussians = num_gaussians

        self.pi = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_gaussians),
            nn.Softmax(dim=-1)
        )
        self.mu = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim * self.num_gaussians)
        )
        self.std = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim * self.num_gaussians)
        )
        self.rho = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_gaussians)
        )

        self.mu[-1].bias.data.copy_(torch.rand_like(self.mu[-1].bias))

    def forward(self, x):
        pi = self.pi(x)

        mu = self.mu(x)
        sigma = 1 + F.elu(self.std(x))
        sigma = torch.clamp(sigma, 0.06, 0.12)
        rho = torch.clamp(self.rho(x), -0.25, 0.25)
        mu = mu.reshape(-1, self.num_gaussians, self.output_dim)
        sigma = sigma.reshape(-1, self.num_gaussians, self.output_dim)
        rho = rho.reshape(-1, self.num_gaussians, 1)

        return pi, mu, sigma, rho


def gaussian_probability(mu, sigma, rho, data):
    mean_x, mean_y = torch.chunk(mu, 2, dim=-1)
    std_x, std_y = torch.chunk(sigma, 2, dim=-1)
    x, y = torch.chunk(data, 2, dim=1)
    dx = x - mean_x
    dy = y - mean_y
    std_xy = std_x * std_y
    z = (dx * dx) / (std_x * std_x) + (dy * dy) / (std_y * std_y) - (2 * rho * dx * dy) / std_xy
    training_stablizer = 20
    norm = 1 / (training_stablizer * math.pi * std_x * std_y * torch.sqrt(1 - rho * rho))
    p = norm * torch.exp(-z / (1 - rho * rho) * 0.5)

    return p


def mixture_probability(pi, mu, sigma, rho, data):
    pi = pi.unsqueeze(-1)
    prob = pi * gaussian_probability(mu, sigma, rho, data)
    prob = torch.sum(prob, dim=1)

    return prob


def sample_mdn(pi, mu, sigma, rho):
    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    cat = Categorical(pi)
    pis = list(cat.sample().data)
    samples = list()
    for i, idx in enumerate(pis):
        loc = mu[i, idx]
        std = sigma[i, idx]
        std_x, std_y = std[0].item(), std[1].item()
        r = rho[i, idx].item()
        cov_mat = torch.tensor([[std_x * std_x, std_x * std_y * r], [std_x * std_y * r, std_y * std_y]]).to(device)
        MN = MultivariateNormal(loc, covariance_matrix=cov_mat)

        samples.append(MN.sample().unsqueeze(0))

    return torch.cat(samples, dim=0)

