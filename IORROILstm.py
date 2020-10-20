import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_lstm_cell import ConvLSTMCell, IORLSTMCell
from utils import ChannelAttention, GaussianFilter


class IORROILstm(nn.Module):
    def __init__(self, input_dim, state_dim):
        super(IORROILstm, self).__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim

        self.iorlstm = IORLSTMCell(self.input_dim, self.state_dim, 128)
        self.roilstm = ConvLSTMCell(self.input_dim, 128)

        self.channel_attention = ChannelAttention(self.input_dim, 128)
        self.smooth = GaussianFilter(1, 3)

    def forward(self, x, current_ROI, fix_duration, fix_tran, ior_state, roi_state):
        batch_size = x.size()[0]
        spatial_size = x.size()[2:]

        if ior_state is None:
            state_size = [batch_size, self.state_dim] + list(spatial_size)
            device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
            ior_state = (
                torch.zeros(state_size).to(device),
                torch.zeros(state_size).to(device)
            )
        if roi_state is None:
            state_size = [batch_size, self.state_dim] + list(spatial_size)
            device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
            roi_state = (
                torch.zeros(state_size).to(device),
                torch.zeros(state_size).to(device)
            )

        with torch.no_grad():
            current_roi = current_ROI.clone()
            current_roi[current_roi > 0.15] = 1
            current_roi = self.smooth(current_roi)
            current_roi = F.interpolate(current_roi, size=spatial_size, mode='bilinear')

        fix_duration = fix_duration.reshape(batch_size, 1, 1, 1)
        fix_tran = fix_tran.reshape(batch_size, 1, 1, 1)
        ior_hidden, ior_cell = self.iorlstm(x, current_roi, fix_duration, fix_tran, ior_state, roi_state[0])

        ior_map = torch.mean(ior_cell, dim=1, keepdim=True)
        xi = x * (1 - ior_cell)

        ca = self.channel_attention(xi, roi_state[0], current_roi)
        xic = xi * ca

        roi_hidden, roi_cell = self.roilstm(xic, roi_state)
        roi_latent = torch.mean(roi_hidden, dim=1, keepdim=True)

        return (ior_hidden, ior_cell), (roi_hidden, roi_cell), ior_map, roi_latent
