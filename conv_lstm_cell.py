import torch
import torch.nn as nn
import torch.nn.functional as F


# define some constants
KSIZE = 5
PADDING = KSIZE // 2


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, state_dim):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim

        self.compute_gates = nn.Conv2d(input_dim + state_dim, 4 * state_dim, KSIZE, padding=PADDING)
        self.compute_gates.bias.data[:state_dim].fill_(1.0)

    def forward(self, x, prev_state):
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        if prev_state is None:
            state_size = [batch_size, self.state_dim] + list(spatial_size)
            device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
            prev_state = (
                torch.zeros(state_size).to(device),
                torch.zeros(state_size).to(device)
            )

        prev_hidden, prev_cell = prev_state

        stacked_input = torch.cat([x, prev_hidden], dim=1)
        gates = self.compute_gates(stacked_input)

        fgate, igate, ogate, g_content = gates.chunk(4, 1)

        igate = torch.sigmoid(igate)
        fgate = torch.sigmoid(fgate)
        ogate = torch.sigmoid(ogate)
        g = torch.tanh(g_content)

        current_cell = fgate * prev_cell + igate * g
        current_hidden = ogate * torch.tanh(current_cell)

        return current_hidden, current_cell


class IORLSTMCell(nn.Module):
    def __init__(self, input_dim, state_dim, roi_state_dim):
        super(IORLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim

        self.forget_gate = nn.Conv2d(input_dim + state_dim + 1 + roi_state_dim, state_dim, KSIZE, padding=PADDING)
        self.update_gate = nn.Conv2d(input_dim + state_dim + 1, state_dim, KSIZE, padding=PADDING)
        self.output_gate = nn.Sequential(
            nn.Conv2d(input_dim + state_dim + 1, self.state_dim, KSIZE, padding=PADDING),
            nn.Sigmoid()
        )
        self.cell_transform = nn.Sequential(
            nn.Conv2d(self.state_dim, self.state_dim, KSIZE, padding=PADDING),
            nn.Tanh()
        )
        self.forget_gate.bias.data.fill_(1.0)

    def forward(self, x, ROI, fix_duration, fix_transition, prev_state, roi_hidden):
        batch_size = x.data.size()[0]
        spatial_size = x.data.size()[2:]

        if prev_state is None:
            state_size = [batch_size, self.state_dim] + list(spatial_size)
            device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
            prev_state = (
                torch.zeros(state_size).to(device),
                torch.zeros(state_size).to(device)
            )

        prev_hidden, prev_cell = prev_state

        with torch.no_grad():
            fix_duration = torch.ones_like(ROI) * fix_duration
            fix_transition = torch.ones_like(ROI) * fix_transition
            duration = fix_duration + fix_transition

        ugate = self.update_gate(torch.cat([x * ROI, prev_hidden, fix_duration], dim=1))
        fgate = self.forget_gate(torch.cat([x, prev_hidden, roi_hidden, duration], dim=1))

        ugate = F.hardtanh(ugate, 0, 1)
        fgate = torch.sigmoid(fgate)

        current_cell = torch.max(fgate * prev_cell, ugate)
        current_hidden = self.cell_transform(current_cell) * self.output_gate(torch.cat([x, prev_hidden, duration], dim=1))

        return current_hidden, current_cell
