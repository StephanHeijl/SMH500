import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class InvestorNetwork(nn.Module):
    def __init__(self):
        super(InvestorNetwork, self).__init__()

        self.saved_actions = []
        self.rewards = []
        # Input channels = 6 (open, close, high, low, volume_from, volume_to)
        self.s_1 = nn.Conv1d(6, 64, 7)
        self.s_1b = nn.BatchNorm1d(64)
        self.s_1a = nn.ReLU()
        self.s_2 = nn.Conv1d(64, 16, 7)
        self.s_2b = nn.BatchNorm1d(16)
        self.s_2a = nn.ReLU()

        n_size = 768 + 2

        self.d_1 = nn.Linear(n_size, 128)
        self.d_1b = nn.BatchNorm1d(128)
        self.d_1a = nn.ReLU()
        self.d_2 = nn.Linear(128, 128)
        self.d_2b = nn.BatchNorm1d(128)
        self.d_2a = nn.ReLU()
        self.d_f = nn.Linear(128, 3)
        self.d_fs = nn.Softmax()  # Softmax forces a distribution of the funds

    def forward(self, x, cf, cc):
        H = self.s_1(x)
        H = self.s_1b(H)
        H = self.s_1a(H)
        H = self.s_2(H)
        H = self.s_2b(H)
        H = self.s_2a(H)

        H = H.view(x.size(0), -1)
        if isinstance(cf, Variable):
            fin_state = torch.cat([cf, cc])
        else:
            fin_state = Variable(Tensor([cf, cc]))
        fin_state = torch.unsqueeze(fin_state, 0)
        H = torch.cat([H, fin_state], -1)

        H = self.d_1(H)
        H = self.d_1b(H)
        H = self.d_1a(H)
        H = self.d_2(H)
        H = self.d_2b(H)
        H = self.d_2a(H)
        H = self.d_f(H)
        return self.d_fs(H)
