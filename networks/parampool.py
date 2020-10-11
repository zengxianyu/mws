from .tools import weight_init
import torch
import torch.nn as nn


class ParamPool(nn.Module):
    def __init__(self, input_c):
        super(ParamPool, self).__init__()
        #self.conv = nn.Conv2d(input_c, 1, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(1, 1, kernel_size=16)

    def forward(self, x):
        w = self.conv.weight
        w = torch.softmax(w.view(1, 1, -1), 2)
        w = w.view(1, 1, 16, 16)
        x = (x*w).sum(3).sum(2)
        return x


