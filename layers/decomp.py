import torch
from torch import nn

from layers.SMA import SMA
from layers.ema import EMA
from layers.dema import DEMA

class DECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type, alpha, beta,kernel_size,channels):
        super(DECOMP, self).__init__()
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)
        elif ma_type == 'sma':
            self.ma = SMA(kernel_size, stride=1)

    def forward(self, x):
        trend = self.ma(x)
        seasonal = x - trend
        return seasonal,trend