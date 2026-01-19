import torch
from torch import nn
import torch.nn.functional as F

from layers.SeriesMix import MultiScaleTrendMixing
from layers.revin import RevIN

class SeasonalTCNStream(nn.Module):
    def __init__(self, c_in: int, seq_len: int, pred_len: int, kernel_size: int = 3, dilations: list = [1, 2, 4, 8]):
        super().__init__()
        # 构建一系列 depthwise dilated conv + ReLU
        layers = []
        for d in dilations:
            # depthwise 卷积
            layers += [
                nn.Conv1d(in_channels=c_in,
                    out_channels=c_in,
                    kernel_size=kernel_size,
                    padding=d*(kernel_size-1)//2,
                    dilation=d,
                    groups=c_in,
                    bias=False),
                nn.ReLU()
                ]
        self.tcn = nn.Sequential(*layers)         # [B,C,L] -> [B,C,L]
        self.proj = nn.Linear(seq_len, pred_len)  # 把时间维 L -> pred_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        y = self.tcn(x)           # [B, C, L]
        # 最后一维做线性投影
        return self.proj(y)       # [B, C, pred_len]

class TrendFFNPoolStream(nn.Module):
    def __init__(self, seq_len: int, pred_len: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.linear1   = nn.Linear(seq_len, hidden_dim)
        self.avgpool1  = nn.AvgPool1d(kernel_size=2, stride=2)
        self.ln1        = nn.LayerNorm(hidden_dim // 2)
        self.linear2   = nn.Linear(hidden_dim // 2, pred_len)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, C, L] ──> linear1 on last dim
        x = self.linear1(x)
        x = self.avgpool1(x)
        x = self.ln1(x)
        x = self.linear2(x)
        return x


class Network_Sea(nn.Module):
    def __init__(
        self, c_in, seq_len, pred_len,seasonal_kernel1=3,ds_window=2, ds_layers=3,initial_tre_w=1.0):
        super(Network_Sea, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in

        # RevIN
        self.revin = RevIN(num_features=c_in)

        # ====== 季节分支 ======
        self.seasonal_stream = SeasonalTCNStream(c_in=c_in, seq_len=seq_len, pred_len=pred_len,
                                                         kernel_size=3, dilations=[1,2,4,8]) 

        # ====== 多尺度趋势分支 ======
        self.trend_stream = TrendFFNPoolStream(seq_len, pred_len, hidden_dim=128)
        self.tre_w = nn.Parameter(torch.FloatTensor([initial_tre_w]*c_in), requires_grad=True)

        # ====== 融合 ======
        self.fc_fuse = nn.Linear(pred_len * 2, pred_len)

    def forward(self, seasonal_init, trend_init):
        # seasonal_init, trend_init: [B, L, C]
        B, L, C = seasonal_init.shape

        # --- 季节分支 (轻量 CNN) ---
        s = seasonal_init.permute(0, 2, 1)        # [B, C, L]
        t = trend_init.permute(0, 2, 1)           # [B, C, L]
        s_out = self.seasonal_stream(s)           # [B, C, pred_len]

        # --- 趋势分支 ---
        # x_tre_out = self.trend_stream(t)        # [B, C, pred_len]
        # t_out = x_tre_out * self.tre_w.view(1, -1, 1)  # [B, C, pred_len]

        # # --- 分支融合 ---
        # feat = torch.cat([s_out, t_out], dim=2)        # [B, C, 2*pred_len]
        # out = self.fc_fuse(feat)                       # [B, c, pred_len]
        out = s_out.permute(0, 2, 1)                   # [B, pred_len, c]

        return out
