import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from layers.decomp import DECOMP
from layers.network import Network
from layers.network_sea import Network_Sea
from layers.network_tre import Network_Tre
from layers.revin import RevIN
from layers.SeriesMix import MultiScaleTrendMixing

class CSM(nn.Module):
    def __init__(self, C, L, dropout_rate, heads=3, hidden=256):
        super().__init__()

        self.C = C         # w^T x_mean  → g_c
        self.heads = heads
        self.extract_com = nn.ModuleList([
            nn.Sequential(
                nn.Linear(C, C),
                nn.LayerNorm(C),
                nn.Dropout(dropout_rate),
                nn.LeakyReLU(),
                nn.Linear(C, 1)
            ) for _ in range(self.heads)
        ])
        self.head_fusion = nn.Linear(self.heads, 1)
        # self.extract_com = nn.Sequential(
        #     nn.Linear(C, C),
        #     nn.LeakyReLU(),
        #     nn.Linear(C, 1)
        # )
        self.ffn_com  = nn.Sequential(
            nn.Linear(L, hidden), nn.GELU(),
            nn.Linear(hidden, L)
        )
        self.ffn_sp   = nn.Sequential(
            nn.Linear(L, hidden), nn.GELU(),
            nn.Linear(hidden, L)
        )

    def forward(self, x):                         # x [B,T,C]
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')
        magnitude = torch.abs(x_freq)
        # com_freq = self.extract_com(magnitude)
        head_outputs = [head(magnitude) for head in self.extract_com] 

        com_freq = self.head_fusion(torch.cat(head_outputs, dim=-1))
        x_com = torch.fft.irfft(com_freq, dim=1, norm='ortho')
        com_out = self.ffn_com(x_com.permute(0,2,1)).permute(0,2,1)

        # FFN on common & specific   
        com_out = com_out.expand_as(x)
        sp_out = x - com_out
        sp_out = self.ffn_sp(sp_out.permute(0, 2, 1)).permute(0, 2, 1)

        out = com_out + sp_out
        # print("是否所有元素都大于0：", (out > 0).all())
        return out        # [B,T,C]

# ================= 主模型 =================
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 参数
        seq_len   = configs.seq_len
        pred_len  = configs.pred_len
        c_in      = configs.enc_in
        patch_len = configs.patch_len
        stride    = configs.stride
        padding_patch = configs.padding_patch
        alpha = configs.alpha
        self.ma_type = configs.ma_type
        self.use_mix = configs.use_mix
        self.hidden_size = configs.hidden_size
        self.heads = configs.heads
        # 归一化
        self.revin        = configs.revin
        self.revin_layer  = RevIN(c_in, affine=True, subtract_last=False)
        self.ds_window = configs.ds_window
        self.ds_layers = configs.ds_layers
        self.down_pool = nn.AvgPool1d(kernel_size=self.ds_window)
        self.ms_mixing = MultiScaleTrendMixing(seq_len, pred_len, c_in, self.ds_layers, self.ds_window)
        # 拆解模块 (搬掉原 decomp，只保留 spectral filter)
        self.mask_matrix = nn.Parameter(torch.ones(int(seq_len / 2) + 1, c_in))
        self.freq_linear = nn.Linear(int(seq_len / 2) + 1, int(pred_len / 2) + 1).to(torch.cfloat)
        # 剩余网络
        self.decomp = DECOMP(configs.ma_type,
                             configs.alpha, configs.beta,
                             configs.std_kernel, c_in)

        self.use_csm = configs.use_csm
        self.csm = CSM(c_in, seq_len, dropout_rate=configs.dropout_rate, heads = self.heads, hidden = self.hidden_size)

        self.net = Network(c_in, seq_len, pred_len, configs.ds_window, configs.ds_layers, configs.initial_tre_w)
        self.net_s = Network_Sea(c_in, seq_len, pred_len, configs.ds_window, configs.ds_layers, configs.initial_tre_w)
        self.net_t = Network_Tre(c_in, seq_len, pred_len, configs.ds_window, configs.ds_layers, configs.initial_tre_w)
        self.ablation_net = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [B, T, C]
        if self.revin:
            x = self.revin_layer(x, 'norm')
        B, T, C = x.shape[0], x.shape[1], x.shape[2]

        if self.use_mix:
            x_in = x.permute(0, 2, 1)         #[B, C, T]                       
            ms_list = []
            ms_list.append(x_in.permute(0, 2, 1))     #[B, T, C] 
            x_ms = x_in                       #[B, C, T] 
            for _ in range(self.ds_layers):
                x_sampling = self.down_pool(x_ms)   
                ms_list.append(x_sampling.permute(0,2,1))  
                x_ms = x_sampling

            x_ms = self.ms_mixing(ms_list)  
            x_out = x_ms[0] 
            x = x_out
        if self.use_csm:
            x = self.csm(x)
        #x = self.ablation_net(x.permute(0,2,1)).permute(0,2,1)
        #传入后续网络
        if self.ma_type == 'reg':
            x = self.net(x, x)
            #x = self.net_s(x, x)
            #x = self.net_t(x, x)
        else:
            s, t = self.decomp(x)
            x = self.net(s, t)

        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x
