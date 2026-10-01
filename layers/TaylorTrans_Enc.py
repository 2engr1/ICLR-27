import concurrent.futures
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from math import sqrt
import math
from layers.TimeSpaceFlip import TimeQueryWithVarQ, VarQueryWithTimeQ

from layers.Taylorder import TaylorPerturbMN



class ND2MN(nn.Module):
    """
    x: (B, N, D)  ->  y: (B, M, N)
    interpret: each variable embedding (D) -> M mode coefficients
    """
    def __init__(self, d_model: int, m_modes: int, dropout: float = 0.1, use_norm: bool = True):
        super().__init__()
        self.proj = nn.Linear(d_model, m_modes, bias=False)  # shared across N
        self.use_norm = use_norm
        self.norm = nn.LayerNorm(d_model) if use_norm else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x_bnd):
        # x_bnd: (B, N, D)
        # x = self.norm(x_bnd)
        # x = self.drop(x)
        x=x_bnd
        y_bnm = self.proj(x)          # (B, N, M)
        y_bmn = y_bnm.transpose(1, 2) # (B, M, N)
        return y_bmn



class LowRankVarInteraction(nn.Module):
    """
    Input/Output: (B, N, D)
    Low-rank latent bottleneck interaction without attention.
    """
    def __init__(self, d_model, rank=16, dropout=0.1):
        super().__init__()
        self.rank = int(d_model/4)
        self.to_latent = nn.Linear(d_model, rank, bias=False)   # (B,N,D)->(B,N,r)
        self.act  = F.relu
        self.to_feat   = nn.Linear(rank, d_model, bias=False)   # (B,N,r)->(B,N,D)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_bnd):
        x=x_bnd
        A = self.to_latent(x)
        g = A.mean(dim=1, keepdim=True)
        A2 = A + g
        A2 =self.act(A2)
        # # Back to D
        delta = self.to_feat(A2)  # (B,N,D)

        return x_bnd + self.dropout(delta)


class LowRankTokenMix(nn.Module):
    def __init__(self, num_vars, d_model, dropout=0.1):
        super().__init__()
        self.rank = int(d_model/2)
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(num_vars, self.rank, bias=False)
        self.act  = F.relu
        self.up   = nn.Linear(self.rank, num_vars, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # (B,N,D)
        y = x.transpose(1,2)      # (B,D,N)
        y = self.up(self.act(self.down(y)))  # (B,D,N)
        y = y.transpose(1,2)                 # (B,N,D)
        return x + self.dropout(y)


class ATTNEncoderLayer(nn.Module):
    def __init__(self, attention, configs, M_len, activation="relu"):
        super(ATTNEncoderLayer, self).__init__()
        
        d_ff = configs.d_ff or 4 * configs.d_model
        self.configs=configs
        
        self.d_keys = configs.d_model // configs.n_heads
        self.d_values = configs.d_model // configs.n_heads
        
        self.attention = attention
        self.L1 = nn.Linear(configs.enc_in, configs.d_model)
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        
        self.conv1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1)

        self.dropout = nn.Dropout(configs.dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
        self.var_q = VarQueryWithTimeQ(num_vars=configs.enc_in,
                                        d_model=configs.d_model,
                                        n_heads=configs.n_heads,
                                        dropout=configs.dropout,
                                        seq_len=configs.seq_len,
                                        use_norm=configs.use_norm)
        self.mn_proj= ND2MN(d_model=configs.d_model, 
                            m_modes=M_len, 
                            dropout=configs.dropout )
        
        self.taylor=TaylorPerturbMN(n_vars=configs.enc_in, max_order=configs.order)
        
        self.var_interact=LowRankVarInteraction(configs.d_model)
        
        self.var_mixer=LowRankTokenMix(num_vars=configs.enc_in, d_model=configs.d_model,dropout=configs.dropout)
            
    def forward(self,enc_x, x, attn_mask=None, tau=None, delta=None):
        x_BLN=x
        x_BLD = self.L1(x_BLN)
        B,L,N = x_BLN.shape
        if self.configs.use_norm:
            x_BLD =self.norm1(x_BLD)
        
        out, attn = self.attention(
            x_BLD, x_BLD, x_BLD,
            attn_mask=attn_mask, tau=tau, delta=delta
        )
                
        Q_blhd, K_blhd, V_blhd = self.attention.last_q, self.attention.last_k, self.attention.last_v
        x_BNL= x.permute(0,2,1)
        var_bnd, beta=self.var_q(x_BNL, Q_blhd, K_blhd, V_blhd)

        if self.configs.use_mixer:
            s_bnd = self.var_mixer(var_bnd)
            s_bnd = self.norm1(enc_x+self.dropout(s_bnd))
        else:
            s_bnd = self.var_interact(var_bnd)
            s_bnd = self.norm1(enc_x+self.dropout(s_bnd))
        
        s_BND = self.dropout(self.activation(self.conv1(s_bnd.transpose(-1, 1))))
        s_BND = self.dropout(self.conv2(s_BND).transpose(-1, 1))
        s_BND = self.norm2(s_bnd+s_BND)
        # s_BND = self.taylor(s_BND.permute(0,2,1)).permute(0,2,1)
        dec_x_bmn=self.mn_proj(s_BND)
        dec_x_bmn = self.taylor(dec_x_bmn)

        return dec_x_bmn, s_BND


class InvEncoder(nn.Module):
    """
    [B,N,D]
    """
    def __init__(self, layers, norm_layer=None, Linear=None,  output_attention=False):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.Linear=Linear
        self.output_attention = output_attention


    def forward(self, enc_x, x, attn_mask=None, tau=None, delta=None):
        """
        x: [B,L,N]
        """
        enc_x_raw = enc_x
        x_re=x
        x_bln = None
        attns = [] if self.output_attention else None

        for layer in self.layers:
            
            pred, enc_x_raw = layer(enc_x_raw,x_re, attn_mask=attn_mask, tau=tau, delta=delta)
            x_re = self.Linear(pred.permute(0,2,1)).permute(0,2,1)
            # x_re = self.Linear(pred).permute(0,2,1)
            enc_x_raw=self.norm(enc_x_raw)

        return pred,x_bln



