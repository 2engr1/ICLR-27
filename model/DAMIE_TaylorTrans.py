import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.TaylorTrans_Enc import InvEncoder, ATTNEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, DataEmbedding
import numpy as np
import math
from layers.Invertible import RevIN



class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs=configs
        self.mamba_d_conv = 4
        self.mamba_expand = 2
        self.taylor_order = configs.order
        self.M_len = configs.M_len
        
        if configs.enc_in < 50:
            self.detach_qkv= True
        else:
            self.detach_qkv=False
    


        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, 
                                                    configs.d_model, 
                                                    configs.embed, 
                                                    configs.freq,
                                                    configs.dropout)

        self.encoder = InvEncoder(
            [
                ATTNEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=True), configs.d_model, configs.n_heads,detach_qkv=self.detach_qkv),
                    configs,
                    M_len=self.M_len,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            Linear = nn.Linear(self.M_len, configs.seq_len),
            # Linear = nn.Linear(configs.d_model, configs.seq_len),
        )

        self.Linear = nn.Linear(self.M_len, configs.pred_len)


        self.rev = RevIN(configs.enc_in)
        self.MLinear = nn.ModuleList([
            nn.Linear(configs.pred_len, configs.pred_len) for _ in range(configs.enc_in)
        ]) 

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
    # B: batch_size;    E: d_model; 
    # L: seq_len;       S: pred_len;
    # N: number of variate (tokens), can also includes covariates
    
        x_enc = self.rev(x_enc, 'norm')
        
         # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_x = self.enc_embedding(x_enc,x_mark=None) # covariates (e.g timestamp) can be also embedded as tokens
        
        # print(enc_x.shape)
        
        dec_out,enc_out=self.encoder(enc_x, x_enc)
        
        # pred = self.projector(dec_out).permute(0, 2, 1)
        
        pred = self.Linear(dec_out.permute(0,2,1)).permute(0,2,1)
        
                
        pre =self.rev(pred, 'denorm')
       
        
       
        return pre,None
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.configs.output_attention:
            return dec_out[:, -self.configs.pred_len:, :], attns
        else:
            return dec_out[:, -self.configs.pred_len:, :]  # [B, L, D]




