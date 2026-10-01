import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Invertible import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.individual = False
        self.Linear = nn.ModuleList([
            nn.Linear(configs.seq_len, configs.pred_len) for _ in range(configs.channel)
        ]) if self.individual else nn.Linear(configs.seq_len, configs.pred_len)
        
        self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(configs.channel) 
        

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y,a,b):
        # x: [B, L, D]
        x = self.rev(x, 'norm') if self.rev else x
        x = self.dropout(x)
      
        pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') 

        return pred