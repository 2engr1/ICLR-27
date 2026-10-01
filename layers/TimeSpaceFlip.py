import torch
import torch.nn as nn
import torch.nn.functional as F
# from typing import Tuple, Optional

class TimeQueryWithVarQ(nn.Module):
    """
    Merge:
      - LearnedPositionalEmbedding (learned abs PE)
      - TimeQueryBuilder (BLN -> q0)
      - QueryMixerWithVarQ (q0 + Qv -> q_time, beta)

    x_bln:    (B, L, N)
    Qv_bnhd:  (B, N, H, d)   where D = H*d

    return:
      q_time: (B, L, D)
      beta:   (B, H, L, N)
    """
    def __init__(
        self,
        num_vars: int,
        d_model: int,
        n_heads: int,
        max_len: int,
        dropout: float = 0.1,
        use_tcn: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.num_vars = num_vars
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.use_tcn = use_tcn

        # 1) BLN -> BLD
        self.in_proj = nn.Linear(num_vars, d_model, bias=True)

        # 2) learned absolute positional embedding: (1, max_len, D)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

        # 3) optional local temporal enhancement (depthwise + pointwise conv on L)
        if use_tcn:
            self.dwconv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
            self.pwconv = nn.Conv1d(d_model, d_model, kernel_size=1)

        # 4) mixer: ctx_proj + gate
        self.ctx_proj = nn.Linear(d_model, d_model, bias=True)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        self.drop_out = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)  # 给 q_time 用（最终）

    def forward(self,x_bln, Qv_bnhd):
        # (B,L,N) Qv_bnhd: torch.Tensor,   # (B,N,H,d)
        B, L, N = x_bln.shape
        assert N == self.num_vars, f"num_vars mismatch: got {N}, expect {self.num_vars}"

        _, N2, H, d = Qv_bnhd.shape
        assert N2 == N and H == self.n_heads and d == self.d_head, "Qv shape mismatch"

        # ---- A) build q0: (B,L,D)
        q0 = self.in_proj(x_bln)                 # (B,L,D)
        q0 = q0 + self.pe[:, :L, :]              # add learned PE

        if self.use_tcn:
            y = q0.transpose(1, 2)               # (B,D,L)
            y = self.pwconv(self.dwconv(y))      # (B,D,L)
            q0 = q0 + y.transpose(1, 2)          # (B,L,D)

     
        # ---- B) route with Qv to get beta and ctx
        q0h = q0.view(B, L, H, d)  # (B,L,H,d)

        # beta_scores: (B,H,L,N)
        beta_scores = torch.einsum("blhd,bnhd->bhln", q0h, Qv_bnhd) * self.scale
        beta = torch.softmax(beta_scores, dim=-1)

        # ctx: (B,L,D)
        ctx = torch.einsum("bhln,bnhd->blhd", beta, Qv_bnhd).contiguous().view(B, L, self.d_model)

        # ---- C) gated fusion -> q_time
        ctx_p = self.ctx_proj(ctx)
        gate = torch.sigmoid(self.gate_proj(ctx))
        q_time = q0 + gate * ctx_p

        q_time = self.norm_out(self.drop_out(q_time))


        return q_time, beta


    
class VarQueryWithTimeQ(nn.Module):
    """
    Flipped version of TimeQueryWithVarQ:
    x_bnl:     (B, N, L)   # var tokens, time as feature
    Q_blhd:    (B, L, H, d)  # time-side projected Q (e.g., last_q from time-attn)
    return:
      q_var:   (B, N, D)
      beta:    (B, H, N, L)  # var->time routing (softmax over L)
    """
    def __init__(self, seq_len: int, num_vars: int, d_model: int, n_heads: int, use_norm: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.seq_len = seq_len
        self.num_vars = num_vars
        self.d_model = d_model
        self.n_heads = n_heads
        self.topk = 3
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        
        self.use_norm=use_norm

        # (B,N,L) -> (B,N,D): each var token summarizes its length-L history
        self.in_proj = nn.Linear(seq_len, d_model, bias=True)

        # self.x_norm = nn.LayerNorm(self.seq_len)
        # self.x_norm = RMSNorm(self.seq_len, eps=1e-5)
        self.x_norm = nn.Identity()
        
        self.x_gate = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len, bias=True),
            nn.GELU(),
            nn.Linear(self.seq_len, self.seq_len, bias=True),
        )
        
        self.out_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        
        self.direct_proj = nn.Linear(self.seq_len, self.d_model, bias=True)
    def forward(self, x_bnl, Q_blhd, K_blhd, V_blhd):
        """
        x_bnl:  (B,N,L)
        Q_blhd: (B,L,H,d)
        """
        B, N, L = x_bnl.shape
        assert N == self.num_vars and L == self.seq_len, f"x_bnl mismatch: got (N={N}, L={L})"
        B2, L2, H, d = Q_blhd.shape
        assert B2 == B and L2 == L and H == self.n_heads and d == self.d_head, "Q_blhd shape mismatch"
        
        
        if self.use_norm:
            x_bnl = self.x_norm(x_bnl)
        xw = self.x_gate(x_bnl)
        
        q0_var = torch.einsum("bnl,blhd->bnhd", xw, Q_blhd)

        beta_scores = torch.einsum("bnhd,blhd->bhnl", q0_var, K_blhd).contiguous() * self.scale
        
        
        beta = torch.softmax(beta_scores, dim=-1)
        var_bnd = torch.einsum("bhnl,blhd->bnhd", beta, V_blhd).contiguous()
        
        var_bnd=var_bnd.view(B,N,self.d_model)
        

        

        return var_bnd, beta


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight