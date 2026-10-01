import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TaylorPerturbMN(nn.Module):
    """
    Input:  x_mn: (B, M, N)
    Output: x_aug: (B, M, N)

    Supports specifying order at forward time:
      - order=0: no perturbation
      - order=1..max_order: use 1st..k-th finite differences
    Taylor residual:
      r = sum_{k=1..order} tanh(c_k) * (dt^k / k!) * d_k
    where d_k is k-th finite difference along M (time-extended axis).
    """
    def __init__(
        self,
        n_vars: int,
        max_order: int = 4,            # prepare parameters up to this order (1..4)
        dt_scale: float = 1.0,
        base_dt: float = 0.2,          # e.g., (L-1)/(M-1) if you upsample time
        apply_last_k: int | None = None,
        smooth_before_diff: bool = False,
        ma_kernel: int = 1,
        dropout = 0.3,
    ):
        super().__init__()
        assert max_order in (0,1, 2, 3, 4), "max_order must be in {0,1,2,3,4}"
        self.max_order = max_order
        self.dt_scale = float(dt_scale)
        self.base_dt = base_dt
        self.apply_last_k = apply_last_k
        self.smooth_before_diff = smooth_before_diff

        # coefficients c_1..c_max_order, each is (1,1,N)
        self.coeffs = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, n_vars)) for _ in range(max_order)
        ])

        # learnable strength (0, dt_scale * base_dt)
        self.logit_dt = nn.Parameter(torch.tensor(-2.0))
        self.drop = nn.Dropout(dropout)

        if smooth_before_diff:
            assert ma_kernel >= 1 and ma_kernel % 2 == 1
            self.ma_kernel = ma_kernel
            self.avg = nn.AvgPool1d(kernel_size=ma_kernel, stride=1, padding=0)

        # factorial buffer for 0..4 (we only use 1..4)
        fact = [1.0, 1.0, 2.0, 6.0, 24.0]
        self.register_buffer("_fact", torch.tensor(fact, dtype=torch.float32), persistent=False)

    @staticmethod
    def _diff_pad_zero(x: torch.Tensor) -> torch.Tensor:
        # x: (B,M,N)
        dx = x[:, 1:, :] - x[:, :-1, :]
        return torch.cat([torch.zeros_like(dx[:, :1, :]), dx], dim=1)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        # smooth along M axis
        x_t = x.transpose(1, 2)  # (B,N,M)
        pad = (self.ma_kernel - 1) // 2
        x_t = F.pad(x_t, (pad, pad), mode="replicate")
        return self.avg(x_t).transpose(1, 2)  # (B,M,N)

    def forward(self, x_mn: torch.Tensor, order: int | None = None) -> torch.Tensor:
        """
        order: 0..max_order. If None, defaults to max_order.
        """
        if order is None:
            order = self.max_order
        assert isinstance(order, int) and 0 <= order <= 4
        assert order <= self.max_order, f"order={order} exceeds max_order={self.max_order}"

        if order == 0:
            return x_mn

        B, M, N = x_mn.shape
        x0 = self._smooth(x_mn) if self.smooth_before_diff else x_mn

        # dt: base_dt * learned_strength
        learned = torch.sigmoid(self.logit_dt) * self.dt_scale
        base = x_mn.new_tensor(self.base_dt) if (self.base_dt is not None) else x_mn.new_tensor(1.0)
        dt = base * learned  # scalar

        # build residual r = sum_{k=1..order} c_k * (dt^k / k!) * d_k
        r = 0.0
        d = x0
        dt_pow = dt  # dt^1

        for k in range(1, order + 1):
            d = self._diff_pad_zero(d)                    # d becomes k-th difference
            ck = torch.tanh(self.coeffs[k - 1])          # (1,1,N) bounded
            scale = dt_pow / self._fact[k].to(dt_pow)    # dt^k / k!
            r = r + ck * (scale * d)                     # broadcast to (B,M,N)
            dt_pow = dt_pow * dt                         # next power

        r = self.drop(r)

        if self.apply_last_k is not None and self.apply_last_k > 0:
            K = min(self.apply_last_k, M)
            out = x_mn.clone()
            out[:, -K:, :] = out[:, -K:, :] + r[:, -K:, :]
            return out

        return x_mn + r
