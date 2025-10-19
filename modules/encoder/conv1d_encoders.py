from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..distributions import DiagonalGaussianDistribution

# 基础块
class ConvBlock1d(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, groups=8, act=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, s, p)
        self.norm = nn.GroupNorm(num_groups=min(groups, cout), num_channels=cout)
        self.act = act()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class Conv1dEncoder(nn.Module):
    """
    输入:  x_BTC ∈ ℝ^{B×T×C} (C = 3J)，内部转为 [B, C, T]
    输出:  DiagonalGaussianDistribution，参数形状 [B, 2*z, T_down, 1]
    - 下采样倍率由 len(channels) 决定（每层 stride=2）
    - 会自动把 T pad 到 2^n 的倍数，并把 pad_len 挂到 posterior.pad_len
    """
    def __init__(
        self,
        in_channels: int,         # C = 3J
        z_channels: int = 64,
        channels: List[int] = [128, 256, 256],
        kernel_size: int = 3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_down = len(channels)
        self.multiple = 2 ** self.n_down

        layers = []
        cprev = in_channels
        for c in channels:
            layers.append(ConvBlock1d(cprev, c, k=kernel_size, s=2, p=kernel_size // 2))
            cprev = c
        self.backbone = nn.Sequential(*layers)
        self.to_params = nn.Conv1d(cprev, 2 * z_channels, kernel_size=1)

    def _pad_time(self, x_bct: torch.Tensor) -> Tuple[torch.Tensor, int]:
        T = x_bct.shape[-1]
        m = self.multiple
        if T % m == 0:
            return x_bct, 0
        need = m - (T % m)
        x_pad = F.pad(x_bct, (0, need), mode="constant", value=0.0)
        return x_pad, need

    def forward(self, x_btc: torch.Tensor) -> DiagonalGaussianDistribution:
        # [B, T, C] -> [B, C, T]
        x_bct = x_btc.transpose(1, 2).contiguous()
        x_pad, pad_len = self._pad_time(x_bct) #保证 x_pad % multiple == 0,

        h = self.backbone(x_pad)            # [B, Cenc, T_down]
        params = self.to_params(h)          # [B, 2*z, T_down]
        posterior = DiagonalGaussianDistribution(parameters=params)  # -> [B, 2*z, T_down, 1]
        posterior.pad_len = pad_len         # 记录 pad，用于解码端还原长度
        print(posterior.pad_len)
        return posterior

if __name__ == "__main__":
    B, T, J = 8, 120, 16
    x = torch.randn(B, T, 3 * J)                 # x_BTC
    enc = Conv1dEncoder(in_channels=3 * J, z_channels=64)
    qz = enc(x)                                  # DiagonalGaussianDistribution over [B, z, T_down, 1]
    z = qz.sample()                              # 采样 z
    kl = qz.kl().mean()