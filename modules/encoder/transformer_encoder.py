# transformer_vae_encoder.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 你之前的实现，确保在路径可 import
from ..distributions import DiagonalGaussianDistribution


def _to_BTC(x: torch.Tensor, C_last: bool = True) -> torch.Tensor:
    """将 [B,T,C] 或 [B,C,T] 统一为 [B,T,C]"""
    if x.dim() != 3:
        raise ValueError(f"Expect 3D tensor, got {x.shape}")
    if C_last:  # [B,T,C]
        return x
    else:       # [B,C,T]
        return x.transpose(1, 2).contiguous()


def sinusoidal_position_encoding(L: int, D: int, device) -> torch.Tensor:
    """[L, D] 正弦位置编码"""
    pe = torch.zeros(L, D, device=device)
    position = torch.arange(0, L, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, D, 2, device=device, dtype=torch.float32) *
                         (-math.log(10000.0) / max(1, D)))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def patchify_1d(x_btc: torch.Tensor, P: int, S: int
                ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    将 [B,T,C] 沿时间维做 1D patch：返回
      tokens: [B, L, C*P]
      token_mask: [B, L]  (True 表示“应当忽略/是填充位” 的 key_padding_mask)
      pad_right: int 右侧补零的步数
    计算方式：L = ceil((T - P)/S) + 1 (T < P 时 L=1)，pad 到 Tpad=(L-1)*S+P
    """
    B, T, C = x_btc.shape
    if P <= 0 or S <= 0:
        raise ValueError("patch_size and stride must be positive integers.")

    # 计算 L 与右侧 padding
    if T >= P:
        L = math.ceil((T - P) / S) + 1
    else:
        L = 1
    Tpad = (L - 1) * S + P
    pad_right = Tpad - T
    if pad_right > 0:
        x_btc = F.pad(x_btc, (0, 0, 0, pad_right), value=0.0)  # pad 时间维右侧

    # 使用 unfold（张量方法）切片：unfold 到 [B, C, L, P]
    x_bct = x_btc.transpose(1, 2).contiguous()           # [B,C,Tpad]
    patches = x_bct.unfold(dimension=2, size=P, step=S)  # [B,C,L,P]
    patches = patches.permute(0, 2, 1, 3).contiguous().view(B, L, C * P)  # [B,L,C*P]

    # key_padding_mask：True = 需要 mask（填充 token）
    # 对第 i 个 patch，起点 s=i*S；当 s >= T（原始长度）时，该 patch 全由 pad 构成 -> mask=True
    starts = torch.arange(L, device=x_btc.device) * S
    valid = (starts.unsqueeze(0) < T).expand(B, L)  # [B,L]
    key_padding_mask = ~valid  # True 表示“忽略/无效”

    return patches, key_padding_mask, pad_right


class TransformerVAEEncoder1D(nn.Module):
    """
    变长友好的 Transformer VAE Encoder。
    输入:
      x: [B,T,C] 或 [B,C,T]
      lengths: [B]（可选），每个样本真实长度（未 pad 前）；若提供，将覆盖自动计算的 mask
      mask: [B,T]（可选），布尔，True 表示有效时间步；与 lengths 二选一
    超参:
      d_model: token 维度
      n_layers, n_heads, dim_ff, dropout
      patch_size P, stride S   （P=1,S=1 = 逐帧）
      out_z: z_channels        （每个 token 输出 2*z 参数）
      project_in: Linear(C*P->d_model)
    输出:
      DiagonalGaussianDistribution，其 parameters 形状为 [B, 2*z, L, 1]
      同时在 posterior 挂载 meta 信息：pad_right, P, S, L
    """
    def __init__(
        self,
        in_channels: int,        # C = 3J
        z_channels: int = 64,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dim_ff: int = 1024,
        dropout: float = 0.1,
        patch_size: int = 1,
        stride: int = 1,
        input_is_BTC: bool = True,   # True: [B,T,C]；False: [B,C,T]
        pos_embed: str = "sine",     # "sine" | "none"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride
        self.input_is_BTC = input_is_BTC
        self.pos_embed = pos_embed

        self.project_in = nn.Linear(in_channels * patch_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,   # 关键：使用 [B, L, D]
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj_params = nn.Linear(d_model, 2 * z_channels)  # 每 token -> mean+logvar
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _mask_from_lengths(lengths: torch.Tensor, T_max: int) -> torch.Tensor:
        """
        从长度生成 key_padding_mask（对 MHA：True=mask / 无效）
        lengths: [B]，每个样本真实长度
        return: [B, T_max]，True=应当忽略
        """
        device = lengths.device
        idx = torch.arange(T_max, device=device).unsqueeze(0)  # [1,T]
        valid = idx < lengths.unsqueeze(1)                     # [B,T]
        return ~valid

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,   # [B]，真实长度（未 pad）
        mask: Optional[torch.Tensor] = None       # [B,T]，True=有效；与lengths二选一
    ) -> DiagonalGaussianDistribution:
        """
        返回 DiagonalGaussianDistribution，parameters 形状 [B, 2*z, L, 1]
        """
        # 统一到 [B,T,C]
        x_btc = _to_BTC(x, C_last=self.input_is_BTC)  # [B,T,C]
        B, T, C = x_btc.shape
        assert C == self.in_channels, f"in_channels={self.in_channels}, but got {C}"

        # 若给了 lengths 或 mask，优先用来构造原始时间的 key_padding_mask（供 patchify 转 token mask）
        # 否则默认都有效
        if mask is not None and lengths is not None:
            raise ValueError("Pass either 'lengths' or 'mask', not both.")
        if lengths is not None:
            # True=应当忽略
            time_kpm = self._mask_from_lengths(lengths.to(x_btc.device), T)  # [B,T], True=pad
        elif mask is not None:
            # 传入的 mask 通常 True=有效；转换为 True=pad
            time_kpm = ~mask.to(torch.bool).to(x_btc.device)                 # [B,T]
        else:
            time_kpm = torch.zeros(B, T, dtype=torch.bool, device=x_btc.device)

        # Patchify 到 tokens
        P, S = self.patch_size, self.stride
        tokens, auto_kpm, pad_right = patchify_1d(x_btc, P, S)               # tokens:[B,L,C*P], auto_kpm: True=pad
        B, L, _ = tokens.shape

        # 将时间级的 mask 聚合到 token 级（更严格：如果该 patch 起点已在 pad 区，则整个 token mask）
        # 这里使用 auto_kpm（基于起点 s>=T）已足够实用；若你希望更精细，可根据窗口覆盖比例再判定。
        token_kpm = auto_kpm  # [B,L], True=pad

        # 嵌入 + 位置
        h = self.project_in(tokens)                   # [B,L,D]
        if self.pos_embed == "sine":
            pe = sinusoidal_position_encoding(L, self.d_model, h.device)  # [L,D]
            h = h + pe.unsqueeze(0)
        h = self.dropout(self.layer_norm(h))

        # TransformerEncoder，传入 key_padding_mask（True=mask 无效）
        h = self.encoder(h, src_key_padding_mask=token_kpm)  # [B,L,D]

        # 每个 token 输出 mean+logvar
        params = self.proj_params(h)                 # [B,L,2*z]
        params = params.transpose(1, 2).unsqueeze(-1)  # [B,2*z,L,1]

        posterior = DiagonalGaussianDistribution(parameters=params, deterministic=False)

        # 附带元信息，便于解码或对齐
        posterior.pad_right = pad_right
        posterior.token_L = L
        posterior.patch_size = P
        posterior.stride = S
        posterior.time_max = T
        posterior.token_key_padding_mask = token_kpm  # [B,L]，True=pad

        return posterior
