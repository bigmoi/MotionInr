import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from utils import reparam, deactivate
from modules.transformerdecoder.transformer import Attention, LocalAttention
import einops

class PoseLatentTokenizer(nn.Module):
    """
    Tokenize 姿态潜在（支持 1D: [B,C,T]；或 2D: [B,C,T,J]）为 [B, N, D] 的 token 序列。
    - 使用 2D unfold 在 (T,J) 上打 patch；1D 情况下令 J=1 即可。
    - 先线性投影到 dim，再做 LocalAttention -> GlobalAttention。
    """
    def __init__(
        self,
        latent_dim: int,               # C_z
        latent_size,                   # int(T) 或 (T, J)
        patch_size,                    # int(Pt) 或 (Pt, Pj)
        dim: int,
        n_head: int,
        head_dim: int,
        stride=None,                   # None→与patch_size相同；或 int / (St,Sj)
        padding=0,                     # int 或 (pad_t, pad_j)
        pos_embed: str = "learned",    # "learned" | "sine"
        add_cls: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        # --- 维度归一化 ---
        if isinstance(latent_size, int):
            latent_size = (latent_size, 1)       # 1D -> (T, 1)
        if isinstance(patch_size, int):
            patch_size = (patch_size, 1)         # 1D -> (Pt, 1)
        if stride is None:
            stride = patch_size
        if isinstance(stride, int):
            stride = (stride, 1)
        if isinstance(padding, int):
            padding = (padding, 0)

        self.latent_dim = latent_dim
        self.latent_size = latent_size      # (T, J)
        self.patch_size = patch_size        # (Pt, Pj)
        self.stride = stride                # (St, Sj)
        self.padding = padding              # (pad_t, pad_j)
        self.dim = dim
        self.add_cls = add_cls
        self.pos_embed_type = pos_embed

        Pt, Pj = patch_size
        self.prefc = nn.Linear(Pt * Pj * latent_dim, dim)

        # n_patches 计算（以 init 时的 latent_size 为准）
        T, J = latent_size
        St, Sj = stride
        pad_t, pad_j = padding
        Tpad, Jpad = T + 2 * pad_t, J + 2 * pad_j
        n_pt = (Tpad - Pt) // St + 1
        n_pj = (Jpad - Pj) // Sj + 1
        n_patches = max(0, n_pt) * max(0, n_pj)

        # 位置编码
        if self.pos_embed_type == "learned":
            self.posemb = nn.Parameter(torch.randn(1, n_patches + (1 if add_cls else 0), dim))
        else:
            self.register_buffer("posemb", None, persistent=False)  # 正弦位置在前向中动态生成

        # 注意力层
        # 按你的原模式，local 的 window_size 用 patch_size[0]（你也可以改成 n_pj 或者 n_pt*n_pj 的某个值）
        self.local_attn = LocalAttention(dim, window_size=max(1, patch_size[0]), n_head=n_head, head_dim=head_dim, dropout=dropout)
        self.global_attn = Attention(dim, n_head=n_head, head_dim=head_dim, dropout=dropout)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if add_cls else None

    def _sinusoidal_pe(self, n, dim, device):
        pe = torch.zeros(n, dim, device=device)
        pos = torch.arange(0, n, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # [n, dim]

    def _ensure_NCHW(self, x):
        """
        接受:
          - [B, C, T]   -> 转成 [B, C, T, 1]
          - [B, T, C]   -> 转成 [B, C, T, 1]
          - [B, C, T, J] 保持
          - [B, T, J, C] -> 转成 [B, C, T, J]
        最终返回 [B, C, T, J]
        """
        if x.dim() == 3:
            B, A, B_or_C = x.shape
            # 猜测 [B, C, T] 或 [B, T, C]
            if A == self.latent_dim:     # [B, C, T]
                x = x.unsqueeze(-1)      # [B, C, T, 1]
            else:                        # [B, T, C]
                x = x.permute(0, 2, 1).unsqueeze(-1)  # [B, C, T, 1]
        elif x.dim() == 4:
            # 可能是 [B, C, T, J] 或 [B, T, J, C]
            if x.shape[1] == self.latent_dim:
                pass  # [B, C, T, J]
            else:
                x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, T, J]
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")
        return x

    def forward(self, x):
        """
        x: [B, C, T] / [B, T, C] / [B, C, T, J] / [B, T, J, C]
        return: tokens [B, N, D]
        """
        x = self._ensure_NCHW(x)                         # [B, C, T, J]

        # 2D unfold on (T,J)
        Pt, Pj = self.patch_size
        St, Sj = self.stride
        pad_t, pad_j = self.padding
        # F.unfold 需要 4D 输入: [B, C, H, W]
        patches = F.unfold(x, kernel_size=(Pt, Pj), stride=(St, Sj), padding=(pad_t, pad_j))
        # patches: [B, C*Pt*Pj, L]
        patches = patches.transpose(1, 2).contiguous()   # [B, L, C*Pt*Pj]

        # 线性投影到 dim
        tokens = self.prefc(patches)                     # [B, L, D]

        # 位置编码
        if self.pos_embed_type == "learned":
            # self.posemb: (1, N, D) 或 (1, N+1, D)（含 CLS）
            pe = self.posemb[:, :tokens.size(1) + (1 if self.add_cls else 0), :]
            tokens = tokens + pe[:, :tokens.size(1), :]
        else:
            pe = self._sinusoidal_pe(tokens.size(1), self.dim, tokens.device)  # [N, D]
            tokens = tokens + pe.unsqueeze(0)

        # 可选 CLS
        if self.add_cls:
            cls = self.cls_token.expand(tokens.size(0), -1, -1)    # [B,1,D]
            tokens = torch.cat([cls, tokens], dim=1)               # [B, 1+N, D]

        tokens = self.drop(self.norm(tokens))

        # Local -> Global
        tokens = self.local_attn(tokens)
        tokens = self.global_attn(tokens)

        return tokens  # [B, N, D] (若 add_cls=True，则 N=1+patches)