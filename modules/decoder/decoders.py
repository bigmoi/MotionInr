from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from Utils.utils import instantiate_from_config

class UpBlock1d(nn.Module):
    def __init__(self, cin, cout, k=4, s=2, p=1, groups=8, act=nn.SiLU):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(cin, cout, k, s, p)
        self.norm = nn.GroupNorm(num_groups=min(groups, cout), num_channels=cout)
        self.act = act()
    def forward(self, x):
        return self.act(self.norm(self.deconv(x)))

class Conv1dDecoder(nn.Module):
    """
    输入:  z ∈ ℝ^{B×Cz×T_down[,×1]}
    输出:  x̂_BTC ∈ ℝ^{B×T×Cout}
    - 逐层上采样 (stride=2)
    - 最后强制对齐到 target_T（裁剪或线性插值）
    """
    def __init__(
        self,
        z_channels: int,
        out_channels: int,             # Cout = 3J
        channels: List[int] = [256, 128, 128],
        kernel_size: int = 3
    ):
        super().__init__()
        self.z_channels = z_channels
        self.out_channels = out_channels
        self.channels = channels

        layers = []
        cprev = z_channels
        for c in channels:
            layers.append(UpBlock1d(cprev, c, k=kernel_size+1, s=2, p=kernel_size//2))
            cprev = c
        self.backbone = nn.Sequential(*layers)
        self.to_out = nn.Conv1d(cprev, out_channels, kernel_size=1)

    def forward(self, z, target_T: int):
        if z.dim() == 4:                  # [B, Cz, T_down, 1] -> [B, Cz, T_down]
            z = z.squeeze(-1)
        h = self.backbone(z)              # [B, Cdec, T_up]
        x_bct = self.to_out(h)            # [B, Cout, T']
        # 对齐目标长度
        if x_bct.size(-1) > target_T:
            x_bct = x_bct[..., :target_T]
        elif x_bct.size(-1) < target_T:
            x_bct = F.interpolate(x_bct, size=target_T, mode="linear", align_corners=False)
        x_btc = x_bct.transpose(1, 2).contiguous()  # -> [B, T, Cout]
        return x_btc


# ---- 一些更新策略（可按需扩展）----
def _expand_groups(x_bg, out_dim):
    # x_bg: [B, in_dim, g]
    g = x_bg.shape[-1]
    repeat_k = out_dim // g
    return x_bg.repeat_interleave(repeat_k, dim=2)  # [B, in_dim, out_dim]

def upd_normalize(w_bim, x_bg, out_dim):
    x_exp = torch.tanh(_expand_groups(x_bg, out_dim))          # [-1,1]
    w_new = w_bim * (1.0 + x_exp)                              # 缩放
    return F.normalize(w_new, dim=1)

def upd_scale(w_bim, x_bg, out_dim):
    x_exp = torch.sigmoid(_expand_groups(x_bg, out_dim)) * 2   # (0,2)
    return w_bim * x_exp

def upd_residual(w_bim, x_bg, out_dim):
    x_exp = torch.tanh(_expand_groups(x_bg, out_dim))
    return w_bim + x_exp

def upd_affine(w_bim, x_bg, out_dim, a=1.0, b=0.0):
    x_exp = _expand_groups(x_bg, out_dim)
    return a * w_bim + b * x_exp

update_strategies = {
    "normalize": upd_normalize,
    "scale": upd_scale,
    "residual": upd_residual,
    "affine": upd_affine,
}

class TransInr1D(nn.Module):
    """
    适配 1D 序列（人体动作）的 HD 超网络：
    - tokenizer(data) -> dtokens: [B, Nd, dim]
    - wtokens: [Nw, dim] （全局可学），按层-分组分配
    - transformer 输出 trans_out_w: [B, Nw, dim]
    - 对每层 param 取出其 g 个 token，经 Linear(dim->in_dim) 得到行门控 x: [B, in_dim, g]
    - 在列维按组复制 -> [B, in_dim, out_dim]，用选择的 update_strategy 调制母版权重
    - set_params -> inr(coord)；coord 支持 [T,1] 形式
    """
    def __init__(self, tokenizer, inr, n_groups, data_shape, transformer,
                 update_strategy="normalize", mod_bias=False):
        super().__init__()
        dim = transformer['params']['dim']

        # self.tokenizer   = instantiate_from_config(tokenizer)

        self.inr         = instantiate_from_config(inr)
        self.transformer = instantiate_from_config(transformer)

        # 1D 坐标网格（假设 data_shape = (T,) 或 int T）
        if isinstance(data_shape, int):
            T = data_shape
        else:
            T = data_shape[0]
        # [-1,1] 归一化时间坐标，形状 [T,1]
        t = torch.linspace(-1.0, 1.0, steps=T)
        self.register_buffer('shared_coord', t[:, None], persistent=False)

        # 基底权重（母版权重 + bias 行）
        self.base_params = nn.ParameterDict()
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = {}
        self.mod_bias = mod_bias

        n_wtokens = 0
        for name, shape in self.inr.param_shapes.items():
            # shape = [in_dim + 1, out_dim]
            in_dim  = shape[0] - 1
            out_dim = shape[1]

            # 母版初始化（你的 inr.init_wb 决定分布/尺度）
            self.base_params[name] = nn.Parameter(self.inr.init_wb(shape, name=name))

            # 分组设置：按列(out_dim)分 g 组
            g = min(n_groups, out_dim)
            assert out_dim % g == 0, f"{name}: out_dim={out_dim} 不能被 g={g} 整除"
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g

            # 每组 token -> 行门控（in_dim）
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, in_dim)
            )

            # 可选：bias 的分组门控（每组一个标量）
            if self.mod_bias:
                setattr(self, f"{name}_bias_map", nn.Sequential(
                    nn.LayerNorm(dim), nn.Linear(dim, 1)
                ))

        # 全局权重 token（可学习）
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))

        # 更新策略函数
        assert update_strategy in update_strategies, f"unknown strategy: {update_strategy}"
        self.update_strategy = update_strategies[update_strategy]

        # 参数统计（可选）
        nparams = sum(p.numel() for p in self.transformer.parameters())
        # nparams += sum(p.numel() for p in self.tokenizer.parameters())
        nparams += sum(p.numel() for p in self.base_params.values())
        nparams += self.wtokens.numel()
        print(f"[TransInr1D] Hypernetwork Params: {nparams/1e6:.2f}M, Nw={n_wtokens}")

    def forward(self, data=None, nsamples=None, coord=None, **kwargs):
        # 1) 数据 token
        # dtokens = self.tokenizer(data, nsamples, **kwargs)   # [B, Nd, dim]
        dtokens=data
        B = dtokens.shape[0]

        # 2) 权重 token
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)  # [B, Nw, dim]

        # 3) Transformer 交互
        if self.transformer.__class__.__name__ == 'TransformerEncoder':
            tout = self.transformer(torch.cat([dtokens, wtokens], dim=1))  # [B, Nd+Nw, dim]
            tout = tout[:, -self.wtokens.shape[0]:, :]                     # 只取 Nw 段（关键修正）
        elif self.transformer.__class__.__name__ == 'Transformer':
            tout = self.transformer(src=dtokens, tgt=wtokens)              # [B, Nw, dim]
        else:
            raise ValueError(f"Unsupported transformer type: {self.transformer.__class__.__name__}")

        # 4) 逐层生成权重
        params = {}
        for name, shape in self.inr.param_shapes.items():
            in_dim  = shape[0] - 1
            out_dim = shape[1]

            # 母版权重
            wb0 = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            w, b = wb0[:, :-1, :], wb0[:, -1:, :]                          # w:[B,in_dim,out_dim], b:[B,1,out_dim]

            # 对应本层的 g 个权重 token
            l, r = self.wtoken_rng[name]
            tw = tout[:, l:r, :]                                           # [B, g, dim]
            x = self.wtoken_postfc[name](tw)                               # [B, g, in_dim]
            x = x.transpose(-1, -2).contiguous()                           # [B, in_dim, g]

            # 调制权重（在列维按组复制）
            w = self.update_strategy(w, x, out_dim)

            # 可选：bias 调制（每组 -> 广播到 out_dim）
            if self.mod_bias:
                b_map = getattr(self, f"{name}_bias_map")(tw)              # [B, g, 1]
                b_map = b_map.transpose(-1, -2)                            # [B, 1, g]
                b_map = b_map.repeat_interleave(out_dim // b_map.shape[-1], dim=2)  # [B,1,out_dim]
                b = b + b_map

            wb = torch.cat([w, b], dim=1)                                  # [B, in_dim+1, out_dim]
            params[name] = wb

        # 5) 注入到 INR
        self.inr.set_params(params)

        # 6) 坐标（支持 1D）
        if coord is None:
            coord = self.shared_coord  # [T, 1]
        if coord.dim() == 2:           # [T, 1] -> [B, T, 1]
            coord = einops.repeat(coord, 't d -> b t d', b=B)
        elif coord.dim() == 3 and coord.shape[0] != B:
            # 允许提前构造好的 [B, T, 1]
            pass


        # 7) INR渲染
        pred = self.inr(coord)

        # # 8) 返回布局：对 1D 序列，推荐 [B, T, C]（不要 permute）
        # if pred.dim() == 4:      # (B, H, W, C)->(B, C, H, W)
        #     pred = pred.permute(0, -1, 1, 2).contiguous()
        # elif pred.dim() == 5:    # (B, X, Y, Z, C)->(B, C, X, Y, Z)
        #     pred = pred.permute(0, -1, 1, 2, 3).contiguous()
        # # dim==3: [B, T, C]，保持不变
        return pred