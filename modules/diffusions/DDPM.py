# ddpm_coeff_hmp.py
import math
from typing import Optional, Dict
import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl

# ---------- 小工具：DCT / IDCT （正交规范） ----------
def dct_basis(T, K, device):
    # 返回 D ∈ R^{T×K}（DCT-II），列正交；IDCT 用 D^T
    n = torch.arange(T, device=device).float().unsqueeze(1)       # [T,1]
    k = torch.arange(K, device=device).float().unsqueeze(0)       # [1,K]
    D = torch.cos(math.pi / T * (n + 0.5) * k)                    # [T,K]
    D[:, 0] *= 1.0 / math.sqrt(2.0)
    D *= math.sqrt(2.0 / T)
    return D  # x∈R^{T} -> c = D^T x,  x = D c

def dct_forward(x_btc, K):
    # x: [B,T,C] -> coeff: [B,K,C]
    B, T, C = x_btc.shape
    D = dct_basis(T, K, x_btc.device)            # [T,K]
    c = torch.einsum("tk,btc->bkc", D, x_btc)    # [B,K,C]
    return c

def idct_recon(coeff_bkc, T_out):
    # coeff: [B,K,C] -> x: [B,T_out,C]（任意长度）
    B, K, C = coeff_bkc.shape
    D_out = dct_basis(T_out, K, coeff_bkc.device)  # [T_out,K]
    x = torch.einsum("tk,bkc->btc", D_out, coeff_bkc)
    return x

# ---------- 条件编码（占位，可替换为你的 tokenizer） ----------
class SimpleCond(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, c):  # c: [B, D]
        return self.net(c)  # [B, dim]

# ---------- 时间步嵌入 ----------
class TimestepEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))
    @staticmethod
    def sinusoid(t, dim, device):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / max(1, half))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2 == 1: emb = F.pad(emb, (0,1))
        return emb
    def forward(self, t, dim):
        e = self.sinusoid(t, dim, t.device)
        return self.proj(e)

# ---------- 极简 1D Transformer Denoisers（可换成你的 U-Net-1D） ----------
class Denoiser1D(nn.Module):
    """
    输入 y_t:[B,C,K]；输出 epŝ:[B,C,K]
    支持两种条件：
      - cond_vec: [B, D] （全局向量）
      - dtokens : [B, N, D]（token 序列，作为 cross-attn 的 key/value）
    """
    def __init__(self, c_in, k_len, d_model=256, n_layers=6, n_heads=8, cond_dim=256, use_tokens=False):
        super().__init__()
        self.use_tokens = use_tokens
        self.c_in = c_in; self.k_len = k_len; self.d_model = d_model
        self.in_proj  = nn.Conv1d(c_in, d_model, kernel_size=1)    # [B,C,K] -> [B,D,K]
        self.pos_emb  = nn.Parameter(torch.zeros(1, k_len, d_model))
        self.time_mlp = TimestepEmbed(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=d_model*4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        if use_tokens:
            self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        else:
            self.cond_proj = nn.Linear(cond_dim, d_model)
        self.out_proj = nn.Conv1d(d_model, c_in, kernel_size=1)

    def forward(self, y_t, t, cond: Dict):
        # y_t: [B,C,K] -> [B,K,D]
        B, C, K = y_t.shape
        h = self.in_proj(y_t)                      # [B,D,K]
        h = h.transpose(1,2).contiguous()          # [B,K,D]
        h = h + self.pos_emb[:, :K, :]
        # timestep 加到每个 token 上
        te = self.time_mlp(t, self.d_model)        # [B,D]
        h = h + te.unsqueeze(1)

        if self.use_tokens:
            # cross-attn: query=h, key/value=dtokens
            q = h
            kv = cond["dtokens"]                    # [B,N,D]
            h,_ = self.cross(query=q, key=kv, value=kv, need_weights=False)
        else:
            # FiLM 风格：全局向量注入
            vec = self.cond_proj(cond["cond_vec"])  # [B,D]
            h = h + vec.unsqueeze(1)

        # 自注意力编码
        h = self.encoder(h)                        # [B,K,D]
        h = h.transpose(1,2)                       # [B,D,K]
        eps_hat = self.out_proj(h)                 # [B,C,K]
        return eps_hat

# ---------- DDPM（在 DCT 系数上训练） ----------
class DDPMCoeffs1D(pl.LightningModule):
    def __init__(self,
                 c_channels,          # = 3J
                 k_len,               # 频率长度 K
                 denoiser_cfg: dict,  # 用于实例化 Denoiser1D（或你的 UNet1D）
                 timesteps=1000,
                 beta_schedule="linear",
                 linear_start=1e-4, linear_end=2e-2,
                 loss_type="l2",
                 lr=1e-4,
                 cond_mode="global",  # "global" | "tokens"
                 aux_time_loss=False, # 是否加时域小正则
                 time_loss_w=0.1,
                 T_future=60          # 训练标签的原时长，用来做 IDCT 对齐
                 ):
        super().__init__()
        self.save_hyperparameters()

        # 噪声表（与原 DDPM 相同）
        betas = self._make_beta(beta_schedule, timesteps, linear_start, linear_end)
        alphas = 1. - betas
        a_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", a_bar)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(a_bar))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - a_bar))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1. / a_bar))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1. / a_bar - 1.))

        # Denoisers
        self.model = Denoiser1D(c_in=c_channels, k_len=k_len, **denoiser_cfg)

        # 条件编码（演示版）：global 模式给一个小 MLP；tokens 模式假设外部已给 tokens
        if cond_mode == "global":
            self.cond_enc = SimpleCond(in_dim=c_channels*self.hparams.k_len, dim=denoiser_cfg.get("d_model", 256))
        self.cond_mode = cond_mode

        self.lr = lr
        self.aux_time_loss = aux_time_loss
        self.time_loss_w = time_loss_w
        self.T_future = T_future

    # ---- 噪声日程 ----
    @staticmethod
    def _make_beta(kind, T, s, e):
        if kind == "linear":
            return torch.linspace(s, e, T)
        elif kind == "cosine":
            # 简化 cosine schedule
            steps = torch.arange(T+1, dtype=torch.float32)
            alphas_cumprod = torch.cos((steps/T + 0.008)/(1+0.008) * math.pi/2)**2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(1e-5, 0.999)
        else:
            raise ValueError(kind)

    # ---- 前向扩散/反推公式 ----
    def q_sample(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        return self.sqrt_alphas_cumprod[t][:,None,None]*x0 + \
               self.sqrt_one_minus_alphas_cumprod[t][:,None,None]*noise

    def predict_x0(self, x_t, t, eps_hat):
        return self.sqrt_recip_alphas_cumprod[t][:,None,None]*x_t - \
               self.sqrt_recipm1_alphas_cumprod[t][:,None,None]*eps_hat

    # ---- 条件构造（示例）----
    def build_cond(self, batch):
        """
        期望 batch 提供：
          - 'hist': [B, T_h, 3J]  历史序列（用于条件）
          - 'fut' : [B, T_f, 3J]  未来序列（监督标签）
        """
        x_hist = batch["hist"]        # [B, T_h, C]
        x_fut  = batch["fut"]         # [B, T_f, C]
        # 未来的 DCT 目标系数
        C_fut = dct_forward(x_fut, K=self.hparams.k_len)        # [B,K,C]
        y0 = C_fut.transpose(1,2).contiguous()                  # [B,C,K] 作为扩散目标

        # 历史条件
        if self.cond_mode == "global":
            C_hist = dct_forward(x_hist, K=self.hparams.k_len)  # [B,K,C]
            cond_vec = C_hist.reshape(x_hist.size(0), -1)       # [B, K*C]
            cond = {"cond_vec": self.cond_enc(cond_vec)}
        else:
            # tokens 模式：假设 DataModule 已构造好 dtokens 放进 batch
            cond = {"dtokens": batch["dtokens"]}                # [B, Nd, D]

        return y0, cond

    # ---- 训练一步 ----
    def training_step(self, batch, _):
        y0, cond = self.build_cond(batch)                       # y0:[B,C,K]
        t = torch.randint(0, self.betas.numel(), (y0.size(0),), device=self.device).long()
        noise = torch.randn_like(y0)
        y_t = self.q_sample(y0, t, noise)

        eps_hat = self.model(y_t, t, cond)                      # [B,C,K]
        loss_eps = F.mse_loss(eps_hat, noise)

        # 可选：在时域上加一点点重建正则（用预测 x0）
        if self.aux_time_loss:
            x0_hat = self.predict_x0(y_t, t, eps_hat)           # [B,C,K]
            x0_hat_bkc = x0_hat.transpose(1,2).contiguous()     # [B,K,C]
            x_hat = idct_recon(x0_hat_bkc, T_out=self.T_future) # [B,Tf,C]
            loss_time = F.l1_loss(x_hat, batch["fut"])          # 简单 L1（可替换为多项物理损失）
            loss = loss_eps + self.time_loss_w * loss_time
            self.log_dict({"train/l_eps":loss_eps, "train/l_time":loss_time, "train/loss":loss})
        else:
            loss = loss_eps
            self.log("train/loss", loss)

        return loss

    # ---- 采样（任意输出长度/帧率）----
    @torch.no_grad()
    def sample(self, batch, T_out=None, steps=None):
        y0_cond_only, cond = self.build_cond(batch)             # 只用条件，忽略真值
        B, C, K = y0_cond_only.shape
        steps = steps or self.betas.numel()
        y = torch.randn(B, C, K, device=self.device)
        for i in reversed(range(steps)):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            eps_hat = self.model(y, t, cond)
            x0_hat  = self.predict_x0(y, t, eps_hat).clamp_(-10, 10)
            if i > 0:
                beta_t = self.betas[t][:,None,None]
                a_t    = self.alphas_cumprod[t][:,None,None]
                a_tm1  = self.alphas_cumprod[t-1][:,None,None] if i>0 else torch.ones_like(a_t)
                # DDPM 采样（你也可换 DDIM）
                mean = torch.sqrt(a_tm1)*x0_hat + torch.sqrt(1-a_tm1)*torch.randn_like(y)
                y = mean
            else:
                y = x0_hat

        # 还原到时域
        T_out = T_out or self.T_future
        C_bkc = y.transpose(1,2).contiguous()                   # [B,K,C]
        x = idct_recon(C_bkc, T_out=T_out)                      # [B,T_out,C]
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
