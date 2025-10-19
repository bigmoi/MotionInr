from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from modules.distributions import DiagonalGaussianDistribution
from Utils.utils import instantiate_from_config

class MotionVAE(pl.LightningModule):
    """
    面向人体动作 B×T×(3J) 的 VAE：
    - 编码器返回 DiagonalGaussianDistribution（含 posterior.pad_len）
    - 解码器根据 target_T 还原到原始帧长
    - 损失: ELBO = recon_loss + beta_kl * KL（支持 KL 退火）
    - 可导出潜在以供 LDM 训练
    """
    def __init__(
        self,
        encoder: Dict[str, Any],          # {"target": "...Conv1dEncoder", "params": {...}}
        decoder: Dict[str, Any],          # {"target": "...Conv1dDecoder", "params": {...}}
        recon_loss: str = "mse",          # "mse" | "l1"
        beta_kl: float = 1.0,
        kl_anneal_steps: int = 0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = instantiate_from_config(encoder)
        self.decoder = instantiate_from_config(decoder)

        self.recon_loss = recon_loss.lower()
        self.beta_kl = beta_kl
        self.kl_anneal_steps = kl_anneal_steps
        self.lr = lr
        self.weight_decay = weight_decay

    # --------- 前向 / 编码 / 解码 ----------
    def encode(self, x_btc) -> DiagonalGaussianDistribution:
        return self.encoder(x_btc)

    def decode(self, z, target_T: int):
        return self.decoder(z, target_T=target_T)

    def forward(self, x_btc):
        B, T, C = x_btc.shape
        posterior = self.encode(x_btc)                 # 返回分布对象
        z = posterior.sample() if self.training else posterior.mode()
        x_hat = self.decode(z, target_T=T)
        return x_hat, posterior

    # --------- 计算损失 ----------
    def _recon(self, x, x_hat):
        if self.recon_loss == "l1":
            return F.l1_loss(x_hat, x, reduction="mean")
        return F.mse_loss(x_hat, x, reduction="mean")

    def _kl_coeff(self):
        if self.kl_anneal_steps and self.global_step is not None:
            return min(1.0, float(self.global_step) / float(self.kl_anneal_steps))
        return 1.0

    # --------- Lightning 训练步骤 ----------
    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch  # 兼容 (x) 或 (x, y)
        x_hat, posterior = self.forward(x)
        recon = self._recon(x, x_hat)
        kl = posterior.kl().mean()
        coef = self._kl_coeff()
        loss = recon + self.hparams.beta_kl * coef * kl
        self.log_dict(
            {"train/loss": loss, "train/recon": recon, "train/kl": kl, "train/kl_coeff": coef},
            on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_hat, posterior = self.forward(x)
        recon = self._recon(x, x_hat)
        kl = posterior.kl().mean()
        loss = recon + self.hparams.beta_kl * kl
        self.log_dict({"val/loss": loss, "val/recon": recon, "val/kl": kl},
                      on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x_hat, posterior = self.forward(x)
        recon = self._recon(x, x_hat)
        kl = posterior.kl().mean()
        loss = recon + self.hparams.beta_kl * kl
        self.log_dict({"test/loss": loss, "test/recon": recon, "test/kl": kl},
                      on_epoch=True, prog_bar=True, logger=True)
        return loss

    # --------- 采样 / 导出潜在 ----------
    @torch.no_grad()
    def sample_from_prior(self, B: int, T: int, C_in: int):
        """
        从标准正态采样一个与 posterior 同形状的 z，并解码。用于 sanity check。
        """
        dummy = torch.zeros(B, T, C_in, device=self.device)
        post = self.encode(dummy)             # 走一遍以确定 T_down
        z_shape = post.mean.shape             # [B, z, T_down, 1]
        params = torch.zeros(z_shape[0], z_shape[1]*2, z_shape[2], z_shape[3], device=self.device)
        prior = DiagonalGaussianDistribution(parameters=params, deterministic=False)
        z = prior.sample()
        x_hat = self.decode(z, target_T=T)
        return x_hat

    @torch.no_grad()
    def encode_to_z(self, x_btc, use_mode=False):
        """
        返回 (z, posterior)，z 形状 [B, Cz, T_down, 1]
        可用于离线导出潜在以训练 LDM。
        """
        post = self.encode(x_btc)
        z = post.mode() if use_mode else post.sample()
        return z, post

    # --------- 优化器 ----------
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
