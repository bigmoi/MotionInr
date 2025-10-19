import copy
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import tensor

from Utils.utils import instantiate_from_config, sample_preprocessing,get_dct_matrix


class MotionINR(pl.LightningModule):
    """

    """
    def __init__(
        self,
        denoiser: Dict[str, Any],          # 实现去噪的网络
        diffusion: Dict[str, Any],          # 这里应该是包含dct变换的diffusion
        decoder: Dict[str, Any],          #  这里应该是HDM
        t_his: int = 25,
        t_pred: int = 100,
        n_pre: int = 20 ,                 # dct系数个数
        recon_loss: str = "mse",          # "mse" | "l1"
        beta_mpjpe: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.denoiser = instantiate_from_config(denoiser)
        self.diffusion=instantiate_from_config(diffusion)
        self.decoder = instantiate_from_config(decoder)

        self.denoiser.eval()
        for p in self.denoiser.parameters():
            p.requires_grad = False
        self.diffusion.eval()
        for p in self.diffusion.parameters():
            p.requires_grad = False

        self.recon_loss = recon_loss.lower()
        self.beta_mpjpe = beta_mpjpe

        self.t_his = t_his
        self.t_pred = t_pred
        self.n_pre = n_pre
        dct_m, idct_m = get_dct_matrix(t_his + t_pred)
        self.register_buffer("dct_m", dct_m.float(), persistent=True)
        self.register_buffer("idct_m", idct_m.float(), persistent=True)
        self.lr = lr
        self.weight_decay = weight_decay

    # --------- 前向 / 编码 / 解码 ----------
    def get_prediction(self,data):
        # traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])
        # traj = tensor(traj_np, device=self.device, dtype=torch.float32)
        # traj = traj.reshape([traj.shape[0], -1, traj.shape[-1]]).transpose(1, 2)
        # traj.shape: [*, t_his + t_pre, 3 * joints_num]
        #得到三个返回值：，后两个完全相同
        mode_dict, traj_dct, traj_dct_cond = sample_preprocessing(data,self.dct_m,t_his=self.t_his,t_pred=self.t_pred,n_pre=self.n_pre, mod_test=1)
        self.diffusion.dct=self.dct_m
        self.diffusion.idct=self.idct_m
        self.diffusion.n_pre=self.n_pre
        sampled_motion = self.diffusion.sample_ddim(self.denoiser,
                                               traj_dct,
                                               traj_dct_cond,
                                               mode_dict)#得到的是全长频域序列（包含观测值）
        return sampled_motion
    def encode(self, x_btc) :
        return self.get_prediction(x_btc)

    def decode(self, z, target_T: int):
        return self.decoder(z, target_T=target_T)


    def forward(self, x_btc):
        B, T, C = x_btc.shape
        x_dct = self.encode(x_btc) #这里应该得到dct编码后预测序列 b*tn*c

        x_hat = self.decode(x_dct, target_T=T)

        return x_hat

    # --------- 计算损失 ----------

    def rec_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 标准 MSE，自动在所有维度求均值
        return F.mse_loss(x, y, reduction='mean')

    def mpjpe_loss(self,pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        assert pred.shape == gt.shape and pred.dim() == 3
        B, T, C = pred.shape
        assert C % 3 == 0, "最后一维应为 3J"
        J = C // 3
        pred = pred.reshape(B, T, J, 3)
        gt = gt.reshape(B, T, J, 3)
        #todo暂时未统一单位
        per_joint = torch.norm(pred - gt, dim=-1)  # [B, T, J]
        return per_joint.mean()


    # --------- Lightning 训练步骤 ----------

    def training_step(self, batch, batch_idx):
        x =batch[:,:self.t_his,:]
        future= batch[:,self.t_his:,:]
        x_hat = self.forward(x)
        rec_loss = self.rec_loss(x_hat, future)
        mpjpe_loss= self.mpjpe_loss(x_hat, future)

        loss = rec_loss+self.beta_mpjpe * mpjpe_loss
        self.log_dict(
            {"train/loss": loss, "train/mse_loss": rec_loss, "train/mpjpe_loss": mpjpe_loss},
            on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x = batch[0] if isinstance(batch, (tuple, list)) else batch
    #     x_hat, posterior = self.forward(x)
    #     recon = self._recon(x, x_hat)
    #
    #     loss=None
    #     # self.log_dict({"val/loss": loss, "val/recon": recon, "val/kl": kl},
    #     #               on_epoch=True, prog_bar=True, logger=True)
    #     return loss


    def test_step(self, batch, batch_idx):
        x = batch[:, :self.t_his, :]
        future = batch[:, self.t_his:-1, :]
        x_hat = self.forward(x)
    # --------- 优化器 ----------
    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]  # 只拿未冻结的，比如 decoder
        opt = torch.optim.Adam(trainable, lr=self.lr, weight_decay=self.weight_decay) #将未冻结参数传入优化器
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def on_after_backward(self):
        unused = [n for n, p in self.named_parameters() if p.requires_grad and p.grad is None]
        if unused:
            self.print(f"[rank{self.global_rank}] Unused params: {unused[:10]}{' ...' if len(unused) > 10 else ''}")
