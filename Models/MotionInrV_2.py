import copy
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import tensor

from Utils.utils import instantiate_from_config, sample_preprocessing,get_dct_matrix,map_to_original_frames

#这个模块实现的是多阶段训练，使用已经训练好的denoiser
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
        denoiser_ckpt_path=None,
        orangefps= 50,
        upsamplefpsrate=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.denoiser = instantiate_from_config(denoiser)
        self.diffusion=instantiate_from_config(diffusion)
        self.decoder = instantiate_from_config(decoder)

        self.recon_loss = recon_loss.lower()
        self.beta_mpjpe = beta_mpjpe

        self.t_his = t_his
        self.t_pred = t_pred
        self.n_pre = n_pre
        self.orangefps= orangefps
        self.upsamplefpsrate= upsamplefpsrate  #提高帧率的倍数
        dct_m, idct_m = get_dct_matrix(t_his+t_pred)
        self.register_buffer("dct_m", dct_m.float(), persistent=True)
        self.register_buffer("idct_m", idct_m.float(), persistent=True)
        self.lr = lr
        self.weight_decay = weight_decay
        if denoiser_ckpt_path is not None:
            denoiser_ckpt = torch.load(denoiser_ckpt_path, map_location='cpu')
            self.denoiser.load_state_dict(denoiser_ckpt)
            print(f"Loaded denoiser checkpoint from {denoiser_ckpt_path}")

        # 冻结去噪网络和扩散模型的参数
        self.denoiser.eval()
        for p in self.denoiser.parameters():
            p.requires_grad = False
        self.diffusion.eval()
        for p in self.diffusion.parameters():
            p.requires_grad = False

        dct4loss, _ = get_dct_matrix(max(upsamplefpsrate*t_pred, t_his+t_pred))
        with torch.no_grad():
            dct4loss = dct4loss.to(dtype=torch.float32)  # CPU 上转 FP32
        self.register_buffer("dct4loss", dct4loss, persistent=True)  # 注册为 buffer（训练时自动到 GPU）

    # --------- 前向 / 编码 / 解码 ----------
    def get_prediction(self,data):

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
    #
    # def rec_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    #     # 标准 MSE，自动在所有维度求均值
    #     return F.mse_loss(x, y, reduction='mean')
    def lowsequance_rec_loss(self, preseq, gtseq, sample_num: int=30):
        preseq_low = torch.matmul(self.dct4loss[:sample_num,:self.upsamplefpsrate*self.t_pred], preseq)
        gtseq_low = torch.matmul(self.dct4loss[:sample_num,:self.t_pred],gtseq)
        return F.mse_loss(preseq_low, gtseq_low, reduction='mean')

    def temporal_consistency_loss(self, pred, gt, time_step: float = 0.02):
        """
        计算时序一致性损失，包括速度和加速度的平滑性

        :param pred: 预测的姿态序列 [B, T, J, 3]，其中 B 为批量大小，T 为时间步数，J 为关节数，3 为每个关节的空间坐标
        :param gt: 真实的姿态序列 [B, T, J, 3]
        :param time_step: 每个时间步的时间间隔，默认 0.02 秒/即 50 FPS
         :return: 时序一致性损失值
        """
        assert pred.shape == gt.shape and pred.dim() == 3
        B, T, C = pred.shape
        assert C % 3 == 0, "最后一维应为 3J"
        J = C // 3
        pred = pred.reshape(B,-1, J, 3)
        gt = gt.reshape(B,-1, J, 3)
        # 计算速度
        pred_velocity = (pred[:, 2:, :, :] - pred[:, :-2, :, :]) / (2 * time_step)
        gt_velocity = (gt[:, 2:, :, :] - gt[:, :-2, :, :]) / (2 * time_step)

        # 计算加速度
        pred_acceleration = (pred_velocity[:, 2:, :, :] - pred_velocity[:, :-2, :, :]) / (2 * time_step)
        gt_acceleration = (gt_velocity[:, 2:, :, :] - gt_velocity[:, :-2, :, :]) / (2 * time_step)

        # 计算速度和加速度的 L2 距离
        velocity_loss = torch.norm(pred_velocity - gt_velocity, p=2, dim=-1)
        acceleration_loss = torch.norm(pred_acceleration - gt_acceleration, p=2, dim=-1)

        # 返回速度和加速度的平均损失
        return (velocity_loss.mean() + acceleration_loss.mean()) / 2

    def anchor_reconstruction_loss(self, pred, gt, target_fps,orig_fps):
        """
        计算锚点重构损失，确保生成的高帧率序列在锚点处与真实数据一致
        :param pred: 预测的高帧率姿态序列 [B, T, J, 3]
        :param gt: 真实的低帧率姿态序列 [B, T, J, 3]
        :param target_fps: 目标帧率（生成的高帧率）
        :param orig_fps: 原始帧率（低帧率数据）
        :return: 锚点重构损失
        """
        assert pred.shape == gt.shape and pred.dim() == 3
        B, T, C = pred.shape
        assert C % 3 == 0, "最后一维应为 3J"
        J = C // 3
        pred = pred.reshape(B, -1, J, 3)
        gt = gt.reshape(B, -1, J, 3)

        # 调用 map_to_original_frames 函数获取原始帧的索引
        orig_indices = map_to_original_frames(orig_fps, target_fps, T)

        # 选取对应的原始帧位置的预测和真实姿态
        pred_anchors = pred[torch.arange(B).view(-1, 1), orig_indices, :, :]
        gt_anchors = gt[torch.arange(B).view(-1, 1), orig_indices, :, :]

        # 计算 L2 距离，衡量预测和真实姿态在锚点位置的差异
        return torch.mean(torch.norm(pred_anchors - gt_anchors, p=2, dim=-1))

    # --------- Lightning 训练步骤 ----------

    def training_step(self, batch, batch_idx):
        x =batch[:,:self.t_his,:]
        gt= batch[:,self.t_his:,:]
        x_hat = self.forward(x)

        lowsequance_rec_loss = self.lowsequance_rec_loss(x_hat, gt,self.n_pre)
        temporal_consistency_loss=self.temporal_consistency_loss(x_hat, gt,1.0/self.orangefps)
        anchor_reconstruction_loss= self.anchor_reconstruction_loss(x_hat, gt,target_fps=self.orangefps*self.upsamplefpsrate, orig_fps=self.orangefps)


        loss = lowsequance_rec_loss+temporal_consistency_loss+anchor_reconstruction_loss
        self.log_dict(
            {"train/loss": loss, "train/lowsequance_rec_loss": lowsequance_rec_loss,
             "train/temporal_consistency_loss": temporal_consistency_loss,
             "train/anchor_reconstruction_loss": anchor_reconstruction_loss},
            on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True,
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
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
