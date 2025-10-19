import torch
import  torch.nn.functional as F
import torch.nn as nn
import numpy as np

def get_dct_matrix(N, is_torch=True):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    if is_torch:
        dct_m = torch.from_numpy(dct_m)
        idct_m = torch.from_numpy(idct_m)
    return dct_m, idct_m

class dct_encoder:

    def __init__ (self, dct_n=20):
        super(dct_encoder,self).__init__()
        self.dct_n = dct_n
        dct_m, idct_m = get_dct_matrix(dct_n) #非学习参数,dct矩阵


    def apply_dct(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        if T < self.dct_n:
            x = F.pad(x, (0, 0, 0, self.dct_n - T), value=0.0)  # pad 时间维右侧
            T = self.dct_n
        x_dct = torch.matmul(self.dct_m[:self.dct_n, :T], x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # [B,C,T] -> [B,C,dct_n] -> [B,dct_n,C]
        return x_dct


    def apply_idct(self, x):
        # x: [B, dct_n, C]
        B, dct_n, C = x.shape
        x_idct = torch.matmul(self.idct_m[:dct_n, :dct_n], x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return x_idct  # [B,C,dct_n] -> [B,C,T] -> [B,T,C]