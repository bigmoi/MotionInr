import torch
import torch.nn as nn
import numpy as np
import math
from .layers import batched_linear_mm


class LearnableFourierFeatures(nn.Module):
    def __init__(self, input_dim, output_dim, sigma=10.0, learnable=True):
        """
        Learnable Random Fourier Features Layer
        Args:
            input_dim: Dimension of the input coordinates (e.g., 2 for (x,y))
            output_dim: Number of Fourier features (should be even)
            sigma: Scaling factor for frequency initialization
        """
        super().__init__()
        self.output_dim = output_dim
        self.sigma = sigma
        if learnable:
            self.B = nn.Parameter(torch.randn(input_dim, output_dim // 2) * sigma, requires_grad=True)  # Learnable frequencies
        else:
            self.register_buffer('B', torch.randn(input_dim, output_dim // 2) * sigma)

    def forward(self, x):
        """
        Apply learnable Fourier feature mapping.
        x: Input coordinates (B, N, in_dim)
        """
        x_proj = 2 * np.pi * x @ self.B  # (B, N, output_dim//2)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (B, N, output_dim)




class InrMlp(nn.Module):
    def __init__(self, depth, in_dim, out_dim, hidden_dim, use_pe=True, pe_dim= 256, out_bias=0, pe_sigma=10.0):
        super().__init__()
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.pe_sigma = pe_sigma
        self.depth = depth
        self.param_shapes = dict()
        if use_pe:
            self.convert_posenc = LearnableFourierFeatures(in_dim, pe_dim, sigma=pe_sigma)
            last_dim = pe_dim
        else:
            last_dim = in_dim
        for i in range(depth): #定义depth深的MLP
            cur_dim = hidden_dim if i < depth - 1 else out_dim
            self.param_shapes[f'wb{i}'] = (last_dim + 1, cur_dim) #得到每一层的weight和bias的shape
            last_dim = cur_dim
        self.relu = nn.ReLU()
        self.params = None
        self.out_bias = out_bias

    def set_params(self, params):
        self.params = params

    def init_wb(self, shape, **kwargs):
        weight = torch.empty(shape[1], shape[0] - 1)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        bias = torch.empty(shape[1], 1)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)

        return torch.cat([weight, bias], dim=1).t().detach()

    def forward(self, x):
        B, query_shape = x.shape[0], x.shape[1: -1]
        x = x.view(B, -1, x.shape[-1])
        if self.use_pe:
            x = self.convert_posenc(x)
        for i in range(self.depth):
            x = batched_linear_mm(x, self.params[f'wb{i}'])
            if i < self.depth - 1:
                x = self.relu(x)
            else:
                x = x + self.out_bias
        x = x.view(B, *query_shape, -1)
        return x

    def get_last_layer(self):
        return self.params[f'wb{self.depth - 1}']
    
