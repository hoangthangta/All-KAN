import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import random
import numpy as np



# For results reproduction
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


class MMEncoder(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.encoder = nn.Linear(in_feature, out_feature)
    def forward(self, x):
        return self.encoder(x)
    
    
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=3,
        spline_order=3,
        scale_noise=0.1,
        scale_spline=1.0,
        grid_eps=0.02,
        grid_range=[-2, 2],
        groups=8,
        need_relu=True
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        assert in_features % groups == 0 or groups == -1, 'Please select the group number divisible to input shape'
        
        if groups == -1:
            # if == -1, means no groups
            # print(in_features)
            self.groups = in_features
        else:
            self.groups = groups
            
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
                ).expand(self.groups, -1).contiguous().unsqueeze(0)
        self.h = h
        self.register_buffer("grid", grid)
        
        self.spline_lin_c = torch.nn.Parameter(
                            torch.Tensor(1, self.groups, grid_size + spline_order)
                            )
        self.spline_weight = torch.nn.Parameter(
                            torch.Tensor(in_features, out_features)
                            )
        
        self.num_neurons_in_g = in_features // self.groups
        
        # self.base_weight = torch.nn.Parameter(
        #                     torch.Tensor(in_features, out_features)
        #                     )
        
        self.grid_range = grid_range
        self.shortcut = nn.SiLU()
        self.grid_bias = torch.nn.Parameter(torch.empty_like(grid).uniform_(-h/8, h/8))
        self.scale_noise = scale_noise
        self.scale_spline = scale_spline
        self.grid_eps = grid_eps
        self.need_relu = need_relu
        self.reset_parameters()



    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5) * self.scale_spline)
        torch.nn.init.kaiming_uniform_(self.spline_lin_c, a=math.sqrt(5) * self.scale_spline)
        # torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_spline)
        # torch.nn.init.kaiming_uniform_(self.grid_bias, a=math.sqrt(5) * self.scale_spline)



    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        
        # grid = (self.grid).repeat(self.num_neurons_in_g, 1, 1)
        
        grid = (self.grid + self.grid_bias)
        grid = torch.sort(grid, dim=2, descending=False).values.repeat(self.num_neurons_in_g, 1, 1)
        grid = torch.clamp(grid, self.grid_range[0] - self.h * self.spline_order, self.grid_range[-1] + self.h * self.spline_order)
        
        B, E = x.shape
        x = x.contiguous().reshape(B, self.num_neurons_in_g, self.groups).unsqueeze(-1)
        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :, : -(k + 1)])
                / (grid[:, :, k:-1] - grid[:, :, : -(k + 1)])
                * bases[:, :, :, :-1]
            ) + (
                (grid[:, :, k + 1 :] - x)
                / (grid[:, :, k + 1 :] - grid[:, :, 1:(-k)])
                * bases[:, :, :, 1:]
            )
            
        return bases.contiguous()



    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)        # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)                        # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(A, B).solution # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(2, 0, 1)           # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()


    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        output = self.b_splines(x)
        output = torch.einsum('begz,qgz->beg', output, self.spline_lin_c).reshape(original_shape[0], -1)
        output = torch.matmul(output, self.spline_weight) 
        if self.need_relu:
            shortcut_x = torch.matmul(self.shortcut(x), self.spline_weight)
            # shortcut_x = torch.matmul(self.shortcut(x), self.base_weight)
            output = output + shortcut_x
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output


class KnotsKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=20,
        spline_order=3,
        scale_noise=0.1,
        scale_spline=1.0,
        grid_eps=0.02,
        grid_range=[-3, 3],
        groups=8,
        need_relu=True
        ):
        super(KnotsKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_hid = len(layers_hidden)
        self.layers = torch.nn.ModuleList()
        self.layer_norm = torch.nn.ModuleList()
        for idx, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            self.layer_norm.append(
                nn.LayerNorm(in_features)
            )
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_spline=scale_spline,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    groups=groups,
                    need_relu=need_relu
                )
            )
    
    
    
    def forward(self, x: torch.Tensor, update_grid=False, normalize=False):
        if normalize:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-6).detach() 
            x /= stdev
        
        x = self.layer_norm[0](x)
        enc_x = self.layers[0](x)
        hid_x = enc_x
        
        for layer, layernorm in zip(self.layers[1:-1], self.layer_norm[1:-1]):
            hid_x = layernorm(hid_x)
            hid_x = layer(hid_x)
            
        hid_x = self.layer_norm[-1](hid_x)
        hid_x = self.layers[-1](hid_x)
        
        if normalize:
            hid_x = hid_x * stdev
            hid_x = hid_x + means
        return hid_x