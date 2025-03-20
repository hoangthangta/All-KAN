import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# A variant of https://github.com/ZiyaoLi/fast-kan/blob/master/fastkan/fastkan.py

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -1.5,
        grid_max: float = 1.5,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class RBF_KANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size = 5,
        spline_order = 3,
        base_activation = torch.nn.SiLU,
        grid_range = [-1.5, 1.5],
        norm_type = 'layer'

    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)
        self.grid_size = grid_size
        self.output_dim = output_dim
        self.base_activation = base_activation()
        self.input_dim = input_dim
        self.base_weight = torch.nn.Parameter(torch.Tensor(self.output_dim, self.input_dim))
        self.spline_weight = torch.nn.Parameter(torch.Tensor(self.output_dim, self.input_dim*(grid_size+spline_order)))
        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size+spline_order)
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))
        
        # Data norm
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(input_size)
        elif(norm_type == 'batch'):
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()  # No-op normalization

    def forward(self, x):
        device = x.device
        x = self.norm(x)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        rbf_output = self.rbf(x).view(x.size(0), -1)
        rbf_output = F.linear(rbf_output, self.spline_weight)
        return base_output + rbf_output


class RBF_KAN(torch.nn.Module):
    
    def __init__(
        self, 
        layers_hidden,
        grid_size = 5, 
        spline_order =3,
        base_activation=torch.nn.SiLU,
        grid_range = [-1.5, 1.5],
        norm_type = 'layer'
    ):
        super(RBF_KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                RBF_KANLayer(
                    input_dim,
                    output_dim,
                    grid_size=grid_size,
                    spline_order = spline_order,
                    base_activation=base_activation,
                    norm_type = norm_type
                )
            )
        
    def forward(self, x: torch.Tensor, normalize=False):
        
        if normalize:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x /= stdev
        
        
        for layer in self.layers: 
            x = layer(x)
            
            
        if normalize:
            x = x * stdev
            x = x + means
        return x