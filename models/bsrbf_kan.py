import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RadialBasisFunction(nn.Module):
    """
    Implements a Radial Basis Function (RBF) layer.
    """
    def __init__(
        self,
        grid_min: float = -1.5,  # original: 2
        grid_max: float = 1.5,  # original: 2
        num_grids: int = 8,  # n_center
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class BSRBF_KANLayer(nn.Module):
    """
    Implements a layer of the BSRBF-KAN network, combining base activation,
    B-splines, and Radial Basis Functions (RBFs).
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size=5,
        spline_order=3,
        base_activation=torch.nn.SiLU,
        grid_range=[-1.5, 1.5],
        norm_type='layer'
    ) -> None:
        super().__init__()

        # Normalization layer
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(input_dim)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(input_dim)
        else:
            self.norm = nn.Identity()  # No-op normalization

        self.spline_order = spline_order
        self.grid_size = grid_size
        self.output_dim = output_dim
        self.base_activation = base_activation()
        self.input_dim = input_dim

        # Base weight (for linear transformation)
        self.base_weight = torch.nn.Parameter(torch.Tensor(self.output_dim, self.input_dim))
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        # Spline weight (for B-spline transformation)
        self.spline_weight = torch.nn.Parameter(torch.Tensor(self.output_dim, self.input_dim * (grid_size + spline_order)))
        torch.nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

        # RBF transformation
        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size + spline_order)

        # Grid setup for B-splines
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(self.input_dim, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.
        """
        assert x.dim() == 2 and x.size(1) == self.input_dim

        grid: torch.Tensor = self.grid  # (input_dim, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.input_dim,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def forward(self, x):
        x = self.norm(x)

        # Base transformation
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # B-splines transformation
        bs_output = self.b_splines(x).view(x.size(0), -1)

        # RBF transformation
        rbf_output = self.rbf(x).view(x.size(0), -1)

        # Combine B-splines and RBF outputs
        bsrbf_output = bs_output + rbf_output
        bsrbf_output = F.linear(bsrbf_output, self.spline_weight)

        return base_output + bsrbf_output


class BSRBF_KAN(nn.Module):
    """
    Implements the full BSRBF-KAN network with multiple layers.
    """
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        base_activation=torch.nn.SiLU,
        norm_type='layer'
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = nn.ModuleList()

        for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                BSRBF_KANLayer(
                    input_dim,
                    output_dim,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                    norm_type=norm_type
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
