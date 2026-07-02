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
        grid_min: float = -2.0,  
        grid_max: float = 2.0,  
        num_grids: int = 8,  # n_center
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class DBG_KAN_Layer(nn.Module):

    activation_fns = {
        "softplus": F.softplus,
        "sigmoid": torch.sigmoid,
        "silu": F.silu,
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "elu": F.elu,
        "gelu": F.gelu,
        "selu": F.selu,
        "tanh": torch.tanh,
    }

    def __init__(
        self,
        input_dim,
        output_dim,
        grid_size=5,
        spline_order=3,
        base_activation="silu",
        grid_min=-2.0,
        grid_max=2.0,
        norm_type="layer",
        basis_type="both",
        gate_type="coupled",
        use_base_update=False,
    ):
        super().__init__()

        self.basis_type = basis_type
        self.gate_type = gate_type
        self.use_base_update = use_base_update
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.spline_order = spline_order
        self.grid_size = grid_size

        # activation
        if base_activation in [None, "none", "identity", ""]:
            self.base_activation = nn.Identity()
        else:
            if base_activation not in self.activation_fns:
                raise ValueError(f"Unknown activation: {base_activation}")
            self.base_activation = self.activation_fns[base_activation]

        self.norm = self._get_norm(norm_type, input_dim)

        # grid
        h = (grid_max - grid_min) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_min
        ).expand(input_dim, -1).contiguous()
        
        #self.grid = nn.Parameter(grid, requires_grad=True)
        self.register_buffer("grid", grid)

        self.rbf = RadialBasisFunction(
            grid_min, grid_max, grid_size + spline_order
        )
        
        self.output_linear = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.kaiming_uniform_(self.output_linear.weight, a=math.sqrt(5))

        self.basis_mixer = nn.Linear(grid_size + spline_order, grid_size + spline_order, bias=True)

        if gate_type == "coupled":
            self.head = nn.Linear(grid_size + spline_order, 2)
        elif gate_type == "decoupled":
            self.value_head = nn.Linear(grid_size + spline_order, 1, bias=True)
            self.gate_head = nn.Linear(grid_size + spline_order, 1, bias=True)

        if use_base_update:
            self.base_linear = nn.Linear(input_dim, output_dim, bias=True)

        self.k = grid_size + spline_order
        
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        #self.alpha = nn.Parameter(torch.zeros(1, input_dim, 1))

    def _get_norm(self, norm_type, dim):
        if norm_type == "layer":
            return nn.LayerNorm(dim)
        elif norm_type == "batch":
            return nn.BatchNorm1d(dim)
        return nn.Identity()

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
        x_ori = x
        #x = self.norm(x)

        if self.basis_type == "b_spline": x = self.b_splines(x)
        elif self.basis_type == "rbf": x = self.rbf(x)
        else: x = self.b_splines(x) + self.rbf(x)
        
        h = self.basis_mixer(x)           

        if self.gate_type == "coupled":
            v, g = self.head(h).chunk(2, dim=-1)
        elif self.gate_type == "decoupled":
            v = self.value_head(h)
            g = self.gate_head(h) 
        elif (self.gate_type == "none"):
            out = h.sum(dim=-1)
        else:
            raise ValueError(f"Unknown gate_type: '{self.gate_type}'. ""Expected one of: 'coupled', 'decoupled', 'none'.")
        
        if self.gate_type in ["coupled", "decoupled"]: 
            #out = (v * torch.sigmoid(g)).squeeze(-1) 
            out = (v * F.silu(g)).squeeze(-1) 

        out = self.norm(out)
        out = self.output_linear(self.base_activation(out))
        if self.use_base_update:
            out = out + self.base_linear(self.base_activation(x_ori))
        
        return out

# Dynamic Basis-Gated KAN
class DBG_KAN(nn.Module):
   
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        base_activation='silu',
        grid_min=-2.0,
        grid_max=2.0,
        norm_type='layer',
        basis_type="both",
        gate_type="shared_head",
        use_base_update=True,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = nn.ModuleList()

        for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                DBG_KAN_Layer(
                    input_dim,
                    output_dim,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_activation=base_activation,
                    norm_type=norm_type,
                    basis_type = basis_type,
                    gate_type = gate_type,
                    use_base_update = use_base_update 
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
