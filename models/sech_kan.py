import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import math
 
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        return self.scale * x_norm

        
class MMNorm(nn.Module):
    def __init__(self, grid_min=-2.0, grid_max=2.0):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.eps = 1e-8

    def forward(self, x):
        x_norm = (x - x.min()) / (x.max() - x.min() + self.eps)
        x_norm = x_norm * (self.grid_max - self.grid_min) + self.grid_min
        return x_norm


class StableGrid(nn.Module):
    def __init__(self, num_grids, grid_min=-2, grid_max=2):
        super().__init__()

        self.num_grids = num_grids
        self.grid_min = grid_min
        self.grid_max = grid_max

        # learn spacing (positive)
        self.delta_raw = nn.Parameter(torch.randn(num_grids))

    def forward(self):
        # positive increments -> monotonic
        delta = F.softplus(self.delta_raw)

        # cumulative -> increasing
        grid = torch.cumsum(delta, dim=0)

        # normalize to [0,1]
        grid = grid - grid.min()
        grid = grid / (grid.max() + 1e-8)

        # scale to [grid_min, grid_max]
        grid = self.grid_min + (self.grid_max - self.grid_min) * grid

        return grid

class SechBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        width: float = 1.0,
        scale: float = 1.0,
        bias: float = 0.0,
        use_width: bool = False,
    ):
        super().__init__()
        assert num_grids > 1, "num_grids must be larger than 1"
        
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.use_width = use_width

        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=True)
        
        #self.grid = StableGrid(num_grids, grid_min, grid_max)
     
        if use_width:
            if width is None:
                width = (grid_max - grid_min) / (num_grids - 1)
            self.log_width = nn.Parameter(torch.log(torch.tensor(width)))
        
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        self.bias = nn.Parameter(torch.tensor(bias), requires_grad=True)

    @property
    def width(self):
        return torch.exp(self.log_width) + 1e-8 
        #return F.softplus(self.log_width) + 1e-8
    
    def forward(self, x):
        #grid = self.grid()
        
        diff = x[..., None] - self.grid  
        
        if self.use_width:
            diff = diff / self.width
            
        basis = 1 / torch.cosh(diff)        

        return basis * self.scale + self.bias
        #return basis


class SechKANLayer(nn.Module):
    """Sech-based KAN transform layer projecting (B, input_dim) → (B, output_dim)."""
    
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
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        norm1_type: str = "layer",
        norm2_type: str = "layer",
        base_activation: str = "silu",
        use_base_update: bool = False,
        spline_weight_init_scale: float = 0.1,
        use_width: bool = False,
    ):
        super().__init__()
        
        # Store ranges for MMNorm if needed
        self.grid_min = grid_min
        self.grid_max = grid_max

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_base_update = use_base_update
        self.num_grids = num_grids
        

        # Normalization
        self.norm1 = self._get_norm(norm1_type, input_dim)
        self.norm2 = self._get_norm(norm2_type, input_dim)

        # Sech bump basis
        self.sbf = SechBasisFunction(grid_min, grid_max, num_grids, use_width = use_width)

        # Project features -> output_dim
        self.feature_linear = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.kaiming_uniform_(self.feature_linear.weight, a=math.sqrt(3))
        #nn.init.uniform_(self.feature_linear.weight, -spline_weight_init_scale, spline_weight_init_scale)

        # Project grids -> 1
        self.grid_linear = nn.Linear(num_grids, 1, bias=True)
        #nn.init.kaiming_uniform_(self.grid_linear.weight, a=math.sqrt(5))
        #nn.init.uniform_(self.grid_linear.weight, -spline_weight_init_scale, spline_weight_init_scale)
        #nn.init.zeros_(self.grid_linear.bias)
        
        self.shared_linear = nn.Linear(num_grids, output_dim, bias=True)
        self.value_head = nn.Linear(output_dim, 1, bias=True)
        self.gate_head  = nn.Linear(output_dim, 1, bias=True)
        self.head = nn.Linear(output_dim, 2, bias=True)
        
        

        # Project features -> output_dim
        self.base_linear = nn.Linear(input_dim, output_dim, bias=True)
        
        
        # Activation
        if base_activation in [None, "none", "identity", ""]:
            self.base_activation = nn.Identity()
        else:
            if base_activation not in SechKANLayer.activation_fns:
                raise ValueError(f"Unknown activation: {base_activation}")
            self.base_activation = SechKANLayer.activation_fns[base_activation]

    def _get_norm(self, norm_type: str, dim: int):
        if norm_type == "layer":
            return nn.LayerNorm(dim)
        elif norm_type == "batch":
            return nn.BatchNorm1d(dim)
        elif norm_type == "mm":
            return MMNorm(self.grid_min, self.grid_max)
        elif norm_type == "rms":
            return RMSNorm(dim)
        else:
            return nn.Identity()

    def forward(self, x):
        # x: (B, input_dim)
        x = self.norm1(x)
      
        spline_basis = self.sbf(x)  # (B, input_dim, G)
       
        '''
        shared = self.shared_linear(spline_basis)                   # (B, input_dim, hidden_dim)
        sb_proj = self.value_head(shared).squeeze(-1)               # (B, input_dim)
        gate = torch.sigmoid(self.gate_head(shared).squeeze(-1))    # (B, input_dim)
        sb_out = sb_proj * gate                                     # (B, input_dim)
        '''

        
        '''shared = self.shared_linear(spline_basis)       # (B, input_dim, hidden_dim)
        out = self.head(shared)                         # (B, input_dim, 2)
        sb_proj = out[..., 0]                           # (B, input_dim)
        gate    = torch.sigmoid(out[..., 1])            # (B, input_dim)
        sb_out  = sb_proj * gate                        # (B, input_dim)'''
        
        

        sb_out = self.grid_linear(spline_basis).squeeze(-1)
        
        sb_out = self.norm2(sb_out)
        ret = self.feature_linear(self.base_activation(sb_out))     # (B, output_dim)
        
        if self.use_base_update: 
            ret = ret + self.base_linear(self.base_activation(x))
            
        return ret


class SechKAN(nn.Module):
    """Stacked SechKAN model."""
    def __init__(
        self,
        net_layers: List[int],
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = False,
        base_activation: str = "silu",
        spline_weight_init_scale: float = 0.1,
        norm1_type: str = "layer",
        norm2_type: str = "layer",
        norm_mode: str = "except_first",
        use_width : bool = False
    ):
        super().__init__()
        assert len(net_layers) >= 2

        self.layers = nn.ModuleList()

        for i, (in_dim, out_dim) in enumerate(zip(net_layers[:-1], net_layers[1:])):

            if norm_mode == "first":
                apply_norm = (i == 0)
            elif norm_mode == "except_first":
                apply_norm = (i != 0)
            elif norm_mode == "all":
                apply_norm = True
            else:
                apply_norm = False  # no norm

            self.layers.append(
                SechKANLayer(
                    input_dim=in_dim,
                    output_dim=out_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    norm1_type=norm1_type if apply_norm else "",
                    norm2_type=norm2_type if apply_norm else "",
                    base_activation=base_activation,
                    use_base_update=use_base_update,
                    spline_weight_init_scale=spline_weight_init_scale,
                    use_width=use_width
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class SechKAN_CNN(nn.Module):
    def __init__(self, data_width, data_height,
                 in_channel=1, middle_channel=8, out_channel=16,
                 base_activation='silu', num_classes=10,
                 classifier_type='mlp', num_grids=8, hidden_size=64,
                 norm1_type='', norm2_type='layer'):
        super().__init__()

        self.classifier_type = classifier_type

        self.conv1 = nn.Conv2d(in_channel, middle_channel, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(middle_channel, out_channel, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        act_map = {
            "softplus": nn.Softplus,
            "sigmoid": nn.Sigmoid,
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "selu": nn.SELU,
            "tanh": nn.Tanh,
        }

        self.act = act_map.get(base_activation, nn.Identity)()

        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, in_channel, data_width, data_height)
            dummy = self.pool1(self.act(self.conv1(dummy)))
            dummy = self.pool2(self.act(self.conv2(dummy)))
            flatten_dim = dummy.view(1, -1).size(1)

        if classifier_type == 'mlp':
            self.fc1 = nn.Linear(flatten_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)

        elif classifier_type == 'kan':
            self.kan1 = SechKANLayer(
                flatten_dim, hidden_size,
                num_grids=num_grids,
                norm1_type=norm1_type, norm2_type=norm2_type,
                base_activation=base_activation,
                use_base_update=False, use_width=False
            )
            self.kan2 = SechKANLayer(
                hidden_size, num_classes,
                num_grids=num_grids,
                norm1_type=norm1_type, norm2_type=norm2_type,
                base_activation=base_activation,
                use_base_update=False, use_width=False
            )
        else:
            raise ValueError("classifier_type must be 'mlp' or 'kan'")

    def forward(self, x):
        x = self.pool1(self.act(self.conv1(x)))
        x = self.pool2(self.act(self.conv2(x)))
        x = self.flatten(x)

        if self.classifier_type == 'mlp':
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
        else:
            x = self.kan2(self.kan1(x))

        return x

