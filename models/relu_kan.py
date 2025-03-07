# Modified from: https://github.com/quiqi/relu_kan
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
        
class ReLUKANLayer(nn.Module):
    def __init__(
                    self, 
                    input_size: int, 
                    g: int, k: int, 
                    output_size: int, 
                    norm_type = 'layer', 
                    base_activation = 'relu',
                    train_ab: bool = True
                ):
        super().__init__()
        self.g, self.k, self.r = g, k, 4*g*g / ((k+1)*(k+1))
        self.input_size, self.output_size = input_size, output_size
        phase_low = torch.arange(-k, g) / g
        phase_high = phase_low + (k + 1) / g

        self.phase_low = nn.Parameter(phase_low[None, :].expand(input_size, -1),
                                      requires_grad=train_ab)
        self.phase_high = nn.Parameter(phase_high[None, :].expand(input_size, -1),
                                         requires_grad=train_ab)

        self.equal_size_conv = nn.Conv2d(1, output_size, (g+k, input_size))
        self.base_activation = base_activation

        # Data normalization
        if (norm_type == 'layer'):
            self.norm = nn.LayerNorm(input_size)
        elif(norm_type == 'batch'):
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity() 
            

    def activation(self, x):
        """
            We found that F.* activation functions produce better performance
            than nn.* activation functions
        """
        af_list = {
            'softplus': F.softplus,
            'sigmoid': F.sigmoid,
            'silu': F.silu,
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'gelu': F.gelu,
            'selu': F.selu,
        }
        af = af_list.get(self.base_activation, lambda x: x)
        return af(x)
        
    def forward(self, x):
        # Thank to "reeyarn" and "davidchern": 
        #   - https://github.com/quiqi/relu_kan/issues/1
        #   - https://github.com/quiqi/relu_kan/issues/2
        
        x = self.norm(x) 
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))
        
        # Perform the subtraction with broadcasting
        x1 = self.activation(x_expanded - self.phase_low)
        x2 = self.activation(self.phase_high - x_expanded)
       
        # Continue with the rest of the operations
        x = x1 * x2 * self.r
        x = x * x 

        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        #x = x.reshape((len(x), self.output_size, 1))
        x = x.reshape((len(x), self.output_size))
       
        return x
        
class ReLUKAN(nn.Module):
    def __init__(self, 
                    width, 
                    grid = 5, 
                    k = 3, 
                    norm_type = 'layer',
                    base_activation = 'relu'):
        super().__init__()
        self.width = width # net structure
        self.grid = grid # grid_size
        self.k = k # spline order
        
        self.base_activation = base_activation
        
        self.rk_layers = []
        for i in range(len(width) - 1):
            self.rk_layers.append(ReLUKANLayer(width[i], 
                                        grid, k, width[i+1], 
                                        norm_type = norm_type, 
                                        base_activation = base_activation
                                        )
                                    )
            #if len(width) - i > 2:
            #   self.rk_layers.append()
        self.rk_layers = nn.ModuleList(self.rk_layers)

    def forward(self, x):
        for rk_layer in self.rk_layers:
            x = rk_layer(x)
        #x = x.reshape((len(x), self.width[-1]))
        return x