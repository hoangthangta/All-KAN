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
        
        '''self.pe = self.sinusoidal_1d_pe().permute(1, 0)
        n = input_size
        values = torch.arange(1, n + 1) / n 
        self.pe = values.view(1, n)'''
        
        #self.base_weight = torch.nn.Parameter(torch.Tensor(self.output_size, self.input_size*(self.g + self.k)))
        #self.base_weight = torch.nn.Parameter(torch.Tensor(self.output_size, self.input_size))
        #torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        
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
    
    def normalize(self, x):
        x = F.normalize(x, p=2, dim=None)  # Normalize using L2 norm
        
        '''target_min = -self.k/self.g
        target_max = (self.k + self.g)/self.g
        original_min = x.min() 
        original_max = x.max()  
        return target_min + (x - original_min) * (target_max - target_min) / (original_max - original_min)'''
        
        return (x - x.min()) / (x.max() - x.min())  # Scale to [0,1]
        
    def sinusoidal_1d_pe(self):
        """Generate 1D positional encodings using sine and cosine functions."""
        
        #channels = self.g + self.k
        channels = 1
        pos = torch.arange(self.input_size).unsqueeze(1)  # Shape: (L, 1)
        #div_term = torch.exp(torch.arange(0, channels, 2) * (-np.log(10000.0) / channels))
        div_term = torch.exp(torch.arange(0, channels, 2) * (-torch.log(torch.tensor(10000.0)) / channels))
        pe = torch.zeros(self.input_size, channels)
        pe[:, 0::2] = torch.sin(pos * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(pos * div_term)  # Apply cos to odd indices
        return pe
    
    def forward(self, x):
        # Thank to "reeyarn" and "davidchern": 
        #   - https://github.com/quiqi/relu_kan/issues/1
        #   - https://github.com/quiqi/relu_kan/issues/2
        
        if (len(x.shape) == 3):
            x = x.squeeze(-1)
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))

        # Perform the subtraction with broadcasting
        x1 = self.activation(x_expanded - self.phase_low)
        x2 = self.activation(self.phase_high - x_expanded)
       
        # Continue with the rest of the operations
        x = x1 * x2 * self.r
        x = x * x

        '''x = x + self.pe.to(device)
        x = self.normalize(x)
        x = x.sum(dim = -1)
        x = self.normalize(x)
        x = x.view(x.shape[0], -1)
        x = F.linear(self.activation(x), self.base_weight)'''

        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
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
        self.norm_type = norm_type
        
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