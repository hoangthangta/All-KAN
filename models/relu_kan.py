import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, norm_type = 'layer', base_activation = 'gelu', train_ab: bool = True):
        super().__init__()
        self.g, self.k, self.r = g, k, 4*g*g / ((k+1)*(k+1))
        self.input_size, self.output_size = input_size, output_size
        phase_low = np.arange(-k, g) / g
        phase_height = phase_low + (k+1) / g
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])),
                                      requires_grad=train_ab)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])),
                                         requires_grad=train_ab)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g+k, input_size))
       
       
        self.base_weight = torch.nn.Parameter(torch.Tensor(self.output_size, self.input_size))
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
       
        self.base_weight_bias = torch.nn.Parameter(torch.Tensor(1, self.output_size))
        torch.nn.init.kaiming_uniform_(self.base_weight_bias, a=math.sqrt(5))
       
        self.norm_type = norm_type
       
        # Use attention linear
        self.attention = nn.Linear(g + k, 1)
       
        # Data norms
        self.layer_norm = nn.LayerNorm(input_size)  
        self.batch_norm = nn.BatchNorm1d(input_size)
       
        self.base_activation = base_activation

    def forward(self, x):
        # Expand dimensions of x to match the shape of self.phase_low
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))
       
        # Perform the subtraction with broadcasting
        #x1 = torch.relu(x_expanded - self.phase_low)
        #x2 = torch.relu(self.phase_height - x_expanded)
        x1 = F.gelu(x_expanded - self.phase_low)
        x2 = F.gelu(self.phase_height - x_expanded)
       
        # Continue with the rest of the operations
        x = x1 * x2 * self.r
        x = x * x
        #x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = x.reshape((len(x), self.input_size, self.g + self.k))
       
        '''print(x.shape)
        x = self.equal_size_conv(x)
        print(x.shape)
        #x = x.reshape((len(x), self.output_size, 1))
        x = x.reshape((len(x), self.output_size))
        print(x.shape)'''
       
        x = self.use_attention(x)
        #x = self.use_dim_sum(x)
       
        return x

    def activation(self, x):
        """
            We found that F.* activation functions produce better performance
            than torch.nn.* activation functions
        """
        if (self.base_activation == 'softplus'):
            return F.softplus(x)
        elif(self.base_activation == 'silu'):
            return F.silu(x)
        elif(self.base_activation == 'relu'):
            return F.relu(x)
        elif(self.base_activation == 'leaky_relu'):
            return F.leaky_relu(x)
        elif(self.base_activation == 'elu'):
            return F.elu(x)
        elif(self.base_activation == 'gelu'):
            return F.gelu(x)
        elif(self.base_activation == 'selu'):
            return F.selu(x)
        else: # default
            return x
   
    def use_dim_sum(self, x):
       
           
        x = torch.sum(x, dim=2)
       
        if (self.norm_type == 'layer'):
            x = self.layer_norm(x)      
        elif (self.norm_type == 'batch'):
            x = self.batch_norm(x)
           
        x = F.linear(self.activation(x), self.base_weight, self.base_weight_bias)
        return x
       
    def use_attention(self, x):

        attn_weights = F.softmax(self.attention(x), dim=-2)  
        x = x * attn_weights
        x = x.sum(dim=-1)
       
        if (self.norm_type == 'layer'):
            x = self.layer_norm(x)      
        elif (self.norm_type == 'batch'):
            x = self.batch_norm(x)
           
        x = F.linear(self.activation(x), self.base_weight, self.base_weight_bias)
       
        return x
   
   
class ReLUKAN(nn.Module):
    def __init__(self, width, grid = 5, k = 3, norm_type = 'layer', base_activation = 'gelu'):
        super().__init__()
        self.width = width # net structure
        self.grid = grid # grid_size
        self.k = k # spline order
        self.rk_layers = []
        for i in range(len(width) - 1):
            self.rk_layers.append(ReLUKANLayer(width[i], grid, k, width[i+1], norm_type = norm_type, base_activation = base_activation))
            # if len(width) - i > 2:
            #     self.rk_layers.append()
        self.rk_layers = nn.ModuleList(self.rk_layers)

    def forward(self, x):
        for rk_layer in self.rk_layers:
            x = rk_layer(x)
        # x = x.reshape((len(x), self.width[-1]))
        return x