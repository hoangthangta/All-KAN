# Modified from: https://github.com/quiqi/relu_kan

import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PRReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, norm_type = 'layer', \
                    base_activation = 'gelu', methods = ['local_attn'], combined_type = 'sum', train_ab: bool = True):
        super().__init__()
        self.g, self.k, self.r = g, k, 4*g*g / ((k+1)*(k+1))
        self.input_size, self.output_size = input_size, output_size
        phase_low = np.arange(-k, g) / g
        phase_height = phase_low + (k+1) / g
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])),
                                      requires_grad=train_ab)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])),
                                         requires_grad=train_ab)

        self.base_weight = torch.nn.Parameter(torch.Tensor(self.output_size, self.input_size))
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
       
        self.base_weight_bias = torch.nn.Parameter(torch.Tensor(1, self.output_size))
        torch.nn.init.kaiming_uniform_(self.base_weight_bias, a=math.sqrt(5))
        
        self.methods = methods
        self.combined_type = combined_type
        self.norm_type = norm_type
        self.base_activation = base_activation
       
        # Use attention linear
        self.attention = nn.Linear(g + k, 1)
       
        # Data norms
        self.layer_norm = nn.LayerNorm(input_size)  
        self.batch_norm = nn.BatchNorm1d(input_size)
        
        # Self-attention
        self.query_linear = nn.Linear(g + k, g + k, bias=False)
        self.key_linear = nn.Linear(g + k, g + k, bias=False)
        self.value_linear = nn.Linear(g + k, g + k, bias=False)
        
        # Spatial attention
        self.conv1d = nn.Conv1d(in_channels=g + k, out_channels=g + k, kernel_size=1, stride=1)
        
        # Multihead attention
        self.mh_attn = nn.MultiheadAttention(embed_dim = g + k, num_heads=2, batch_first=True)
        

    def forward(self, x):
        # Thank to reeyarn, https://github.com/quiqi/relu_kan/issues/1
        
        # Expand dimensions of x to match the shape of self.phase_low
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))
       
        # Perform the subtraction with broadcasting
        x1 = self.activation(x_expanded - self.phase_low)
        x2 = self.activation(self.phase_height - x_expanded)
       
        # Continue with the rest of the operations
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), self.input_size, self.g + self.k))
       
        # Convert (B, D, G + k) to (B, D)
        output = torch.zeros(len(self.methods), x.size(0), self.output_size).to(x.device)
        for i, method in zip(range(len(self.methods)), self.methods):
            temp = x.clone()
            if (method == 'global_attn'):
                temp = self.global_attn(x)
            elif (method == 'local_attn'): 
                temp = self.local_attn(x)
            elif (method == 'spatial_attn'):
                temp = self.spatial_attn(x) 
            elif (method == 'self_attn'):
                temp = self.self_attn(x) 
            elif (method == 'multihead_attn'):
                temp = self.multihead_attn(x) 
            else:
                raise Exception('The method "' + method + '" does not support!')
            output[i] = temp
        
        # This is for a single method
        if (len(self.methods) == 1):
            return output[0]
        
        # Several simple combinations
        if (self.combined_type == 'sum'): 
            output = torch.sum(output, dim=0)
        elif (self.combined_type == 'product'):
            output = torch.prod(output, dim=0)
        elif (self.combined_type == 'sum_product'): 
            output = torch.sum(output, dim=0) +  torch.prod(output, dim=0)
        else:
            raise Exception('The combined type "' + self.combined_type + '" does not support!')
            # Write more combinations here...
            
        return output

    def activation(self, x):
        """
            We found that F.* activation functions produce better performance
            than torch.nn.* activation functions
        """
        af_list = {
            'softplus': F.softplus,
            'silu': F.silu,
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'gelu': F.gelu,
            'selu': F.selu,
        }
        af = af_list.get(self.base_activation, lambda x: x)
        return af(x)

    def local_attn(self, x, window_size = 2):

        # Unfold the last dimension (G + k) into local windows
        # Shape: (B, D, num_windows, window_size)
        local_windows = x.unfold(dimension=2, size=window_size, step=1) 

        # Use mean as a simple query
        query = local_windows.mean(dim=-1, keepdim=True)  
        key = local_windows
        
        # Shape: (B, D, num_windows, window_size)
        attn_score = torch.matmul(query.transpose(-1, -2), key)  
        
        # Apply softmax to get weights
        attn_weight = F.softmax(attn_score, dim=-1)    
        
        # Shape: (B, D, num_windows)
        weighted_sum = torch.sum(attn_weight * local_windows, dim=-1)  
        x = weighted_sum.sum(dim=-1)  
        
        # Data normalization
        if (self.norm_type == 'layer'):
            x = self.layer_norm(x)            
        elif (self.norm_type == 'batch'):
            x = self.batch_norm(x)
        
        # Linear transformation
        x = F.linear(self.activation(x), self.base_weight, self.base_weight_bias)
        
        return x


    def global_attn(self, x):
        
        # Apply softmax to get weights
        attn_weights = F.softmax(self.attention(x), dim=-1)  
        
        # Shape: (B, D)
        x = torch.sum(x * attn_weights, dim=-1)
        
        # Data normalization
        if (self.norm_type == 'layer'):
            x = self.layer_norm(x)            
        elif (self.norm_type == 'batch'):
            x = self.batch_norm(x)
           
        x = F.linear(self.activation(x), self.base_weight, self.base_weight_bias)
        return x
   
    def spatial_attn(self, x):
        # Shape: (B, D, G + k)
    
        # Calculate attention scores by using conv1d
        x = x.permute(0, 2, 1)
        attn_scores = self.conv1d(x)  
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
    
        # Shape: (B, D, G + k)
        attn_weights = attn_weights.expand_as(x)
        
        # Apply attention weights
        x = torch.sum(x * attn_weights, dim=-2)

        # Data normalization
        if (self.norm_type == 'layer'):
            x = self.layer_norm(x)
        elif (self.norm_type == 'batch'):
            x = self.batch_norm(x)
    
        # Linear transformation
        x = F.linear(self.activation(x), self.base_weight, self.base_weight_bias)
    
        return x
    
    
    def multihead_attn(self, x):
        """
            Multi-head Attention
        """
        # x: (B, D, G + k)

        # Apply multihead attention
        attn_output, _ = self.mh_attn(x, x, x)

        # Sum over the last dimension to get shape (B, D)
        x = attn_output.sum(dim=-1)

        # Data normalization
        if self.norm_type == 'layer':
            x = self.layer_norm(x)
        elif self.norm_type == 'batch':
            x = self.batch_norm(x)
            
        # Linear transformation
        x = F.linear(self.activation(x), self.base_weight, self.base_weight_bias)
        
        return x
    
    def self_attn(self, x):
        
        """
            Self Attention (Scaled Dot-Product Attention)
            Take long time
        """
        B, D, G_plus_k = x.size()
        
        # Shape: (B, G + k, D)
        #x = x.permute(0, 2, 1)

        # Linear transformations for query, key, and value
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)

        # Compute attention scores, G_plus_k is too small so better not include term (G_plus_k ** 0.5)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) #/ (G_plus_k ** 0.5)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)
        
        # Summing along the last dimension to convert (B, G + k, D) to (B, D)
        x = attn_output.sum(dim=-1)

        # Data normalization
        if self.norm_type == 'layer':
            x = self.layer_norm(x)
        elif self.norm_type == 'batch':
            x = self.batch_norm(x)
            
        # Linear transformation
        x = F.linear(self.activation(x), self.base_weight, self.base_weight_bias)
        
        return x

class PRReLUKAN(nn.Module):
    def __init__(self, width, grid = 5, k = 3, norm_type = 'layer', methods = ['local_attn'], combined_type = 'sum', base_activation = 'gelu'):
        super().__init__()
        self.width = width # net structure
        self.grid = grid # grid_size
        self.k = k # spline order
        self.norm_type = norm_type
        self.base_activation = base_activation
        self.combined_type = combined_type
        
        self.rk_layers = []
        for i in range(len(width) - 1):
            self.rk_layers.append(PRReLUKANLayer(width[i], grid, k, width[i+1], norm_type = norm_type, base_activation = base_activation, methods = methods, combined_type = combined_type))
            # if len(width) - i > 2:
            #     self.rk_layers.append()
        self.rk_layers = nn.ModuleList(self.rk_layers)

    def forward(self, x):
        for rk_layer in self.rk_layers:
            x = rk_layer(x)
        # x = x.reshape((len(x), self.width[-1]))
        return x
   
   
class ReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, norm_type = 'layer', train_ab: bool = True):
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

        # Data normalization
        self.norm_type = norm_type
        self.layer_norm = nn.LayerNorm(input_size)  
        self.batch_norm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        
        # Thank to reeyarn, https://github.com/quiqi/relu_kan/issues/1
        if (self.norm_type == 'layer'):
            x = self.layer_norm(x)            
        elif (self.norm_type == 'batch'):
            x = self.batch_norm(x)
            
        # Expand dimensions of x to match the shape of self.phase_low
        x_expanded = x.unsqueeze(2).expand(-1, -1, self.phase_low.size(1))
       
        # Perform the subtraction with broadcasting
        x1 = torch.relu(x_expanded - self.phase_low)
        x2 = torch.relu(self.phase_height - x_expanded)
       
        # Continue with the rest of the operations
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))

        x = self.equal_size_conv(x)
        #x = x.reshape((len(x), self.output_size, 1))
        x = x.reshape((len(x), self.output_size))
       
        return x
        
class ReLUKAN(nn.Module):
    def __init__(self, width, grid = 5, k = 3, norm_type = 'layer'):
        super().__init__()
        self.width = width # net structure
        self.grid = grid # grid_size
        self.k = k # spline order
        
        self.rk_layers = []
        for i in range(len(width) - 1):
            self.rk_layers.append(ReLUKANLayer(width[i], grid, k, width[i+1], norm_type = norm_type))
            #if len(width) - i > 2:
            #   self.rk_layers.append()
        self.rk_layers = nn.ModuleList(self.rk_layers)

    def forward(self, x):
        for rk_layer in self.rk_layers:
            x = rk_layer(x)
        #x = x.reshape((len(x), self.width[-1]))
        return x
