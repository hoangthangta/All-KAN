import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base_activation='silu',
        norm_type='layer',
        use_attn=False
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_activation = base_activation
        self.norm_type = norm_type
        self.use_attn = use_attn
        
        self.base_weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        
        self.temperature = nn.Parameter(torch.tensor(math.sqrt(input_dim)))
        
        # Normalization
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(input_dim)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(input_dim)
        else:
            self.norm = nn.Identity()  # No-op normalization
        
        self.attn_proj = nn.Linear(input_dim, input_dim)
    
    def activation(self, x):
        """
        We found that F.* activation functions produce better performance
        than torch.nn.* activation functions.
        """
        activation_funcs = {
            'softplus': F.softplus,
            'sigmoid': F.sigmoid,
            'silu': F.silu,
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'gelu': F.gelu,
            'selu': F.selu,
        }
        return activation_funcs.get(self.base_activation, lambda x: x)(x)
    
    def global_attn(self, x):
        attn_scores = self.attn_proj(x)
        attn_weights = F.softmax(attn_scores / self.temperature.clamp(min=1.0), dim=-1)
        return x * attn_weights
        
    def forward(self, x):
        if self.use_attn:
            x = self.global_attn(x)
        x = self.norm(x)
        return self.activation(F.linear(x, self.base_weight))

class MLP(nn.Module):
    def __init__(
        self,
        layers_hidden,
        base_activation='silu',
        norm_type='layer',
        use_attn=False
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MLPLayer(input_dim, output_dim, base_activation, norm_type, use_attn)
            for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:])
        ])
    
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
