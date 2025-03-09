import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import random
import numpy as np

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


class ChebyKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        degree=3
        ):
        super(ChebyKAN, self).__init__()
        
        
        self.layers = torch.nn.ModuleList()
        self.layer_norm = torch.nn.ModuleList()
        for idx, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            self.layer_norm.append(
                nn.LayerNorm(in_features)
            )
            self.layers.append(
                ChebyKANLayer(
                    in_features,
                    out_features,
                    degree
                )
            )
    
    def forward(self, x: torch.Tensor, normalize=False):
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