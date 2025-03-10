# https://github.com/IcurasLW/FR-KAN/blob/main/src/efficient_kan/rational_kan.py
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import random
import numpy as np

# For results reproduction
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


class MMEncoder(torch.nn.Module):
    def __init__(self, args, in_feature, out_feature):
        super().__init__()
        self.args = args
        self.encoder = nn.Linear(in_feature, out_feature)
    def forward(self, x):
        return self.encoder(x)
    
    
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        P_order=3,
        Q_order=3,
        groups=8,
        need_relu=True
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert in_features % groups == 0 or groups == -1, 'Please select the group number divisible to input shape'
        
        if groups == -1:
            # if == -1, means no groups
            self.groups = in_features
        else:
            self.groups = groups
        
        
        self.num_neurons_in_g = in_features // self.groups
        self.P_order = P_order
        self.Q_order = Q_order
        self.w_a = torch.nn.Parameter(torch.Tensor(self.P_order, self.groups))
        self.w_b = torch.nn.Parameter(torch.Tensor(self.Q_order - 1, self.groups))
        
        
        self.ln_weight = torch.nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.shortcut = nn.SiLU()
        self.need_relu = need_relu
        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.w_a, a=math.sqrt(5) * 1)
        torch.nn.init.kaiming_uniform_(self.w_b, a=math.sqrt(5) * 1)
        torch.nn.init.kaiming_uniform_(self.ln_weight, a=math.sqrt(5) * 1)



    def rational_basis(self, x: torch.Tensor):
        '''
        x --> [B, E], where E is input dimension
        self.ax in shape [P_order, E]
        self.bx in shape [Q_order, E]
        self.order = 3
        '''
        
        weight_a = self.w_a.repeat(1, self.num_neurons_in_g)
        weight_b = self.w_b.repeat(1, self.num_neurons_in_g)
        P_x = torch.stack([x**i for i in range(len(self.w_a))], dim=-1)
        Q_x = torch.stack([x**i for i in range(len(self.w_b))], dim=-1)
        # P_x = torch.sum(torch.matmul(weight_a, P_x), dim=-1)
        # Q_x = 1 + torch.sum(torch.matmul(weight_b, Q_x), dim=-1).abs()
        
        P_x = torch.einsum('ke,bek->be', weight_a, P_x)
        Q_x = 1 + (torch.einsum('ke,bek->be', weight_b, Q_x)).abs()
        
        
        
        
        return P_x/Q_x 


    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        output = self.rational_basis(x)
        
        output = torch.matmul(output, self.ln_weight) 
        if self.need_relu:
            shortcut_x = torch.matmul(self.shortcut(x), self.ln_weight)
            output = output + shortcut_x
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output


class RationalKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        P_order=3,
        Q_order=3,
        groups=8,
        need_relu=True
        ):
        super(RationalKAN, self).__init__()
        self.n_hid = len(layers_hidden)
        self.layers = torch.nn.ModuleList()
        for idx, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            self.layers.append(
                    KANLinear(
                        in_features,
                        out_features,
                        P_order=P_order,
                        Q_order=Q_order,
                        groups=groups,
                        need_relu=need_relu
                    )
                )
    
    
    def forward(self, x: torch.Tensor, update_grid=False, normalize=False):
        if normalize:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-6).detach() 
            x /= stdev
        
        enc_x = self.layers[0](x)
        hid_x = enc_x
        
        for layer in self.layers[1:-1]:
            hid_x = layer(hid_x)
            
        hid_x = self.layers[-1](hid_x)
        
        if normalize:
            hid_x = hid_x * stdev
            hid_x = hid_x + means
        return hid_x


