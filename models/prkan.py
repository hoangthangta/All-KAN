import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

    
class PRKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size = 5,
        spline_order = 3,
        grid_range = [-1.5, 1.5],
        num_grids: int = 8,
        func = 'rbf',
        norm_type = 'layer',  
        denominator: float = None,  # larger denominators lead to smoother basis
        base_activation = 'silu',
        methods = ['conv'],
        combined_type = 'sum'

    ) -> None:
        super().__init__()
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_activation = base_activation
        self.func = func
        self.norm_type = norm_type
        self.methods = methods
        self.combined_type = combined_type

        self.base_weight = torch.nn.Parameter(torch.Tensor(self.output_dim, self.input_dim))
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        
        self.base_weight_bias = torch.nn.Parameter(torch.Tensor(1, self.output_dim))
        torch.nn.init.kaiming_uniform_(self.base_weight_bias, a=math.sqrt(5))
        
        feature_dim = grid_size + spline_order
        if (func == 'rbf'): feature_dim = num_grids
        self.feature_weight = torch.nn.Parameter(torch.Tensor(feature_dim, 1))
        torch.nn.init.kaiming_uniform_(self.feature_weight, a=math.sqrt(5))
        
        # Norms
        self.layer_norm = nn.LayerNorm(self.input_dim)  
        self.batch_norm = nn.BatchNorm1d(self.input_dim)

        '''self.layer_norm1 = nn.LayerNorm(self.output_dim)  
        self.batch_norm1 = nn.BatchNorm1d(self.output_dim)'''
        
        # RBF
        self.grid_min = grid_range[0]
        self.grid_max = grid_range[1]
        self.num_grids = num_grids
        rbf_grid = torch.linspace(self.grid_min, self.grid_max, self.num_grids)
        self.denominator = denominator or (self.grid_max - self.grid_min) / (self.num_grids - 1)
        self.register_buffer("rbf_grid", rbf_grid)

        # B-spline
        h = (grid_range[1] - grid_range[0]) / grid_size 
        bs_grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h 
                + grid_range[0]
            )
            .expand(self.input_dim, -1)
            .contiguous()
        )
        self.register_buffer("bs_grid", bs_grid)

        # Use a conv layer
        self.conv1d_1 = nn.Conv1d(in_channels=feature_dim, out_channels=1, kernel_size=1, stride=1)
        
        # Use conv + maxpool layers, then linear
        self.conv1d_2 = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1, stride=1) 
        self.pool_2 = nn.MaxPool1d(kernel_size=feature_dim, stride=feature_dim)
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        
        # Use attention linear
        self.attention = nn.Linear(feature_dim, 1)
        
        #self.drop = nn.Dropout(p=0.1) # dropout

    def b_splines(self, x: torch.Tensor):
        """
            Compute the B-spline bases for the given input tensor.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            Returns:
                torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.input_dim

        grid: torch.Tensor = (
            self.bs_grid
        )  
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        
        assert bases.size() == (
            x.size(0),
            self.input_dim,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()   
    
    def rbf(self, x):
        return torch.exp(-((x[..., None] - self.rbf_grid) / self.denominator) ** 2)
    

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
    
    def add_features(self, x):
        x = x.view(x.size(0), -1)
        x = F.linear(self.activation(x), self.spline_weight)
        return x
        
    
    def use_conv1d_1(self, x):
        
        if (self.norm_type == 'layer'):
            x = self.layer_norm(x)       
        elif (self.norm_type == 'batch'):
            x = self.batch_norm(x)
            
        if (self.func == 'rbf'):
            x = self.rbf(x) 
        elif (self.func == 'bs'): 
            x = self.b_splines(x)
        else:
            raise Exception('The function "' + self.func + '" does not support!')
        
        x = x.permute(0, 2, 1) 
        x = self.conv1d_1(x) 
        x = x.squeeze(1)  
        x = F.linear(self.activation(x), self.base_weight, self.base_weight_bias)
        
        return x
    
    def use_conv1d_2(self, x):
        
        if (self.norm_type == 'layer'):
            x = self.layer_norm(x)       
        elif (self.norm_type == 'batch'):
            x = self.batch_norm(x)
            
        if (self.func == 'rbf'):
            x = self.rbf(x) 
        elif (self.func == 'bs'): 
            x = self.b_splines(x)
        else:
            raise Exception('The function "' + self.func + '" does not support!')

        x = x.permute(0, 2, 1)  
        x = self.conv1d_2(x)  
        #print(x.shape)
        x = self.pool_2(x) 
        x = x.permute(0, 2, 1)  
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)

        return x
        
    def use_dim_sum(self, x):
        
        if (self.func == 'rbf'):
            x = self.rbf(x) 
        elif (self.func == 'bs'): 
            x = self.b_splines(x)
        else:
            raise Exception('The function "' + self.func + '" does not support!')
            
        x = torch.sum(x, dim=2) 
        
        if (self.norm_type == 'layer'):
            x = self.layer_norm(x)       
        elif (self.norm_type == 'batch'):
            x = self.batch_norm(x)
            
        x = F.linear(self.activation(x), self.base_weight, self.base_weight_bias)
        return x
    
    def use_feature_weight(self, x):
        
        if (self.norm_type == 'layer'):
            x = self.layer_norm(x)       
        elif (self.norm_type == 'batch'):
            x = self.batch_norm(x)
            
        if (self.func == 'rbf'):
            x = self.rbf(x) 
        elif (self.func == 'bs'): 
            x = self.b_splines(x)
        else:
            raise Exception('The function "' + self.func + '" does not support!')
            
        x = torch.matmul(x, self.feature_weight)
        x = x.view(x.size(0), -1)
        x = F.linear(self.activation(x), self.base_weight, self.base_weight_bias)
        
        return x
    
    
    def use_attention(self, x):

        if (self.func == 'rbf'):
            x = self.rbf(x) 
        elif (self.func == 'bs'): 
            x = self.b_splines(x)
        else:
            raise Exception('The function "' + self.func + '" does not support!')
        
        attn_weights = F.softmax(self.attention(x), dim=-2)  
        x = x * attn_weights
        x = x.sum(dim=-1) 
        
        if (self.norm_type == 'layer'):
            x = self.layer_norm(x)       
        elif (self.norm_type == 'batch'):
            x = self.batch_norm(x)
            
        x = F.linear(self.activation(x), self.base_weight, self.base_weight_bias)
        
        return x
    
    
    def use_base(self, x):
        
        if (self.norm_type == 'layer'):
            x = self.layer_norm(x)       
        elif (self.norm_type == 'batch'):
            x = self.batch_norm(x)
            
        x = self.activation(F.linear(x, self.base_weight, self.base_weight_bias))
        return x
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        output = torch.zeros(len(self.methods), x.shape[0], self.output_dim).to(x.device)
        for i, method in zip(range(len(self.methods)), self.methods):
            temp = x.clone()
            if (method == 'base'):
                temp = self.use_base(x)
            elif (method == 'conv1d_1'): # convolution
                temp = self.use_conv1d_1(x)
            elif (method == 'conv1d_2'): # convolution
                temp = self.use_conv1d_2(x)
            elif (method == 'attention'): # convolution
                temp = self.use_attention(x)
            elif (method == 'fw'): # feature weight
                temp = self.use_feature_weight(x)
            elif (method == 'ds'): # dim sum
                temp = self.use_dim_sum(x)
            else:
                raise Exception('The method "' + method + '" does not support!')
            output[i] = temp
            
        # This is for a single method. When using "base" alone, it is a MLP.
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
               
class PRKAN(torch.nn.Module):
    
    def __init__(
        self, 
        layers_hidden,
        grid_size=5,
        spline_order=3, 
        grid_range = [-1.5,  1.5],
        num_grids: int = 8,
        func = 'rbf',
        norm_type = 'layer', 
        base_activation='silu',
        methods = [],
        combined_type = 'sum'
    ):
        super(PRKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        #self.drop = torch.nn.Dropout(p=0.1) # dropout

        for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                PRKANLayer(
                    input_dim,
                    output_dim,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    grid_range = grid_range,
                    num_grids = num_grids,
                    func = func,
                    norm_type = norm_type, 
                    base_activation=base_activation,
                    methods = methods,
                    combined_type = combined_type
                )
            )
    
    def forward(self, x: torch.Tensor):
        #x = self.drop(x)
        
        for layer in self.layers: 
            x = layer(x)
        return x
        
