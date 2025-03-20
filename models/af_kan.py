# Modified from: https://github.com/quiqi/relu_kan
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
        
class AF_KANLayer(nn.Module):
    def __init__(   
                    self, 
                    input_size: int, 
                    g: int, k: int, 
                    output_size: int, 
                    norm_type = 'layer',
                    base_activation = 'silu', 
                    methods = ['global_attn'], 
                    combined_type = 'sum', 
                    window_size = 2, 
                    step = 2, 
                    train_ab: bool = True,
                    func = 'quad1', # the function used to extract data features
                    func_norm = True,
                ):
        super().__init__()
        self.g, self.k = g, k
        self.input_size, self.output_size = input_size, output_size
        self.base_activation = base_activation # set before compute r
        self.func = func
        self.func_norm = func_norm

        phase_low = torch.arange(-k, g) / g
        phase_high = phase_low + (k + 1) / g
        self.phase_low = nn.Parameter(phase_low, requires_grad=train_ab)
        self.phase_high = nn.Parameter(phase_high, requires_grad=train_ab)
        
        #self.phase_low = nn.Parameter(phase_low[None, :].expand(input_size, -1), requires_grad=train_ab)                          
        #self.register_buffer("phase_low", phase_low[None, :].expand(input_size, -1))
        #self.phase_high = nn.Parameter(phase_high[None, :].expand(input_size, -1), requires_grad=train_ab)
        #self.register_buffer("phase_high", phase_high[None, :].expand(input_size, -1))
        
        #relu_r = torch.tensor(4*g*g / ((k+1)*(k+1)))
        #self.r = self.compute_r()
        
        # Controls attention sharpness: higher values smooth, lower values sharpen.
        self.temperature = nn.Parameter(torch.tensor(math.sqrt(self.input_size))) 
        #self.temperature = nn.Parameter(torch.tensor(1.0)) 
        
        self.base_weight = nn.Parameter(torch.Tensor(self.output_size, self.input_size))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
       
        self.base_weight_bias = nn.Parameter(torch.Tensor(1, self.output_size))
        nn.init.zeros_(self.base_weight_bias)
        
        self.methods = methods
        self.combined_type = combined_type
        
        # Data norm
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(input_size)
        elif(norm_type == 'batch'):
            self.norm = nn.BatchNorm1d(input_size)
        else:
            self.norm = nn.Identity()  # No-op normalization
         
        # Self-attention
        self.query_self = nn.Linear(g + k, g + k)
        self.key_self = nn.Linear(g + k, g + k)
        self.value_self = nn.Linear(g + k, g + k)
        
        # Spatial attention
        self.conv1d = nn.Conv1d(in_channels=g + k, out_channels=g + k, kernel_size=3, stride=1, padding=1, groups=g + k)
        
        # Multihead attention
        self.mh_attn = nn.MultiheadAttention(embed_dim = g + k, num_heads=g + k, batch_first=True)
        
        # Local attention
        self.window_size = window_size
        self.step = step
        self.query_local = nn.Linear(window_size, window_size)
        self.key_local = nn.Linear(window_size, window_size)
        self.value_local = nn.Linear(window_size, window_size) 

        local_dim = int((g + k - window_size)/step + 1)*window_size
        self.local_linear = nn.Linear(local_dim, 1)
        
        # Global attention +  Simple Linear
        self.gk_linear = nn.Linear(g + k, 1)
    
    def forward(self, X):

        device = X.device
        output = torch.zeros(X.shape[0], X.shape[1], self.output_size).to(device)
        
        for i, method in zip(range(X.shape[0]), self.methods):
            x = X[i, :, :].squeeze(0)
            x = self.extract_features(x)

            if (method == 'global_attn'):
                x = self.global_attn(x)
            elif (method == 'multistep'):
                x = self.multistep(x)
            elif (method == 'spatial_attn'):
                x = self.spatial_attn(x) 
            elif (method == 'local_attn'): # slow, memory error
                x = self.local_attn(x)
            elif (method == 'self_attn'): # slow, memory error
                x = self.self_attn(x) 
            elif (method == 'multihead_attn'): # slow, memory error
                x = self.multihead_attn(x)
            else:
                raise Exception('The method "' + method + '" does not support!')
            output[i] = x
        
        #print(self.temperature)
        return output
    
    '''def normalize_x(self, x):
        """
            # Scale to [-self.k/self.g,(self.k + self.g)/self.g]
        """
        #target_min = -self.k/self.g
        #target_max = (self.k + self.g)/self.g
        #target_max = 1/self.g # same result
        
        #target_min = 0
        #target_max = 1
        eps = 1e-2
        target_min = (self.k - 2)/self.g + eps
        target_max = (self.k - 1)/self.g - eps
        
        original_min = x.min() 
        original_max = x.max()  
        
        return target_min + (x - original_min) * (target_max - target_min) / (original_max - original_min)'''
    
    def normalize(self, x):
        x = F.normalize(x, p=2, dim=None)  # Normalize using L2 norm
        return (x - x.min()) / (x.max() - x.min())  # Scale to [0,1]

    '''def extract_features(self, x):
        # Thank to "reeyarn" and "davidchern": 
        #   - https://github.com/quiqi/relu_kan/issues/1
        #   - https://github.com/quiqi/relu_kan/issues/2
        
        low = self.phase_low[None, :].expand(x.size(0), self.input_size, -1)
        high = self.phase_high[None, :].expand(x.size(0), self.input_size, -1)
        
        # Expand dimensions of x to match the shape of self.phase_low
        if (len(x.shape) == 2):
            x_expanded = x.unsqueeze(2).expand(-1, -1, low.size(-1))
            
        else: 
            # Avoid an error in calculating FLOPs only
            x_expanded = x.unsqueeze(0).unsqueeze(-1).expand(-1, -1, low.size(1))
        
        x = self.cal_output(low, high, x_expanded)
        return x'''
    
    def extract_features(self, x):
        # Thank to "reeyarn" and "davidchern": 
        #   - https://github.com/quiqi/relu_kan/issues/1
        #   - https://github.com/quiqi/relu_kan/issues/2
        
        low = self.phase_low[None, :].expand(self.input_size, -1)
        high = self.phase_high[None, :].expand(self.input_size, -1)
        
        # Expand dimensions of x to match the shape of self.phase_low
        if (len(x.shape) == 2):
            x_expanded = x.unsqueeze(2).expand(-1, -1, low.size(1))
        else: 
            # Avoid an error in calculating FLOPs only
            x_expanded = x.unsqueeze(0).unsqueeze(-1).expand(-1, -1, low.size(1))
        
        x = self.cal_output(low, high, x_expanded)
        return x
    
    def cal_output(self, low, high, x):
        """
            Compute the output based on func.
        """
        x1 = self.activation(x - low)
        x2 = self.activation(high - x)

        x1_x2 = None
        x1_sq = None
        x2_sq = None

        if self.func in ['quad1', 'quad2', 'prod', 'sum_prod', 'cubic2']:
            x1_x2 = x1 * x2  # Only compute when necessary

        if self.func in ['quad2', 'cubic1']:
            x1_sq = x1 * x1
            x2_sq = x2 * x2

        func_dict = {
            'quad1': lambda: x1_x2**2,
            'quad2': lambda: x1_x2 + x1_sq + x2_sq,
            'sum': lambda: x1 + x2,
            'prod': lambda: x1_x2,
            'sum_prod': lambda: x1 + x2 + x1_x2,
            'cubic1': lambda: (x1 + x2) * (x1_sq + x2_sq),
            'cubic2': lambda: x1_x2**3,
        }
        
        try:
            if (self.func_norm == True):
                return self.normalize(func_dict[self.func]())
            return func_dict[self.func]()  
        except KeyError:
            raise ValueError(f"Unknown func: {self.func}")

    '''def compute_r(self):
        """ 
            Compute the parameter r for normalizing to the range [0, 1]
        """
        a = - self.k / self.g
        b = 1 / self.g
        max_x = (1-self.k) / (2*self.g)
        max_value = self.cal_output(torch.tensor(a), torch.tensor(b), torch.tensor(max_x))
        r = 1/max_value
        print(a, b, max_x, max_value, r)
        return r'''
    
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
            'tanh': F.tanh
        }
        af = af_list.get(self.base_activation, lambda x: x)
        return af(x)

    def multistep(self, x):
        """
            Function linearization
            Convert g + k (the number of functions) to 1.
        """
        # Linear transformation
        x = self.gk_linear(x)
        #x = x.view(x.size(0), -1) # worse than using torch.reshape
        x = x.reshape(x.size(0), -1)
        
        # Data normalization
        x = self.norm(x)   
        
        # Linear transformation
        x = F.linear(self.activation(x), self.base_weight)
        #x = F.linear(self.activation(x), self.base_weight, self.base_weight_bias)
        
        return x

    def global_attn(self, x):
        """
            Global Attention
        """
        x_linear = self.gk_linear(x)
        
        # Apply softmax to get weights + temperature scaling
        attn_weights = F.softmax(x_linear/self.temperature.clamp(min=1.0), dim=-2)  
        #attn_weights = F.softmax(x_linear/self.temperature, dim=-2)  
        
        # Shape: (B, D)
        x = torch.sum(x * attn_weights, dim=-1)
        
        # Data normalization
        x = self.norm(x)  
        
        # Linear transformation
        x = F.linear(self.activation(x), self.base_weight)
        return x
   
    def spatial_attn(self, x):
        """
            Spatial Attention
        """
        
        # Calculate attention scores by using conv1d
        attn_scores = self.conv1d(x.permute(0, 2, 1) )  # Shape: (B, G + k, D)

         # Apply softmax along spatial dimension (G + k)
        attn_weights = F.softmax(attn_scores / self.temperature.clamp(min=1.0), dim=1)
        #attn_weights = F.softmax(attn_scores / self.temperature.clamp(min=0.1), dim=1)
        
        # Apply attention weights and sum over spatial dimension
        x = torch.sum(x * attn_weights.permute(0, 2, 1), dim=-1)
        
        # Data normalization
        x = self.norm(x)  
    
        # Linear transformation
        x = F.linear(self.activation(x), self.base_weight)
    
        return x
    
    
    def local_attn(self, x):
        """
            Local Attention, slow
        """

        # Unfold the last dimension (G + k) into local windows
        local_windows = x.unfold(dimension=-1, size=self.window_size, step=self.step) # Shape: (B, D, num_windows, window_size)
        
        # Compute query and key
        query = self.query_local(local_windows)  # (B, D, num_windows, window_size)
        key = self.key_local(local_windows) # (B, D, num_windows, window_size)
        value = self.value_local(local_windows) # (B, D, num_windows, window_size)
        
        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-1, -2))  # Shape: (B, D, num_windows, num_windows)
        
        # Apply softmax to get weights
        attn_weights = F.softmax(attn_scores / self.temperature.clamp(min=1.0), dim=-1)  # (B, D, num_windows, num_windows)

        # Compute weighted sum correctly using matmul
        weighted_sum = torch.matmul(attn_weights, value)  # (B, D, num_windows, window_size)

        # Reduce dimensionality
        # x = weighted_sum.sum(dim=(-1, -2))  # May lose data features
        x = weighted_sum.reshape(x.size(0), x.size(1), -1)
        x = self.local_linear(x)
        x = x.squeeze(-1)
    
        # Data normalization
        x = self.norm(x)  
     
        # Linear transformation
        x = F.linear(self.activation(x), self.base_weight)
        
        return x
    
    def multihead_attn(self, x):
        """
            Multi-head Attention, slow
        """
        # x: (B, D, G + k)

        # Apply multihead attention
        attn_output, _ = self.mh_attn(x, x, x)

        # Sum over the last dimension to get shape (B, D)
        x = attn_output.sum(dim=-1)

        # Data normalization
        x = self.norm(x)  
            
        # Linear transformation
        x = F.linear(self.activation(x), self.base_weight)
        
        return x
    
    def self_attn(self, x):
        
        """
            Self Attention (Scaled Dot-Product Attention), slow
        """
        B, D, G_plus_k = x.size()
        
        # Inout shape: (B, G + k, D)
        #x = x.permute(0, 2, 1)

        # Linear transformations for query, key, and value
        Q = self.query_linear(x)
        K = self.key_linear(x)
        V = self.value_linear(x)

        # Compute attention scores, G_plus_k is too small so better not include term (G_plus_k ** 0.5)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) #/ (G_plus_k ** 0.5)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores / self.temperature.clamp(min=1.0), dim=-1)
        
        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)
        
        # Summing along the last dimension to convert (B, G + k, D) to (B, D)
        x = attn_output.sum(dim=-1)

        # Data normalization
        x = self.norm(x)  
            
        # Linear transformation
        x = F.linear(self.activation(x), self.base_weight)
        
        return x

          
class AF_KAN(nn.Module):
    def __init__(
                    self, 
                    width, 
                    grid = 5, 
                    k = 3, 
                    norm_type = 'layer',
                    base_activation = 'silu',
                    methods = ['global_attn'], 
                    combined_type = 'sum',
                    func = 'quad1',
                    func_norm = True,
                ):
        super().__init__()
        self.width = width # net structure
        self.grid = grid # grid_size
        self.k = k # spline order
        self.norm_type = norm_type
        self.base_activation = base_activation
        self.methods = methods
        self.func = func
        self.func_norm = func_norm
        
        self.rk_layers = []
        for i in range(len(width) - 1):
            self.rk_layers.append(
                AF_KANLayer(
                    width[i], grid, k, width[i+1], 
                    norm_type = norm_type,
                    base_activation = base_activation,
                    func = func,
                    func_norm = func_norm,
                )
            )
        self.rk_layers = nn.ModuleList(self.rk_layers)

    def forward(self, x):
  
        X = torch.stack([x] * len(self.methods)) # size (number of methods, batch_size, input_dim)
        for rk_layer in self.rk_layers: 
            X = rk_layer(X)
        
        if (len(self.methods) == 1): return X[0]
        
        output = X.detach().clone()
        if (self.combined_type == 'sum'): output = torch.sum(X, dim=0)
        elif (self.combined_type == 'product'):  
            '''
            # Use only for very large tensors. This is slower and can have cumulative numerical errors
            output_prod = torch.ones(X.shape[1:], device=X.device)
            for i in range(X.shape[0]):
                output_prod *= X[i, :, :]
            '''
            output = torch.prod(X, dim=0)
        elif (self.combined_type == 'sum_product'): output = torch.sum(X, dim=0) +  torch.prod(X, dim=0)
        elif (self.combined_type == 'quadratic'): 
            output = torch.sum(X, dim=0) +  torch.prod(X, dim=0) 
            for i in range(X.shape[0]):
                output = output + X[i, :, :].squeeze(0)*X[i, :, :].squeeze(0)
            #output += torch.sum(X ** 2, dim=0) # can lead to memory error
        else:
            raise Exception('The combined type "' + self.combined_type + '" does not support!')
            # write more combinations here

        return output