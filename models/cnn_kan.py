import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class CNN_KANC(nn.Module):
    def __init__(self, 
        layer_list = [784, 64, 10],
        nets = ['conv2','rbf'], 
        net_type = 'independent', 
        kan_norm = 'layer', 
        conv_norm = 'batch', 
        combined_norm = 'none',
        combined_type = 'rbf', 
        base_activation = 'selu',
        batch_size = 64,
        grid_size = 5,
        spline_order = 3,
        grid_range = [-1.5, 1.5],
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
            
        super(CNN_KANC, self).__init__()

        self.layer_list = layer_list
        
        if (len(self.layer_list) != 3):
            raise Exception('The number of layers should be 3, not "' + len(self.layer_list) + '".')
            
        self.input_dim = layer_list[0]
        self.hidden_dim = layer_list[1]
        self.output_dim = layer_list[-1]
        
        self.nets = nets
        if (len(self.nets) < 1): #len(self.nets) > 2 or 
            raise Exception('The number of nets should be at least 1, not "' + len(self.nets) + '".')
        if (len(self.nets) == 1): 
            if (self.nets[0] == ''):
                print('The model now becomes an MLP!')
                
        self.kan_norm = kan_norm
        self.conv_norm = conv_norm
        self.combined_norm = combined_norm
        self.combined_type = combined_type
        self.net_type = net_type
        self.base_activation = base_activation
        self.batch_size = batch_size # for instance normalization
        self.grid_range = grid_range
        self.spline_order = spline_order
        self.grid_size = grid_size
        
        self.scale = nn.Parameter(torch.ones(self.output_dim, self.input_dim))
        self.translation = nn.Parameter(torch.zeros(self.output_dim, self.input_dim))
        
        self.base_weight = torch.nn.Parameter(torch.Tensor(self.output_dim, self.input_dim))
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
           
        # 1 convolutional layer
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # 16 filters
        self.pool1_1 = nn.MaxPool2d(4, 4)
        
        # 2 convolutional layers
        self.conv2_1 = nn.Conv2d(1, 8, kernel_size=3, padding=1) # 8 filters
        self.pool2_1 = nn.MaxPool2d(2, 2)
        self.conv2_2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # 16 filters
        self.pool2_2 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim) # Reduced input size and output size
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim) # Final output layer
        
        # Layer Normalization
        self.layer_norm1_1 = nn.LayerNorm(12544)
        self.layer_norm2_1 = nn.LayerNorm(6272)
        self.layer_norm2_2 = nn.LayerNorm(3136)
        self.layer_norm = nn.LayerNorm(self.input_dim)  # 28*28

        # Batch Normalization
        self.batch_norm1_1 = nn.BatchNorm2d(16)
        self.batch_norm2_1 = nn.BatchNorm2d(8)
        self.batch_norm2_2 = nn.BatchNorm2d(16)
        self.batch_norm = nn.BatchNorm1d(self.input_dim) # 28*28
        
        # Instance Normalization
        self.instance_norm1_1 = nn.InstanceNorm2d(16)
        self.instance_norm2_1 = nn.InstanceNorm2d(8)
        self.instance_norm2_2 = nn.InstanceNorm2d(16)
        self.instance_norm = nn.InstanceNorm1d(batch_size, affine=True)
        
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
        
        #self.dropout = nn.Dropout(0.05) 
        
    def combine_attention(self, x_set):
        """
        Combine a set of tensors using an attention mechanism.
        
        Args:
            x_set (torch.Tensor): Tensor of shape (n, batch_size, feature_dim).
                                  n = number of tensors.
        
        Returns:
            torch.Tensor: Combined tensor of shape (batch_size, feature_dim).
        """
        n, batch_size, feature_dim = x_set.shape

        # Compute pairwise attention scores
        # Flatten tensors along batch and feature dimensions
        queries = x_set.view(n, -1)  # Shape: (n, batch_size * feature_dim)
        keys = x_set.view(n, -1).T  # Shape: (batch_size * feature_dim, n)

        attention_scores = F.softmax(torch.matmul(queries, keys), dim=1)  # Shape: (n, n)

        # Apply attention scores to combine tensors
        # Expand dimensions for broadcasting
        attention_scores = attention_scores.unsqueeze(-1).unsqueeze(-1)  # Shape: (n, n, 1, 1)
        weighted_tensors = attention_scores * x_set.unsqueeze(0)  # Shape: (n, n, batch_size, feature_dim)

        # Sum and prod over the set dimension
        combined_tensor = torch.sum(weighted_tensors, dim=0)
        combined_tensor = torch.prod(combined_tensor, dim=0)
        
        return combined_tensor
    
    def wavelet_transform(self, x, wavelet_type = 'dog'):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded
            
        # Implementation of different wavelet types
        if wavelet_type == 'mexh':
            term1 = ((x_scaled ** 2)-1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.base_weight.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=1)
        elif wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.base_weight.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=1)
            
        elif wavelet_type == 'dog':
            # Implementing Derivative of Gaussian Wavelet 
            dog = - x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            #dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            #dog = 2 * x_scaled * (1 - x_scaled ** 2) * dog
    
            wavelet = dog
            wavelet_weighted = wavelet * self.base_weight.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=1)
            #wavelet_output = wavelet_weighted
                
        elif wavelet_type == 'meyer':
            # Implement Meyer Wavelet here
            # Constants for the Meyer wavelet transition boundaries
            v = torch.abs(x_scaled)
            pi = math.pi

            def meyer_aux(v):
                return torch.where(v <= 1/2,torch.ones_like(v),torch.where(v >= 1,torch.zeros_like(v),torch.cos(pi / 2 * nu(2 * v - 1))))

            def nu(t):
                return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)
            # Meyer wavelet calculation using the auxiliary function
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.base_weight.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=1)
        elif wavelet_type == 'shannon':
            # Windowing the sinc function to limit its support
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)  # sinc(x) = sin(pi*x) / (pi*x)

            # Applying a Hamming window to limit the infinite support of the sinc function
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype, device=x_scaled.device)
            # Shannon wavelet is the product of the sinc function and the window
            wavelet = sinc * window
            wavelet_weighted = wavelet * self.base_weight.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=1)
            #You can try many more wavelet types ...
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output
    
    def rbf(self, x):
        return torch.exp(-((x[..., None] - self.rbf_grid) / self.denominator) ** 2)
        
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
    
    def extract_features(self, x, nets = ['conv1', 'rbf'], net_type = 'composition', kan_norm = 'layer', 
                            conv_norm = 'batch', combined_type = 'product'):
        
        x_ori = x
        device = x.device
        output = torch.zeros(len(nets), x.shape[0], x.shape[1]).to(device)
        
        # Returning to x, the model becomes an MLP
        if (len(nets) == 0): return x
        if (len(nets) == 1): 
            if (nets[0] == ''):
                return x
        
        if (net_type == 'independent'):
            for i, net in enumerate(nets):
                if (net == 'conv1'):
                    x_out = self.extract_conv1(x, use_norm = conv_norm)     
                elif (net == 'conv2'):
                    x_out = self.extract_conv2(x, use_norm = conv_norm)     
                elif (net in ['rbf', 'bs', 'dog']):
                    x_out = self.extract_kan(x, use_norm = kan_norm, function = net)   
                else:
                    raise Exception('The network "' + net + '" does not support!')
                x_out = x_out.view(x_out.size(0), -1) 
                output[i] = x_out
        else: # composition
            for i, net in enumerate(nets):
                if (net == 'conv1'):
                    x = self.extract_conv1(x, use_norm = conv_norm)  
                elif (net == 'conv2'):
                    x = self.extract_conv2(x, use_norm = conv_norm)
                elif (net in ['rbf', 'bs', 'dog']):
                    x = self.extract_kan(x, use_norm = kan_norm, function = net)
                else:
                    raise Exception('The network "' + net + '" does not support!')
                
                x = x.view(x.size(0), -1) 
                output[i] = x
        
        #print(output[0].shape, output[1].shape)
        
        # Combine outputs
        if (output.size(0) == 1): return output[0] # only 1 output
        if (combined_type == 'product'):
            output = torch.prod(output, dim=0)
        elif (combined_type == 'sum'):
            output = torch.sum(output, dim=0) 
        elif (combined_type == 'sum_product'):
            output = torch.sum(output, dim=0) +  torch.prod(output, dim=0)
        elif (combined_type == 'attention'):
            output = self.combine_attention(output)
        else:
            raise Exception('The combination type "' + combined_type + '" does not support!')
                
        return output
    
    def extract_kan(self, x, use_norm = 'layer', function = 'rbf'):
        
        # Flattened original input
        x = x.view(x.size(0), -1)      
        if (use_norm == 'layer'):
            x = self.layer_norm(x)       
        elif (use_norm == 'batch'):
            x = self.batch_norm(x) 
        elif (use_norm == 'instance'):
            # for the last batch
            if (x.size(0) != self.batch_size):
                padding = torch.zeros(self.batch_size - x.size(0), x.size(1)).to(x.device)
                new_x = torch.cat((x, padding), dim=0)
                new_x = self.instance_norm(new_x)
                x = new_x[:x.size(0), :]
            else:
                x = self.instance_norm(x)
        
        if (function == 'rbf'):
            x = self.rbf(x) 
        elif (function == 'bs'): 
            x = self.b_splines(x)
        elif (function in ['dog', 'mexh']): 
            x = self.wavelet_transform(x, wavelet_type = function)
            return x
        else:
            raise Exception('The function "' + function + '" does not support!')
            
        # Summing along one dimension
        x = torch.sum(x, dim=2) 

        return x
    
    
    def extract_conv1(self, x, use_norm = 'batch'):
        
        # Reshape the input to (batch, channel, height, width)
        n = int(math.sqrt(x.size(1)))
        x = x.view(-1, 1, n, n)

        # Convolutional layers with SELU activation and pooling
        x = self.conv1_1(x) # (batch, 16, 28, 28)
        
        if (use_norm == 'instance'):
            x = self.instance_norm1_1(x) 
        elif(use_norm == 'batch'):
            x = self.batch_norm1_1(x)  
        elif(use_norm == 'layer'):
            y = x.view(x.size(0), -1) 
            y = self.layer_norm1_1(y)
            #y_reshaped = y.view(y.shape[0], *x.shape[1:])  (batch, 16, 7, 7)
            x = y.view(*x.shape)  # or y.reshape(*x.shape)
        x = self.activation(x)        
        x = self.pool1_1(x) # (batch, 16, 7, 7)
        return x
    
    def extract_conv2(self, x, use_norm = 'batch'):
        # Reshape the input to (batch, channel, height, width)
        n = int(math.sqrt(x.size(1)))
        x = x.view(-1, 1, n, n)

        # Convolutional layers with SELU activation and pooling
        x = self.conv2_1(x) # (batch, 8, 28, 28)
        if (use_norm == 'instance'):
            x = self.instance_norm2_1(x) 
        elif(use_norm == 'batch'):
            x = self.batch_norm2_1(x)  
        elif(use_norm == 'layer'):
            y = x.view(x.size(0), -1) 
            y = self.layer_norm2_1(y)
            #y_reshaped = y.view(y.shape[0], *x.shape[1:])  (batch, 8, 14, 14)
            x = y.view(*x.shape)  # or y.reshape(*x.shape)
        x = self.activation(x)        
        x = self.pool2_1(x) # (batch, 8, 14, 14)
        
        x = self.conv2_2(x) # (batch, 16, 14, 14)
        if (use_norm == 'instance'):
            x = self.instance_norm2_2(x) 
        elif(use_norm == 'batch'):
            x = self.batch_norm2_2(x)
        elif(use_norm == 'layer'):
            y = x.view(x.size(0), -1) 
            y = self.layer_norm2_2(y)
            #y_reshaped = y.view(y.shape[0], *x.shape[1:])  (batch, 8, 14, 14)
            x = y.view(*x.shape)  # or y.reshape(*x.shape)
        x = self.activation(x) # (batch, 16, 14, 14)
        x = self.pool2_2(x) # (batch, 16, 7, 7)
        
        return x

    def forward(self, x):
        
        #x = self.dropout(x)
        # Extract features
        x = self.extract_features(x, nets = self.nets, net_type = self.net_type, kan_norm = self.kan_norm, 
                                    conv_norm = self.conv_norm, combined_type = self.combined_type)

        # Fully connected layers
        x = x.view(x.size(0), -1)

        # Do normalization again
        if (self.combined_norm == 'layer'):
            x = self.layer_norm(x)       
        elif (self.combined_norm == 'batch'):
            x = self.batch_norm(x) 
        elif (self.combined_norm == 'instance'):
            # for the last batch
            if (x.size(0) != self.batch_size):
                padding = torch.zeros(self.batch_size - x.size(0), x.size(1)).to(x.device)
                new_x = torch.cat((x, padding), dim=0)
                new_x = self.instance_norm(new_x)
                x = new_x[:x.size(0), :]
            else:
                x = self.instance_norm(x)
        else:
            pass
        
        x = self.activation(self.fc1(x))          
        x = self.fc2(x)                
        
        return x