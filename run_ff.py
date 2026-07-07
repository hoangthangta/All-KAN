import argparse
import random
import time
from datetime import datetime
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import MLP, ReLUKAN, EfficientKAN, FastKAN, BSRBF_KAN, FasterKAN
# SechKAN, MSKAN

from utils import *
from file_io import *

# Function 1: Localized oscillation (Gaussian envelope + medium frequency)
# Tests: ability to model smooth, localized patterns
def function1(x):
    return torch.exp(-4 * x**2) * torch.sin(6 * torch.pi * x)

# Function 2: Warped oscillation (input-dependent frequency)
# Tests: handling of non-uniform frequency / nonlinear phase distortion
def function2(x):
    return torch.sin(4 * torch.pi * x**2) * torch.cos(2 * torch.pi * x)

# Function 3: 2D anisotropic structure (different behavior per dimension)
# Tests: separability and dimension-wise frequency variation
def function3_2d(x1, x2):
    return torch.sin(3 * torch.pi * x1**2) * torch.cos(3 * torch.pi * x2)

# Function 4: Localized + nonlinear interaction (3D)
# Tests: coupling between variables and spatial decay
def function4_3d(x1, x2, x3):
    return torch.exp(-3 * (x1**2 + x2**2)) * torch.sin(4 * torch.pi * x3 + x1 * x2)

# Function 5: High-dimensional coupled nonlinear structure (4D)
# Tests: complex variable interactions and compositional structure
def function5_4d(x1, x2, x3, x4):
    return torch.sin(3 * torch.pi * (x1 * x2 + x3)) * torch.cos(2 * torch.pi * (x2 * x4))

# Function 6: Non-smooth oscillation (kink at x = 0)
# Tests: ability to approximate non-differentiable points
def function6(x):
    return torch.abs(x) * torch.sin(3 * torch.pi * x)


# Fractal functions ---------------------------------|
# ---------------------------------------------------|
def weierstrass_1d(x, a=0.5, b=5, terms=10):
    """
    Classical Weierstrass function.
    """
    return sum(
        (a ** n) * torch.cos((b ** n) * math.pi * x)
        for n in range(terms)
    )


def takagi_1d(x, terms=20):
    """
    Classical Takagi function.
    """
    return sum(
        torch.minimum(
            (2 ** n * x) - torch.floor(2 ** n * x),
            1 - ((2 ** n * x) - torch.floor(2 ** n * x))
        ) / (2 ** n)
        for n in range(terms)
    )


def cantor_1d(x, terms=20):
    """
    Cantor function (Devil's staircase).
    Assumes x in [0, 1].
    """
    y = torch.zeros_like(x)
    xx = x.clone()

    for n in range(terms):
        digit = torch.floor(3 * xx)
        y += (digit == 2).float() / (2 ** (n + 1))
        xx = torch.clamp(3 * xx - digit, 0, 1)

    return y

    
def riemann_1d(x, terms=40):
    """
    Riemann nondifferentiable function.
    """
    return sum(
        torch.sin((n ** 2) * math.pi * x) / (n ** 2)
        for n in range(1, terms + 1)
    )

def weierstrass_2d(x1, x2, a=0.5, b=3, terms=5):
    """
    Separable 2D Weierstrass function.
    """
    return sum(
        (a ** n)
        * torch.cos((b ** n) * math.pi * x1)
        * torch.cos((b ** n) * math.pi * x2)
        for n in range(terms)
    )

    
def fbm_2d(x1, x2, octaves=6):
    """
    Fractal Brownian motion style spectral synthesis.
    """
    return sum(
        (0.5 ** i)
        * torch.sin((2 ** i) * math.pi * x1)
        * torch.cos((2 ** i) * math.pi * x2)
        for i in range(octaves)
    )


def takagi_2d(x1, x2, terms=20):
    y = torch.zeros_like(x1)

    for n in range(terms):
        fx = (2**n * x1) - torch.floor(2**n * x1)
        fy = (2**n * x2) - torch.floor(2**n * x2)

        tx = torch.minimum(fx, 1 - fx)
        ty = torch.minimum(fy, 1 - fy)

        y += (tx + ty) / (2**n)

    return y
    
def kiesswetter_2d(x, y, terms=12):
    """
    2D Kiesswetter fractal function.
    Extremely rough, nowhere differentiable.
    """
    z = torch.zeros_like(x)
    
    for n in range(terms):
        # Higher base (4^n) creates sharper features
        sx = 4**n * x
        sy = 4**n * y
        
        fx = sx - torch.floor(sx)
        fy = sy - torch.floor(sy)
        
        # Kiesswetter kernel
        tx = torch.abs(fx - 0.5) - 0.25
        ty = torch.abs(fy - 0.5) - 0.25
        
        # Combine with multiplicative or additive mixing
        z += (tx * ty) / (4**n)        # multiplicative = more interesting patterns
    
    return z
# ---------------------------------------------------|
# ---------------------------------------------------|

def normalize(z, eps=1e-8):
    return (z - z.min()) / (z.max() - z.min() + eps)
                    
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    seed = args.seed
    print('Seed: ', seed)
    set_seed(seed, device)
   
    # Create model
    if args.model_name == 'mlp':
        model = MLP(args.layers, norm_type='', base_activation=args.base_activation)
    elif args.model_name == 'fast_kan':
        model = FastKAN(args.layers, num_grids = args.num_grids, use_layernorm=False) # num_grids = 8
    elif args.model_name == 'faster_kan':
        model = FasterKAN(args.layers, num_grids = args.num_grids, norm_type='') # num_grids = 8
    elif args.model_name == 'bsrbf_kan':
        model = BSRBF_KAN(args.layers, grid_size=args.grid_size, spline_order=args.spline_order, norm_type='')
    elif args.model_name == 'relu_kan':
        model = ReLUKAN(args.layers, norm_type='', k=args.spline_order, grid=args.grid_size)
    elif args.model_name == 'efficient_kan':
        model = EfficientKAN(args.layers, grid_size=args.grid_size, spline_order=args.spline_order) # base_activation = silu
    else:
        raise ValueError(f"The model '{args.model_name}' is not supported.")
    '''elif args.model_name == 'sech_kan':
        model = SechKAN(args.layers, norm1_type = args.norm1_type, norm2_type = args.norm2_type, base_activation = args.base_activation, use_width = True, norm_mode = "first", num_grids = args.num_grids, use_base_update = False) # default: norm1_type = 'mm', norm2_type = '' '''
    '''elif args.model_name == 'ms_kan':
        model = MSKAN(
            args.layers,
            norm1_type=args.norm1_type,
            norm2_type=args.norm2_type,
            grids=[int(x) for x in args.grids.split(',')],
            orders=[int(x) for x in args.orders.split(',')],
            base_activation=args.base_activation,
            shared_phases=True
        )'''

    # Move model to device
    model = model.to(device)

    # Create dataset
    input_dim = 1024
    x1 = torch.tensor(np.arange(0, input_dim) / input_dim, dtype=torch.float32).unsqueeze(1)
    x2 = torch.tensor(np.arange(0, input_dim) / input_dim, dtype=torch.float32).unsqueeze(1)
    x3 = torch.tensor(np.arange(0, input_dim) / input_dim, dtype=torch.float32).unsqueeze(1)
    x4 = torch.tensor(np.arange(0, input_dim) / input_dim, dtype=torch.float32).unsqueeze(1)

    func_map = {
        "func1": (function1, [x1]),
        "func2": (function2, [x1]),
        "func3_2d": (function3_2d, [x1, x2]),
        "func4_3d": (function4_3d, [x1, x2, x3]),
        "func5_4d": (function5_4d, [x1, x2, x3, x4]),
        "func6": (function6, [x1]),

        # Fractal functions
        # 1D
        "weierstrass_1d": (weierstrass_1d, [x1]),
        "takagi_1d": (takagi_1d, [x1]),
        "cantor_1d": (cantor_1d, [x1]),
        "riemann_1d": (riemann_1d, [x1]),

        # 2D
        "weierstrass_2d": (weierstrass_2d, [x1, x2]),
        "fbm_2d": (fbm_2d, [x1, x2]),
        "takagi_2d": (takagi_2d, [x1, x2]),
        "kiesswetter_2d": (kiesswetter_2d, [x1, x2])

    }

    if args.func not in func_map:
        raise ValueError(f"Function {args.func} is not defined.")

    func, inputs = func_map[args.func]
    dim = len(inputs)

    if dim == 1:
        x = torch.linspace(0, 1, input_dim, dtype=torch.float32, device=device).unsqueeze(1)
        #y = func(x).unsqueeze(1)
        y = func(x)

    else:
        # resolution
        res = {2: 32, 3: 10, 4: 6}[dim]
        coords = [torch.linspace(-1, 1, res,dtype=torch.float32, device=device) for _ in range(dim)]
        grids = torch.meshgrid(*coords, indexing='ij')
        x = torch.stack([g.reshape(-1) for g in grids], dim=1)
        y = func(*[g.reshape(-1) for g in grids]).unsqueeze(1)
        

    '''if args.func in func_map:
        func, inputs = func_map[args.func]
        
        # 2D functions
        if '2d' in args.func.lower():
            res = 300  
            x1 = torch.linspace(-1, 1, res, dtype=torch.float32, device=device)
            x2 = torch.linspace(-1, 1, res, dtype=torch.float32, device=device)
            X, Y = torch.meshgrid(x1, x2, indexing='ij')
            
            # Flatten for model input
            x = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)   # shape: (res*res, 2)
            
            # Compute target
            y = func(X.reshape(-1), Y.reshape(-1)).unsqueeze(1)       # shape: (res*res, 1)
            
        # 1D functions
        else:
            
            x = torch.linspace(0, 1, input_dim, dtype=torch.float32, device=device).unsqueeze(1)
            y = func(x).unsqueeze(1)
        
        # Input dimension check
        if x.shape[1] != args.layers[0]:
            raise ValueError(
                f"Input dimension mismatch: expected {args.layers[0]}, "
                f"but got {x.shape[1]}. Check your \"layers\" input."
            )
    else:
        raise ValueError(f"Function {args.func} is not defined.")'''

    # Move inputs and targets to device
    x = x.to(device)
    y = y.to(device)

    # Optimizer and loss
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    mse = torch.nn.MSELoss()

    plt.ion()
    plt.figure(figsize=(10, 4))
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.perf_counter()
    losses = []

    for e in range(args.epochs):
        opt.zero_grad()
        pred = model(x)
        pred = pred.unsqueeze(-1)
       
        loss = mse(pred[:, :, 0], y)
        losses.append(loss.item())
        loss.backward()
        opt.step()
       
        
        # Plot
        pred_plot = pred.detach().cpu()
        x_plot = x.cpu()
        y_plot = y.cpu()
         
        if e % 10 == 0 or e == args.epochs - 1:
            if '2d' in args.func.lower():
                plt.clf()

                

                n = int(math.sqrt(y_plot.numel()))

                plt.subplot(1, 2, 1)
                plt.imshow(
                    y_plot.squeeze().reshape(n, n),
                    cmap='inferno',
                    origin='lower',
                    extent=[-1, 1, -1, 1]
                )
                plt.title("Target")
                plt.colorbar()

                plt.subplot(1, 2, 2)
                plt.imshow(
                    normalize(pred_plot.squeeze()).reshape(n, n),
                    cmap='inferno',
                    origin='lower',
                    extent=[-1, 1, -1, 1]
                )
                plt.title("Prediction")
                plt.colorbar()

                plt.tight_layout()
                plt.pause(0.01)
            else:        
                
                plt.clf()
                #plt.plot(x_plot, y_plot, label="Target")
                plt.plot(x_plot, y_plot, color='lightgray', label='Target', linewidth=2)
                plt.plot(x_plot, pred_plot[:, :, 0], label="Prediction")
                plt.grid()
                plt.pause(0.01)


    if torch.cuda.is_available(): torch.cuda.synchronize()
    end = time.perf_counter()
   
    #args.model_name, args.func
    run_time = end - start
    used_params = count_params(model)
    final_loss = losses[-1]
    print('-'*50)
    print(f"Model: {args.model_name}")
    print(f"Function: {args.func}")
    print(f"Training time (s): {run_time:.2f}")
    print(f"Used params: {used_params}")
    print(f"Final loss: {losses[-1]:.2e}")
    print('-'*50)
   
    data_dict = {}
    unix_time = time.time()
    readable_time = datetime.fromtimestamp(unix_time).strftime("%d-%m-%Y %H:%M:%S")
    data_dict['time'] = readable_time

    data_dict['model_name'] = args.model_name
    data_dict['function'] = args.func
    data_dict['loss'] =   f"{losses[-1]:.2e}"
    data_dict['used_params'] = used_params
    data_dict['run_time'] = run_time
    data_dict['seed'] = args.seed
   
    if (args.model_name == 'sech_kan'):
        data_dict['base_activation'] =  args.base_activation
        data_dict['num_grids'] =  args.num_grids 
        data_dict['norm1_type'] =  args.norm1_type 
        data_dict['norm2_type'] =  args.norm2_type 
    if (args.model_name == 'ms_kan'):
        data_dict['order'] =  args.orders
        data_dict['grids'] =  args.grids
    if (args.model_name in ['efficient_kan', 'bsrbf_kan']):
        data_dict['grid_size'] =  args.grid_size
        data_dict['spline_order'] =  args.spline_order
    if (args.model_name == 'fast_kan'):
        data_dict['num_grids'] =  args.num_grids
        
   
    # append to the result file
    output_path = 'output/function_fitting/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    write_single_dict_to_jsonl('output/function_fitting/result.json', data_dict, file_access = 'a')

    '''plt.ioff()
    plt.plot(x_plot, y_plot, label="Target")
    plt.plot(x_plot, pred_plot[:, :, 0], label="Prediction")
    plt.grid()
    plt.legend()
    plt.show()'''


def main(args):
    layers = args.layers.split(',')
    layers = [int(x) for x in layers]
    args.layers = layers
    if args.mode == 'train':
        train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_name', type=str, default='efficient_kan')
    parser.add_argument('--layers', type=str, default="2,1")
    parser.add_argument('--func', type=str, default="func1")
    parser.add_argument('--norm1_type', type=str, default="mm")
    parser.add_argument('--norm2_type', type=str, default="")
    parser.add_argument('--num_grids', type=int, default=8)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--spline_order', type=int, default=3)
    
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_activation', type=str, default="relu")
    parser.add_argument('--grids', type=str, default='1,4', help='Grid sizes for MSKAN')
    parser.add_argument('--orders', type=str, default='3,3', help='Spline orders for MSKAN')
    args = parser.parse_args()
    main(args)


