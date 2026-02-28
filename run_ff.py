import argparse
import matplotlib.pyplot as plt
from models import (
    EfficientKAN, FastKAN, BSRBF_KAN, FasterKAN, MLP, FC_KAN, GottliebKAN,
    SKAN, PRKAN, ReLUKAN, AF_KAN, ChebyKAN, FourierKAN, KnotsKAN,
    RationalKAN, RBF_KAN
)

# SechKAN: will be updated
import torch

import numpy as np
import time
from utils import *
from file_io import *
import random
from pathlib import Path

from datetime import datetime

# 1D functions
def function1(x):
    return torch.sin(torch.pi * x)
   
def function2(x):
    return torch.sin(5 * torch.pi * x) + x

# 2D function
def function3(x1, x2):
    return torch.sin(3 * torch.pi * x1**2) * torch.cos(3 * torch.pi * x2)
   
# 3D functions, consider to replace this function
def function4(x1, x2, x3):
    k = 3.0   # higher = more curve
    return torch.sin(k * torch.pi * x1) * x2 * torch.cos(k * torch.pi * x3)

# 4D function
def function5(x1, x2, x3, x4):
    return torch.exp(
        torch.sin(torch.pi * (x1**2 + x2**2)) +
        torch.sin(torch.pi * (x3**2 + x4**2))
    )

       
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    seed = args.seed
    print('Seed: ', seed)
    set_seed(seed, device)
   
    # Create model
    if args.model_name == 'fc_kan':
        model = FC_KAN(args.layers, norm_type='')
    elif args.model_name == 'efficient_kan':
        model = EfficientKAN(args.layers)
    elif args.model_name == 'bsrbf_kan':
        model = BSRBF_KAN(args.layers, norm_type='')
    elif args.model_name == 'fast_kan':
        model = FastKAN(args.layers, use_layernorm=False)
    elif args.model_name == 'faster_kan':
        model = FasterKAN(args.layers, norm_type='')
    elif args.model_name == 'mlp':
        model = MLP(args.layers, norm_type='', base_activation=args.base_activation)
    elif args.model_name == 'relu_kan':
        model = ReLUKAN(args.layers, norm_type='')
    elif args.model_name == 'rbf_kan':
        model = RBF_KAN(args.layers, norm_type='')
    elif args.model_name == 'prkan':
        model = PRKAN(args.layers, norm_type='')
    elif args.model_name == 'af_kan':
        model = AF_KAN(args.layers, norm_type='')
    else:
        raise ValueError(f"The model '{args.model_name}' is not supported.")
    # SechKAN: will be updated
    '''elif args.model_name == 'sech_kan':
        model = SechKAN(args.layers, norm_type='mm', base_activation=args.base_activation, first_layer_norm = True, num_grids = args.num_grids)'''
    

    # Move model to device
    model = model.to(device)

    # Create dataset
    x1 = torch.tensor(np.arange(0, 1024) / 1024, dtype=torch.float32).unsqueeze(1)
    x2 = torch.tensor(np.arange(0, 1024) / 1024, dtype=torch.float32).unsqueeze(1)
    x3 = torch.tensor(np.arange(0, 1024) / 1024, dtype=torch.float32).unsqueeze(1)
    x4 = torch.tensor(np.arange(0, 1024) / 1024, dtype=torch.float32).unsqueeze(1)

    func_map = {
        "func1": (function1, [x1]),
        "func2": (function2, [x1]),
        "func3": (function3, [x1, x2]),
        "func4": (function4, [x1, x2, x3]),
        "func5": (function5, [x1, x2, x3, x4])
    }

    if args.func in func_map:
        func, inputs = func_map[args.func]
        y = func(*inputs)
        x = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=1)
        if x.shape[1] != args.layers[0]:
            raise ValueError(
                f"Input dimension mismatch: expected {args.layers[0]}, "
                f"but got {x.shape[1]}. Check your \"layers\" input."
            )
    else:
        raise ValueError(f"Function {args.func} is not defined.")

    # Move inputs and targets to device
    x = x.to(device)
    y = y.to(device)

    # Optimizer and loss
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    mse = torch.nn.MSELoss()

    plt.ion()
    
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
        plt.clf()
        plt.plot(x_plot, y_plot, label="Target")
        plt.plot(x_plot, pred_plot[:, :, 0], label="Prediction")
        plt.grid()
        plt.pause(0.01)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    end = time.perf_counter()
   
    #args.model_name, args.func
    run_time = end - start
    used_params = count_params(model)
    final_loss = losses[-1]
    print('--------------------------------------')
    print(f"Model: {args.model_name}")
    print(f"Function: {args.func}")
    print(f"Training time (s): {run_time:.2f}")
    print(f"Used params: {used_params}")
    print(f"Final loss: {losses[-1]:.2e}")
    print('--------------------------------------')
   
    data_dict = {}
    unix_time = time.perf_counter()
    readable_time = datetime.fromtimestamp(unix_time).strftime("%d-%m-%Y %H:%M:%S")
    data_dict['time'] = readable_time

    data_dict['model_name'] = args.model_name
    data_dict['function'] = args.func
    data_dict['loss'] =   f"{losses[-1]:.2e}"
    data_dict['used_params'] = used_params
    data_dict['run_time'] = run_time
   
    if (args.model_name == 'ms_kan'):
        data_dict['order'] =  args.orders
        data_dict['grids'] =  args.grids
   
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
    parser.add_argument('--norm1_type', type=str, default="l2mm")
    parser.add_argument('--norm2_type', type=str, default="")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_activation', type=str, default="relu")
    parser.add_argument('--num_grids', type=int, default=8, help='Number of grids for SechKAN')
    parser.add_argument('--grids', type=str, default='1,4', help='Grid sizes for MSKAN')
    parser.add_argument('--orders', type=str, default='3,3', help='Spline orders for MSKAN')
    args = parser.parse_args()
    main(args)

# 1D
# python run_ff2.py --mode "train" --model_name "mlp" --layers "1,1" --func "func1" --epochs 500 --base_activation "relu"
# python run_ff2.py --mode "train" --model_name "relu_kan" --layers "1,1" --func "func1" --epochs 500 --base_activation "relu"

# python run_ff2.py --mode "train" --model_name "fast_kan" --layers "1,1" --func "func1" --epochs 500

# python run_ff2.py --mode "train" --model_name "sech_kan" --layers "1,1" --func "func1" --epochs 500 --base_activation "selu" --num_grids 8


# python run_ff2.py --mode "train" --model_name "mlp" --layers "1,1" --func "func2" --epochs 500 --base_activation "relu"
# python run_ff2.py --mode "train" --model_name "relu_kan" --layers "1,1" --func "func2" --epochs 500 --base_activation "relu"
# python run_ff2.py --mode "train" --model_name "sech_kan" --layers "1,1" --func "func2" --epochs 500 --base_activation "selu" --num_grids 8

# 2D
# python run_ff2.py --mode "train" --model_name "mlp" --layers "2,1" --func "func3" --epochs 500 --base_activation "relu"
# python run_ff2.py --mode "train" --model_name "relu_kan" --layers "2,1" --func "func3" --epochs 500 --base_activation "relu"
# python run_ff2.py --mode "train" --model_name "sech_kan" --layers "2,1" --func "func3" --epochs 500 --base_activation "selu" --num_grids 16

# 3D
# python run_ff2.py --mode "train" --model_name "mlp" --layers "3,2,1" --func "func4" --epochs 500 --base_activation "relu"
# python run_ff2.py --mode "train" --model_name "relu_kan" --layers "3,2,1" --func "func4" --epochs 500 --base_activation "relu"
# python run_ff2.py --mode "train" --model_name "sech_kan" --layers "3,2,1" --func "func4" --epochs 500 --base_activation "selu" --num_grids 32

# 4D
# python run_ff2.py --mode "train" --model_name "mlp" --layers "4,2,1" --func "func5" --epochs 500 --base_activation "relu"
# python run_ff2.py --mode "train" --model_name "relu_kan" --layers "4,2,1" --func "func5" --epochs 500 --base_activation "relu"
# python run_ff2.py --mode "train" --model_name "sech_kan" --layers "4,2,1" --func "func5" --epochs 500 --base_activation "selu" --num_grids 8
