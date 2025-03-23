import argparse
import matplotlib.pyplot as plt
from models import (
    EfficientKAN, FastKAN, BSRBF_KAN, FasterKAN, MLP, FC_KAN, GottliebKAN,
    SKAN, PRKAN, ReLUKAN, AF_KAN, ChebyKAN, FourierKAN, KnotsKAN,
    RationalKAN, RBF_KAN
)
import torch
import numpy as np
import time

def function1(x):
    return torch.sin(torch.pi * x)

def function2(x):
    return torch.sin(5 * torch.pi * x) + x

def function3(x1, x2):
    return torch.sin(torch.pi * x1 + torch.pi * x2)
    
def function4(x1, x2):
    return torch.exp(torch.sin(torch.pi * x1) + x2**2)
    
def function5(x1, x2, x3):
    return torch.exp(torch.sin(torch.pi * x1) + x2**2) * torch.tanh(x3)
    
def function6(x1, x2, x3, x4):
    return torch.exp(torch.sin(torch.pi * (x1**2 + x2**2)) + torch.sin(torch.pi * (x3**2 + x4**2)))

    
def train(args):
    
    # Create model
    # We must turn off "data normalization" to do this task
    if (args.model_name == 'fc_kan'):
        model = FC_KAN(args.layers, norm_type = '')
    elif(args.model_name == 'efficient_kan'):
        model = EfficientKAN(args.layers)
    elif(args.model_name == 'bsrbf_kan'):
        model = BSRBF_KAN(args.layers, norm_type = '') 
    elif(args.model_name == 'fast_kan'):
        model = FastKAN(args.layers, use_layernorm = False)
    elif(args.model_name == 'faster_kan'):
        model = FasterKAN(args.layers, norm_type = '')
    elif(args.model_name == 'mlp'): # recheck?
        model = MLP(args.layers, norm_type = '')
    elif(args.model_name == 'relu_kan'):
        model = ReLUKAN(args.layers, norm_type = '')
    elif(args.model_name == 'rbf_kan'):
        model = ReLUKAN(args.layers, norm_type = '')
    elif(args.model_name == 'prkan'):
        model = PRKAN(args.layers, norm_type = '')
    elif(args.model_name == 'af_kan'):
        model = AF_KAN(args.layers, norm_type = '')
    else:
        raise ValueError(f"The model '{args.model_name}' is not supported.")
    
    # Create dataset    
    x1 = torch.tensor(np.arange(0, 1024) / 1024, dtype=torch.float32).unsqueeze(1)
    x2 = torch.tensor(np.arange(0, 1024) / 1024, dtype=torch.float32).unsqueeze(1)
    x3 = torch.tensor(np.arange(0, 1024) / 1024, dtype=torch.float32).unsqueeze(1)
    x4 = torch.tensor(np.arange(0, 1024) / 1024, dtype=torch.float32).unsqueeze(1)

    func_map = {
        "func1": (function1, [x1]),
        "func2": (function2, [x1]),
        "func3": (function3, [x1, x2]),
        "func4": (function4, [x1, x2]),
        "func5": (function5, [x1, x2, x3]),
        "func6": (function6, [x1, x2, x3, x4]),
    }

    if args.func in func_map:
        func, inputs = func_map[args.func]
        y = func(*inputs)
        x = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=1)
        
        # check inputs + layers
        if (x.shape[1] != args.layers[0]):
            raise ValueError(f"Expected input dimension {args.layers[0]}, but got {x.shape[1]}")
    
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        y = y.cuda()

    # Optimization + Loss
    opt = torch.optim.Adam(model.parameters())
    mse = torch.nn.MSELoss()

    # Train
    start = time.time() # should be here better
    losses = []
    for e in range(args.epochs): # start from 0
        opt.zero_grad()
        pred = model(x)
        pred = pred.unsqueeze(-1) 
        
        loss = mse(pred[:, :, 0], y)
        losses.append(loss.item())
        loss.backward()
        opt.step()
        # print(time.time() - t)
        pred = pred.detach()
        
        plt.clf()
        plt.plot(x.cpu(), y.cpu())
        plt.plot(x.cpu(), pred[:, :, 0].cpu())
        plt.pause(0.01)
    
    # Result    
    end = time.time()
    print(f"Training time (s): {end-start}")
    print('Final loss: ', losses[-1])
    plt.plot(x.cpu(), y.cpu())
    plt.plot(x.cpu(), pred[:, :, 0].cpu())
    plt.show()


def main(args):

    layers = args.layers.split(',')
    layers = [int(x) for x in layers]
    args.layers = layers
    
    if (args.mode == 'train'): train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_name', type=str, default='efficient_kan')
    parser.add_argument('--layers', type=str, default="2,1")
    parser.add_argument('--func', type=str, default="function")
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()
    main(args)
    
# python run_ff.py --mode "train" --model_name "efficient_kan" --layers "1,1" --func "func1" --epochs 1