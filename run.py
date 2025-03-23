import argparse
import copy
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import scipy.io
import requests

from ptflops import get_model_complexity_info
from PIL import Image

from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from file_io import *
from utils import *
from schedulers import *
from storage import *
from models import (
    EfficientKAN, FastKAN, BSRBF_KAN, FasterKAN, MLP, FC_KAN, GottliebKAN,
    SKAN, PRKAN, ReLUKAN, AF_KAN, ChebyKAN, FourierKAN, KnotsKAN,
    RationalKAN, RBF_KAN
)

# MNIST: Mean=0.1307, Std=0.3081
# Fashion-MNIST: Mean=0.2860, Std=0.3530
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    
# CIFAR10
transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
'''transform_cifar_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert (32,32,3) -> (32,32,1)
    transforms.ToTensor(),  
    transforms.Normalize((0.5), (0.5))  
    ])'''

# Omniglot
transform_omniglot = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.Grayscale(num_output_channels=1),  # Ensure images are grayscale
    transforms.ToTensor()  # Convert images to PyTorch tensors
    ])
    
    
# use for CalTech 101 Silhouettes  
def convert_to_tensors(dataset):
    images, labels = [], []
    for image, label in tfds.as_numpy(dataset):
        images.append(image)
        labels.append(label)
    return torch.tensor(images, dtype=torch.float32).unsqueeze(1), torch.tensor(labels)
    
    
def run(args):
    
    
    
    trainset, valset = [], []
    if (args.ds_name == 'mnist'):
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        valset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif(args.ds_name == 'fashion_mnist'):
        trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )

        valset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif(args.ds_name == 'cal_si'): # "caltech101_silhouettes
        # Load the .mat file
        data = scipy.io.loadmat("data/caltech101_silhouettes/caltech101_silhouettes_28_split1.mat")

        # Extract data and labels
        train_data = data['train_data']
        train_labels = data['train_labels'].flatten()

        val_data = data['val_data']
        val_labels = data['val_labels'].flatten()

        test_data = data['test_data']
        test_labels = data['test_labels'].flatten()

        # Convert to PyTorch tensors
        trainset = TensorDataset(
            torch.tensor(train_data, dtype=torch.float32).reshape(-1, 1, 28, 28),  
            torch.tensor(train_labels, dtype=torch.long)  
        )
        valset = TensorDataset(
            torch.tensor(val_data, dtype=torch.float32).reshape(-1, 1, 28, 28),
            torch.tensor(val_labels, dtype=torch.long)
        )
        
        testset = TensorDataset(
            torch.tensor(test_data, dtype=torch.float32).reshape(-1, 1, 28, 28),
            torch.tensor(test_labels, dtype=torch.long)
        )
    
    elif(args.ds_name == 'cifar10'):
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True,  transform=transform_cifar
        )

        valset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_cifar
        )
        
    # add other datasets here
    '''elif(args.ds_name == 'omniglot'):
        trainset = torchvision.datasets.Omniglot(
            root="./data", background=True, download=True, transform=transform_omniglot
        )

        valset = torchvision.datasets.Omniglot(
            root="./data", background=False, download=True, transform=transform_omniglot
        )
        
        #print(len(trainset._characters)) # 964
        #print(len(valset._characters)) # 659
        
        all_classes = set(trainset._characters + valset._characters)
        n_output = len(all_classes)'''
        
    if (args.n_examples > 0):
        if (args.n_examples/args.batch_size > 1):
            trainset = torch.utils.data.Subset(trainset, range(args.n_examples))
        else:
            print('The number of examples is too small!')
            return
    elif(args.n_part > 0):
        if (len(trainset)*args.n_part > args.batch_size):
            trainset = torch.utils.data.Subset(trainset, range(int(len(trainset)*args.n_part)))
        else:
            print('args.n_part is too small!')
            return

    print('trainset: ', len(trainset))
    print('valset: ', len(valset))
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False) 
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    
    # If having a test set
    if (args.ds_name == 'cal_si'): # or other datasets
        testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
        print('testset: ', len(testset))

    # Create model storage
    output_path = 'output/' + args.ds_name + '/' + args.model_name + '/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    output_path, saved_model_name, saved_model_history = create_model_storage(args)

    # Define models
    model = {}
    print('model_name: ', args.model_name)
    if (args.model_name == 'bsrbf_kan'):
        model = BSRBF_KAN(args.layers, grid_size = args.grid_size, spline_order = args.spline_order)
    elif(args.model_name == 'fast_kan'):
        model = FastKAN(args.layers, num_grids = args.num_grids)
    elif(args.model_name == 'faster_kan'):
        model = FasterKAN(args.layers, num_grids = args.num_grids)
    elif(args.model_name == 'gottlieb_kan'):
        model = GottliebKAN(args.layers, spline_order = args.spline_order)
    elif(args.model_name == 'mlp'):
        model = MLP(args.layers, base_activation = args.base_activation, norm_type = args.norm_type, use_attn  = args.use_attn)
    elif(args.model_name == 'fc_kan'):
        model = FC_KAN(args.layers, args.func_list, combined_type = args.combined_type, grid_size = args.grid_size, spline_order = args.spline_order)
    elif(args.model_name == 'efficient_kan'):
        model = EfficientKAN(args.layers, grid_size = args.grid_size, spline_order = args.spline_order)
    elif(args.model_name == 'prkan'):
        model = PRKAN(args.layers, grid_size = args.grid_size, spline_order = args.spline_order, num_grids = args.num_grids, func = args.func, norm_type = args.norm_type, base_activation = args.base_activation, methods = args.methods, combined_type = args.combined_type, norm_pos = args.norm_pos)
    elif(args.model_name == 'skan'):
        model = SKAN(args.layers, basis_function = args.basis_function) # lshifted_softplus, larctan 
    elif(args.model_name == 'relu_kan'):
        model = ReLUKAN(args.layers, grid = args.grid_size , k = args.spline_order, norm_type = args.norm_type, base_activation = args.base_activation) 
    elif(args.model_name == 'af_kan'):
        model = AF_KAN(args.layers, grid = args.grid_size , k = args.spline_order, norm_type = args.norm_type, base_activation = args.base_activation, methods = args.methods, combined_type = args.combined_type, func = args.func, func_norm = args.func_norm)
    elif(args.model_name == 'cheby_kan'):
        model = ChebyKAN(args.layers, degree = args.spline_order) 
    elif(args.model_name == 'fourier_kan'):
        model = FourierKAN(args.layers, grid_size = args.grid_size, spline_order = args.spline_order) 
    elif(args.model_name == 'knots_kan'):
        model = KnotsKAN(args.layers, grid_size = args.grid_size, spline_order = args.spline_order) 
    elif(args.model_name == 'rational_kan'):
        model = RationalKAN(args.layers, P_order = args.p_order, Q_order = args.q_order, groups = args.groups) 
    elif(args.model_name == 'rbf_kan'):
        model = RBF_KAN(args.layers, grid_size = args.grid_size, spline_order = args.spline_order) 
    else:
        # add other KANs here
        raise ValueError("Unsupported network type.")
    model.to(device)

    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wc)

    # Define learning rate scheduler
    if(args.scheduler == 'StepLR'):
        scheduler = get_scheduler(optimizer, name="StepLR", step_size = args.epochs//3)
    elif(args.scheduler == 'CosineAnnealingLR'):
        scheduler = get_scheduler(optimizer, name="CosineAnnealingLR", epochs = args.epochs)
    elif(args.scheduler == 'OneCycleLR'):
        scheduler = get_scheduler(optimizer, name="OneCycleLR", step_size=len(trainloader)*args.epochs)
    elif(args.scheduler == 'ExponentialLR'):
        scheduler = get_scheduler(optimizer, name="ExponentialLR")
    elif(args.scheduler == 'CyclicLR'):
        scheduler = get_scheduler(optimizer, name="CyclicLR", step_size=len(trainloader)*2)
    else:
        print('You should choose a scheduler (StepLR, CosineAnnealingLR, OneCycleLR, ExponentialLR).')
        return
    
    # Define loss
    criterion = nn.CrossEntropyLoss()

    best_epoch, best_accuracy = 0, 0
    y_true = [labels.tolist() for images, labels in valloader]
    y_true = sum(y_true, [])
    
    if(args.ds_name == 'cal_si'):
        y_true_test = [labels.tolist() for images, labels in testloader]
        y_true_test = sum(y_true_test, [])
    
    grad_history = []
    
    start = time.time() # should be here better
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_accuracy, train_loss = 0, 0
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):

                images = images.view(-1, args.layers[0]).to(device)
                optimizer.zero_grad()
                output = model(images.to(device))
                loss = criterion(output, labels.to(device))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                # Update learning rate
                if(args.scheduler not in ['StepLR', 'ExponentialLR', 'CosineAnnealingLR']):
                    scheduler.step()
                
                #accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                train_accuracy += (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                pbar.set_postfix(loss=train_loss/len(trainloader), accuracy=train_accuracy/len(trainloader), lr=optimizer.param_groups[0]['lr'])
        
        # Update learning rate
        if(args.scheduler in ['StepLR', 'ExponentialLR', 'CosineAnnealingLR']): 
            scheduler.step()
        
        train_loss /= len(trainloader)
        train_accuracy /= len(trainloader)
        
        grad_norm = cal_grad_norm(model)
        if grad_norm < 1e-5:
            print("Warning: Gradient norm is very low. The model might be at a local minimum or saddle point.")
            
        grad_mean = cal_grad_mean(model)
        grad_history.append(grad_mean.item())
        
        # Validation
        model.eval()
        val_loss, val_accuracy = 0, 0
        
        y_pred = []
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, args.layers[0]).to(device)
                output = model(images.to(device))
                val_loss += criterion(output, labels.to(device)).item()
                y_pred += output.argmax(dim=1).tolist()
                val_accuracy += ((output.argmax(dim=1) == labels.to(device)).float().mean().item())
       
        # calculate F1, Precision and Recall
        #f1 = f1_score(y_true, y_pred, average='micro')
        #pre = precision_score(y_true, y_pred, average='micro')
        #recall = recall_score(y_true, y_pred, average='micro')
        
        f1 = f1_score(y_true, y_pred, average='macro')
        pre = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        val_loss /= len(valloader)
        val_accuracy /= len(valloader)

        # Choose best model
        if (val_accuracy > best_accuracy):
            best_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model, output_path + '/' + saved_model_name)
              
        print(f"Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.6f}, Grad mean: {grad_mean.item():.6f}, Grad L2 Norm: {grad_norm:.6f}")
        print(f"Epoch [{epoch}/{args.epochs}], Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.6f}, F1: {f1:.6f}, Precision: {pre:.6f}, Recall: {recall:.6f}")
        
        test_loss, test_accuracy = 0, 0
        if (args.ds_name == 'cal_si'):
            
            y_pred_test = []
            with torch.no_grad():
                for images, labels in testloader:
                    images = images.view(-1, args.layers[0]).to(device)
                    output = model(images.to(device))
                    test_loss += criterion(output, labels.to(device)).item()
                    y_pred_test += output.argmax(dim=1).tolist()
                    test_accuracy += ((output.argmax(dim=1) == labels.to(device)).float().mean().item())
           
            # calculate F1, Precision and Recall
            #f1_test = f1_score(y_true_test, y_pred_test, average='micro')
            #pre_test = precision_score(y_true_test, y_pred_test, average='micro')
            #recall_test = recall_score(y_true_test, y_pred_test, average='micro')
            
            f1_test = f1_score(y_true_test, y_pred_test, average='macro')
            pre_test = precision_score(y_true_test, y_pred_test, average='macro')
            recall_test = recall_score(y_true_test, y_pred_test, average='macro')

            test_loss /= len(testloader)
            test_accuracy /= len(testloader)
   
            
            print(f"Epoch [{epoch}/{args.epochs}], Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.6f}, F1: {f1_test:.6f}, Precision: {pre_test:.6f}, Recall: {recall_test:.6f}")

        if test_accuracy != 0: # there has a test set
            write_single_dict_to_jsonl(output_path + '/' + saved_model_history, {'epoch':epoch, 'test_accuracy':test_accuracy, 'val_accuracy':val_accuracy, 'train_accuracy':train_accuracy, 'test_f1_macro':f1_test, 'test_pre_macro':pre_test, 'test_re_macro':recall_test,  'val_f1_macro':f1, 'val_pre_macro':pre, 'val_re_macro':recall, 'best_epoch':best_epoch, 'test_loss': test_loss, 'val_loss': val_loss, 'train_loss':train_loss, 'learning_rate': optimizer.param_groups[0]['lr'], 'grad_mean': grad_mean.item(), 'grad_L2_norm': grad_norm}, file_access = 'a')
        else:
            write_single_dict_to_jsonl(output_path + '/' + saved_model_history, {'epoch':epoch, 'val_accuracy':val_accuracy, 'train_accuracy':train_accuracy, 'f1_macro':f1, 'pre_macro':pre, 're_macro':recall, 'best_epoch':best_epoch, 'val_loss': val_loss, 'train_loss':train_loss, 'learning_rate': optimizer.param_groups[0]['lr'], 'grad_mean': grad_mean.item(), 'grad_L2_norm': grad_norm}, file_access = 'a')
    
    
    end = time.time()
    print(f"Training time (s): {end-start}")
    
    # Plot gradient flow over epochs
    plt.plot(grad_history)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Gradient Norm")
    plt.title("Gradient Flow Over Training")
    plt.show()

    # # Calculate parameters
    # remove unused parameters and count the number of parameters after that
    remove_unused_params(model)
    torch.save(model, output_path + '/' + saved_model_name)
    count_params(model)
    
    model = copy.deepcopy(model).cpu() # for more correct count
    # Calculate FLOPs
    flops, _ = get_model_complexity_info(model, (args.layers[0],), as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {flops}")
    write_single_dict_to_jsonl(output_path + '/' + saved_model_history, {'training time':end-start, 'flops':str(flops)}, file_access = 'a')
    

def predict_set(args):
    """
        Predict a given dataset using a trained model.
    """
    
    # Load the model
    model = torch.load(args.model_path)
    model.eval()  
    
    # Load the val/test set
    if args.ds_name == 'mnist':
        dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif args.ds_name == 'fashion_mnist':
        dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        
    elif(args.ds_name == 'cal_si'): # "caltech101_silhouettes
        data = scipy.io.loadmat("data/caltech101_silhouettes/caltech101_silhouettes_28_split1.mat")
        test_data = data['test_data']
        test_labels = data['test_labels'].flatten()
        dataset = TensorDataset(
            torch.tensor(test_data, dtype=torch.float32).reshape(-1, 1, 28, 28),
            torch.tensor(test_labels, dtype=torch.long)
        )
    else:
        # Customize the code to load any dataset you want to test
        raise ValueError("Unsupported dataset name.")
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Define loss
    criterion = nn.CrossEntropyLoss()

    # Initialize validation loss and accuracy
    set_loss, set_accuracy = 0, 0   
    
    # List to store predictions
    y_pred = []
    
    # Get true labels
    y_true = [labels.tolist() for images, labels in loader]
    y_true = sum(y_true, [])
    
    with torch.no_grad():  # Disable gradient calculation
        for images, labels in loader:
            batch_size, _, height, width = images.shape # extract all dimensions
            images = images.view(-1, height*width).to(device)
            output = model(images.to(device))
            set_loss += criterion(output, labels.to(device)).item()
            y_pred += output.argmax(dim=1).tolist()
            set_accuracy += ((output.argmax(dim=1) == labels.to(device)).float().mean().item())
    
    # Calculate F1
    f1 = f1_score(y_true, y_pred, average='macro')
    pre = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    
    # Calculate set loss and set accuracy
    set_loss /= len(loader)
    set_accuracy /= len(loader)
    
    result_dict = {}
    result_dict['set_loss'] = round(set_loss, 6)
    result_dict['set_accuracy'] = round(set_accuracy, 6)
    result_dict['f1'] = round(f1, 6)
    result_dict['pre'] = round(pre, 6)
    result_dict['recall'] = round(recall, 6)
    
    # Create a false inference dictionary
    false_dict = {}
    for x, y in zip(y_true, y_pred):
        if (x != y):
            if (y not in false_dict):
                false_dict[y] = 1
            else:
                false_dict[y] += 1
    false_dict = dict(sorted(false_dict.items(), key=lambda x: x[1], reverse = True))
    
    # Print results
    print(f"Set Loss: {set_loss:.6f}, Set Accuracy: {set_accuracy:.6f}, F1: {f1:.6f}, Precision: {pre:.6f}, Recall: {recall:.6f}")
    print(f"False inference dict: {false_dict}")
    
    return result_dict, false_dict

# this one is only for my own works
'''def compare(args, base_output = 'papers//FC-KAN//fc_kan_paper//'):
    
    # base_output = 'output//bsrbf_paper//'

    models = ['efficient_kan', 'fast_kan', 'bsrbf_kan',  'faster_kan',  'mlp', 'mfc_kan']
    
    dict_list = []
    for m in models:
        if (m == 'mfc_kan'):
            model_path = base_output + args.ds_name + '//' + m + '//' + m + '__' + args.ds_name + '__dog-bs__quadratic__full_1.pth'
        else:
            model_path = base_output + args.ds_name + '//' + m + '//' + m + '__' + args.ds_name + '__full_1.pth'
        #false_dict = predict_set(m, model_path, dataset, batch_size = 64)
        args.model_path = model_path
        false_dict = predict_set(args)
        dict_list.append({m:false_dict})
    print(dict_list)'''
   
def main(args):
    
    # Network layers
    layers = args.layers.split(',')
    layers = [int(x) for x in layers]
    args.layers = layers
    
    # FC-KAN
    func_list = args.func_list.split(',')
    func_list = [x.strip() for x in func_list]
    args.func_list = func_list
    
    # PRKAN + AF-KAN
    methods = args.methods.split(',')
    methods = [x.strip() for x in methods]
    args.methods = methods
    
    if (args.mode == 'train'):
        run(args)
    elif(args.mode == 'predict_set'):
        predict_set(args)
    '''else:
        compare(args)'''
    
if __name__ == "__main__":
    
    '''import torch
    from fvcore.nn import FlopCountAnalysis

    # Assuming FC_KAN is your model
    #model = FC_KAN([784, 64, 10])
    model = FC_KAN([784, 64, 10])
    dummy_input = torch.randn(1, 784)  # Adjust based on your input shape

    flop_counter = FlopCountAnalysis(model, dummy_input)
    flops = flop_counter.total()

    print(f"Total FLOPs: {flops:.2e}")'''

    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--mode', type=str, default='train') # or predict_set
    parser.add_argument('--model_name', type=str, default='efficient_kan')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--layers', type=str, default="784,64,10")
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, default='output/model.pth')
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--num_grids', type=int, default=8)
    parser.add_argument('--spline_order', type=int, default=3)
    parser.add_argument('--ds_name', type=str, default='mnist')
    parser.add_argument('--n_examples', type=int, default=0)
    parser.add_argument('--note', type=str, default='full')
    parser.add_argument('--n_part', type=float, default=0)
    parser.add_argument('--func_list', type=str, default='dog,rbf') # for FC-KAN
    parser.add_argument('--combined_type', type=str, default='quadratic')
    
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wc', type=float, default=1e-4, help='Weight decay')

    # SKAN
    parser.add_argument('--basis_function', type=str, default='sin')
    
    # PRKAN
    parser.add_argument('--func', type=str, default='rbf')
    parser.add_argument('--methods', type=str, default='attention')
    parser.add_argument('--norm_type', type=str, default='layer') 
    parser.add_argument('--base_activation', type=str, default='silu')
    parser.add_argument('--norm_pos', type=int, default=1)
    
    # RationalKAN
    parser.add_argument('--p_order', type=int, default=3)
    parser.add_argument('--q_order', type=int, default=3)
    parser.add_argument('--groups', type=int, default=8)
    
    # All
    parser.add_argument('--scheduler', type=str, default='ExponentialLR')
    
    # AF-KAN
    parser.add_argument('--func_norm', type=int, default=0, help='Function norm')
    
    # MLP
    parser.add_argument('--use_attn', type=int, default=0, help='Attention mechanism')
    args = parser.parse_args()
    
    # ReLUKAN
    args.use_attn = bool(args.use_attn) # Attention mechanism in MLP 
    
    # AF-KAN
    args.func_norm = bool(args.func_norm) # Function norm

    global device
    device = args.device
    if (args.device == 'cuda'): # check available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main(args)

# Some examples
#python run.py --mode "train" --model_name "fc_kan" --epochs 1 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --func_list "bs,dog" --combined_type "quadratic"

#python run.py --mode "train" --model_name "bsrbf_kan" --epochs 10 --batch_size 64 --layers "784,64,10" --grid_size 5 --spline_order 3 --ds_name "mnist"

#python run.py --mode "train" --model_name "bsrbf_kan" --epochs 1 --batch_size 16 --layers "3072,64,10" --grid_size 5 --spline_order 3 --ds_name "cifar10"

#python run.py --mode "train" --model_name "rbf_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --grid_size 5 --spline_order 3

#python run.py --mode "train" --model_name "rational_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --p_order 3 --q_order 3 --groups 8

#python run.py --mode "train" --model_name "knots_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --grid_size 20 --spline_order 3 --ds_name "mnist"

#python run.py --mode "train" --model_name "cheby_kan" --epochs 1 --batch_size 64 --layers "784,64,10" --spline_order 3 --ds_name "fashion_mnist"

#python run.py --mode "train" --model_name "fourier_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --grid_size 5 --spline_order 3 --ds_name "fashion_mnist"

#python run.py --mode "train" --model_name "relu_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --grid_size 3 --spline_order 3 --ds_name "mnist" --norm_type "layer" --base_activation "relu"

#python run.py --mode "train" --model_name "af_kan" --epochs 25 --batch_size 64 --layers "784,392,102" --grid_size 3 --spline_order 3 --ds_name "cal_si" --norm_type "layer" --base_activation "gelu" --methods "function_linear"

#python run.py --mode "train" --model_name "af_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --grid_size 3 --spline_order 3 --ds_name "mnist" --norm_type "layer" --base_activation "gelu" --methods "global_attn," --combined_type "sum_product"

#python run.py --mode "train" --model_name "af_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --grid_size 3 --spline_order 3 --ds_name "mnist" --norm_type "layer" --base_activation "silu" --methods "global_attn" --func "quad1"

#python run.py --mode "train" --model_name "mlp" --epochs 25 --batch_size 64 --layers "784,392,102" --ds_name "cal_si" --note "full"

#python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full" --norm_type "layer" --base_activation "silu"

# python run.py --mode "train" --model_name "mlp" --epochs 35 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full" --norm_type "layer" --base_activation "silu"

#python run.py --mode "train" --model_name "mlp" --epochs 25 --batch_size 64 --layers "784,392,102" --ds_name "cal_si" --note "full" --norm_type "layer" --base_activation "silu"

#python run.py --mode "train" --model_name "efficient_kan" --epochs 1 --batch_size 64 --layers "784,64,10" --grid_size 5 --spline_order 3 --ds_name "mnist"

#python run.py --mode "train" --model_name "skan" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --basis_function "sin"

#python run.py --mode "train" --model_name "fast_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --num_grids 8 --ds_name "mnist"

#python run.py --mode "train" --model_name "faster_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --num_grids 8 --ds_name "mnist"

#python run.py --mode "train" --model_name "gottlieb_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --spline_order 3 --ds_name "mnist"

#python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 16 --layers "784,392,102" --ds_name "cifar10" --note "full"

# python run.py --mode "train" --model_name "prkan" --epochs 25 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full" --grid_size 5 --spline_order 3 --num_grids 8 --func "rbf" --norm_type "" --base_activation "silu" --methods "conv2d" --combined_type "product"

#python run.py --mode "predict_set" --model_name "bsrbf_kan" --model_path='papers//BSRBF-KAN//bsrbf_paper//mnist//bsrbf_kan//bsrbf_kan__mnist__full_0.pth' --ds_name "mnist" --batch_size 64

# python run.py --mode "train" --model_name "prkan" --epochs 1 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_0" --n_part 0 --func "rbf" --base_activation "silu" --methods "attention" --norm_type "layer" --norm_pos 2 --scheduler "ExponentialLR" --lr 5e-7 

# python run.py --mode "train" --model_name "prkan" --epochs 15 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full_0" --n_part 0 --func "rbf" --base_activation "silu" --methods "attention" --norm_type "layer" --norm_pos 2 --scheduler "OneCycleLR"

#python run.py --mode "train" --model_name "af_kan" --epochs 25 --batch_size 64 --layers "784,64,10" --grid_size 3 --spline_order 3 --ds_name "mnist" --norm_type "layer" --base_activation "silu" --methods "global_attn" --func "quad1" --scheduler "OneCycleLR"

# python run.py --mode "train" --model_name "bsrbf_kan" --epochs 10 --batch_size 64 --layers "784,64,10" --grid_size 5 --spline_order 3 --ds_name "mnist" --scheduler "OneCycleLR"

#python run.py --mode "train" --model_name "rational_kan" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --p_order 3 --q_order 3 --groups 8 --scheduler "OneCycleLR"

# python run.py --mode "train" --model_name "fc_kan" --epochs 10 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --func_list "bs,dog" --combined_type "quadratic" --scheduler "OneCycleLR"

#python run.py --mode "train" --model_name "mlp" --epochs 5 --batch_size 64 --layers "784,64,10" --ds_name "mnist" --note "full" --scheduler "OneCycleLR"

#python run.py --mode "train" --model_name "bsrbf_kan" --epochs 5 --batch_size 16 --layers "3072,64,10" --grid_size 5 --spline_order 3 --ds_name "cifar10" --scheduler "OneCycleLR"

#python run.py --mode "train" --model_name "bsrbf_kan" --epochs 25 --batch_size 64 --layers "784,392,102" --grid_size 5 --spline_order 3 --ds_name "cal_si" --scheduler "CyclicLR"

# python run.py --mode "train" --model_name "fc_kan" --epochs 10 --batch_size 64 --layers "784,392,102" --ds_name "cal_si" --func_list "bs,dog" --combined_type "quadratic" --scheduler "OneCycleLR"
