import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from ptflops import get_model_complexity_info
import copy

import scipy.io
import requests

#import numpy as np
from file_io import *
from models import EfficientKAN, FastKAN, BSRBF_KAN, FasterKAN, MLP, FC_KAN, GottliebKAN, SKAN, PRKAN, ReLUKAN

from pathlib import Path
from PIL import Image
from prettytable import PrettyTable
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import matplotlib.pyplot as plt

# MNIST: Mean=0.1307, Std=0.3081
# Fashion-MNIST: Mean=0.2860, Std=0.3530
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])
    
# CIFAR10
transform_cifar = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
        ])

# Omniglot
transform_omniglot = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.Grayscale(num_output_channels=1),  # Ensure images are grayscale
    transforms.ToTensor()  # Convert images to PyTorch tensors
    ])

def remove_unused_params(model):
    """
        Remove unused parameters from the trained model
    """
    unused_params, _ = count_unused_params(model)
    for name in unused_params:
        #attr_name = name.split('.')[0]  # Get the top-level attribute name (e.g., 'unused')
        if hasattr(model, name):
            #print(f"Removing unused layer: {name}")
            delattr(model, name)  # Dynamically remove the unused layer
    return model

def count_unused_params(model):
    """
        Detect and count unused parameters
    """
    unused_params = []
    unused_param_count = 0

    for name, param in model.named_parameters():
        if param.grad is None:
            unused_params.append(name)
            unused_param_count += param.numel()  # Add the number of elements in this parameter
    
    return unused_params, unused_param_count

# use for CalTech 101 Silhouettes  
def convert_to_tensors(dataset):
    images, labels = [], []
    for image, label in tfds.as_numpy(dataset):
        images.append(image)
        labels.append(label)
    return torch.tensor(images, dtype=torch.float32).unsqueeze(1), torch.tensor(labels)
    
def count_params(model):
    """
        Count the model's parameters
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    
    # Detect and count unused parameters
    unused_params, unused_param_count = count_unused_params(model)
    
    if (unused_param_count != 0):
        print("Unused Parameters:", unused_params)
        print(f"Total Trainable Params: {total_params}")
        print(f"Total Number of Unused Parameters: {unused_param_count}")
        print(f"Total Number of Used Parameters: {total_params - unused_param_count}")
    else:
        print(f"Total Trainable Params: {total_params}")
        print(f"Total Number of Used Parameters: {total_params - unused_param_count}")
    
    return total_params
    
def run(args):
    
    start = time.time()
    
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
        
    '''elif(args.ds_name == 'cifar10'):
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True,  transform=transform_cifar
        )

        valset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_cifar
        )'''
    

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
    
    saved_model_name, saved_model_history = '', ''
    if (args.model_name == 'fc_kan'):
        saved_model_name = args.model_name + '__' + args.ds_name + '__' + '-'.join(x for x in args.func_list) + '__' + args.combined_type + '__' + args.note + '.pth'
        saved_model_history = args.model_name + '__' + args.ds_name + '__' + '-'.join(x for x in args.func_list) + '__' + args.combined_type + '__' + args.note + '.json' 
    elif(args.model_name == 'skan'):
        # args.basis_function
        saved_model_name = args.model_name + '__' + args.ds_name + '__' + args.basis_function + '__' + args.note + '.pth'
        saved_model_history =  args.model_name + '__' + args.ds_name + '__' + args.basis_function + '__' + args.note + '.json'
    elif(args.model_name == 'prkan'):
        # model = PRKAN([args.n_input, args.n_hidden, args.n_output], grid_size = args.grid_size, spline_order = args.spline_order, num_grids = args.num_grids, func = args.func, norm_type = args.norm_type, base_activation = args.base_activation, methods = args.methods, combined_type = args.combined_type)
        
        if (len(args.methods) == 1): args.combined_type = 'none'
        
        saved_model_name = args.model_name + '__' + args.ds_name + '__' + args.func + '__' + args.norm_type  + '__' + args.base_activation + '__' +'-'.join(x for x in args.methods)  + '__' + args.combined_type + '__' + args.note + '.pth'
        saved_model_history =  args.model_name + '__' + args.ds_name + '__' + args.func + '__' + args.norm_type  + '__' + args.base_activation + '__' +'-'.join(x for x in args.methods)  + '__' + args.combined_type + '__' + args.note + '.json'

    else:
        saved_model_name = args.model_name + '__' + args.ds_name + '__' + args.note + '.pth'
        saved_model_history =  args.model_name + '__' + args.ds_name + '__' + args.note + '.json'
    with open(os.path.join(output_path, saved_model_history), 'w') as fp: pass

    # Define models
    model = {}
    print('model_name: ', args.model_name)
    if (args.model_name == 'bsrbf_kan'):
        model = BSRBF_KAN([args.n_input, args.n_hidden, args.n_output], grid_size = args.grid_size, spline_order = args.spline_order)
    elif(args.model_name == 'fast_kan'):
        model = FastKAN([args.n_input, args.n_hidden, args.n_output], num_grids = args.num_grids)
    elif(args.model_name == 'faster_kan'):
        model = FasterKAN([args.n_input, args.n_hidden, args.n_output], num_grids = args.num_grids)
    elif(args.model_name == 'gottlieb_kan'):
        model = GottliebKAN([args.n_input, args.n_hidden, args.n_output], spline_order = args.spline_order)
    elif(args.model_name == 'mlp'):
        model = MLP([args.n_input, args.n_hidden, args.n_output])
    elif(args.model_name == 'fc_kan'):
        model = FC_KAN([args.n_input, args.n_hidden, args.n_output], args.func_list, combined_type = args.combined_type, grid_size = args.grid_size, spline_order = args.spline_order)
    elif(args.model_name == 'efficient_kan'):
        model = EfficientKAN([args.n_input, args.n_hidden, args.n_output], grid_size = args.grid_size, spline_order = args.spline_order)
    elif(args.model_name == 'prkan'):
        model = PRKAN([args.n_input, args.n_hidden, args.n_output], grid_size = args.grid_size, spline_order = args.spline_order, num_grids = args.num_grids, func = args.func, norm_type = args.norm_type, base_activation = args.base_activation, methods = args.methods, combined_type = args.combined_type, norm_pos = args.norm_pos)
    elif(args.model_name == 'skan'):
        model = SKAN([args.n_input, args.n_hidden, args.n_output], basis_function = args.basis_function) # lshifted_softplus, larctan 
    elif(args.model_name == 'relu_kan'):
        model = ReLUKAN([args.n_input, args.n_hidden, args.n_output], grid = args.grid_size , k = args.spline_order, norm_type = args.norm_type) 
    elif(args.model_name == 'pr_relu_kan'):
        model = PRReLUKAN([args.n_input, args.n_hidden, args.n_output], grid = args.grid_size , k = args.spline_order, norm_type = args.norm_type, base_activation = args.base_activation, methods = args.methods, combined_type = args.combined_type)     
    else:
        # add other KANs here
        raise ValueError("Unsupported network type.")
    model.to(device)

    # Define optimizer
    lr = 1e-3
    wc = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wc)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    
    # Define loss
    criterion = nn.CrossEntropyLoss()

    best_epoch, best_accuracy = 0, 0
    y_true = [labels.tolist() for images, labels in valloader]
    y_true = sum(y_true, [])
    
    if(args.ds_name == 'cal_si'):
        y_true_test = [labels.tolist() for images, labels in testloader]
        y_true_test = sum(y_true_test, [])
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_accuracy, train_loss = 0, 0
        with tqdm(trainloader) as pbar:
            for i, (images , labels) in enumerate(pbar):

                images = images.view(-1, args.n_input).to(device)
                optimizer.zero_grad()
                output = model(images.to(device))
                loss = criterion(output, labels.to(device))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                #accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                train_accuracy += (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                pbar.set_postfix(loss=train_loss/len(trainloader), accuracy=train_accuracy/len(trainloader), lr=optimizer.param_groups[0]['lr'])
        
        train_loss /= len(trainloader)
        train_accuracy /= len(trainloader)
            
        # Validation
        model.eval()
        val_loss, val_accuracy = 0, 0
        
        y_pred = []
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, args.n_input).to(device)
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

        # Update learning rate
        scheduler.step()

        # Choose best model
        if (val_accuracy > best_accuracy):
            best_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model, output_path + '/' + saved_model_name)
              
        print(f"Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.6f}")
        print(f"Epoch [{epoch}/{args.epochs}], Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.6f}, F1: {f1:.6f}, Precision: {pre:.6f}, Recall: {recall:.6f}")
        
        test_loss, test_accuracy = 0, 0
        if (args.ds_name == 'cal_si'):
            
            y_pred_test = []
            with torch.no_grad():
                for images, labels in testloader:
                    images = images.view(-1, args.n_input).to(device)
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
            write_single_dict_to_jsonl(output_path + '/' + saved_model_history, {'epoch':epoch, 'test_accuracy':test_accuracy, 'val_accuracy':val_accuracy, 'train_accuracy':train_accuracy, 'test_f1_macro':f1_test, 'test_pre_macro':pre_test, 'test_re_macro':recall_test,  'val_f1_macro':f1, 'val_pre_macro':pre, 'val_re_macro':recall, 'best_epoch':best_epoch, 'test_loss': test_loss, 'val_loss': val_loss, 'train_loss':train_loss}, file_access = 'a')
        else:
            write_single_dict_to_jsonl(output_path + '/' + saved_model_history, {'epoch':epoch, 'val_accuracy':val_accuracy, 'train_accuracy':train_accuracy, 'f1_macro':f1, 'pre_macro':pre, 're_macro':recall, 'best_epoch':best_epoch, 'val_loss': val_loss, 'train_loss':train_loss}, file_access = 'a')
    
    end = time.time()
    print(f"Training time (s): {end-start}")
    write_single_dict_to_jsonl(output_path + '/' + saved_model_history, {'training time':end-start}, file_access = 'a')
    
    # # Calculate parameters
    # remove unused parameters and count the number of parameters after that
    remove_unused_params(model)
    torch.save(model, output_path + '/' + saved_model_name)
    count_params(model)
    
    model = copy.deepcopy(model).cpu() # for more correct count
    # Calculate FLOPs
    flops, _ = get_model_complexity_info(model, (args.n_input,), as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {flops}")
    

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
    
    # for FC-KAN
    func_list = args.func_list.split(',')
    func_list = [x.strip() for x in func_list]
    args.func_list = func_list
    
    # for PRKAN + PRReLUKAN
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

    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--mode', type=str, default='train') # or predict_set
    parser.add_argument('--model_name', type=str, default='efficient_kan')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_input', type=int, default=28*28)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_output', type=int, default=10)
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

    # use for SKAN
    parser.add_argument('--basis_function', type=str, default='sin')
    
    # use for PRKAN
    parser.add_argument('--func', type=str, default='rbf')
    parser.add_argument('--methods', type=str, default='attention')
    parser.add_argument('--norm_type', type=str, default='layer') 
    parser.add_argument('--base_activation', type=str, default='silu')
    parser.add_argument('--norm_pos', type=int, default=1)

    args = parser.parse_args()
    
    global device
    device = args.device
    if (args.device == 'cuda'): # check available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main(args)


# Some examples

#python run.py --mode "train" --model_name "fc_kan" --epochs 35 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "fashion_mnist" --func_list "dog,sin" --combined_type "sum"

#python run.py --mode "train" --model_name "bsrbf_kan" --epochs 1 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 25 --grid_size 5 --spline_order 3 --ds_name "sl_mnist"

#python run.py --mode "train" --model_name "relu_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 3 --spline_order 3 --ds_name "mnist" --norm_type "layer"

#python run.py --mode "train" --model_name "pr_relu_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 392 --n_output 102 --grid_size 3 --spline_order 3 --ds_name "cal_si" --norm_type "layer" --base_activation "gelu" --methods "local_attn"

#python run.py --mode "train" --model_name "relu_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 392 --n_output 102 --grid_size 3 --spline_order 3 --ds_name "cal_si" 

#python run.py --mode "train" --model_name "mlp" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 392 --n_output 102 --ds_name "cal_si" --note "full"

#python run.py --mode "train" --model_name "bsrbf_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 392 --n_output 102 --grid_size 5 --spline_order 3 --ds_name "cal_si"

#python run.py --mode "train" --model_name "skan" --epochs 10 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --basis_function "sin"

#python run.py --mode "train" --model_name "fast_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8 --ds_name "mnist"

#python run.py --mode "train" --model_name "faster_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8 --ds_name "mnist"

#python run.py --mode "train" --model_name "gottlieb_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --spline_order 3 --ds_name "mnist"

#python run.py --mode "train" --model_name "mlp" --epochs 10 --batch_size 16 --n_input 3072 --n_hidden 64 --n_output 10 --ds_name "cifar10" --note "full"

# python run.py --mode "train" --model_name "prkan" --epochs 1 --batch_size 64 --n_input 3072 --n_hidden 64 --n_output 10 --ds_name "fashion_mnist" --note "full" --grid_size 5 --spline_order 3 --num_grids 8 --func "rbf" --norm_type "" --base_activation "silu" --methods "conv2d" --combined_type "product"

#python run.py --mode "predict_set" --model_name "bsrbf_kan" --model_path='papers//BSRBF-KAN//bsrbf_paper//mnist//bsrbf_kan//bsrbf_kan__mnist__full_0.pth' --ds_name "mnist" --batch_size 64

