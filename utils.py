import torch
from prettytable import PrettyTable

def cal_grad_mean(model):
    grad_mean = torch.mean(torch.stack([p.grad.abs().mean() for p in model.parameters() if p.grad is not None]))
    return grad_mean
        
def cal_grad_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
    return total_norm ** 0.5  # L2 norm            
            
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
