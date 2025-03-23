import torch
import torch.optim as optim

def get_scheduler(optimizer, name="ExponentialLR", step_size=1000, epochs=25):
    """
    Returns a learning rate scheduler with fixed parameters based on the given schedule type.

    Args:
        optimizer: PyTorch optimizer.
        name (str): Name of the scheduler. Options: 
            'ExponentialLR', 'StepLR', 'MultiStepLR', 'CosineAnnealingLR', 
            'OneCycleLR', 'CyclicLR', 'ReduceLROnPlateau'.
        step_size (int): Step size for applicable schedulers.

    Returns:
        A PyTorch learning rate scheduler.
    """
    
    if name == "ExponentialLR":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)  # 0.8, 0.85, 0.9, 0.95, 0.99

    elif name == "StepLR":  # bad
        # epochs // 3
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    elif name == "MultiStepLR":  # bad
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)

    elif name == "CosineAnnealingLR":
        # eta_min: Minimum learning rate. Default: 0.
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    elif name == "OneCycleLR":
        return optim.lr_scheduler.OneCycleLR(
            # epochs, steps_per_epoch 
            optimizer, 
            max_lr=3e-3,  # Fixed max_lr
            total_steps=step_size,  # Total steps for annealing
            anneal_strategy='cos',
            div_factor = 3
            #final_div_factor=1e4 
        )
    
    elif name == "CyclicLR": 
        
        return optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=3e-6,  # Lower bound
            max_lr=3e-3,  # Upper bound
            step_size_up=step_size,  # Steps to peak LR
            mode='triangular2'
        )
        
    else:
        raise ValueError(f"Invalid schedule type '{name}'. Choose from: ExponentialLR, StepLR, MultiStepLR, CosineAnnealingLR, OneCycleLR, CyclicLR.")
