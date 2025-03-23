import os
from pathlib import Path

def create_model_storage(args):
    """
    Creates model storage directories and generates model file names based on the provided arguments.
    
    Args:
        args: An object containing attributes such as model_name, ds_name, func_list, combined_type, 
              note, basis_function, norm_type, base_activation, func, methods, and grid parameters.
              
    Returns:
        output_path (str): The path where models will be stored.
        saved_model_name (str): The filename for saving the model.
        saved_model_history (str): The filename for saving the training history.
    """
    output_path = os.path.join('output', args.ds_name, args.model_name)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    saved_model_name = ''
    saved_model_history = ''
    
    if args.model_name == 'fc_kan':
        model_suffix = f"{'-'.join(args.func_list)}__{args.combined_type}__{args.note}"
    elif args.model_name == 'skan':
        model_suffix = f"{args.basis_function}__{args.note}"
    elif args.model_name == 'prkan':
        if len(args.methods) == 1:
            args.combined_type = 'none'
        model_suffix = f"{args.func}__{args.norm_type}__{args.base_activation}__{'-'.join(args.methods)}__{args.combined_type}__{args.note}"
    elif args.model_name == 'af_kan':
        if len(args.methods) == 1:
            args.combined_type = 'none'
        model_suffix = f"{args.norm_type}__{args.base_activation}__{'-'.join(args.methods)}__{args.combined_type}__{args.func}__{args.note}"
    else:
        model_suffix = args.note
    
    saved_model_name = f"{args.model_name}__{args.ds_name}__{model_suffix}.pth"
    saved_model_history = f"{args.model_name}__{args.ds_name}__{model_suffix}.json"
    
    # Create empty history file
    with open(os.path.join(output_path, saved_model_history), 'w') as fp:
        pass
    
    return output_path, saved_model_name, saved_model_history
