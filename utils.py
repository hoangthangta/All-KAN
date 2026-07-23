import torch
from prettytable import PrettyTable
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape

import random
import numpy as np

def cal_grad_mean(model):
    grad_mean = torch.mean(torch.stack([p.grad.abs().mean() for p in model.parameters() if p.grad is not None]))
    return grad_mean

def cal_grad_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm(2).item() ** 2
    return total_norm ** 0.5  # L2 norm

def count_params(model, display = True):
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
    if (display == True): print(table)

    unused_params, unused_param_count = count_unused_params(model)

    used_params = max(0, total_params - unused_param_count)  # Prevent negative used params
    
    if (display == False): return total_params
    if unused_param_count > 0:
        print("Unused Parameters:", unused_params)
        print(f"Total Trainable Params: {total_params}")
        print(f"Total Number of Unused Parameters: {unused_param_count}")
        print(f"Total Number of Used Parameters: {used_params}")
    else:
        print(f"Total Trainable Params: {total_params}")
        print(f"Total Number of Used Parameters: {used_params}")

    return used_params
   

def count_unused_params(model):
    """
    Detect and count unused parameters
    """
    unused_params = []
    unused_param_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:  # Only count trainable params without gradients
            unused_params.append(name)
            unused_param_count += param.numel()

    return unused_params, unused_param_count

def remove_unused_params(model):
    """
    Remove unused parameters from the trained model
    """
    unused_params, _ = count_unused_params(model)
    for name in unused_params:
        if hasattr(model, name):
            delattr(model, name)  # Dynamically remove the unused layer
    return model
    
    
def set_seed(seed : int = 42, device = 'cuda'):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
       

def calculate_flops(model, dummy_input):
    """
    Calculate FLOPs using fvcore with custom handlers for unsupported
    element-wise, pooling, and custom activation operators.
    """
    def get_numel(outputs):
        if not outputs:
            return 0
        shape = get_shape(outputs[0])
        if shape is None:
            return 0
        numel = 1
        for s in shape:
            if s is None:
                return 0
            numel *= s
        return numel

    def elementwise_flop(inputs, outputs):
        """Generic elementwise: usually 1 op per output element."""
        return get_numel(outputs)

    def exp_flop(inputs, outputs):
        return get_numel(outputs)

    def cosh_flop(inputs, outputs):
        # cosh(x) = (exp(x) + exp(-x)) / 2  -> roughly 2 exp + 1 add + 1 div
        return 4 * get_numel(outputs)   # Slightly more conservative

    def silu_flop(inputs, outputs):
        # SiLU(x) = x * sigmoid(x) -> 1 mul + sigmoid cost
        return 4 * get_numel(outputs)

    def pow_flop(inputs, outputs):
        return 4 * get_numel(outputs)   # Conservative estimate

    def max_pool2d_flop(inputs, outputs):
        """Approximate max pooling flops (mainly comparisons)."""
        try:
            # inputs[1] = kernel_size (usually a list in JIT)
            kernel_size = inputs[1].toIValue()
            if isinstance(kernel_size, (int, float)):
                kernel_size = [kernel_size] * 2  # common 1D->2D fallback
            kernel_ops = 1
            for k in kernel_size:
                kernel_ops *= int(k)
            kernel_ops = max(kernel_ops - 1, 1)  # comparisons
        except Exception:
            kernel_ops = 3  # fallback (e.g., 2x2 pool)

        return kernel_ops * get_numel(outputs)

    def rswaff_flop(inputs, outputs):
        """Custom FasterKAN RSWAF activation."""
        return 5 * get_numel(outputs)

    def sigmoid_flop(inputs, outputs):
        # sigmoid(x)=1/(1+exp(-x))
        # approx: neg + exp + add + reciprocal
        return 4 * get_numel(outputs)
    
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy_input)

        # Element-wise operators
        elementwise_ops = [
            "aten::add", "aten::sub", "aten::mul", "aten::div",
            "aten::reciprocal", "aten::min", "aten::max",
            "aten::lt", "aten::le", "aten::gt", "aten::ge",
            "aten::neg", "aten::where",
        ]
        for op in elementwise_ops:
            flops.set_op_handle(op, elementwise_flop)

        # Nonlinear / special ops
        flops.set_op_handle("aten::exp", exp_flop)
        flops.set_op_handle("aten::cosh", cosh_flop)
        flops.set_op_handle("aten::silu", silu_flop)
        flops.set_op_handle("aten::sigmoid", sigmoid_flop)
        flops.set_op_handle("aten::pow", pow_flop)

        # Pooling
        flops.set_op_handle("aten::max_pool2d", max_pool2d_flop)

        # Custom ops (FasterKAN)
        flops.set_op_handle("prim::PythonOp.RSWAFFunction", rswaff_flop)

        total_flops = flops.total()
        unsupported = flops.unsupported_ops()

        if unsupported:
            print("Remaining unsupported operators:", unsupported)

    return total_flops
