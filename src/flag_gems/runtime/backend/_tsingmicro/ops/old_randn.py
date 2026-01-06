import torch
import triton
import triton.language as tl
import functools

def hetero_device_support(func):
    @functools.wraps(func)
    def wrapper(*args, device='cpu', **kwargs):
        if device is not None and str(device) != 'cpu':
            ori_kwargs = kwargs.copy()
            ori_kwargs['device'] = 'cpu'
            cpu_tensor = func(*args, **ori_kwargs)
            return cpu_tensor.to(device)
        return func(*args, device=device, **kwargs)
    return wrapper

def hetero_like_support(func):
    @functools.wraps(func)
    def wrapper(input: torch.Tensor,
                **kwargs) -> torch.Tensor:
        target_device = input.device

        if target_device == 'cpu':
            return func(input, **kwargs)
        
        cpu_input = input.to('cpu')
        if cpu_input.dtype == torch.float64:
            cpu_input = cpu_input.to(torch.float32)
        cpu_out = func(cpu_input,  **kwargs)
        return cpu_out.to(target_device)
    return wrapper

torch.randn = hetero_device_support(torch.randn)
torch.randint = hetero_device_support(torch.randint)
torch.randn_like = hetero_like_support(torch.randn_like)
torch.zeros_like = hetero_like_support(torch.zeros_like)