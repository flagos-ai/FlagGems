import torch
from .leaky_relu import leaky_relu_kernel

def leaky_relu_(input, negative_slope=0.01):
    n_elements = input.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
    leaky_relu_kernel[grid](input, input, n_elements, negative_slope)
    return input
