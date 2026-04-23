import torch


def select_scatter(inp, src, dim, index):
    dim = dim % inp.ndim
    index = index % inp.size(dim)
    out = inp.clone()
    out.select(dim, index).copy_(src)
    return out
