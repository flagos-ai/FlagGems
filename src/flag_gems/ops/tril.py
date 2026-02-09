# Copyright 2024 FlagGems Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

import torch
import triton
import triton.language as tl

@triton.jit
def tril_kernel(
    inp_ptr, out_ptr,
    M, N, stride_m, num_elements,
    diagonal,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Calculate row/col from linear index
    # Assuming contiguous last dim (row-major)
    # row = offsets // N, col = offsets % N
    row = offsets // N
    col = offsets % N
    
    val = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    
    # Tril condition: col <= row + diagonal
    condition = col <= (row + diagonal)
    
    res = tl.where(condition, val, 0.0)
    tl.store(out_ptr + offsets, res, mask=mask)

def tril(input, diagonal=0):
    # Handle batch dimensions by flattening to 2D (NxM) effectively
    # if we only care about the last two dims.
    # PyTorch tril works on the last two dimensions.
    assert input.dim() >= 2
    row_dim = input.dim() - 2
    col_dim = input.dim() - 1
    
    M = input.size(row_dim)
    N = input.size(col_dim)
    num_elements = input.numel()
    
    output = torch.empty_like(input)
    grid = lambda META: (triton.cdiv(num_elements, META['BLOCK_SIZE']),)
    
    tril_kernel[grid](
        input, output,
        M, N, input.stride(row_dim), num_elements,
        diagonal,
        BLOCK_SIZE=1024
    )
    return output