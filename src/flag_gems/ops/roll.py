# Copyright 2024 FlagGems Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

import torch
import triton
import triton.language as tl

@triton.jit
def roll_kernel(
    in_ptr, out_ptr,
    num_elements,
    shift, 
    middle_size, inner_size,
    BLOCK_SIZE: tl.constexpr
):
    # This kernel treats the tensor as a flattened 3D volume: [Outer, Middle, Inner]
    # We roll along the 'Middle' dimension.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Decompose linear index 'offsets' into (outer, middle, inner)
    # The linear index 'idx' maps to:
    #   idx = outer * (middle_size * inner_size) + middle * inner_size + inner
    
    # 1. Get 'inner' coordinate
    inner_idx = offsets % inner_size
    
    # 2. Get 'middle' coordinate
    # temp = offsets // inner_size
    # middle_idx = temp % middle_size
    # Combined:
    middle_idx = (offsets // inner_size) % middle_size
    
    # 3. Get 'outer' coordinate (implicitly handled by the rest of the offset math)
    outer_offset = offsets - (middle_idx * inner_size) - inner_idx
    
    # 4. Calculate source 'middle' index with wrap-around
    # src_middle = (middle_idx - shift) % middle_size
    shifted_middle = middle_idx - shift
    src_middle = ((shifted_middle % middle_size) + middle_size) % middle_size
    
    # 5. Reconstruct source linear offset
    # src_offset = outer_offset + src_middle * inner_size + inner_idx
    src_offsets = outer_offset + (src_middle * inner_size) + inner_idx
    
    val = tl.load(in_ptr + src_offsets, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)

def roll(input, shifts, dims=None):
    if dims is None:
        # Flatten logic
        input_view = input.flatten()
        output = torch.empty_like(input_view)
        grid = lambda META: (triton.cdiv(input.numel(), META['BLOCK_SIZE']),)
        # Shift must be passed as scalar
        s = shifts[0] if isinstance(shifts, (tuple, list)) else shifts
        # Treat as 1D: Outer=1, Middle=Numel, Inner=1
        roll_kernel[grid](input_view, output, input.numel(), s, input.numel(), 1, BLOCK_SIZE=1024)
        return output.view_as(input)

    if isinstance(shifts, int): shifts = (shifts,)
    if isinstance(dims, int): dims = (dims,)
    
    curr_input = input
    
    # Process dimensions sequentially
    for s, d in zip(shifts, dims):
        output = torch.empty_like(curr_input)
        num_el = curr_input.numel()
        
        # Calculate 3D view dimensions [Outer, Middle, Inner]
        # Middle is the size of the dim we are rolling
        middle_size = curr_input.size(d)
        
        # Inner is the product of all dims AFTER d
        inner_size = 1
        for i in range(d + 1, curr_input.dim()):
            inner_size *= curr_input.size(i)
            
        grid = lambda META: (triton.cdiv(num_el, META['BLOCK_SIZE']),)
        
        roll_kernel[grid](
            curr_input, output, 
            num_el, s, 
            middle_size, inner_size, 
            BLOCK_SIZE=1024
        )
        curr_input = output
        
    return output