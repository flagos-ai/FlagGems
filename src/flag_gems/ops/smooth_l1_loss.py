# Copyright 2024 FlagGems Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

import torch
import triton
import triton.language as tl

@triton.jit
def smooth_l1_loss_kernel(
    input_ptr, target_ptr, output_ptr,
    num_elements,
    beta,
    reduction_mode, # 0: none, 1: mean, 2: sum
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Grid-Stride Loop to handle tensors larger than grid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Load data
    inp = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tgt = tl.load(target_ptr + offsets, mask=mask, other=0.0)

    # Math: Smooth L1 Logic
    # if |diff| < beta: 0.5 * diff^2 / beta
    # else: |diff| - 0.5 * beta
    diff = tl.abs(inp - tgt)
    cond = diff < beta
    
    squared_loss = 0.5 * diff * diff / beta
    linear_loss = diff - 0.5 * beta
    
    val = tl.where(cond, squared_loss, linear_loss)
    
    # Reduction Logic
    # reduction_mode 0 (None): Store every element
    if reduction_mode == 0:
        tl.store(output_ptr + offsets, val, mask=mask)
    
    # reduction_mode 1 (Mean) or 2 (Sum): Sum elements in block
    else:
        # Mask out invalid elements for reduction
        val = tl.where(mask, val, 0.0)
        block_sum = tl.sum(val, axis=0)
        
        # Atomic add to global accumulator
        # Note: For very large tensors, a two-pass reduction is faster, 
        # but for typical sizes, atomic_add is competitive and simpler.
        if pid == 0:
             # Just for demonstration of atomic safety, normally we use atomic_add everywhere
             # But here we can just atomic add the block sum to the 0-th element output
             pass
        tl.atomic_add(output_ptr, block_sum)

def smooth_l1_loss(input, target, beta=1.0, reduction='mean'):
    assert input.is_contiguous() and target.is_contiguous()
    num_elements = input.numel()
    
    # Map string reduction to int for kernel
    red_mode = 0
    if reduction == 'mean': red_mode = 1
    elif reduction == 'sum': red_mode = 2
    
    # Output setup
    if red_mode == 0:
        output = torch.empty_like(input)
    else:
        output = torch.zeros(1, device=input.device, dtype=input.dtype)

    grid = lambda META: (triton.cdiv(num_elements, META['BLOCK_SIZE']),)
    
    smooth_l1_loss_kernel[grid](
        input, target, output,
        num_elements,
        beta,
        red_mode,
        BLOCK_SIZE=1024,
    )
    
    if reduction == 'mean':
        return output / num_elements
    return output