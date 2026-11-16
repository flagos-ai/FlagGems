import torch
import triton
import triton.language as tl


@triton.jit
def _masked_scale_kernel(
    input_ptr,
    mask_ptr,
    out_ptr,
    n_elements,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=valid)
    m = tl.load(mask_ptr + offsets, mask=valid)
    res = tl.where(m != 0, x * scale, 0.0)
    tl.store(out_ptr + offsets, res, mask=valid)


@triton.jit
def _masked_scale_kernel_nomask(
    input_ptr,
    mask_ptr,
    out_ptr,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + offsets)
    m = tl.load(mask_ptr + offsets)
    res = tl.where(m != 0, x * scale, 0.0)
    tl.store(out_ptr + offsets, res)


@triton.jit
def _masked_scale_kernel_nomask_stream(
    input_ptr,
    mask_ptr,
    out_ptr,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    # Streaming variant: .cg loads bypass L1/L2 caching for single-use
    # inputs, .cs store marks the output as streaming to avoid write
    # allocate pollution. Best on very large (>100M element) workloads.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(
        input_ptr + offsets,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    m = tl.load(
        mask_ptr + offsets,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    res = tl.where(m != 0, x * scale, 0.0)
    tl.store(
        out_ptr + offsets,
        res,
        cache_modifier=".cs",
        eviction_policy="evict_first",
    )


def run(input, mask, scale):
    out = torch.empty_like(input)
    n_elements = input.numel()
    scale = float(scale)
    if n_elements % 1024 == 0:
        if n_elements >= 100_000_000:
            grid = (n_elements // 1024,)
            _masked_scale_kernel_nomask_stream[grid](
                input, mask, out, scale, BLOCK_SIZE=1024, num_warps=8
            )
        else:
            grid = (n_elements // 1024,)
            _masked_scale_kernel_nomask[grid](input, mask, out, scale, BLOCK_SIZE=1024)
    else:
        grid = (triton.cdiv(n_elements, 1024),)
        _masked_scale_kernel[grid](input, mask, out, n_elements, scale, BLOCK_SIZE=1024)
    return out


# Alias for FlagGems import convention
masked_scale = run
