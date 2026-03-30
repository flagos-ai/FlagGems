import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def repeat_kernel_3d(
    inp_ptr, out_ptr,
    inp_s0, inp_s1, inp_s2,
    out_s1, out_s2,
    inp_st0, inp_st1,
    rep_s2,
    BLOCK_SIZE: tl.constexpr,
):
    """3D repeat: each program handles one output row, writing all dim2 repeats."""
    row_id = tl.program_id(0)

    o0 = row_id // out_s1
    o1 = row_id % out_s1

    i0 = o0 % inp_s0
    i1 = o1 % inp_s1

    inp_row_base = i0 * inp_st0 + i1 * inp_st1
    out_row_base = row_id * out_s2

    # Load input row once
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < inp_s2
    vals = tl.load(inp_ptr + inp_row_base + offsets, mask=mask)

    # Write to all dim2 repeats
    for t in range(rep_s2):
        tl.store(out_ptr + out_row_base + t * inp_s2 + offsets, vals, mask=mask)


@triton.jit
def repeat_kernel_3d_tiled(
    inp_ptr, out_ptr,
    inp_s0, inp_s1, inp_s2,
    out_s1, out_s2,
    inp_st0, inp_st1,
    BLOCK_SIZE: tl.constexpr,
):
    """3D repeat tiled: each program handles one (row, tile) pair."""
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    o0 = row_id // out_s1
    o1 = row_id % out_s1

    i0 = o0 % inp_s0
    i1 = o1 % inp_s1

    inp_row_base = i0 * inp_st0 + i1 * inp_st1
    out_row_base = row_id * out_s2 + tile_id * inp_s2

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < inp_s2
    vals = tl.load(inp_ptr + inp_row_base + offsets, mask=mask)
    tl.store(out_ptr + out_row_base + offsets, vals, mask=mask)


def _next_power_of_2(n):
    """Compute next power of 2."""
    n = int(n)
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def repeat(inp: torch.Tensor, *sizes) -> torch.Tensor:
    """Repeat operator for Iluvatar platform.

    Repeat tensor elements along specified dimensions. Uses optimized Triton kernel
    for 3D tensors with loop or tiled strategy based on repeat size.

    Args:
        inp: Input tensor to be repeated
        *sizes: The number of times to repeat this tensor along each dimension

    Returns:
        Output tensor with repeated elements
    """
    logger.debug("GEMS_ILUVATAR REPEAT")

    # Convert sizes to tuple if not already
    if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
        sizes = tuple(sizes[0])
    else:
        sizes = tuple(sizes)

    ndim = inp.ndim
    sizes_ndim = len(sizes)

    if sizes_ndim > ndim:
        for _ in range(sizes_ndim - ndim):
            inp = inp.unsqueeze(0)
        ndim = inp.ndim

    # 1D: PyTorch is already fast
    if ndim == 1:
        return inp.repeat(sizes)

    # Normalize to 3D for unified kernel path
    orig_ndim = ndim
    orig_sizes = sizes
    while ndim < 3:
        inp = inp.unsqueeze(0)
        sizes = (1,) + sizes
        ndim += 1

    # For >3D, fall back to PyTorch
    if ndim > 3:
        return inp.repeat(sizes)

    inp = inp.contiguous()
    inp_shape = inp.shape
    out_shape = (
        inp_shape[0] * sizes[0],
        inp_shape[1] * sizes[1],
        inp_shape[2] * sizes[2],
    )

    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    if out.numel() == 0:
        return out

    num_rows = out_shape[0] * out_shape[1]
    inp_s2 = inp_shape[2]
    rep_s2 = sizes[2]

    BLOCK_SIZE = _next_power_of_2(inp_s2)
    if BLOCK_SIZE < 128:
        BLOCK_SIZE = 128
    if BLOCK_SIZE > 4096:
        BLOCK_SIZE = 4096

    # Fall back to PyTorch if single element is larger than BLOCK_SIZE
    if inp_s2 > BLOCK_SIZE:
        result = inp.repeat(sizes)
        if orig_ndim == 2:
            return result.reshape(out_shape[1], out_shape[2])
        return result

    num_warps = 4 if BLOCK_SIZE >= 512 else (2 if BLOCK_SIZE >= 128 else 1)

    with torch_device_fn.device(inp.device.index):
        # For small rep_s2, use loop kernel (load once, store multiple times)
        # For large rep_s2, use tiled kernel (better parallelism)
        if rep_s2 <= 8:
            grid = (num_rows,)
            repeat_kernel_3d[grid](
                inp, out,
                inp_shape[0], inp_shape[1], inp_s2,
                out_shape[1], out_shape[2],
                inp.stride(0), inp.stride(1),
                rep_s2,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                num_stages=2,
            )
        else:
            grid = (num_rows, rep_s2)
            repeat_kernel_3d_tiled[grid](
                inp, out,
                inp_shape[0], inp_shape[1], inp_s2,
                out_shape[1], out_shape[2],
                inp.stride(0), inp.stride(1),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                num_stages=4,
            )

    # Reshape back to original dimensionality
    if orig_ndim == 2:
        return out.reshape(out_shape[1], out_shape[2])
    return out
