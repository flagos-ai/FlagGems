import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24


@libentry()
@triton.jit
def repeat_interleave_kernel(
    x_ptr,
    out_ptr,
    N_total,
    inner_size,
    repeat_inner: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    for block_id in tl.range(pid, NUM_BLOCKS, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total

        inner_idx = off % inner_size
        outer_idx = off // repeat_inner
        src_idx = outer_idx * inner_size + inner_idx

        vals = tl.load(x_ptr + src_idx, mask=mask)
        tl.store(out_ptr + off, vals, mask=mask)


@libentry()
@triton.jit
def repeat_interleave_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    repeats: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    for block_id in tl.range(pid, NUM_BLOCKS, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total

        src_idx = off // repeats

        vals = tl.load(x_ptr + src_idx, mask=mask)
        tl.store(out_ptr + off, vals, mask=mask)


def repeat_interleave_self_int(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS REPEAT_INTERLEAVE_SELF_INT")
    if dim is None:
        inp = inp.contiguous().flatten()
        dim = 0
    else:
        if dim < -inp.ndim or dim >= inp.ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )
        if dim < 0:
            dim = dim + inp.ndim

    inp_shape = list(inp.shape)
    output_shape = list(inp.shape)
    output_shape[dim] *= repeats

    if output_size is not None and output_size != output_shape[dim]:
        raise RuntimeError(
            "repeat_interleave: Invalid output_size, expected {} but got {}".format(
                output_shape[dim], output_size
            )
        )

    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    if repeats == 0:
        return output

    inp = inp.contiguous()
    N_total = output.numel()

    BLOCK = 2048
    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, NUM_SIPS * 4)

    inner_size = 1
    for i in range(dim + 1, len(inp_shape)):
        inner_size *= inp_shape[i]

    with torch_device_fn.device(inp.device):
        if inner_size == 1:
            repeat_interleave_flat_kernel[(grid_size,)](
                inp, output, N_total,
                repeats=repeats, NUM_BLOCKS=NUM_BLOCKS, BLOCK=BLOCK,
            )
        else:
            repeat_inner = repeats * inner_size
            repeat_interleave_kernel[(grid_size,)](
                inp, output, N_total,
                inner_size,
                repeat_inner=repeat_inner, NUM_BLOCKS=NUM_BLOCKS, BLOCK=BLOCK,
            )

    return output


@triton.jit
def repeat_interleave_tensor_kernel(
    repeats_ptr, cumsum_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    mask = pid < size
    cumsum = tl.load(cumsum_ptr + pid, mask, other=0)
    repeats = tl.load(repeats_ptr + pid, mask, other=0)
    out_offset = cumsum - repeats

    tl.device_assert(repeats >= 0, "repeats can not be negative")

    out_ptr += out_offset
    for start_k in range(0, repeats, BLOCK_SIZE):
        offsets_k = start_k + tl.arange(0, BLOCK_SIZE)
        mask_k = offsets_k < repeats
        tl.store(out_ptr + offsets_k, pid, mask=mask_k)


def repeat_interleave_tensor(repeats, *, output_size=None):
    logger.debug("GEMS REPEAT_INTERLEAVE_TENSOR")
    assert repeats.ndim == 1, "repeat_interleave only accept 1D vector as repeat"
    cumsum = repeats.cumsum(axis=0)
    result_size = cumsum[-1].item()
    assert result_size >= 0, "repeats can not be negative"
    out = torch.empty((result_size,), dtype=repeats.dtype, device=repeats.device)
    size = repeats.size(0)
    grid = (size,)
    BLOCK_SIZE = 32
    repeat_interleave_tensor_kernel[grid](
        repeats, cumsum, out, size, BLOCK_SIZE=BLOCK_SIZE, num_warps=1,
    )
    return out


def repeat_interleave_self_tensor(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS REPEAT_INTERLEAVE_SELF_TENSOR")
    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if dim < -inp.ndim or dim >= inp.ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )
    if repeats.ndim == 0 or (repeats.ndim == 1 and repeats.size(0) == 1):
        return repeat_interleave_self_int(
            inp, repeats.item(), dim=dim, output_size=output_size
        )
    elif repeats.ndim > 1:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")
    inp_shape = list(inp.shape)
    if dim < 0:
        dim = dim + len(inp_shape)
    if repeats.size(0) != inp_shape[dim]:
        raise RuntimeError(
            "repeats must have the same size as input along dim"
        )
    indices = repeat_interleave_tensor(repeats)
    res = torch.index_select(inp, dim, indices)
    return res
