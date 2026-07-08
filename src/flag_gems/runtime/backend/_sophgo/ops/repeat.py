import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils.libentry import libentry


def repeat_block_m(n_rows):
    if n_rows >= 1024:
        return 8
    if n_rows >= 256:
        return 4
    return 2


def repeat_block_n(n_cols):
    if n_cols >= 1024:
        return 512
    if n_cols >= 256:
        return 256
    return triton.next_power_of_2(max(1, n_cols))


@libentry()
@triton.jit
def repeat_2d_kernel(
    in_ptr,
    out_ptr,
    in_rows,
    in_cols,
    out_rows,
    out_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < out_rows) & (cols < out_cols)

    src_rows = rows - (rows // in_rows) * in_rows
    src_cols = cols - (cols // in_cols) * in_cols
    vals = tl.load(in_ptr + src_rows * in_cols + src_cols, mask=mask, other=0.0)
    tl.store(out_ptr + rows * out_cols + cols, vals, mask=mask)


def repeat(inp: torch.Tensor, sizes) -> torch.Tensor:
    logging.debug("GEMS REPEAT (Sophgo backend)")

    in_rank = inp.dim()
    sizes_rank = len(sizes)
    in_shape = list(inp.shape)
    sizes_shape = list(sizes)

    assert (
        sizes_rank >= in_rank
    ), "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor"
    if sizes_rank > in_rank:
        diff = sizes_rank - in_rank
        in_shape = [1 for _ in range(diff)] + in_shape
        inp = inp.reshape(in_shape)

    is_empty = False
    out_shape = []
    for in_size, repeat_size in zip(in_shape, sizes_shape):
        assert (
            repeat_size >= 0
        ), f"the number of repetitions per dimension out of range (expected to >= 0) but got {repeat_size}"
        if repeat_size == 0:
            is_empty = True
        out_shape.append(in_size * repeat_size)

    out0 = torch.empty(out_shape, device=inp.device, dtype=inp.dtype)
    if is_empty:
        return out0

    if inp.ndim == 2 and inp.is_contiguous() and out0.is_contiguous():
        block_m = repeat_block_m(out0.shape[0])
        block_n = repeat_block_n(out0.shape[1])
        grid = (triton.cdiv(out0.shape[0], block_m), triton.cdiv(out0.shape[1], block_n))
        with torch_device_fn.device(inp.device.index):
            repeat_2d_kernel[grid](
                inp,
                out0,
                inp.shape[0],
                inp.shape[1],
                out0.shape[0],
                out0.shape[1],
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )
        return out0

    out = inp
    for dim, repeat_size in enumerate(sizes_shape):
        if repeat_size <= 1:
            continue
        out = torch.cat([out] * repeat_size, dim=dim)
    return out
