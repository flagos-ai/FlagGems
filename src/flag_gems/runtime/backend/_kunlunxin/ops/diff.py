import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# The generic diff uses @libtuner (key=["M","N"] with 45 configs on kunlunxin)
# so every distinct (M, N) shape re-autotunes all configs -> huge compile +
# IR explosion (13.6M-line dump). Worse, its diff_kernel_2d addresses a 2D
# strided tile `M_offsets[:,None]*M_STRIDE + offs` whose runtime row stride
# defeats XPU contiguity analysis -> fully discrete access (0.003-0.03x torch
# on every 2D/3D shape).
#
# Fix (no libtuner, fixed BLOCK): drive one program per (row, chunk) with a
# pre-offset base pointer so each program does a purely contiguous 1D block-DMA
# `out[row, j:j+BLOCK] = in[row, j+1:...] - in[row, j:...]`. A fixed BLOCK=8192
# beats an N-adaptive block on XPU (large tiles stay well utilized; smaller
# tiles regress small-N cases). 1D inputs keep the fast flat-DMA path.
BLOCK = 8192


@libentry()
@triton.jit
def diff_kernel_1d(in_ptr, out_ptr, N_OUT, BLOCK: tl.constexpr):
    pid = tle.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_OUT
    a = tl.load(in_ptr + offs, mask)
    b = tl.load(in_ptr + offs + 1, mask)
    tl.store(out_ptr + offs, b - a, mask)


@libentry()
@triton.jit
def diff_kernel_2d(
    in_ptr,
    out_ptr,
    N_OUT,
    M_STRIDE_IN,
    M_STRIDE_OUT,
    BLOCK: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_c = tle.program_id(1)
    row_in = in_ptr + pid_m * M_STRIDE_IN
    row_out = out_ptr + pid_m * M_STRIDE_OUT
    offs = pid_c * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_OUT
    a = tl.load(row_in + offs, mask)
    b = tl.load(row_in + offs + 1, mask)
    tl.store(row_out + offs, b - a, mask)


def diff(input, n=1, dim=-1, prepend=None, append=None) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN DIFF")

    if prepend is not None:
        input = torch.cat([prepend, input], dim=dim)
    if append is not None:
        input = torch.cat([input, append], dim=dim)

    if n <= 0:
        return input

    shape = list(input.shape)
    dim = dim % input.ndim
    reduce_len = shape[dim]

    if n >= reduce_len:
        empty_tensor = torch.tensor([], dtype=input.dtype, device=input.device)
        return torch.reshape(empty_tensor, shape[:dim] + [0] + shape[(dim + 1) :])

    input = dim_compress(input, dim)
    N = reduce_len
    M = input.numel() // N

    is_1d = len(shape) == 1

    def _launch(src, dst, in_stride_m, out_stride_m, n_bound):
        n_out = n_bound - 1
        with torch_device_fn.device(src.device):
            if is_1d:
                grid = (triton.cdiv(n_out, BLOCK),)
                diff_kernel_1d[grid](src, dst, n_out, BLOCK=BLOCK)
            else:
                grid = (M, triton.cdiv(n_out, BLOCK))
                diff_kernel_2d[grid](
                    src, dst, n_out, in_stride_m, out_stride_m, BLOCK=BLOCK
                )

    out_shape = list(input.shape)
    out_shape[-1] = N - n
    output = torch.empty(out_shape, device=input.device, dtype=input.dtype)

    if n == 1:
        _launch(input, output, N, N - 1, N)
        return torch.moveaxis(output, -1, dim)

    # n >= 2: ping-pong between two scratch buffers, writing the last iteration
    # directly into `output` (size N-n).
    scratch_a_shape = list(input.shape)
    scratch_a_shape[-1] = N - 1
    scratch_a = torch.empty(scratch_a_shape, device=input.device, dtype=input.dtype)
    if n >= 3:
        scratch_b_shape = list(input.shape)
        scratch_b_shape[-1] = N - 2
        scratch_b = torch.empty(scratch_b_shape, device=input.device, dtype=input.dtype)

    _launch(input, scratch_a, N, N - 1, N)
    src, src_stride = scratch_a, N - 1

    for k in range(1, n):
        if k == n - 1:
            dst, dst_stride = output, N - n
        elif k % 2 == 1:
            dst, dst_stride = scratch_b, N - 2
        else:
            dst, dst_stride = scratch_a, N - 1
        _launch(src, dst, src_stride, dst_stride, N - k)
        src, src_stride = dst, dst_stride

    return torch.moveaxis(output, -1, dim)
