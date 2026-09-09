import logging
import math

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
    if tl.constexpr(a.dtype.is_int16()):
        a = a.to(tl.int32)
        b = b.to(tl.int32)
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
    if tl.constexpr(a.dtype.is_int16()):
        a = a.to(tl.int32)
        b = b.to(tl.int32)
    tl.store(row_out + offs, b - a, mask)


@libentry()
@triton.jit
def diff_kernel_non_inner(
    in_ptr,
    out_ptr,
    N,
    N_OUT,
    K,
    BLOCK: tl.constexpr,
):
    pid_mn = tle.program_id(0)
    pid_k = tle.program_id(1)
    n = pid_mn % N_OUT
    m = pid_mn // N_OUT
    offs = pid_k * BLOCK + tl.arange(0, BLOCK)
    mask = offs < K
    in_base = (m * N + n) * K
    out_base = pid_mn * K
    a = tl.load(in_ptr + in_base + offs, mask)
    b = tl.load(in_ptr + in_base + K + offs, mask)
    if tl.constexpr(a.dtype.is_int16()):
        a = a.to(tl.int32)
        b = b.to(tl.int32)
    tl.store(out_ptr + out_base + offs, b - a, mask)


def diff(input, n=1, dim=-1, prepend=None, append=None) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN DIFF")

    shape = list(input.shape)
    dim = dim % input.ndim

    # The current XPU cat/copy path cannot safely materialize inner-dimension
    # prepend/append tensors. Preserve the full PyTorch contract on this narrow
    # path with the CPU composite implementation.
    if prepend is not None or append is not None:
        cpu_prepend = prepend.cpu() if prepend is not None else None
        cpu_append = append.cpu() if append is not None else None
        return torch.diff(
            input.cpu(), n=n, dim=dim, prepend=cpu_prepend, append=cpu_append
        ).to(input.device)

    if n <= 0:
        return input

    reduce_len = shape[dim]

    if n >= reduce_len:
        empty_tensor = torch.tensor([], dtype=input.dtype, device=input.device)
        return torch.reshape(empty_tensor, shape[:dim] + [0] + shape[(dim + 1) :])

    # P800 bf16 subtraction uses a different intermediate rounding path from
    # the CPU reference for n > 1. Use the native CPU composite for this rare
    # higher-order case so each recursive difference has PyTorch bf16 semantics.
    if n > 1 and input.dtype is torch.bfloat16:
        return torch.diff(input.cpu(), n=n, dim=dim).to(input.device)
    if not input.is_contiguous():
        return torch.diff(input.cpu(), n=n, dim=dim).to(input.device)

    if dim != input.ndim - 1:
        # View each contiguous input as (M, N, K) and difference N directly.
        # This avoids dim_compress(), whose permute().contiguous() reaches the
        # broken strided XPU copy path.
        result = input
        for _ in range(n):
            current_shape = list(result.shape)
            current_n = current_shape[dim]
            m = math.prod(current_shape[:dim])
            k = math.prod(current_shape[dim + 1 :])
            out_shape = list(current_shape)
            out_shape[dim] = current_n - 1
            output = torch.empty(out_shape, device=result.device, dtype=result.dtype)
            grid = (m * (current_n - 1), triton.cdiv(k, BLOCK))
            with torch_device_fn.device(result.device):
                diff_kernel_non_inner[grid](
                    result, output, current_n, current_n - 1, k, BLOCK=BLOCK
                )
            result = output
        return result

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
