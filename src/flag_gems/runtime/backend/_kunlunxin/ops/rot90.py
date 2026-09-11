import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# NOTE (kunlunxin/XPU): the generic rot90 kernel is decorated with
# `@triton.autotune(configs=get_tuned_config("rot90"), key=["n_elements"])`.
# On XPU triton, autotune re-benchmarks ALL configs for every distinct
# n_elements (benchmark has many shapes/dtypes -> many distinct n) -> the launch
# path recompiles per (config, n) -> IR explosion (196MB / 10512 modules, see
# ir-rot90-dev5.log). Same family as the bernoulli_/uniform_ "don't let
# autotune/heuristics supply launch params on XPU" lesson. Fix: drop autotune,
# compute BLOCK_SIZE/num_warps in the Python wrapper (size-banded) and pass them
# explicitly. Kernel body is byte-for-byte identical to generic -> zero numeric
# change.
@triton.jit
def rot90_kernel_2d(
    in_ptr,
    out_ptr,
    n_elements,
    M,
    N,
    k_norm,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    m_minus_1 = M - 1
    n_minus_1 = N - 1

    if k_norm == 0:
        stride_0 = n_elements // M
        out_dim0 = offsets // stride_0
        remainder = offsets % stride_0
        out_dim1 = remainder % N

        in_dim0 = out_dim0
        in_dim1 = out_dim1

        stride_0_in = n_elements // M
        in_offset = in_dim0 * stride_0_in + in_dim1 * (stride_0_in // N)

    elif k_norm == 1:
        stride_0 = n_elements // N
        out_dim0 = offsets // stride_0
        remainder = offsets % stride_0
        out_dim1 = remainder % M

        in_dim0 = out_dim1
        in_dim1 = n_minus_1 - out_dim0

        stride_0_in = n_elements // M
        in_offset = in_dim0 * stride_0_in + in_dim1 * (stride_0_in // N)

    elif k_norm == 2:
        stride_0 = n_elements // M
        out_dim0 = offsets // stride_0
        remainder = offsets % stride_0
        out_dim1 = remainder % N

        in_dim0 = m_minus_1 - out_dim0
        in_dim1 = n_minus_1 - out_dim1

        stride_0_in = n_elements // M
        in_offset = in_dim0 * stride_0_in + in_dim1 * (stride_0_in // N)

    else:  # k_norm == 3
        stride_0 = n_elements // N
        out_dim0 = offsets // stride_0
        remainder = offsets % stride_0
        out_dim1 = remainder % M

        in_dim0 = m_minus_1 - out_dim1
        in_dim1 = out_dim0

        stride_0_in = n_elements // M
        in_offset = in_dim0 * stride_0_in + in_dim1 * (stride_0_in // N)

    x = tl.load(in_ptr + in_offset, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


def _launch_config(n_elements):
    # Size-banded BLOCK_SIZE / num_warps (mirrors the nvidia rot90 tune configs)
    # computed in Python and passed explicitly, so no autotune recompiles on XPU.
    if n_elements <= 4096:
        return 512, 2
    elif n_elements <= 65536:
        return 1024, 4
    elif n_elements <= 1048576:
        return 2048, 8
    else:
        return 4096, 16


def rot90_2d(inp, k, dims, out):
    """Handle the case when dims = [0, 1] using optimized Triton kernel."""
    M = inp.shape[dims[0]]
    N = inp.shape[dims[1]]
    n_elements = out.numel()
    if n_elements == 0:
        return

    k_norm = ((k % 4) + 4) % 4

    BLOCK_SIZE, num_warps = _launch_config(n_elements)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    with torch_device_fn.device(inp.device):
        rot90_kernel_2d[grid](
            inp,
            out,
            n_elements,
            M,
            N,
            k_norm,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )


def rot90(input, k=1, dims=[0, 1]):
    logger.debug("GEMS_KUNLUNXIN ROT90")
    x = input
    if not x.is_contiguous():
        x = x.contiguous()

    dim0, dim1 = dims[0], dims[1]
    M = x.shape[dim0]
    N = x.shape[dim1]

    k_norm = ((k % 4) + 4) % 4

    if k_norm == 0 or k_norm == 2:
        out_shape = list(x.shape)
    else:
        out_shape = list(x.shape)
        out_shape[dim0] = N
        out_shape[dim1] = M

    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)

    if dim0 == 0 and dim1 == 1:
        rot90_2d(x, k, dims, out)
    else:
        ndim = x.ndim

        perm = [dim0, dim1]
        for i in range(ndim):
            if i != dim0 and i != dim1:
                perm.append(i)

        inverse_perm = [0] * ndim
        inverse_perm[dim0] = 0
        inverse_perm[dim1] = 1
        idx = 2
        for i in range(ndim):
            if i != dim0 and i != dim1:
                inverse_perm[i] = idx
                idx += 1

        x_transposed = x.permute(perm)
        out_transposed = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        rot90_2d(x_transposed, k, [0, 1], out_transposed)
        out.copy_(out_transposed.permute(inverse_perm))

    return out
