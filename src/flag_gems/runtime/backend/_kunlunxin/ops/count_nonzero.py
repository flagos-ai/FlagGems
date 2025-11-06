import logging
import os

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@triton.jit
def count_nonzero_kernel_1(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int64)
    nonzero_count = tl.sum(is_nonzero, axis=0)
    tl.atomic_add(out_ptr, nonzero_count)


"""***************************** TROTITON XPU KERNEL *****************************"""


@libentry()
@triton.jit
def count_nonzero_kernel_1_part0_xpu(x_ptr, out_ptr, numel, BLOCK_SIZE_0: tl.constexpr):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE_0
    offsets = block_start + tl.arange(0, BLOCK_SIZE_0)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int64)
    nonzero_count = tl.sum(is_nonzero, axis=0)
    tl.store(out_ptr + pid, nonzero_count)


@triton.jit
def count_nonzero_kernel_1_part1_xpu(x_ptr, out_ptr, numel, BLOCK_SIZE_1: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE_1)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    nonzero_count = tl.sum(x, axis=0)
    tl.store(out_ptr, nonzero_count)


"""***************************** TROTITON XPU KERNEL *****************************"""


def heur_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["numel"], 12))


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("count_nonzero"), key=["numel"])
@triton.heuristics(
    {
        "BLOCK_SIZE": heur_block_size,
    }
)
@triton.jit
def count_nonzero_kernel(x_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr):
    pid_x = tle.program_id(0)

    nonzero_count = tl.full((), value=0, dtype=out_ptr.dtype.element_ty)
    for start_n in range(0, N, BLOCK_SIZE):
        cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        offset = pid_x * N + cols_offsets
        mask = offset < numel and cols_offsets < N
        x = tl.load(x_ptr + offset, mask=mask, other=0)
        is_nonzero = (x != 0).to(tl.int64)
        nonzero_count += tl.sum(is_nonzero)

    tl.store(out_ptr + pid_x, nonzero_count)


"""***************************** TROTITON XPU KERNEL *****************************"""


@triton.jit
def count_nonzero_kernel_xpu(
    x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_x = tl.program_id(0)
    row = pid_x * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    row_mask = row < M

    _nonzero_count = tl.zeros([BLOCK_M, BLOCK_N], dtype=out_ptr.dtype.element_ty)
    for start_n in range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        x = tl.load(x_ptr + row * N + cols, mask=mask, other=0)
        is_nonzero = (x != 0).to(tl.int64)
        _nonzero_count += is_nonzero

    nonzero_count = tl.sum(_nonzero_count, axis=1)[:, None]
    tl.store(out_ptr + row, nonzero_count, row_mask)


"""***************************** TROTITON XPU KERNEL *****************************"""


@libentry()
# @triton.autotune(configs=runtime.get_tuned_config("count_nonzero"), key=["numel"])
@triton.heuristics(
    {
        "BLOCK_SIZE": heur_block_size,
    }
)
@triton.jit
def count_nonzero_combin_kernel_1(x_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr):
    pid_x = tle.program_id(0)
    nonzero_count = tl.full((), value=0, dtype=out_ptr.dtype.element_ty)
    for start_n in range(0, N, BLOCK_SIZE):
        cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        offset = pid_x * N + cols_offsets
        mask = offset < numel and cols_offsets < N
        x = tl.load(x_ptr + offset, mask=mask, other=0)
        nonzero_count += tl.sum(x)
    tl.store(out_ptr + pid_x, nonzero_count)


@libentry()
@triton.jit
def count_nonzero_combin_kernel(
    x_ptr,
    combin_ptr,
    N: tl.constexpr,
    combin_N: tl.constexpr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x = tle.program_id(0)
    pid_y = tle.program_id(1)
    cols_offsets = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset = pid_x * N + cols_offsets
    mask = offset < numel and cols_offsets < N
    x = tl.load(x_ptr + offset, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int64)
    nonzero_count = tl.sum(is_nonzero)
    tl.store(combin_ptr + pid_x * combin_N + pid_y, nonzero_count)


def count_nonzero(x, dim=None):
    logger.debug("GEMS COUNT NONZERO")

    CORE_NUM = 64
    SIZE_PER_CORE = 512
    SIZE_PER_CLUSTER = CORE_NUM * SIZE_PER_CORE

    elem_bytes = x.element_size()
    if dim is not None:
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        shape = x.shape
        numel = x.numel()
        # premute
        os.environ["TRITONXPU_IS_SCATTER_SLICE"] = "1"
        x = dim_compress(x, dim)
        x = x.contiguous().flatten()
        del os.environ["TRITONXPU_IS_SCATTER_SLICE"]
        # 2D count_nonzero
        out_shape = list(shape)
        del out_shape[dim]
        os.environ["TRITONXPU_ELEMBYTES"] = "8"
        out = torch.zeros(out_shape, dtype=torch.int64, device=x.device)
        del os.environ["TRITONXPU_ELEMBYTES"]
        N = shape[dim]
        M = triton.cdiv(numel, shape[dim])
        BLOCK_M = CORE_NUM
        BLOCK_N = SIZE_PER_CORE
        grid = lambda meta: (triton.cdiv(M, BLOCK_M),)
        os.environ["TRITONXPU_ELEMBYTES"] = "8"
        count_nonzero_kernel_xpu[grid](
            x,
            out,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            groups_per_cluster=CORE_NUM,
            buffer_size_limit=SIZE_PER_CORE * 8,
            is_use_mask_zero=True,
        )
        del os.environ["TRITONXPU_ELEMBYTES"]
        return out
    else:
        # 1D count_nonzero
        x = x.contiguous().flatten()
        numel = x.numel()
        gridX = triton.cdiv(numel, SIZE_PER_CLUSTER)
        os.environ["TRITONXPU_ELEMBYTES"] = "8"
        out_mid = torch.zeros(gridX, dtype=torch.int64, device=x.device)
        del os.environ["TRITONXPU_ELEMBYTES"]
        count_nonzero_kernel_1_part0_xpu[(gridX,)](
            x,
            out_mid,
            numel,
            BLOCK_SIZE_0=SIZE_PER_CLUSTER,
            buffer_size_limit=SIZE_PER_CORE * elem_bytes,
            is_use_mask_zero=True,
        )
        BLOCK_SIZE_1 = triton.next_power_of_2(gridX)
        os.environ["TRITONXPU_ELEMBYTES"] = "8"
        out = torch.zeros(1, dtype=torch.int64, device=x.device)
        count_nonzero_kernel_1_part1_xpu[(1,)](
            out_mid,
            out,
            gridX,
            BLOCK_SIZE_1=BLOCK_SIZE_1,
            buffer_size_limit=SIZE_PER_CORE * 8,
            is_use_mask_zero=True,
        )
        del os.environ["TRITONXPU_ELEMBYTES"]

        return out[0]
