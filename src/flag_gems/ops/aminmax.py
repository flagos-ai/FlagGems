import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def aminmax_kernel_1(
    inp,
    mid_min,
    mid_max,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M

    dtype = inp.type.element_ty
    min_init = get_dtype_max(dtype)
    max_init = get_dtype_min(dtype)

    inp_val = tl.load(inp_ptrs, mask=mask, other=min_init)
    min_val = tl.min(inp_val)
    # Reload with different other value for max
    inp_val_max = tl.load(inp_ptrs, mask=mask, other=max_init)
    max_val = tl.max(inp_val_max)

    mid_min_ptr = mid_min + pid
    mid_max_ptr = mid_max + pid
    tl.store(mid_min_ptr, min_val)
    tl.store(mid_max_ptr, max_val)


@libentry()
@triton.jit
def aminmax_kernel_2(
    mid_min,
    mid_max,
    out_min,
    out_max,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size

    dtype = mid_min.type.element_ty
    min_init = get_dtype_max(dtype)
    max_init = get_dtype_min(dtype)

    mid_min_ptrs = mid_min + offset
    mid_max_ptrs = mid_max + offset

    mid_min_val = tl.load(mid_min_ptrs, mask=mask, other=min_init)
    mid_max_val = tl.load(mid_max_ptrs, mask=mask, other=max_init)

    min_val = tl.min(mid_min_val)
    max_val = tl.max(mid_max_val)

    tl.store(out_min, min_val)
    tl.store(out_max, max_val)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def aminmax_kernel(
    inp,
    out_min,
    out_max,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.type.element_ty
    min_init = get_dtype_max(dtype)
    max_init = get_dtype_min(dtype)

    pid = tle.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out_min = out_min + rows
    out_max = out_max + rows
    row_mask = rows < M

    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    all_min = tl.full([BLOCK_M, BLOCK_N], value=min_init, dtype=acc_type)
    all_max = tl.full([BLOCK_M, BLOCK_N], value=max_init, dtype=acc_type)

    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask

        a = tl.load(inp + cols, mask, other=min_init)
        all_min = tl.minimum(all_min, a)

        a_max = tl.load(inp + cols, mask, other=max_init)
        all_max = tl.maximum(all_max, a_max)

    min_result = tl.min(all_min, axis=1)[:, None]
    max_result = tl.max(all_max, axis=1)[:, None]

    tl.store(out_min, min_result, row_mask)
    tl.store(out_max, max_result, row_mask)


def aminmax(inp, dim=None, keepdim=False):
    logger.debug("GEMS AMINMAX")
    Aminmax_out = namedtuple("aminmax", ["min", "max"])

    if dim is None:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype

        mid_min = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_max = torch.empty((mid_size,), dtype=dtype, device=inp.device)

        if not keepdim:
            out_min = torch.empty([], dtype=dtype, device=inp.device)
            out_max = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(inp.dim()):
                shape[i] = 1
            out_min = torch.empty(shape, dtype=dtype, device=inp.device)
            out_max = torch.empty(shape, dtype=dtype, device=inp.device)

        with torch_device_fn.device(inp.device):
            aminmax_kernel_1[(mid_size, 1)](
                inp,
                mid_min,
                mid_max,
                M,
                block_size,
            )
            aminmax_kernel_2[(1, 1)](
                mid_min,
                mid_max,
                out_min,
                out_max,
                mid_size,
                block_mid,
            )

        return Aminmax_out(min=out_min, max=out_max)
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = dim % inp.ndim
        inp = dim_compress(inp, dim)
        N = shape[dim]
        shape[dim] = 1
        M = inp.numel() // N

        out_min = torch.empty(shape, dtype=dtype, device=inp.device)
        out_max = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            aminmax_kernel[grid](inp, out_min, out_max, M, N)

        if not keepdim:
            out_min = out_min.squeeze(dim=dim)
            out_max = out_max.squeeze(dim=dim)

        return Aminmax_out(min=out_min, max=out_max)
