import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_min


@libentry()
@triton.jit
def max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)


@libentry()
@triton.jit
def max_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("max"),
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    values={
        "BLOCK_M": lambda args: 32,  # Reduced from default to save memory
        "BLOCK_N": lambda args: 64,  # Reduced from default to save memory
    }
)
@triton.jit
def max_kernel_value_only(
    inp,
    out_value,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Only compute max values, not indices.
    This avoids the unsupported linalg.reduce with multiple outputs.
    Uses smaller block sizes to avoid TPU local memory overflow.
    """
    # set offset
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    acc_type = tl.float32 if (dtype is tl.bfloat16 or dtype is tl.float16) else dtype
    min_value = get_dtype_min(dtype)
    result_value = tl.full([BLOCK_M], value=min_value, dtype=acc_type)

    for i in range(0, N, BLOCK_N):
        n_offset = i + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        # set mask
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)

        # Only compute max value, no indices
        max_value = tl.max(inp_vals, axis=1)

        # Update result
        result_value = tl.maximum(result_value, max_value)

    mask1 = m_offset < M
    offset_index = m_offset * K + pid_k
    out_value_ptrs = out_value + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)


def max(inp):
    logging.debug("GEMS MAX")
    inp = inp.contiguous()

    # Use dimension-wise reduction instead of global reduction
    # to avoid TPU emulator bug with axis=0 in global kernels
    result = inp
    for dim in range(inp.ndim):
        # Always reduce the first dimension (0) since shape changes after each reduction
        result = max_dim(result, dim=0, keepdim=False).values

    return result


def max_dim(inp, dim=None, keepdim=False):
    logging.debug("GEMS MAX DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch_device_fn.device(inp.device):
        # Use value-only kernel to avoid multi-output linalg.reduce issue
        max_kernel_value_only[grid](inp, out_value, M, N, K)

    # Compute indices on CPU to avoid TPU memory issues with expand().contiguous()
    # Move tensors to CPU for index computation
    inp_cpu = inp.cpu()
    out_value_cpu = out_value.cpu()
    expanded_values_cpu = out_value_cpu.expand_as(inp_cpu)
    mask_cpu = inp_cpu == expanded_values_cpu
    out_index_cpu = mask_cpu.to(torch.int32).argmax(dim=dim, keepdim=True)
    # Move indices back to TPU
    out_index = out_index_cpu.to(inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    Max_out = namedtuple("max", ["values", "indices"])
    out = Max_out(values=out_value, indices=out_index)
    return out
