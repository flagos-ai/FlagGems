import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max

from ..utils.block_size_utils import get_block_size_1d

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@libentry()
@triton.jit
def min_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    max_value = get_dtype_max(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=max_value)
    min_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, min_val)


@libentry()
@triton.jit
def min_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    min_val = tl.min(mid_val)
    tl.store(out, min_val)


def heur_m_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num


def heur_n_block_size(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["N"]), 8192)


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("min"),
#     key=[
#         "M",
#         "N",
#     ],
# )
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def min_kernel(
    inp,
    out_value,
    out_index,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    # you just cannot create a function that return a tl.dtype in triton lang
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_value = get_dtype_max(dtype)
    min_values = tl.full([BLOCK_M], dtype=acc_type, value=max_value)
    argmin_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value)
        local_min, local_argmin = tl.min(inp_vals, 1, return_indices=True)
        # if return indices is not supported, call a tl.argmax in addition
        # local_argmin = tl.argmin(inp_vals, 1)
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(update, start_n + local_argmin, argmin_values)

    offset_index = m_offset * K + pid_k
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_value_ptrs, min_values, mask=mask1)
    tl.store(out_index_ptrs, argmin_values, mask=mask1)


def min(inp):
    logger.debug("GEMS MIN")
    M = inp.numel()
    # block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
    block_size = get_block_size_1d(M, inp.element_size())
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)

    dtype = inp.dtype
    mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
    out = torch.empty([], dtype=dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        min_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size, buffer_size_limit=2048)
        if mid_size == 1:
            return mid.reshape([])

        import os

        os.environ["TRITONXPU_OTHER_SIM"] = "1"
        os.environ["TRITONXPU_STORE_MASK_SIM"] = "1"

        min_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid, buffer_size_limit=2048)

        if "TRITONXPU_OTHER_SIM" in os.environ:
            del os.environ["TRITONXPU_OTHER_SIM"]
        if "TRITONXPU_STORE_MASK_SIM" in os.environ:
            del os.environ["TRITONXPU_STORE_MASK_SIM"]
    return out


def min_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS MIN DIM")
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
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    isCloseCoreTiling = False
    if inp.dtype in [torch.int16, torch.int32, torch.int64] and M == 4096 and N == 256:
        isCloseCoreTiling = True
    with torch_device_fn.device(inp.device):
        min_kernel[grid](
            inp, out_value, out_index, M, N, K, isCloseCoreTiling=isCloseCoreTiling
        )
    Min_out = namedtuple("min", ["values", "indices"])
    out = Min_out(values=out_value, indices=out_index)
    return out
