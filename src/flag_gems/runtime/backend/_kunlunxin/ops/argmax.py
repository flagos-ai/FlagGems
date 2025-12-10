import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))
torch_dtype_to_tl_dtype_and_min_value = {
    torch.int16: (tl.int16, torch.iinfo(torch.int16).min),
    torch.int32: (tl.int32, torch.iinfo(torch.int32).min),
    torch.float16: (tl.float16, torch.finfo(torch.float16).min),
    torch.float32: (tl.float32, torch.finfo(torch.float32).min),
    torch.bfloat16: (tl.float32, torch.finfo(torch.float32).min),
}


@libentry()
@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
    dtype_min_value: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=dtype_min_value)
    max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
    max_index = max_index + pid * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    max_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, max_val)
    tl.store(max_index_ptr, max_index)


@libentry()
@triton.jit
def argmax_kernel_2(
    mid_value,
    mid_index,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
    dtype_min_value: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=dtype_min_value)
    index_val = tl.argmax(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


def heur_block_n(args):
    return min(4096, triton.next_power_of_2(args["N"]))


@libentry()
@triton.heuristics(runtime.get_heuristic_config("argmax"))
@triton.jit
def argmax_kernel(
    inp,
    out_index,
    M,
    N,
    K,
    tl_dtype: tl.constexpr,
    dtype_min_value: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    if tl_dtype is tl.int16:
        tl_dtype = tl.int32
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    offset_index = m_offset * K + pid_k
    # set mask
    mask1 = m_offset < M
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=dtype_min_value)
    _, result_index = tl.max(inp_vals, axis=1, return_indices=True)

    out_index_ptrs = out_index + offset_index

    tl.store(out_index_ptrs, result_index, mask=mask1)


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS ARGMAX")
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)

        mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        tl_dtype, dtype_min_value = torch_dtype_to_tl_dtype_and_min_value[inp.dtype]
        with torch_device_fn.device(inp.device):
            argmax_kernel_1[(mid_size, 1, 1)](
                inp,
                mid_value,
                mid_index,
                M,
                block_size,
                dtype_min_value,
            )
            argmax_kernel_2[(1, 1, 1)](
                mid_value,
                mid_index,
                out,
                mid_size,
                block_mid,
                dtype_min_value,
            )
        return out
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        shape = inp.shape
        dim = dim % inp.ndim
        if inp.numel() == 0:
            out_shape = list(shape)
            if keepdim:
                out_shape[dim] = 1
            else:
                del out_shape[dim]
            return torch.zeros(out_shape, dtype=torch.int64, device=inp.device)
        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N

        inp = inp.contiguous()

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)

        tl_dtype, dtype_min_value = torch_dtype_to_tl_dtype_and_min_value[inp.dtype]

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )

        with torch_device_fn.device(inp.device):
            argmax_kernel[grid](
                inp,
                out_index,
                M,
                N,
                K,
                tl_dtype,
                dtype_min_value,
            )

        return out_index
