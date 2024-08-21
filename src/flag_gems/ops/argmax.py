import logging
import math

import torch
import triton
import triton.language as tl

from ..utils import libentry
from ..utils.shape_utils import can_use_int32_index


@libentry()
@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
    max_index = max_index + pid * BLOCK_SIZE
    mid_value_ptr = mid_value + pid
    max_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, max_val)
    tl.store(max_index_ptr, max_index)


@libentry()
@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid_value + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    index_val = tl.argmax(mid_val, axis=0)
    mid_index_ptrs = mid_index + index_val
    out_val = tl.load(mid_index_ptrs)
    tl.store(out, out_val)


def heur_block_n(args):
    return min(4096, triton.next_power_of_2(args["N"]))


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 16}, num_warps=8),
        triton.Config({"BLOCK_M": 32}, num_warps=8),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def argmax_kernel(
    inp,
    out_index,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    # set offset
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if INT64_INDEX:
        pid_m = pid_m.to(tl.int64)
        pid_k = pid_k.to(tl.int64)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    max_values = tl.full([BLOCK_M], dtype=tl.float32, value=float("-inf"))
    argmax_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
        local_max, local_argmax = tl.max(
            inp_vals, 1, return_indices=True, return_indices_tie_break_left=True
        )
        # if return indices is not supported, call a tl.argmax in addition
        # local_argmax = tl.argmax(inp_vals, 1)
        update = local_max > max_values
        max_values = tl.where(update, local_max, max_values)
        argmax_values = tl.where(update, start_n + local_argmax, argmax_values)

    offset_index = m_offset * K + pid_k
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_index_ptrs, argmax_values, mask=mask1)


def argmax(inp, dim=None, keepdim=False, *, dtype=None):
    logging.debug("GEMS ARGMAX")
    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        use_int64_index = not can_use_int32_index(inp)

        mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=torch.int64, device=inp.device)
        if keepdim:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=torch.int64, device=inp.device)
        else:
            out = torch.empty([], dtype=torch.int64, device=inp.device)

        with torch.cuda.device(inp.device):
            argmax_kernel_1[(mid_size, 1, 1)](
                inp,
                mid_value,
                mid_index,
                M,
                block_size,
                INT64_INDEX=use_int64_index,
            )
            argmax_kernel_2[(1, 1, 1)](mid_value, mid_index, out, mid_size, block_mid)
        return out
    else:
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        shape = inp.shape
        dim = dim % inp.ndim
        N = shape[dim]
        M = math.prod(shape[:dim])
        K = inp.numel() // M // N

        inp = inp.contiguous()
        use_int64_index = not can_use_int32_index(inp)

        shape_list = list(shape)
        shape_list[dim] = 1
        out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
        if not keepdim:
            out_index = torch.squeeze(out_index, dim)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        with torch.cuda.device(inp.device):
            argmax_kernel[grid](
                inp,
                out_index,
                M,
                N,
                K,
                INT64_INDEX=use_int64_index,
            )

        return out_index
