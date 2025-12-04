import builtins
import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Favor wider column tiles for long rows and more rows per block for tall shapes.
ARGMIN_REDUCTION_CONFIGS = [
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
]


def _prune_reduction_configs(configs, nargs, **meta):
    n = meta.get("N", nargs["N"])
    if n <= 128:
        max_block_n = 128
    elif n <= 2048:
        max_block_n = 256
    else:
        max_block_n = 512
    return [cfg for cfg in configs if cfg.kwargs["BLOCK_N"] <= max_block_n]


def _flatten_dim(shape, dim):
    dim = dim % len(shape)
    n = shape[dim]
    inner = math.prod(shape[dim + 1 :]) if dim + 1 < len(shape) else 1
    outer = math.prod(shape[:dim]) if dim > 0 else 1
    return dim, n, inner, outer


@libentry()
@triton.jit
def argmin_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M

    max_value = get_dtype_max(inp.type.element_ty)
    inp_val = tl.load(inp + offset, mask=mask, other=max_value, cache_modifier=".cg")
    min_val, min_index = tl.min(
        inp_val, axis=0, return_indices=True, return_indices_tie_break_left=True
    )
    tl.store(mid_value + pid, min_val)
    tl.store(mid_index + pid, min_index + pid * BLOCK_SIZE)


@libentry()
@triton.jit
def argmin_kernel_2(
    mid_value,
    mid_index,
    out,
    mid_size,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < mid_size
    max_value = get_dtype_max(mid_value.type.element_ty)
    mid_val = tl.load(mid_value + offset, mask=mask, other=max_value)
    _, index_val = tl.min(
        mid_val,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    out_val = tl.load(mid_index + index_val)
    tl.store(out, out_val)


@libentry()
@triton.autotune(
    configs=ARGMIN_REDUCTION_CONFIGS,
    key=["M", "N"],
    warmup=8,
    rep=40,
    prune_configs_by={"early_config_prune": _prune_reduction_configs},
)
@triton.jit
def argmin_kernel(
    inp,
    out_index,
    M,
    N,
    INNER,
    STRIDE_OUTER,
    STRIDE_REDUCE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = rows.to(tl.int64)
    row_mask = rows < M

    outer_idx = rows // INNER
    inner_idx = rows % INNER
    base_ptr = inp + outer_idx * STRIDE_OUTER + inner_idx

    dtype = inp.type.element_ty
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_value = get_dtype_max(dtype)
    min_values = tl.full([BLOCK_M], dtype=acc_type, value=max_value)
    argmin_values = tl.full([BLOCK_M], dtype=tl.int32, value=0)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        n_offset = n_offset.to(tl.int64)
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        inp_ptrs = base_ptr[:, None] + n_offset[None, :] * STRIDE_REDUCE
        inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value, cache_modifier=".cg")
        local_min, local_argmin = tl.min(
            inp_vals,
            1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        local_argmin = local_argmin.to(tl.int32)
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(
            update, (start_n + local_argmin).to(tl.int32), argmin_values
        )

    out_index_ptrs = out_index + rows
    tl.store(out_index_ptrs, argmin_values, mask=row_mask)


def argmin(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_MTHREADS ARGMIN")
    if not inp.is_contiguous():
        inp = inp.contiguous()

    if dim is None:
        M = inp.numel()
        if dtype is None:
            dtype = inp.dtype
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        block_size = builtins.min(block_size * 4, 4096, triton.next_power_of_2(M))
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

        num_warps_block = builtins.min(8, max(1, block_size // 128))
        num_warps_mid = builtins.min(8, max(1, block_mid // 128))

        with torch_device_fn.device(inp.device):
            argmin_kernel_1[(mid_size, 1, 1)](
                inp,
                mid_value,
                mid_index,
                M,
                block_size,
                num_warps=num_warps_block,
                num_stages=2,
            )
            argmin_kernel_2[(1, 1, 1)](
                mid_value,
                mid_index,
                out,
                mid_size,
                block_mid,
                num_warps=num_warps_mid,
                num_stages=2,
            )
        return out

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim

    shape = list(inp.shape)
    dim, N, inner, outer = _flatten_dim(shape, dim)
    M = outer * inner
    stride = inp.stride()
    stride_reduce = stride[dim]
    stride_outer = stride_reduce * N

    out_index = torch.empty((M,), dtype=torch.int32, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        argmin_kernel[grid](
            inp,
            out_index,
            M,
            N,
            max(inner, 1),
            stride_outer,
            stride_reduce,
        )

    out_shape = shape.copy()
    out_shape[dim] = 1
    out_index = out_index.view(out_shape).to(torch.int64)
    if not keepdim:
        out_index = torch.squeeze(out_index, dim)
    return out_index


__all__ = ["argmin"]
