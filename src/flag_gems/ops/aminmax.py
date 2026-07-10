import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max, get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def aminmax_kernel_1(
    inp,
    min_out,
    max_out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_fill = get_dtype_max(inp.type.element_ty)
    max_fill = get_dtype_min(inp.type.element_ty)
    min_val = tl.load(inp_ptrs, mask=mask, other=min_fill)
    max_val = tl.load(inp_ptrs, mask=mask, other=max_fill)

    min_val = tl.min(min_val)
    max_val = tl.max(max_val)

    min_ptr = min_out + pid
    max_ptr = max_out + pid
    tl.store(min_ptr, min_val)
    tl.store(max_ptr, max_val)


@libentry()
@triton.jit
def aminmax_kernel_2(
    min_inp, max_inp, min_out, max_out, mid_size, BLOCK_MID: tl.constexpr
):
    offset = tl.arange(0, BLOCK_MID)
    min_ptrs = min_inp + offset
    max_ptrs = max_inp + offset
    mask = offset < mid_size
    min_fill = get_dtype_max(min_inp.type.element_ty)
    max_fill = get_dtype_min(max_inp.type.element_ty)
    min_val = tl.load(min_ptrs, mask=mask, other=min_fill)
    max_val = tl.load(max_ptrs, mask=mask, other=max_fill)

    min_val = tl.min(min_val)
    max_val = tl.max(max_val)

    tl.store(min_out, min_val)
    tl.store(max_out, max_val)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def aminmax_dim_kernel_non_inner(
    inp,
    min_out,
    max_out,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # Split the input into (M, N, K) and reduce over N with a strided read,
    # avoiding the dim_compress copy. Computes the min and the max in one pass.
    dtype = inp.dtype.element_ty
    cdtype = tl.float32 if dtype is tl.bfloat16 or dtype is tl.float16 else dtype
    min_value = get_dtype_min(dtype)
    max_value = get_dtype_max(dtype)

    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        min_in = tl.load(inp + inp_offset, mask=mask, other=max_value).to(cdtype)
        max_in = tl.load(inp + inp_offset, mask=mask, other=min_value).to(cdtype)
        min_result = tl.min(min_in, axis=0, keep_dims=True)
        max_result = tl.max(max_in, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(min_out + out_offset, min_result, mask=k_offsets < K)
        tl.store(max_out + out_offset, max_result, mask=k_offsets < K)
    else:
        min_acc = tl.full([TILE_N, TILE_K], value=max_value, dtype=cdtype)
        max_acc = tl.full([TILE_N, TILE_K], value=min_value, dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            min_in = tl.load(inp + inp_offsets, mask=mask, other=max_value).to(cdtype)
            max_in = tl.load(inp + inp_offsets, mask=mask, other=min_value).to(cdtype)
            min_acc = tl.minimum(min_acc, min_in)
            max_acc = tl.maximum(max_acc, max_in)
        min_result = tl.min(min_acc, axis=0, keep_dims=True)
        max_result = tl.max(max_acc, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(min_out + out_offset, min_result, mask=k_offsets < K)
        tl.store(max_out + out_offset, max_result, mask=k_offsets < K)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def aminmax_dim_kernel_inner(
    inp,
    min_out,
    max_out,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    dtype = inp.dtype.element_ty
    cdtype = tl.float32 if dtype is tl.bfloat16 or dtype is tl.float16 else dtype
    min_value = get_dtype_min(dtype)
    max_value = get_dtype_max(dtype)

    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        inp_offset = pid_m * N + n_offsets
        mask = n_offsets < N
        min_in = tl.load(inp + inp_offset, mask=mask, other=max_value).to(cdtype)
        max_in = tl.load(inp + inp_offset, mask=mask, other=min_value).to(cdtype)
        min_result = tl.min(min_in, axis=0)
        max_result = tl.max(max_in, axis=0)
        tl.store(min_out + pid_m, min_result)
        tl.store(max_out + pid_m, max_result)
    else:
        min_acc = tl.full([TILE_N], value=max_value, dtype=cdtype)
        max_acc = tl.full([TILE_N], value=min_value, dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_offsets = pid_m * N + n_offsets
            mask = n_offsets < N
            min_in = tl.load(inp + inp_offsets, mask=mask, other=max_value).to(cdtype)
            max_in = tl.load(inp + inp_offsets, mask=mask, other=min_value).to(cdtype)
            min_acc = tl.minimum(min_acc, min_in)
            max_acc = tl.maximum(max_acc, max_in)
        min_result = tl.min(min_acc, axis=0)
        max_result = tl.max(max_acc, axis=0)
        tl.store(min_out + pid_m, min_result)
        tl.store(max_out + pid_m, max_result)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def aminmax_kernel(
    inp,
    min_out,
    max_out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)
    max_value = get_dtype_max(dtype)

    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    min_out = min_out + rows
    max_out = max_out + rows
    row_mask = rows < M

    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    _min = tl.full([BLOCK_M, BLOCK_N], value=max_value, dtype=acc_type)
    _max = tl.full([BLOCK_M, BLOCK_N], value=min_value, dtype=acc_type)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask & col_mask
        a = tl.load(inp + cols, mask=mask, other=min_value)
        _min = tl.where(mask, tl.minimum(_min, a), _min)
        _max = tl.where(mask, tl.maximum(_max, a), _max)
    min_result = tl.min(_min, axis=1)[:, None]
    max_result = tl.max(_max, axis=1)[:, None]
    tl.store(min_out, min_result, row_mask)
    tl.store(max_out, max_result, row_mask)


def aminmax(inp, dim=None, keepdim=False, *, out=None):
    logger.debug("GEMS AMINMAX")

    if dim is None:
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        min_mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        max_mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)

        if out is not None:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
            if not keepdim:
                min_out = min_out.squeeze()
                max_out = max_out.squeeze()
        else:
            if not keepdim:
                min_out = torch.empty([], dtype=dtype, device=inp.device)
                max_out = torch.empty([], dtype=dtype, device=inp.device)
            else:
                shape = [1] * inp.dim()
                min_out = torch.empty(shape, dtype=dtype, device=inp.device)
                max_out = torch.empty(shape, dtype=dtype, device=inp.device)

        with torch_device_fn.device(inp.device):
            aminmax_kernel_1[(mid_size, 1)](
                inp,
                min_mid,
                max_mid,
                M,
                block_size,
            )
            aminmax_kernel_2[(1, 1)](
                min_mid, max_mid, min_out, max_out, mid_size, block_mid
            )
        return min_out, max_out
    else:
        if isinstance(dim, int):
            dim = [dim]
        assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
        dtype = inp.dtype

        shape = list(inp.shape)
        dim = [d % inp.ndim for d in dim]

        if len(dim) == 1:
            # Single-dim reduction: split into (M, N, K) and reduce over N with
            # a strided kernel, avoiding the dim_compress copy. K == 1 means the
            # reduced dim is innermost (contiguous); K > 1 means it is a middle
            # or outer dim, where the copy used to dominate the runtime.
            d = dim[0]
            N = shape[d]
            if N == 0:
                raise IndexError(
                    f"aminmax: Expected reduction dim {d} to have non-zero size."
                )
            M = math.prod(shape[:d])
            K = math.prod(shape[d + 1 :])
            shape[d] = 1
            if out is not None:
                min_out = out[0] if isinstance(out, tuple) else out
                max_out = out[1] if isinstance(out, tuple) else out
            else:
                min_out = torch.empty(shape, dtype=dtype, device=inp.device)
                max_out = torch.empty(shape, dtype=dtype, device=inp.device)
            inp = inp.contiguous()
            with torch_device_fn.device(inp.device):
                if K == 1:
                    grid = lambda meta: (M, 1, 1)
                    aminmax_dim_kernel_inner[grid](inp, min_out, max_out, M, N)
                else:
                    grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                    aminmax_dim_kernel_non_inner[grid](inp, min_out, max_out, M, N, K)
            if not keepdim:
                min_out = min_out.squeeze(dim=d)
                max_out = max_out.squeeze(dim=d)
            return min_out, max_out

        inp = dim_compress(inp, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = inp.numel() // N

        if out is not None:
            min_out = out[0] if isinstance(out, tuple) else out
            max_out = out[1] if isinstance(out, tuple) else out
        else:
            min_out = torch.empty(shape, dtype=dtype, device=inp.device)
            max_out = torch.empty(shape, dtype=dtype, device=inp.device)

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(inp.device):
            aminmax_kernel[grid](inp, min_out, max_out, M, N)
        if not keepdim:
            min_out = min_out.squeeze(dim=dim)
            max_out = max_out.squeeze(dim=dim)
        return min_out, max_out
