import logging
import math
from functools import reduce

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_min

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def amax_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    min_value = get_dtype_min(inp.type.element_ty)
    inp_val = tl.load(inp_ptrs, mask=mask, other=min_value)
    amax_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, amax_val)


@libentry()
@triton.jit
def amax_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    min_value = get_dtype_min(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=min_value)
    amax_val = tl.max(mid_val)
    tl.store(out, amax_val)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def amax_dim_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    dtype = input_ptr.dtype.element_ty
    cdtype = tl.float32 if dtype is tl.bfloat16 or dtype is tl.float16 else dtype
    min_value = get_dtype_min(dtype)

    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)[None, :]

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)[:, None]
        inp_offset = pid_m * N * K + n_offsets * K + k_offsets
        mask = (n_offsets < N) & (k_offsets < K)
        inp = tl.load(input_ptr + inp_offset, mask=mask, other=min_value).to(cdtype)
        out = tl.max(inp, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)
    else:
        acc = tl.full([TILE_N, TILE_K], value=min_value, dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)[:, None]
            inp_offsets = pid_m * N * K + n_offsets * K + k_offsets
            mask = (n_offsets < N) & (k_offsets < K)
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=min_value).to(
                cdtype
            )
            acc = tl.maximum(acc, inp)
        out = tl.max(acc, axis=0, keep_dims=True)
        out_offset = pid_m * K + k_offsets
        tl.store(output_ptr + out_offset, out, mask=k_offsets < K)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def amax_dim_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    dtype = input_ptr.dtype.element_ty
    cdtype = tl.float32 if dtype is tl.bfloat16 or dtype is tl.float16 else dtype
    min_value = get_dtype_min(dtype)

    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        inp_offset = pid_m * N + n_offsets
        mask = n_offsets < N
        inp = tl.load(input_ptr + inp_offset, mask=mask, other=min_value).to(cdtype)
        out = tl.max(inp, axis=0)
        tl.store(output_ptr + pid_m, out)
    else:
        acc = tl.full([TILE_N], value=min_value, dtype=cdtype)
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp_offsets = pid_m * N + n_offsets
            mask = n_offsets < N
            inp = tl.load(input_ptr + inp_offsets, mask=mask, other=min_value).to(
                cdtype
            )
            acc = tl.maximum(acc, inp)
        out = tl.max(acc, axis=0)
        tl.store(output_ptr + pid_m, out)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def amax_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)

    # Map the program id to the row of inp it should compute.
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    _all = tl.full([BLOCK_M, BLOCK_N], value=min_value, dtype=acc_type)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        a = tl.load(inp + cols, mask, other=min_value)
        _all = tl.maximum(_all, a)
    all = tl.max(_all, axis=1)[:, None]
    tl.store(out, all, row_mask)


def amax(inp, dim=None, keepdim=False):
    logger.debug("GEMS AMAX")
    if dim is None or (hasattr(dim, "__len__") and len(dim) == 0):
        M = inp.numel()
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        if not keepdim:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            amax_kernel_1[(mid_size, 1)](
                inp,
                mid,
                M,
                block_size,
            )
            amax_kernel_2[(1, 1)](
                mid, out, mid_size, block_mid
            )  # max block size is 128k, so mid does not requires int64 index
        return out

    if isinstance(dim, int):
        dim = [dim]
    assert ((i >= -inp.ndim and i < inp.ndim) for i in dim), "Invalid dim"
    dtype = inp.dtype
    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]

    if len(dim) == 1:
        # Single-dim reduction: split into (M, N, K) and reduce over N with a
        # strided kernel, avoiding the dim_compress copy. K == 1 means the
        # reduced dim is innermost (contiguous); K > 1 means it is a middle or
        # outer dim, where the copy used to dominate the runtime.
        d = dim[0]
        N = inp.shape[d]
        M = reduce(lambda x, y: x * y, shape[:d], 1)
        inp = inp.contiguous()
        K = inp.numel() // M // N
        shape[d] = 1
        out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            if K > 1:
                grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
                amax_dim_kernel_non_inner[grid](out, inp, M, N, K)
            else:
                grid = (M, 1, 1)
                amax_dim_kernel_inner[grid](out, inp, M, N)
        if not keepdim:
            out = out.squeeze(dim=d)
        return out

    # Multi-dim reduction keeps the original dim_compress path.
    inp = dim_compress(inp, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N
    out = torch.empty(shape, dtype=dtype, device=inp.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        amax_kernel[grid](inp, out, M, N)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out
