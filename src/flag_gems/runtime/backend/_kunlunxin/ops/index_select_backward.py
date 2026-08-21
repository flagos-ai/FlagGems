import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.index_select_backward import (
    index_select_backward as generic_index_select_backward,
)
from flag_gems.utils import dim_compress, libentry

from .mm import mm

logger = logging.getLogger(__name__)

_MAX_ONE_HOT_ELEMENTS = 20_000_000

# One-hot build tile: each program covers R rows = 32 of the exact inner
# length. The inner length is auto-padded by the backend internally; the XPU
# lowering is only correct when the masked-out lane budget stays small
# (heavy-masked column chunks miscompile), so never chunk the inner
# dimension -- instead bound the inner length and fall back to the generic
# path for huge dims.
_MAX_ROW_TILE = 32
_MAX_ONE_HOT_INNER = 8192

# The one-hot gemm is the hot path for the large benchmark shapes. The
# general-mm wrapper picks BLOCK_K=256 for M,N > 512; the direct launch with
# BLOCK_K=128 measures ~1.3-1.4x faster on 4096^3-class tiles (2026-08-16
# XPU6 tile sweep), so launch the mm kernel directly for those shapes.
_DIRECT_MM_MIN = 1024
_DIRECT_MM_BM = 256
_DIRECT_MM_BN = 256
_DIRECT_MM_BK = 128
_DIRECT_MM_WARPS = 8


def _mm_large(a, b):
    """mm with the tuned tile for large K-multiples-of-128 shapes."""
    M, K = a.shape
    _, N = b.shape
    if (
        M < _DIRECT_MM_MIN
        or N < _DIRECT_MM_MIN
        or K % _DIRECT_MM_BK != 0
        or M % _DIRECT_MM_BM != 0
        or N % _DIRECT_MM_BN != 0
    ):
        return mm(a, b)
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    grid = (
        (M // _DIRECT_MM_BM) * (N // _DIRECT_MM_BN),
        1,
    )
    from .mm import mm_kernel

    mm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        dot_out_dtype=tl.float32,
        BLOCK_M=_DIRECT_MM_BM,
        BLOCK_N=_DIRECT_MM_BN,
        BLOCK_K=_DIRECT_MM_BK,
        GROUP_M=1,
        SPLIT_K=1,
        EVEN_K=True,
        num_warps=_DIRECT_MM_WARPS,
        num_stages=2,
    )
    return c

@libentry()
@triton.jit
def _make_one_hot_rows_kernel(
    out,
    index,
    index_len,
    dim_size_out,
    R: tl.constexpr,
    C: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # one-hot (index_len, dim_size_out); rows are the index positions.
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    cols = tl.arange(0, C)
    rmask = rows < index_len
    cmask = cols < dim_size_out
    idx = tl.load(index + rows, mask=rmask, other=-1)
    val = idx[:, None] == cols[None, :]
    offs = rows[:, None] * dim_size_out + cols[None, :]
    mask = rmask[:, None] & cmask[None, :]
    if IS_FP16:
        tl.store(out + offs, val.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out + offs, val.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out + offs, val.to(tl.float32), mask=mask)


@libentry()
@triton.jit
def _make_one_hot_cols_kernel(
    out,
    index,
    dim_size_out,
    index_len,
    R: tl.constexpr,
    C: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # Transposed one-hot (dim_size_out, index_len): out[n, i] = (index[i] == n).
    # Rows are the output buckets; columns are the index positions.
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    cols = tl.arange(0, C)
    rmask = rows < dim_size_out
    cmask = cols < index_len
    idx = tl.load(index + cols, mask=cmask, other=-1)
    val = rows[:, None] == idx[None, :]
    offs = rows[:, None] * index_len + cols[None, :]
    mask = rmask[:, None] & cmask[None, :]
    if IS_FP16:
        tl.store(out + offs, val.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out + offs, val.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out + offs, val.to(tl.float32), mask=mask)


def index_select_backward(grad, self_sizes, dim, index):
    logger.debug("GEMS_KUNLUNXIN INDEX_SELECT_BACKWARD")

    dim = dim % grad.ndim
    index_len = index.numel()
    dim_size_out = self_sizes[dim]
    one_hot_elements = index_len * dim_size_out

    if (
        index_len == 0
        or one_hot_elements > _MAX_ONE_HOT_ELEMENTS
        or index_len > _MAX_ONE_HOT_INNER
        or dim_size_out > _MAX_ONE_HOT_INNER
        or grad.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return generic_index_select_backward(grad, self_sizes, dim, index)

    index = index.to(torch.int64)
    orig_dtype = grad.dtype
    is_fp16 = orig_dtype == torch.float16
    is_bf16 = orig_dtype == torch.bfloat16
    r_tile = _MAX_ROW_TILE

    if dim == grad.ndim - 1:
        # out[..., k] = sum_i grad[..., i] * (index[i] == k)
        M = grad.numel() // index_len
        grad_flat = grad.reshape(M, index_len)
        one_hot = torch.empty(
            (index_len, dim_size_out), dtype=orig_dtype, device=grad.device
        )
        _make_one_hot_rows_kernel[(triton.cdiv(index_len, r_tile),)](
            one_hot,
            index,
            index_len,
            dim_size_out,
            R=r_tile,
            C=dim_size_out,
            IS_FP16=is_fp16,
            IS_BF16=is_bf16,
        )
        out = _mm_large(grad_flat, one_hot)
        return out.reshape(self_sizes)

    if dim == 0:
        # out[k, ...] = sum_i grad[i, ...] * (index[i] == k)
        M = grad.numel() // index_len
        grad_flat = grad.reshape(index_len, M)
        one_hot_t = torch.empty(
            (dim_size_out, index_len), dtype=orig_dtype, device=grad.device
        )
        _make_one_hot_cols_kernel[(triton.cdiv(dim_size_out, r_tile),)](
            one_hot_t,
            index,
            dim_size_out,
            index_len,
            R=r_tile,
            C=index_len,
            IS_FP16=is_fp16,
            IS_BF16=is_bf16,
        )
        out = _mm_large(one_hot_t, grad_flat)
        return out.reshape(self_sizes)

    # mid-dim: compressed (permute + one-hot + mm) path, exact but copies.
    grad_compressed = dim_compress(grad, dim)
    grad_flat = grad_compressed.reshape(-1, index_len)

    one_hot = torch.empty(
        (index_len, dim_size_out),
        dtype=orig_dtype,
        device=grad.device,
    )
    _make_one_hot_rows_kernel[(triton.cdiv(index_len, r_tile),)](
        one_hot,
        index,
        index_len,
        dim_size_out,
        R=r_tile,
        C=dim_size_out,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )
    out_flat = _mm_large(grad_flat, one_hot)

    compressed_shape = list(grad_compressed.shape)
    compressed_shape[-1] = dim_size_out
    out_flat = out_flat.reshape(compressed_shape)
    order = [i for i in range(out_flat.ndim - 1)]
    order.insert(dim, out_flat.ndim - 1)
    out = out_flat.permute(order).contiguous()
    return out