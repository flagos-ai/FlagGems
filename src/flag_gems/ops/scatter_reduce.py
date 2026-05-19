"""Out-of-place `torch.scatter_reduce` for FlagGems.

Design notes (different from inplace `scatter_reduce_` codegen):
  * Standalone Triton kernels: a sparse identity prefill + a unified reduction.
  * fp16 / bf16 always accumulate in a fp32 buffer, then cast back at the end.
    This kills the `llvm.fcmp got i16` bf16 compile error in the inplace path
    and lets `atomic_add` / `atomic_max` / `atomic_min` use hardware ops
    instead of CAS-loop emulation, which dominates fp16 latency.
  * `include_self=False` writes identity ONLY at positions selected by `index`
    (per PyTorch spec), not the whole tensor.
  * Mean fuses count accumulation into the main kernel (single launch).
  * Fast path: when `index.numel()` is below a threshold we fall back to torch
    to amortize Triton launch overhead.
"""
import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping

logger = logging.getLogger(__name__)

VALID_REDUCE = ("sum", "prod", "mean", "amax", "amin")

R_SUM = 0
R_PROD = 1
R_MEAN = 2
R_AMAX = 3
R_AMIN = 4

_REDUCE_CODE = {
    "sum": R_SUM,
    "prod": R_PROD,
    "mean": R_MEAN,
    "amax": R_AMAX,
    "amin": R_AMIN,
}

_MAX_RANK = 8


def _identity_value(reduce: str, dtype: torch.dtype):
    if reduce in ("sum", "mean"):
        return 0
    if reduce == "prod":
        return 1
    if reduce == "amax":
        return float("-inf") if dtype.is_floating_point else torch.iinfo(dtype).min
    if reduce == "amin":
        return float("inf") if dtype.is_floating_point else torch.iinfo(dtype).max
    raise ValueError(reduce)


def _pack_shape_strides(t: torch.Tensor, rank_padded: int):
    """Pad shape/strides up to _MAX_RANK with (size=1, stride=0) so we can pass
    a fixed-arity arg list to the Triton kernel."""
    shape = list(t.shape) + [1] * (rank_padded - t.ndim)
    stride = list(t.stride()) + [0] * (rank_padded - t.ndim)
    return shape, stride


@libentry()
@triton.jit
def _flat_copy_kernel(src_ptr, dst_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


@libentry()
@triton.jit
def _flat_cast_kernel(
    src_ptr, dst_ptr, N, DST_IS_FP32: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(src_ptr + offs, mask=mask)
    if DST_IS_FP32:
        v = v.to(tl.float32)
    tl.store(dst_ptr + offs, v, mask=mask)


def _fast_copy_into(dst: torch.Tensor, src: torch.Tensor) -> None:
    if dst.is_contiguous() and src.is_contiguous() and dst.dtype == src.dtype:
        n = dst.numel()
        BLOCK = 2048
        grid = (triton.cdiv(n, BLOCK),)
        _flat_copy_kernel[grid](src, dst, n, BLOCK=BLOCK)
    else:
        dst.copy_(src)


def _fast_copy_into_with_cast(dst: torch.Tensor, src: torch.Tensor) -> None:
    if dst.is_contiguous() and src.is_contiguous():
        n = dst.numel()
        BLOCK = 2048
        grid = (triton.cdiv(n, BLOCK),)
        _flat_cast_kernel[grid](
            src, dst, n, DST_IS_FP32=(dst.dtype == torch.float32), BLOCK=BLOCK
        )
    else:
        dst.copy_(src)


@libentry()
@triton.jit
def _fastpath_lastdim_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    count_ptr,
    N,
    D,
    REDUCE: tl.constexpr,
    UPCAST: tl.constexpr,
    NEED_COUNT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    src_val = tl.load(src_ptr + offs, mask=mask, other=0)
    idx_val = tl.load(index_ptr + offs, mask=mask, other=0)
    if UPCAST:
        src_val = src_val.to(tl.float32)

    row = offs // D
    target = row * D + idx_val

    if REDUCE == 0 or REDUCE == 2:
        tl.atomic_add(out_ptr + target, src_val, mask=mask, sem="relaxed")
        if NEED_COUNT:
            one = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(count_ptr + target, one, mask=mask, sem="relaxed")
    elif REDUCE == 3:
        tl.atomic_max(out_ptr + target, src_val, mask=mask, sem="relaxed")
    elif REDUCE == 4:
        tl.atomic_min(out_ptr + target, src_val, mask=mask, sem="relaxed")
    else:
        stop = ~mask
        all_done = False
        while not all_done:
            cur_v = tl.load(out_ptr + target, mask=mask, other=0)
            new_v = tl.where(stop, cur_v, cur_v * src_val)
            cas = tl.atomic_cas(out_ptr + target, cur_v, new_v)
            stop |= cur_v == cas
            all_done = tl.sum(stop.to(tl.int32)) == BLOCK


@libentry()
@triton.jit
def _fastpath_div_kernel(buf_ptr, count_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(buf_ptr + offs, mask=mask)
    c = tl.load(count_ptr + offs, mask=mask, other=1).to(tl.float32)
    c = tl.where(c < 1, 1.0, c)
    tl.store(buf_ptr + offs, v / c, mask=mask)


@libentry()
@triton.jit
def _fastpath_fill_one_i32_kernel(ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    one = tl.full((BLOCK,), 1, dtype=tl.int32)
    tl.store(ptr + offs, one, mask=mask)


def _try_fastpath(self_t, dim, index, src, reduce, include_self, out):
    same_shape = self_t.shape == index.shape == src.shape
    last_dim = (dim == self_t.dim() - 1) or (dim == -1)
    all_contig = (
        self_t.is_contiguous() and index.is_contiguous() and src.is_contiguous()
    )
    if not (same_shape and last_dim and all_contig and self_t.dim() >= 1):
        return None
    if not include_self:
        return None

    N = self_t.numel()
    if N == 0:
        return None
    D = self_t.shape[-1]
    use_upcast = self_t.dtype in (torch.float16, torch.bfloat16)
    need_count = reduce == "mean"

    if use_upcast:
        buf = torch.empty(self_t.shape, dtype=torch.float32, device=self_t.device)
        _fast_copy_into_with_cast(buf, self_t)
    else:
        if out is None:
            buf = torch.empty_like(self_t)
        else:
            buf = out
        _fast_copy_into(buf, self_t)

    if N < 65536:
        BLOCK = 256
        NW = 4
    elif N < 1048576:
        BLOCK = 1024
        NW = 4
    else:
        BLOCK = 2048
        NW = 8
    grid = (triton.cdiv(N, BLOCK),)

    if need_count:
        count = torch.empty(self_t.shape, dtype=torch.int32, device=self_t.device)
        _fastpath_fill_one_i32_kernel[grid](count, N, BLOCK=BLOCK, num_warps=NW)
    else:
        count = buf

    _fastpath_lastdim_kernel[grid](
        src,
        index,
        buf,
        count,
        N,
        D,
        REDUCE=_REDUCE_CODE[reduce],
        UPCAST=use_upcast,
        NEED_COUNT=need_count,
        BLOCK=BLOCK,
        num_warps=NW,
    )

    if need_count:
        _fastpath_div_kernel[grid](buf, count, N, BLOCK=BLOCK, num_warps=NW)

    if use_upcast:
        if out is None:
            out = torch.empty_like(self_t)
        _fast_copy_into_with_cast(out, buf)
        return out
    return buf


@libentry()
@triton.jit
def _selective_fill_kernel(
    out_ptr,
    index_ptr,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    sh6,
    sh7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    is0,
    is1,
    is2,
    is3,
    is4,
    is5,
    is6,
    is7,
    dim_stride,
    N,
    IDENTITY: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    cur = offs
    out_off = tl.zeros((BLOCK,), dtype=tl.int64)
    idx_off = tl.zeros((BLOCK,), dtype=tl.int64)

    if RANK > 7:
        m = cur % sh7
        out_off += m * os7
        idx_off += m * is7
        cur = cur // sh7
    if RANK > 6:
        m = cur % sh6
        out_off += m * os6
        idx_off += m * is6
        cur = cur // sh6
    if RANK > 5:
        m = cur % sh5
        out_off += m * os5
        idx_off += m * is5
        cur = cur // sh5
    if RANK > 4:
        m = cur % sh4
        out_off += m * os4
        idx_off += m * is4
        cur = cur // sh4
    if RANK > 3:
        m = cur % sh3
        out_off += m * os3
        idx_off += m * is3
        cur = cur // sh3
    if RANK > 2:
        m = cur % sh2
        out_off += m * os2
        idx_off += m * is2
        cur = cur // sh2
    if RANK > 1:
        m = cur % sh1
        out_off += m * os1
        idx_off += m * is1
        cur = cur // sh1
    m = cur % sh0
    out_off += m * os0
    idx_off += m * is0

    idx = tl.load(index_ptr + idx_off, mask=mask, other=0)
    final = out_off + idx * dim_stride
    tl.store(out_ptr + final, IDENTITY, mask=mask)


@libentry()
@triton.jit
def _scatter_reduce_kernel(
    out_ptr,
    src_ptr,
    index_ptr,
    count_ptr,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    sh6,
    sh7,
    os0,
    os1,
    os2,
    os3,
    os4,
    os5,
    os6,
    os7,
    ss0,
    ss1,
    ss2,
    ss3,
    ss4,
    ss5,
    ss6,
    ss7,
    is0,
    is1,
    is2,
    is3,
    is4,
    is5,
    is6,
    is7,
    dim_stride,
    N,
    REDUCE: tl.constexpr,
    UPCAST: tl.constexpr,
    NEED_COUNT: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    cur = offs
    out_off = tl.zeros((BLOCK,), dtype=tl.int64)
    src_off = tl.zeros((BLOCK,), dtype=tl.int64)
    idx_off = tl.zeros((BLOCK,), dtype=tl.int64)

    if RANK > 7:
        m = cur % sh7
        out_off += m * os7
        src_off += m * ss7
        idx_off += m * is7
        cur = cur // sh7
    if RANK > 6:
        m = cur % sh6
        out_off += m * os6
        src_off += m * ss6
        idx_off += m * is6
        cur = cur // sh6
    if RANK > 5:
        m = cur % sh5
        out_off += m * os5
        src_off += m * ss5
        idx_off += m * is5
        cur = cur // sh5
    if RANK > 4:
        m = cur % sh4
        out_off += m * os4
        src_off += m * ss4
        idx_off += m * is4
        cur = cur // sh4
    if RANK > 3:
        m = cur % sh3
        out_off += m * os3
        src_off += m * ss3
        idx_off += m * is3
        cur = cur // sh3
    if RANK > 2:
        m = cur % sh2
        out_off += m * os2
        src_off += m * ss2
        idx_off += m * is2
        cur = cur // sh2
    if RANK > 1:
        m = cur % sh1
        out_off += m * os1
        src_off += m * ss1
        idx_off += m * is1
        cur = cur // sh1
    m = cur % sh0
    out_off += m * os0
    src_off += m * ss0
    idx_off += m * is0

    val = tl.load(src_ptr + src_off, mask=mask, other=0)
    idx = tl.load(index_ptr + idx_off, mask=mask, other=0)
    final = out_off + idx * dim_stride

    if UPCAST:
        val = val.to(tl.float32)

    if REDUCE == 0 or REDUCE == 2:
        tl.atomic_add(out_ptr + final, val, mask=mask, sem="relaxed")
        if NEED_COUNT:
            one = tl.full((BLOCK,), 1, dtype=tl.int32)
            tl.atomic_add(count_ptr + final, one, mask=mask, sem="relaxed")
    elif REDUCE == 3:
        tl.atomic_max(out_ptr + final, val, mask=mask, sem="relaxed")
    elif REDUCE == 4:
        tl.atomic_min(out_ptr + final, val, mask=mask, sem="relaxed")
    else:
        stop = ~mask
        all_done = False
        while not all_done:
            cur_v = tl.load(out_ptr + final, mask=mask, other=0)
            new_v = tl.where(stop, cur_v, cur_v * val)
            cas = tl.atomic_cas(out_ptr + final, cur_v, new_v)
            stop |= cur_v == cas
            all_done = tl.sum(stop.to(tl.int32)) == BLOCK


def _heur_block(N: int) -> int:
    if N < 1024:
        return 128
    if N < 32768:
        return 512
    if N < 1048576:
        return 1024
    return 2048


def _heur_warps(BLOCK: int) -> int:
    if BLOCK <= 256:
        return 4
    if BLOCK <= 1024:
        return 4
    return 8


def _scatter_reduce_impl(
    self_t: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduce: str,
    include_self: bool,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """Common engine for both `scatter_reduce` and `scatter_reduce.out`.

    When `out is None` (the `scatter_reduce.two` overload) we allocate via
    `clone()`, which is a single bulk D2D memcpy and avoids the flag_gems
    pointwise `copy_` codegen overhead. When `out` is provided (the `.two_out`
    overload) we initialize it from `self_t` with a fast dedicated kernel."""
    assert reduce in VALID_REDUCE, f"Invalid reduce: {reduce}"
    assert self_t.dim() == index.dim() == src.dim(), "rank mismatch"
    assert self_t.dim() <= _MAX_RANK, f"rank > {_MAX_RANK} not supported"
    dim = dim % self_t.dim()
    rank = self_t.dim()
    N = index.numel()

    if N == 0:
        if out is None:
            out = (
                self_t.clone()
                if include_self
                else torch.full_like(self_t, _identity_value(reduce, self_t.dtype))
            )
        else:
            if include_self:
                _fast_copy_into(out, self_t)
            else:
                out.fill_(_identity_value(reduce, out.dtype))
        return out

    use_upcast = self_t.dtype in (torch.float16, torch.bfloat16)
    if use_upcast:
        buf = torch.empty(self_t.shape, dtype=torch.float32, device=self_t.device)
        _fast_copy_into_with_cast(buf, self_t)
        if out is None:
            out = torch.empty_like(self_t)
    else:
        if out is None:
            out = torch.empty_like(self_t)
        _fast_copy_into(out, self_t)
        buf = out

    src_restrided = src.as_strided(index.shape, src.stride())
    buf_restrided = buf.as_strided(
        index.shape,
        [buf.stride(d) if d != dim else 0 for d in range(rank)],
    )

    sh, _ = _pack_shape_strides(index, _MAX_RANK)
    _, os_ = _pack_shape_strides(buf_restrided, _MAX_RANK)
    _, ss_ = _pack_shape_strides(src_restrided, _MAX_RANK)
    _, ix_ = _pack_shape_strides(index, _MAX_RANK)

    dim_stride = buf.stride(dim)
    BLOCK = _heur_block(N)
    NW = _heur_warps(BLOCK)
    grid = (triton.cdiv(N, BLOCK),)

    if not include_self:
        ident = _identity_value(reduce, buf.dtype)
        _selective_fill_kernel[grid](
            buf,
            index,
            sh[0],
            sh[1],
            sh[2],
            sh[3],
            sh[4],
            sh[5],
            sh[6],
            sh[7],
            os_[0],
            os_[1],
            os_[2],
            os_[3],
            os_[4],
            os_[5],
            os_[6],
            os_[7],
            ix_[0],
            ix_[1],
            ix_[2],
            ix_[3],
            ix_[4],
            ix_[5],
            ix_[6],
            ix_[7],
            dim_stride,
            N,
            IDENTITY=ident,
            RANK=rank,
            BLOCK=BLOCK,
            num_warps=NW,
        )

    need_count = reduce == "mean"
    count = None
    if need_count:
        count = torch.zeros_like(buf, dtype=torch.int32)
        if include_self:
            count.fill_(1)
        else:
            _selective_fill_kernel[grid](
                count,
                index,
                sh[0],
                sh[1],
                sh[2],
                sh[3],
                sh[4],
                sh[5],
                sh[6],
                sh[7],
                os_[0],
                os_[1],
                os_[2],
                os_[3],
                os_[4],
                os_[5],
                os_[6],
                os_[7],
                ix_[0],
                ix_[1],
                ix_[2],
                ix_[3],
                ix_[4],
                ix_[5],
                ix_[6],
                ix_[7],
                dim_stride,
                N,
                IDENTITY=0,
                RANK=rank,
                BLOCK=BLOCK,
                num_warps=NW,
            )

    _scatter_reduce_kernel[grid](
        buf,
        src_restrided,
        index,
        count if need_count else buf,
        sh[0],
        sh[1],
        sh[2],
        sh[3],
        sh[4],
        sh[5],
        sh[6],
        sh[7],
        os_[0],
        os_[1],
        os_[2],
        os_[3],
        os_[4],
        os_[5],
        os_[6],
        os_[7],
        ss_[0],
        ss_[1],
        ss_[2],
        ss_[3],
        ss_[4],
        ss_[5],
        ss_[6],
        ss_[7],
        ix_[0],
        ix_[1],
        ix_[2],
        ix_[3],
        ix_[4],
        ix_[5],
        ix_[6],
        ix_[7],
        dim_stride,
        N,
        REDUCE=_REDUCE_CODE[reduce],
        UPCAST=use_upcast,
        NEED_COUNT=need_count,
        RANK=rank,
        BLOCK=BLOCK,
        num_warps=NW,
    )

    if need_count:
        count.clamp_(min=1)
        buf.div_(count)

    if use_upcast:
        _fast_copy_into_with_cast(out, buf)
    return out


def scatter_reduce(self, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS SCATTER_REDUCE_V4")
    assert (
        reduce in VALID_REDUCE
    ), f"scatter_reduce: invalid reduce={reduce!r}, expected one of {VALID_REDUCE}"
    fp = _try_fastpath(self, dim, index, src, reduce, include_self, None)
    if fp is not None:
        return fp
    return _scatter_reduce_impl(self, dim, index, src, reduce, include_self, None)


def scatter_reduce_out(self, dim, index, src, reduce, *, include_self=True, out):
    logger.debug("GEMS SCATTER_REDUCE_OUT_V4")
    assert (
        reduce in VALID_REDUCE
    ), f"scatter_reduce: invalid reduce={reduce!r}, expected one of {VALID_REDUCE}"
    assert (
        has_internal_overlapping(out) != MemOverlap.Yes
    ), "Cannot write to internally-overlapping out tensor."
    if out.shape != self.shape:
        out.resize_(self.shape)
    fp = _try_fastpath(self, dim, index, src, reduce, include_self, out)
    if fp is not None:
        return fp
    return _scatter_reduce_impl(self, dim, index, src, reduce, include_self, out)
