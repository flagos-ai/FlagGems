import math
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_REDUCTIONS = {"prod": 0, "mean": 1, "amax": 2, "amin": 3}
_BLOCK = 256


@libentry()
@triton.jit
def _index_reduce_scan_kernel(
    inp,
    index,
    source,
    output,
    outer,
    dim_size,
    inner,
    S,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    # One program per output element; scans the whole source dim. No atomics
    # (required for prod and for duplicate indices, whose same-address atomics
    # are unreliable on this backend).
    pid = tl.program_id(0)
    inner_o = pid % inner
    t = pid // inner
    dim_o = t % dim_size
    outer_o = t // dim_size

    self_val = tl.load(inp + pid).to(ACC_DTYPE)

    if REDUCE == 0:
        acc = self_val if INCLUDE_SELF else 1.0
    elif REDUCE == 1:
        acc = self_val if INCLUDE_SELF else 0.0
    elif REDUCE == 2:
        acc = self_val if INCLUDE_SELF else float("-inf")
    else:
        acc = self_val if INCLUDE_SELF else float("inf")
    count = 1 if INCLUDE_SELF else 0

    for s in range(0, S):
        src_off = (outer_o * S + s) * inner + inner_o
        v = tl.load(source + src_off).to(ACC_DTYPE)
        idx = tl.load(index + s).to(tl.int32)
        if idx == dim_o:
            if REDUCE == 0:
                acc = acc * v
            elif REDUCE == 1:
                acc = acc + v
            elif REDUCE == 2:
                acc = tl.maximum(acc, v)
            else:
                acc = tl.minimum(acc, v)
            count = count + 1

    if REDUCE == 1:
        denom = tl.maximum(count, 1)
        acc = acc / denom.to(ACC_DTYPE)
    if not INCLUDE_SELF:
        if count == 0:
            acc = self_val

    tl.store(output + pid, acc)


@libentry()
@triton.jit
def _index_reduce_scatter_kernel(
    out,
    index,
    src,
    count,
    touched,
    outer,
    dim_size,
    inner,
    S,
    REDUCE: tl.constexpr,
    USE_COUNT: tl.constexpr,
    USE_TOUCHED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Scatter each source element to its destination slot with an atomic. This
    # is only used for unique indices, so the destinations within a program are
    # distinct and the (software-simulated) atomics are correct.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = outer * S * inner
    mask = offs < total
    inner_o = offs % inner
    t = offs // inner
    s = t % S
    outer_o = t // S
    idx = tl.load(index + s, mask=mask, other=0).to(tl.int32)
    dst = (outer_o * dim_size + idx) * inner + inner_o
    v = tl.load(src + offs, mask=mask, other=0.0)

    if REDUCE == 1:
        tl.atomic_add(out + dst, v, mask=mask, sem="relaxed")
        ones = tl.full((BLOCK,), 1, dtype=tl.int32)
        tl.atomic_add(count + dst, ones, mask=mask, sem="relaxed")
    elif REDUCE == 2:
        tl.atomic_max(out + dst, v, mask=mask, sem="relaxed")
    elif REDUCE == 3:
        tl.atomic_min(out + dst, v, mask=mask, sem="relaxed")

    if USE_TOUCHED:
        ones = tl.full((BLOCK,), 1, dtype=tl.int32)
        tl.atomic_add(touched + dst, ones, mask=mask, sem="relaxed")


@libentry()
@triton.jit
def _index_reduce_mean_finalize_kernel(
    out,
    acc,
    orig,
    count,
    TOTAL,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    cnt = tl.load(count + offs, mask=mask, other=0).to(tl.float32)
    av = tl.load(acc + offs, mask=mask, other=0.0).to(tl.float32)
    if INCLUDE_SELF:
        res = av / (cnt + 1.0)
    else:
        res = av / tl.maximum(cnt, 1.0)
        ov = tl.load(orig + offs, mask=mask, other=0.0)
        res = tl.where(cnt > 0, res, ov)
    tl.store(out + offs, res, mask=mask)


@libentry()
@triton.jit
def _index_reduce_select_finalize_kernel(
    out,
    val,
    orig,
    touched,
    TOTAL,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    t = tl.load(touched + offs, mask=mask, other=0)
    v = tl.load(val + offs, mask=mask, other=0.0)
    o = tl.load(orig + offs, mask=mask, other=0.0)
    tl.store(out + offs, tl.where(t > 0, v, o), mask=mask)


def _validate_args(inp, dim, index, source, reduce):
    assert reduce in _REDUCTIONS, f"Unsupported reduce: {reduce}"
    assert inp.ndim > 0, "Expected self to have at least one dimension"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim == 1, "Index is supposed to be a vector"
    assert index.dtype in (torch.int32, torch.int64), "Expected dtype int32/int64 for index"
    assert inp.is_floating_point(), "index_reduce_(): Expected self to be floating point"
    assert source.dtype == inp.dtype, "index_reduce_(): Expected self and source to have same dtype"
    assert inp.ndim == source.ndim, "Self and source should have the same number of dimensions"
    assert index.numel() == source.size(dim), (
        "The dimth dimension of source must have the same size as the length of index"
    )
    assert all(
        inp.size(i) == source.size(i) for i in range(inp.ndim) if i != dim
    ), "source.size(d) == self.size(d) for all dimensions d != dim"


def _index_is_unique(index, out_dim):
    if index.numel() > out_dim:
        return False
    if index.numel() <= 1:
        return True
    return index.unique().numel() == index.numel()


def _compute_scan(inp_c, dim, index_c, source_c, reduce, include_self):
    outer = 1
    for d in range(dim):
        outer *= inp_c.shape[d]
    dim_size = inp_c.shape[dim]
    inner = 1
    for d in range(dim + 1, inp_c.ndim):
        inner *= inp_c.shape[d]
    S = index_c.numel()
    result = torch.empty_like(inp_c)
    acc_dtype = tl.float64 if inp_c.dtype == torch.float64 else tl.float32
    total = inp_c.numel()
    with torch_device_fn.device(inp_c.device):
        _index_reduce_scan_kernel[(total,)](
            inp_c,
            index_c,
            source_c,
            result,
            outer,
            dim_size,
            inner,
            S,
            REDUCE=_REDUCTIONS[reduce],
            INCLUDE_SELF=include_self,
            ACC_DTYPE=acc_dtype,
        )
    return result


def _compute_atomic(inp_c, dim, index_c, source_c, reduce, include_self):
    outer = 1
    for d in range(dim):
        outer *= inp_c.shape[d]
    dim_size = inp_c.shape[dim]
    inner = 1
    for d in range(dim + 1, inp_c.ndim):
        inner *= inp_c.shape[d]
    S = index_c.numel()

    # amax/amin have no native fp16/bf16 atomic on this backend: upcast to fp32.
    use_fp32_ws = reduce in ("amax", "amin") and inp_c.dtype in (
        torch.float16,
        torch.bfloat16,
    )
    ws_dtype = torch.float32 if use_fp32_ws else inp_c.dtype

    inp_w = inp_c.to(ws_dtype)
    src_w = source_c.to(ws_dtype)

    # ``count`` is only used for mean (denominator). It always starts at zero:
    # the finalize kernel adds +1 for ``include_self`` explicitly.
    count = (
        torch.zeros_like(inp_c, dtype=torch.int32)
        if reduce == "mean"
        else torch.empty(1, dtype=torch.int32, device=inp_c.device)
    )
    touched = (
        torch.zeros_like(inp_c, dtype=torch.int32)
        if reduce != "mean" and not include_self
        else torch.empty(1, dtype=torch.int32, device=inp_c.device)
    )

    if include_self:
        out_w = inp_w.clone()
    else:
        if reduce == "mean":
            out_w = torch.zeros_like(inp_w)
        elif reduce == "amax":
            out_w = torch.full_like(inp_w, float("-inf"))
        else:
            out_w = torch.full_like(inp_w, float("inf"))

    total = outer * S * inner
    grid = (triton.cdiv(total, _BLOCK),)
    with torch_device_fn.device(inp_c.device):
        _index_reduce_scatter_kernel[grid](
            out_w,
            index_c,
            src_w,
            count,
            touched,
            outer,
            dim_size,
            inner,
            S,
            REDUCE=_REDUCTIONS[reduce],
            USE_COUNT=(reduce == "mean"),
            USE_TOUCHED=(reduce != "mean" and not include_self),
            BLOCK=_BLOCK,
        )

    result = torch.empty_like(inp_w)
    with torch_device_fn.device(inp_c.device):
        if reduce == "mean":
            _index_reduce_mean_finalize_kernel[
                (lambda meta: (triton.cdiv(inp_w.numel(), meta["BLOCK"]),))
            ](
                result,
                out_w,
                inp_w,
                count,
                inp_w.numel(),
                include_self,
                BLOCK=_BLOCK,
            )
        elif not include_self:
            _index_reduce_select_finalize_kernel[
                (lambda meta: (triton.cdiv(inp_w.numel(), meta["BLOCK"]),))
            ](
                result,
                out_w,
                inp_w,
                touched,
                inp_w.numel(),
                BLOCK=_BLOCK,
            )
        else:
            result.copy_(out_w)

    return result.to(inp_c.dtype)


def _compute(inp_c, dim, index_c, source_c, reduce, include_self):
    dim_size = inp_c.shape[dim]
    if reduce != "prod" and _index_is_unique(index_c, dim_size):
        return _compute_atomic(inp_c, dim, index_c, source_c, reduce, include_self)
    return _compute_scan(inp_c, dim, index_c, source_c, reduce, include_self)


def index_reduce_(inp, dim, index, source, reduce, *, include_self=True):
    logger.debug("GEMS_KUNLUNXIN INDEX_REDUCE_")
    _validate_args(inp, dim, index, source, reduce)
    dim %= inp.ndim
    if index.numel() == 0:
        return inp
    inp_c = inp.contiguous()
    source_c = source.contiguous()
    index_c = index.contiguous()
    result = _compute(inp_c, dim, index_c, source_c, reduce, include_self)
    inp.copy_(result)
    return inp


def index_reduce(inp, dim, index, source, reduce, *, include_self=True):
    logger.debug("GEMS_KUNLUNXIN INDEX_REDUCE")
    _validate_args(inp, dim, index, source, reduce)
    dim %= inp.ndim
    out = inp.clone(memory_format=torch.contiguous_format)
    if index.numel() == 0:
        return out
    source_c = source.contiguous()
    index_c = index.contiguous()
    return _compute(out, dim, index_c, source_c, reduce, include_self)


def index_reduce_out(inp, dim, index, source, reduce, *, include_self=True, out=None):
    logger.debug("GEMS_KUNLUNXIN INDEX_REDUCE_OUT")
    _validate_args(inp, dim, index, source, reduce)
    if out is None:
        return index_reduce(inp, dim, index, source, reduce, include_self=include_self)
    if out.dtype != inp.dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {inp.dtype}, but got {out.dtype} instead"
        )
    if out.device != inp.device:
        raise RuntimeError(
            f"Expected out tensor to be on device {inp.device}, but got {out.device} instead"
        )
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)
    dim %= inp.ndim
    if out.data_ptr() != inp.data_ptr() or out.stride() != inp.stride():
        out.copy_(inp)
    if index.numel() == 0:
        return out
    out_c = out.contiguous()
    source_c = source.contiguous()
    index_c = index.contiguous()
    result = _compute(out_c, dim, index_c, source_c, reduce, include_self)
    if result is not out:
        out.copy_(result)
    return out
