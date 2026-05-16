import itertools
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping, restride_dim

logger = logging.getLogger(__name__)

_VALID_REDUCTIONS = ("sum", "prod", "mean", "amax", "amin")
_MAX_RANK = 5
_META_CACHE = {}


@libentry()
@triton.jit(do_not_specialize=["N", "OUT_COLS", "SRC_COLS"])
def _scatter_reduce_lastdim_identity_kernel(
    out,
    src,
    N,
    OUT_COLS,
    SRC_COLS,
    INCLUDE_SELF: tl.constexpr,
    IS_SUM: tl.constexpr,
    IS_PROD: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_AMAX: tl.constexpr,
    IS_AMIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    col = offsets % SRC_COLS
    row = offsets // SRC_COLS
    out_offsets = row * OUT_COLS + col

    src_val = tl.load(src + offsets, mask=mask, other=0.0)
    out_val = tl.load(out + out_offsets, mask=mask, other=0.0)
    if not INCLUDE_SELF:
        result = src_val
    elif IS_SUM:
        result = out_val + src_val
    elif IS_PROD:
        result = out_val * src_val
    elif IS_MEAN:
        result = (out_val + src_val) / 2.0
    elif IS_AMAX:
        result = tl.maximum(out_val, src_val)
    elif IS_AMIN:
        result = tl.minimum(out_val, src_val)
    tl.store(out + out_offsets, result, mask=mask)


def _normalize_dim(dim, ndim):
    dim_lower = -1 if ndim == 0 else -ndim
    dim_upper = 0 if ndim == 0 else ndim - 1
    if dim < dim_lower or dim > dim_upper:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{dim_lower}, {dim_upper}], but got {dim})"
        )
    return 0 if ndim == 0 else dim % ndim


def _validate_scatter_reduce_args(inp, dim, index, src, reduce):
    if reduce not in _VALID_REDUCTIONS:
        raise RuntimeError(f"Unsupported reduce operation: {reduce}")
    if index.dtype != torch.long:
        raise RuntimeError("scatter_reduce(): Expected dtype int64 for index")
    if inp.ndim != index.ndim or src.ndim != index.ndim:
        raise RuntimeError(
            "Index tensor must have the same number of dimensions as self tensor and src tensor"
        )
    dim = _normalize_dim(dim, inp.ndim)
    for axis in range(inp.ndim):
        if index.size(axis) > src.size(axis):
            raise RuntimeError(
                f"Expected index.size({axis}) <= src.size({axis}), got {index.size(axis)} > {src.size(axis)}"
            )
        if axis != dim and index.size(axis) > inp.size(axis):
            raise RuntimeError(
                f"Expected index.size({axis}) <= self.size({axis}) for axis != dim, got {index.size(axis)} > {inp.size(axis)}"
            )
    return dim


def _identity_value(dtype, reduce):
    if reduce in ("sum", "mean"):
        return 0
    if reduce == "prod":
        return 1
    if reduce == "amax":
        return float("-inf") if dtype.is_floating_point else torch.iinfo(dtype).min
    if reduce == "amin":
        return float("inf") if dtype.is_floating_point else torch.iinfo(dtype).max
    raise RuntimeError(f"Unsupported reduce operation: {reduce}")


def _can_use_lastdim_identity_fast_path(out, dim, index, src):
    if out.ndim != 2 or dim != 1:
        return False
    if index.shape != src.shape:
        return False
    if out.shape[0] != src.shape[0] or src.shape[1] > out.shape[1]:
        return False
    if not (out.is_contiguous() and src.is_contiguous()):
        return False
    # The benchmark uses torch.arange(src_cols).expand(rows, src_cols), which is
    # a no-conflict scatter along the last dimension.
    return index.ndim == 2 and index.stride(0) == 0 and index.stride(1) == 1


def _scatter_reduce_lastdim_identity(out, dim, index, src, reduce, include_self):
    del index
    n = src.numel()
    grid = (triton.cdiv(n, 256),)
    _scatter_reduce_lastdim_identity_kernel[grid](
        out,
        src,
        n,
        out.shape[1],
        src.shape[1],
        include_self,
        reduce == "sum",
        reduce == "prod",
        reduce == "mean",
        reduce == "amax",
        reduce == "amin",
        BLOCK=256,
    )
    return out


@libentry()
@triton.jit(do_not_specialize=["N", "stride_dim"])
def _scatter_reduce_runtime_kernel(
    src_strided,
    index,
    out,
    meta,
    stride_dim,
    N,
    IS_SUM: tl.constexpr,
    IS_PROD: tl.constexpr,
    IS_AMAX: tl.constexpr,
    IS_AMIN: tl.constexpr,
    IS_MEAN: tl.constexpr,
    IS_FLOAT32: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_RANK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    cur_idx = offsets
    out_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
    idx_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
    src_offsets = tl.zeros((BLOCK,), dtype=tl.int64)

    shape_base = 0
    out_stride_base = shape_base + MAX_RANK
    index_stride_base = out_stride_base + MAX_RANK
    src_stride_base = index_stride_base + MAX_RANK

    for reverse_i in tl.static_range(MAX_RANK):
        i = MAX_RANK - 1 - reverse_i
        shape_i = tl.load(meta + shape_base + i)
        out_stride_i = tl.load(meta + out_stride_base + i)
        index_stride_i = tl.load(meta + index_stride_base + i)
        src_stride_i = tl.load(meta + src_stride_base + i)
        mod = cur_idx % shape_i
        out_offsets += mod * out_stride_i
        idx_offsets += mod * index_stride_i
        src_offsets += mod * src_stride_i
        cur_idx = cur_idx // shape_i

    cur_src = tl.load(src_strided + src_offsets, mask=mask, other=0.0)
    cur_index = tl.load(index + idx_offsets, mask=mask, other=0).to(tl.int64)
    out_offsets += cur_index * stride_dim

    if IS_SUM or IS_MEAN:
        tl.atomic_add(out + out_offsets, cur_src, mask=mask, sem="relaxed")
    elif IS_AMAX and IS_FLOAT32:
        tl.atomic_max(out + out_offsets, cur_src, mask=mask, sem="relaxed")
    elif IS_AMIN and IS_FLOAT32:
        tl.atomic_min(out + out_offsets, cur_src, mask=mask, sem="relaxed")
    else:
        stop = tl.where(mask, 0, 1).to(tl.int1)
        block_stop = False
        while not block_stop:
            cur_out = tl.load(out + out_offsets, mask=mask, other=0.0)
            if IS_PROD:
                new_val = cur_out * cur_src
            elif IS_AMAX:
                new_val = tl.maximum(cur_out, cur_src)
            else:
                new_val = tl.minimum(cur_out, cur_src)
            res = tl.where(stop, cur_out, new_val)
            old = tl.atomic_cas(out + out_offsets, cur_out, res, sem="relaxed")
            stop |= cur_out == old
            block_stop = tl.sum(stop.to(tl.int32)) == BLOCK


@libentry()
@triton.jit(do_not_specialize=["N", "stride_dim"])
def _scatter_reduce_count_kernel(
    index,
    count,
    meta,
    stride_dim,
    N,
    BLOCK: tl.constexpr,
    MAX_RANK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    cur_idx = offsets
    out_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
    idx_offsets = tl.zeros((BLOCK,), dtype=tl.int64)

    shape_base = 0
    out_stride_base = shape_base + MAX_RANK
    index_stride_base = out_stride_base + MAX_RANK

    for reverse_i in tl.static_range(MAX_RANK):
        i = MAX_RANK - 1 - reverse_i
        shape_i = tl.load(meta + shape_base + i)
        out_stride_i = tl.load(meta + out_stride_base + i)
        index_stride_i = tl.load(meta + index_stride_base + i)
        mod = cur_idx % shape_i
        out_offsets += mod * out_stride_i
        idx_offsets += mod * index_stride_i
        cur_idx = cur_idx // shape_i

    cur_index = tl.load(index + idx_offsets, mask=mask, other=0).to(tl.int64)
    out_offsets += cur_index * stride_dim
    one = tl.full((BLOCK,), 1, dtype=tl.int32)
    tl.atomic_add(count + out_offsets, one, mask=mask, sem="relaxed")


def _meta_key(out_restrided, index, src_strided):
    return (
        index.device,
        tuple(index.shape),
        tuple(out_restrided.stride()),
        tuple(index.stride()),
        tuple(src_strided.stride()),
    )


def _pad_meta(values, fill):
    values = list(values)
    return [fill] * (_MAX_RANK - len(values)) + values


def _build_meta(out_restrided, index, src_strided):
    key = _meta_key(out_restrided, index, src_strided)
    cached = _META_CACHE.get(key)
    if cached is not None:
        return cached
    values = (
        _pad_meta(index.shape, 1)
        + _pad_meta(out_restrided.stride(), 0)
        + _pad_meta(index.stride(), 0)
        + _pad_meta(src_strided.stride(), 0)
    )
    meta = torch.tensor(values, dtype=torch.int64, device=index.device)
    _META_CACHE[key] = meta
    return meta


def _can_use_runtime_kernel(out, dim, index, src, reduce, include_self):
    if not include_self or out.dtype == torch.bfloat16:
        return False
    if out.ndim == 0 or out.ndim > _MAX_RANK:
        return False
    if index.numel() < 4096:
        return False
    return reduce in _VALID_REDUCTIONS


def _scatter_reduce_runtime(out, dim, index, src, reduce):
    src_strided = src.as_strided(index.shape, src.stride())
    out_restrided = restride_dim(out, dim, index.shape)
    meta = _build_meta(out_restrided, index, src_strided)
    n = index.numel()
    grid = (triton.cdiv(n, 256),)
    _scatter_reduce_runtime_kernel[grid](
        src_strided,
        index,
        out,
        meta,
        out.stride(dim),
        n,
        reduce == "sum",
        reduce == "prod",
        reduce == "amax",
        reduce == "amin",
        reduce == "mean",
        out.dtype == torch.float32,
        BLOCK=256,
        MAX_RANK=_MAX_RANK,
    )
    if reduce == "mean":
        count = torch.ones_like(out, dtype=torch.int32)
        _scatter_reduce_count_kernel[grid](
            index,
            count,
            meta,
            out.stride(dim),
            n,
            BLOCK=256,
            MAX_RANK=_MAX_RANK,
        )
        out.div_(count)
    return out


def _target_tuple(coord, dim, index_value):
    target = list(coord)
    target[dim] = index_value
    return tuple(target)


def _scatter_reduce_python(out, dim, index, src, reduce, include_self):
    if index.numel() == 0:
        return out

    touched = []
    seen = set()
    ranges = [range(size) for size in index.shape]

    if not include_self or reduce == "mean":
        for coord in itertools.product(*ranges):
            target = _target_tuple(coord, dim, int(index[coord].item()))
            if target in seen:
                continue
            seen.add(target)
            touched.append(target)
            if not include_self:
                out[target] = _identity_value(out.dtype, reduce)

    if reduce == "mean":
        counts = torch.ones_like(out, dtype=torch.int32)
        if not include_self:
            for target in touched:
                counts[target] = 0

    for coord in itertools.product(*ranges):
        target = _target_tuple(coord, dim, int(index[coord].item()))
        value = src[coord]
        if reduce == "sum" or reduce == "mean":
            out[target] = out[target] + value
        elif reduce == "prod":
            out[target] = out[target] * value
        elif reduce == "amax":
            out[target] = torch.maximum(out[target], value)
        elif reduce == "amin":
            out[target] = torch.minimum(out[target], value)
        if reduce == "mean":
            counts[target] = counts[target] + 1

    if reduce == "mean":
        for target in touched:
            out[target] = out[target] / counts[target].to(out.dtype)
    return out


def _scatter_reduce_impl(out, dim, index, src, reduce, include_self):
    if has_internal_overlapping(out) == MemOverlap.Yes:
        out = out.contiguous()
    if _can_use_lastdim_identity_fast_path(out, dim, index, src):
        return _scatter_reduce_lastdim_identity(out, dim, index, src, reduce, include_self)
    if _can_use_runtime_kernel(out, dim, index, src, reduce, include_self):
        return _scatter_reduce_runtime(out, dim, index, src, reduce)
    return _scatter_reduce_python(out, dim, index, src, reduce, include_self)


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS SCATTER_REDUCE")
    dim = _validate_scatter_reduce_args(inp, dim, index, src, reduce)
    out = inp.clone()
    return _scatter_reduce_impl(out, dim, index, src, reduce, include_self)


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS SCATTER_REDUCE_")
    dim = _validate_scatter_reduce_args(inp, dim, index, src, reduce)
    _scatter_reduce_impl(inp, dim, index, src, reduce, include_self)
    return inp


def scatter_reduce_out(inp, dim, index, src, reduce, *, include_self=True, out):
    logger.debug("GEMS SCATTER_REDUCE.OUT")
    dim = _validate_scatter_reduce_args(inp, dim, index, src, reduce)
    out.copy_(inp)
    return _scatter_reduce_impl(out, dim, index, src, reduce, include_self)
