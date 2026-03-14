import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.scatter import scatter_
from flag_gems.ops.scatter_add_ import scatter_add_
from flag_gems.utils import libentry
from flag_gems.utils.shape_utils import (
    MemOverlap,
    has_internal_overlapping,
    restride_dim,
)

logger = logging.getLogger(__name__)

_VALID_REDUCTIONS = {"sum", "prod", "mean", "amax", "amin"}
_RUNTIME_MAX_RANK = 8
_SCATTER_REDUCE_META_CACHE = {}
_SCATTER_REDUCE_PROD_FAST_DTYPES = (
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.int32,
)


def _validate_scatter_reduce_args(inp, dim, index, src, reduce):
    if reduce not in _VALID_REDUCTIONS:
        raise RuntimeError(
            f"reduce argument must be one of {_VALID_REDUCTIONS}, got {reduce!r}"
        )
    if index.dtype != torch.long:
        raise RuntimeError("scatter_reduce(): Expected dtype int64 for index")
    if inp.ndim != index.ndim or src.ndim != index.ndim:
        raise RuntimeError(
            "Index tensor must have the same number of dimensions as self tensor and src tensor"
        )
    dim_lower = -1 if inp.ndim == 0 else -inp.ndim
    dim_upper = 0 if inp.ndim == 0 else inp.ndim - 1
    if dim < dim_lower or dim > dim_upper:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{dim_lower}, {dim_upper}], but got {dim})"
        )
    dim = 0 if inp.ndim == 0 else dim % inp.ndim
    for d in range(inp.ndim):
        if index.size(d) > src.size(d):
            raise RuntimeError(
                f"Expected index.size({d}) <= src.size({d}), got {index.size(d)} > {src.size(d)}"
            )
        if d != dim and index.size(d) > inp.size(d):
            raise RuntimeError(
                f"Expected index.size({d}) <= self.size({d}) for d != dim, got {index.size(d)} > {inp.size(d)}"
            )
    return dim


def _make_fill_tensor(index, inp, fill_value):
    return torch.full(index.shape, fill_value, dtype=inp.dtype, device=inp.device)


def _reset_scatter_targets(out, dim, index, fill_value):
    fill_src = _make_fill_tensor(index, out, fill_value)
    return scatter_(out, dim, index, fill_src)


def _reduction_identity(dtype, reduce):
    if reduce == "sum" or reduce == "mean":
        return 0
    if reduce == "prod":
        return 1
    if reduce == "amax":
        if dtype.is_floating_point:
            return float("-inf")
        return torch.iinfo(dtype).min
    if reduce == "amin":
        if dtype.is_floating_point:
            return float("inf")
        return torch.iinfo(dtype).max
    raise RuntimeError(f"Unsupported reduction {reduce}")


def _mean_division(out, counts):
    if out.dtype.is_floating_point:
        return out / counts.to(out.dtype)
    return torch.div(out, counts.to(out.dtype), rounding_mode="floor")


def _scatter_reduce_cpu_fallback(inp, dim, index, src, reduce, include_self):
    out = torch.scatter_reduce(
        inp.cpu(),
        dim,
        index.cpu(),
        src.cpu(),
        reduce,
        include_self=include_self,
    )
    return out.to(inp.device)


def _requires_scatter_reduce_copy(inp, src):
    return torch._C._overlaps(inp, src)


def _is_same_tensor_view(lhs, rhs):
    return (
        lhs.data_ptr() == rhs.data_ptr()
        and lhs.storage_offset() == rhs.storage_offset()
        and lhs.shape == rhs.shape
        and lhs.stride() == rhs.stride()
    )


def _scatter_reduce_out_requires_temp(inp, out, src):
    if _requires_scatter_reduce_copy(out, src):
        return True
    return _requires_scatter_reduce_copy(out, inp) and not _is_same_tensor_view(
        out, inp
    )


def _atomic_accumulator_dtype(dtype, reduce):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if dtype == torch.int16:
        return torch.int32
    return dtype


def _make_strided_copy(tensor, dtype):
    out = torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=dtype, device=tensor.device
    )
    out.copy_(tensor)
    return out


def _scatter_reduce_sum(inp, dim, index, src, include_self):
    if inp.dtype == torch.int16:
        return _scatter_reduce_cpu_fallback(inp, dim, index, src, "sum", include_self)
    out = inp.clone()
    if not include_self:
        _reset_scatter_targets(out, dim, index, 0)
    scatter_add_(out, dim, index, src)
    return out


def _scatter_reduce_sum_(inp, dim, index, src, include_self):
    if inp.dtype == torch.int16 or _requires_scatter_reduce_copy(inp, src):
        inp.copy_(
            _scatter_reduce_cpu_fallback(inp, dim, index, src, "sum", include_self)
        )
        return inp
    assert (
        has_internal_overlapping(inp) != MemOverlap.Yes
    ), "Unsupported operation: trying to inplace write to an internally overlapping tensor."
    if not include_self:
        _reset_scatter_targets(inp, dim, index, 0)
    scatter_add_(inp, dim, index, src)
    return inp


def _scatter_reduce_sum_out(inp, dim, index, src, include_self, out):
    if _is_same_tensor_view(out, inp):
        return _scatter_reduce_sum_(out, dim, index, src, include_self)
    if inp.dtype == torch.int16 or _scatter_reduce_out_requires_temp(inp, out, src):
        out.copy_(_scatter_reduce_sum(inp, dim, index, src, include_self))
        return out
    out.copy_(inp)
    if not include_self:
        _reset_scatter_targets(out, dim, index, 0)
    scatter_add_(out, dim, index, src)
    return out


def _can_use_scatter_reduce_prod_fast_path(inp, dim, index, src, out=None):
    if dim != inp.ndim - 1 or inp.ndim < 2:
        return False
    if src.ndim != inp.ndim or index.ndim != inp.ndim:
        return False
    if not (inp.is_contiguous() and src.is_contiguous() and index.is_contiguous()):
        return False
    if out is not None and not out.is_contiguous():
        return False
    if index.shape != src.shape:
        return False
    if inp.shape[:-1] != index.shape[:-1]:
        return False
    return inp.dtype in _SCATTER_REDUCE_PROD_FAST_DTYPES


def _reshape_scatter_reduce_prod_fast_tensors(inp, index, src, out):
    flat_inp = inp.view(-1, inp.shape[-1])
    flat_index = index.view(-1, index.shape[-1])
    flat_src = src.view(-1, src.shape[-1])
    flat_out = out.view(-1, out.shape[-1])
    return flat_inp, flat_index, flat_src, flat_out


def _scatter_reduce_prod_fast_launch_config(rows, src_cols, dtype):
    # Small and medium problems benefit from lower launch overhead. Once the row
    # count gets large, the lighter config under-utilizes the kernel and hurts
    # the wide 4096x4096-style cases, so route those back to a heavier launch.
    if src_cols <= 512:
        block = 64
    elif rows <= 1024 and src_cols <= 1024:
        block = 64
    elif rows <= 1024 and src_cols >= 4096:
        block = 128
    elif rows >= 2048:
        block = 128
    else:
        block = 128

    if rows >= 2048 and dtype in (torch.float32, torch.int32):
        num_warps = 8
    else:
        num_warps = 2 if block == 64 else 4

    return block, num_warps


@libentry()
@triton.jit(do_not_specialize=["num_cols_out", "num_cols_src"])
def _scatter_reduce_prod_fast_2d_kernel(
    src,
    index,
    out,
    num_cols_out,
    num_cols_src,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    pid_col = tl.program_id(1)
    cols = pid_col * BLOCK + tl.arange(0, BLOCK)
    mask = cols < num_cols_src

    src_offsets = row * num_cols_src + cols
    cur_src = tl.load(src + src_offsets, mask=mask, other=0)
    cur_index = tl.load(index + src_offsets, mask=mask, other=0).to(tl.int64)
    out_offsets = row * num_cols_out + cur_index

    stop = tl.where(mask, 0, 1).to(tl.int1)
    block_stop = False
    while not block_stop:
        cur_out = tl.load(out + out_offsets, mask=mask, other=0)
        res = tl.where(stop, cur_out, cur_out * cur_src)
        cas_res = tl.atomic_cas(out + out_offsets, cur_out, res, sem="relaxed")
        cur_out_is_nan = cur_out != cur_out
        cas_res_is_nan = cas_res != cas_res
        cas_succeeded = (cur_out == cas_res) | (cur_out_is_nan & cas_res_is_nan)
        stop |= cas_succeeded
        block_stop = tl.sum(stop.to(tl.int32)) == BLOCK


def _scatter_reduce_prod_fast_path(inp, index, src, include_self, out):
    (
        flat_inp,
        flat_index,
        flat_src,
        flat_out,
    ) = _reshape_scatter_reduce_prod_fast_tensors(inp, index, src, out)
    if not include_self:
        _reset_scatter_targets(out, inp.ndim - 1, index, 1)
    block, num_warps = _scatter_reduce_prod_fast_launch_config(
        flat_inp.shape[0], flat_src.shape[1], inp.dtype
    )
    grid = (flat_inp.shape[0], triton.cdiv(flat_src.shape[1], block))
    _scatter_reduce_prod_fast_2d_kernel[grid](
        flat_src,
        flat_index,
        flat_out,
        flat_inp.shape[1],
        flat_src.shape[1],
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


def _scatter_reduce_prod(inp, dim, index, src, include_self):
    if _can_use_scatter_reduce_prod_fast_path(inp, dim, index, src):
        return _scatter_reduce_prod_fast_path(
            inp, index, src, include_self, inp.clone()
        )
    return _scatter_reduce_atomic(inp, dim, index, src, "prod", include_self)


def _scatter_reduce_mean(inp, dim, index, src, include_self):
    if inp.dtype == torch.int16:
        return _scatter_reduce_cpu_fallback(inp, dim, index, src, "mean", include_self)
    out = inp.clone()
    counts = torch.ones_like(inp, dtype=torch.int32)
    if not include_self:
        _reset_scatter_targets(out, dim, index, 0)
        zero_src = torch.zeros(index.shape, dtype=counts.dtype, device=counts.device)
        scatter_(counts, dim, index, zero_src)

    scatter_add_(out, dim, index, src)
    count_src = torch.ones(index.shape, dtype=counts.dtype, device=counts.device)
    scatter_add_(counts, dim, index, count_src)
    return _mean_division(out, counts)


def _scatter_reduce_mean_out(inp, dim, index, src, include_self, out):
    if _is_same_tensor_view(out, inp):
        res = scatter_reduce(inp, dim, index, src, "mean", include_self=include_self)
        out.copy_(res)
        return out
    if inp.dtype == torch.int16 or _scatter_reduce_out_requires_temp(inp, out, src):
        out.copy_(_scatter_reduce_mean(inp, dim, index, src, include_self))
        return out
    out.copy_(inp)
    counts = torch.ones_like(inp, dtype=torch.int32)
    if not include_self:
        _reset_scatter_targets(out, dim, index, 0)
        zero_src = torch.zeros(index.shape, dtype=counts.dtype, device=counts.device)
        scatter_(counts, dim, index, zero_src)

    scatter_add_(out, dim, index, src)
    count_src = torch.ones(index.shape, dtype=counts.dtype, device=counts.device)
    scatter_add_(counts, dim, index, count_src)
    out.copy_(_mean_division(out, counts))
    return out


def heur_block(args):
    return 128


@libentry()
@triton.heuristics(
    {
        "BLOCK": heur_block,
    }
)
@triton.jit(do_not_specialize=["N", "stride_dim"])
def _scatter_reduce_runtime_kernel(
    src_strided,
    index,
    out,
    meta,
    stride_dim,
    N,
    IS_MAX: tl.constexpr,
    IS_PROD: tl.constexpr,
    BLOCK: tl.constexpr,
    INT32_OFFSET: tl.constexpr,
    MAX_RANK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    cur_idx = offsets

    if INT32_OFFSET:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        stride_dim = stride_dim.to(tl.int32)
    else:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int64)

    # Metadata layout:
    # [rank, shape[0:MAX_RANK], inp_stride[0:MAX_RANK],
    #  index_stride[0:MAX_RANK], src_stride[0:MAX_RANK]]
    # Unused dimensions are padded with shape=1 and stride=0.
    shape_base = 1
    inp_stride_base = shape_base + MAX_RANK
    index_stride_base = inp_stride_base + MAX_RANK
    src_stride_base = index_stride_base + MAX_RANK

    for idx in tl.static_range(MAX_RANK):
        i = MAX_RANK - 1 - idx
        shape_i = tl.load(meta + shape_base + i)
        inp_stride_i = tl.load(meta + inp_stride_base + i)
        index_stride_i = tl.load(meta + index_stride_base + i)
        src_stride_i = tl.load(meta + src_stride_base + i)

        if INT32_OFFSET:
            shape_i = shape_i.to(tl.int32)
            inp_stride_i = inp_stride_i.to(tl.int32)
            index_stride_i = index_stride_i.to(tl.int32)
            src_stride_i = src_stride_i.to(tl.int32)

        mod = cur_idx % shape_i
        inp_offsets += mod * inp_stride_i
        idx_offsets += mod * index_stride_i
        src_offsets += mod * src_stride_i
        cur_idx = cur_idx // shape_i

    cur_src = tl.load(src_strided + src_offsets, mask=mask, other=0)
    cur_index = tl.load(index + idx_offsets, mask=mask, other=0)
    if INT32_OFFSET:
        cur_index = cur_index.to(tl.int32)
    inp_offsets += cur_index * stride_dim

    if IS_MAX:
        tl.atomic_max(out + inp_offsets, cur_src, mask=mask, sem="relaxed")
    elif IS_PROD:
        stop = tl.where(mask, 0, 1).to(tl.int1)
        block_stop = False
        while not block_stop:
            cur_out = tl.load(out + inp_offsets, mask=mask, other=0)
            res = tl.where(stop, cur_out, cur_out * cur_src)
            cas_res = tl.atomic_cas(out + inp_offsets, cur_out, res, sem="relaxed")
            cur_out_is_nan = cur_out != cur_out
            cas_res_is_nan = cas_res != cas_res
            cas_succeeded = (cur_out == cas_res) | (cur_out_is_nan & cas_res_is_nan)
            stop |= cas_succeeded
            block_stop = tl.sum(stop.to(tl.int32)) == BLOCK
    else:
        tl.atomic_min(out + inp_offsets, cur_src, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics(
    {
        "BLOCK": heur_block,
    }
)
@triton.jit(do_not_specialize=["N", "stride_dim"])
def _scatter_reduce_nan_kernel(
    src_strided,
    index,
    out,
    meta,
    stride_dim,
    N,
    BLOCK: tl.constexpr,
    INT32_OFFSET: tl.constexpr,
    MAX_RANK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    cur_idx = offsets

    if INT32_OFFSET:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int32)
        stride_dim = stride_dim.to(tl.int32)
    else:
        inp_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        idx_offsets = tl.zeros((BLOCK,), dtype=tl.int64)
        src_offsets = tl.zeros((BLOCK,), dtype=tl.int64)

    shape_base = 1
    inp_stride_base = shape_base + MAX_RANK
    index_stride_base = inp_stride_base + MAX_RANK
    src_stride_base = index_stride_base + MAX_RANK

    for idx in tl.static_range(MAX_RANK):
        i = MAX_RANK - 1 - idx
        shape_i = tl.load(meta + shape_base + i)
        inp_stride_i = tl.load(meta + inp_stride_base + i)
        index_stride_i = tl.load(meta + index_stride_base + i)
        src_stride_i = tl.load(meta + src_stride_base + i)

        if INT32_OFFSET:
            shape_i = shape_i.to(tl.int32)
            inp_stride_i = inp_stride_i.to(tl.int32)
            index_stride_i = index_stride_i.to(tl.int32)
            src_stride_i = src_stride_i.to(tl.int32)

        mod = cur_idx % shape_i
        inp_offsets += mod * inp_stride_i
        idx_offsets += mod * index_stride_i
        src_offsets += mod * src_stride_i
        cur_idx = cur_idx // shape_i

    cur_src = tl.load(src_strided + src_offsets, mask=mask, other=0)
    nan_mask = mask & (cur_src != cur_src)
    cur_index = tl.load(index + idx_offsets, mask=mask, other=0)
    if INT32_OFFSET:
        cur_index = cur_index.to(tl.int32)
    inp_offsets += cur_index * stride_dim
    tl.atomic_xchg(out + inp_offsets, cur_src, mask=nan_mask, sem="relaxed")


def _scatter_reduce_meta_cache_key(inp_restrided, index, src_strided, meta_dtype):
    device_index = index.device.index if index.device.index is not None else -1
    return (
        index.device.type,
        device_index,
        meta_dtype,
        index.ndim,
        tuple(index.shape),
        tuple(inp_restrided.stride()),
        tuple(index.stride()),
        tuple(src_strided.stride()),
    )


def _build_scatter_reduce_meta(inp_restrided, index, src_strided, meta_dtype):
    cache_key = _scatter_reduce_meta_cache_key(
        inp_restrided, index, src_strided, meta_dtype
    )
    cached_meta = _SCATTER_REDUCE_META_CACHE.get(cache_key)
    if cached_meta is not None:
        return cached_meta

    meta = torch.empty(1 + 4 * _RUNTIME_MAX_RANK, dtype=meta_dtype, device=index.device)
    meta[0] = index.ndim
    shape_base = 1
    inp_stride_base = shape_base + _RUNTIME_MAX_RANK
    index_stride_base = inp_stride_base + _RUNTIME_MAX_RANK
    src_stride_base = index_stride_base + _RUNTIME_MAX_RANK

    meta[shape_base:] = 1
    meta[inp_stride_base:] = 0
    meta[index_stride_base:] = 0
    meta[src_stride_base:] = 0

    rank = index.ndim
    if rank > 0:
        meta[shape_base : shape_base + rank] = torch.tensor(
            index.shape, dtype=meta_dtype, device=index.device
        )
        meta[inp_stride_base : inp_stride_base + rank] = torch.tensor(
            inp_restrided.stride(), dtype=meta_dtype, device=index.device
        )
        meta[index_stride_base : index_stride_base + rank] = torch.tensor(
            index.stride(), dtype=meta_dtype, device=index.device
        )
        meta[src_stride_base : src_stride_base + rank] = torch.tensor(
            src_strided.stride(), dtype=meta_dtype, device=index.device
        )
    _SCATTER_REDUCE_META_CACHE[cache_key] = meta
    return meta


def _scatter_reduce_atomic_kernel_impl(out, dim, index, src, reduce, include_self):
    if reduce not in ("prod", "amax", "amin"):
        raise RuntimeError(f"Unsupported atomic reduction {reduce}")
    if not include_self:
        _reset_scatter_targets(out, dim, index, _reduction_identity(out.dtype, reduce))
    if has_internal_overlapping(out) == MemOverlap.Yes:
        out = out.contiguous()
    src_strided = src.as_strided(index.shape, src.stride())
    out_restrided = restride_dim(out, dim, index.shape)
    dim_stride = out.stride(dim)
    n = index.numel()
    int32_size_dim = lambda x: x.stride(dim) * x.size(dim) < 2**32
    use_int32_offset = all(map(int32_size_dim, (out, index, src)))
    meta_dtype = torch.int32 if use_int32_offset else torch.int64
    meta = _build_scatter_reduce_meta(out_restrided, index, src_strided, meta_dtype)
    is_max = reduce == "amax"
    is_prod = reduce == "prod"
    grid = lambda meta_args: (triton.cdiv(n, meta_args["BLOCK"]),)
    _scatter_reduce_runtime_kernel[grid](
        src_strided,
        index,
        out,
        meta,
        dim_stride,
        n,
        is_max,
        is_prod,
        INT32_OFFSET=use_int32_offset,
        MAX_RANK=_RUNTIME_MAX_RANK,
    )
    if reduce in ("amax", "amin") and out.dtype.is_floating_point:
        _scatter_reduce_nan_kernel[grid](
            src_strided,
            index,
            out,
            meta,
            dim_stride,
            n,
            INT32_OFFSET=use_int32_offset,
            MAX_RANK=_RUNTIME_MAX_RANK,
        )
    return out


def _scatter_reduce_atomic_impl(inp, dim, index, src, reduce, include_self, out):
    if inp.ndim > _RUNTIME_MAX_RANK:
        return _scatter_reduce_cpu_fallback(inp, dim, index, src, reduce, include_self)

    accum_dtype = _atomic_accumulator_dtype(inp.dtype, reduce)
    if accum_dtype != inp.dtype:
        promoted_out = _make_strided_copy(out, accum_dtype)
        promoted_src = _make_strided_copy(src, accum_dtype)
        res = _scatter_reduce_atomic_kernel_impl(
            promoted_out, dim, index, promoted_src, reduce, include_self
        )
        out.copy_(res)
        return out

    return _scatter_reduce_atomic_kernel_impl(
        out, dim, index, src, reduce, include_self
    )


def _scatter_reduce_atomic(inp, dim, index, src, reduce, include_self):
    return _scatter_reduce_atomic_impl(
        inp, dim, index, src, reduce, include_self, inp.clone()
    )


def _scatter_reduce_atomic_(inp, dim, index, src, reduce, include_self):
    if (
        reduce == "prod"
        and _can_use_scatter_reduce_prod_fast_path(inp, dim, index, src)
        and not _requires_scatter_reduce_copy(inp, src)
    ):
        assert (
            has_internal_overlapping(inp) != MemOverlap.Yes
        ), "Unsupported operation: trying to inplace write to an internally overlapping tensor."
        return _scatter_reduce_prod_fast_path(inp, index, src, include_self, inp)
    if inp.ndim > _RUNTIME_MAX_RANK:
        inp.copy_(
            _scatter_reduce_cpu_fallback(inp, dim, index, src, reduce, include_self)
        )
        return inp
    if _requires_scatter_reduce_copy(inp, src):
        inp.copy_(_scatter_reduce_atomic(inp, dim, index, src, reduce, include_self))
        return inp
    assert (
        has_internal_overlapping(inp) != MemOverlap.Yes
    ), "Unsupported operation: trying to inplace write to an internally overlapping tensor."
    return _scatter_reduce_atomic_impl(inp, dim, index, src, reduce, include_self, inp)


def _scatter_reduce_atomic_out(inp, dim, index, src, reduce, include_self, out):
    if reduce == "prod" and _can_use_scatter_reduce_prod_fast_path(
        inp, dim, index, src, out
    ):
        if _is_same_tensor_view(out, inp):
            return _scatter_reduce_atomic_(out, dim, index, src, reduce, include_self)
        if _scatter_reduce_out_requires_temp(inp, out, src):
            out.copy_(_scatter_reduce_prod(inp, dim, index, src, include_self))
            return out
        out.copy_(inp)
        return _scatter_reduce_prod_fast_path(inp, index, src, include_self, out)
    if _is_same_tensor_view(out, inp):
        return _scatter_reduce_atomic_(out, dim, index, src, reduce, include_self)
    if _scatter_reduce_out_requires_temp(inp, out, src):
        out.copy_(_scatter_reduce_atomic(inp, dim, index, src, reduce, include_self))
        return out
    out.copy_(inp)
    return _scatter_reduce_atomic_impl(inp, dim, index, src, reduce, include_self, out)


def scatter_reduce(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS SCATTER_REDUCE")
    dim = _validate_scatter_reduce_args(inp, dim, index, src, reduce)
    if inp.ndim == 0:
        return _scatter_reduce_cpu_fallback(inp, dim, index, src, reduce, include_self)
    if reduce == "sum":
        return _scatter_reduce_sum(inp, dim, index, src, include_self)
    if reduce == "prod":
        return _scatter_reduce_prod(inp, dim, index, src, include_self)
    if reduce == "mean":
        return _scatter_reduce_mean(inp, dim, index, src, include_self)
    return _scatter_reduce_atomic(inp, dim, index, src, reduce, include_self)


def scatter_reduce_(inp, dim, index, src, reduce, *, include_self=True):
    logger.debug("GEMS SCATTER_REDUCE_")
    dim = _validate_scatter_reduce_args(inp, dim, index, src, reduce)
    if inp.ndim == 0:
        inp.copy_(
            _scatter_reduce_cpu_fallback(inp, dim, index, src, reduce, include_self)
        )
        return inp
    if reduce == "sum":
        return _scatter_reduce_sum_(inp, dim, index, src, include_self)
    if reduce in ("prod", "amax", "amin"):
        return _scatter_reduce_atomic_(inp, dim, index, src, reduce, include_self)
    res = scatter_reduce(inp, dim, index, src, reduce, include_self=include_self)
    inp.copy_(res)
    return inp


def scatter_reduce_out(inp, dim, index, src, reduce, *, include_self=True, out=None):
    logger.debug("GEMS SCATTER_REDUCE_OUT")
    dim = _validate_scatter_reduce_args(inp, dim, index, src, reduce)
    if inp.ndim == 0:
        out.copy_(
            _scatter_reduce_cpu_fallback(inp, dim, index, src, reduce, include_self)
        )
        return out
    if reduce == "sum":
        return _scatter_reduce_sum_out(inp, dim, index, src, include_self, out)
    if reduce == "mean":
        return _scatter_reduce_mean_out(inp, dim, index, src, include_self, out)
    return _scatter_reduce_atomic_out(inp, dim, index, src, reduce, include_self, out)
