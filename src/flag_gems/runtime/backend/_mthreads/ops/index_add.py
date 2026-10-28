# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_INDEX_OUT_OF_BOUNDS_MESSAGE = "0 <= index < self.size(dim)"
_META_BLOCK = 256
_META_INIT_BLOCK = 1024
_INT32_MAX = torch.iinfo(torch.int32).max
_CONTIGUOUS_GATHER_MIN_UPDATES = {
    torch.float16: 4 * (1 << 20),
    torch.bfloat16: 16 * (1 << 20),
    torch.float32: 4 * (1 << 20),
}
_STRIDED_TILE_ELEMS = 128
_STRIDED_LARGE_TILE_ELEMS = 512
_STRIDED_MIN_BLOCKS = 32
_STRIDED_MAX_BLOCK_S = 128
_VALIDATION_CACHE_LIMIT = 4
_validation_cache = {}
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _resolve_index_for_kernel(index):
    if index.is_neg():
        return torch.neg(torch._neg_view(index))
    return index


def _resolve_value_for_kernel(tensor):
    if tensor.is_neg():
        return torch.neg(torch._neg_view(tensor))
    if tensor.is_conj():
        return tensor.resolve_conj()
    return tensor


def _has_lazy_view(tensor):
    return tensor.is_neg() or tensor.is_conj()


def _has_inplace_alias(inp, index, src):
    return (
        src is inp
        or index is inp
        or torch._C._is_alias_of(inp, src)
        or torch._C._is_alias_of(inp, index)
    )


def _needs_native_semantic_fallback(inp, index, src, inplace):
    return (
        _has_lazy_view(inp)
        or _has_lazy_view(index)
        or _has_lazy_view(src)
        or (inplace and _has_inplace_alias(inp, index, src))
    )


def _native_index_add(inp, dim, index, src, alpha):
    return torch.ops.aten.index_add.default.redispatch(
        _FALLBACK_KEYSET, inp, dim, index, src, alpha=alpha
    )


def _native_index_add_(inp, dim, index, src, alpha):
    return torch.ops.aten.index_add_.default.redispatch(
        _FALLBACK_KEYSET, inp, dim, index, src, alpha=alpha
    )


def _run_native_semantic_fallback(inp, dim, index, src, alpha, inplace):
    if inplace and _has_inplace_alias(inp, index, src):
        return _native_index_add_(inp, dim, index, src, alpha)

    work_index = _resolve_index_for_kernel(index)
    work_src = _resolve_value_for_kernel(src)

    if inplace:
        if inp.is_neg():
            raw_inp = torch._neg_view(inp)
            _native_index_add_(raw_inp, dim, work_index, work_src, -alpha)
            return inp
        return _native_index_add_(inp, dim, work_index, work_src, alpha)

    work_inp = _resolve_value_for_kernel(inp)
    return _native_index_add(work_inp, dim, work_index, work_src, alpha)


def _metadata_fits_int32(dim_size, index_len):
    return dim_size <= _INT32_MAX and index_len < _INT32_MAX


@libentry()
@triton.jit
def _index_add_validate_kernel(
    status, index, index_len, upper_bound, BLOCK: tl.constexpr
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < index_len
    values = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)
    bad = mask & ((values < 0) | (values >= upper_bound))
    tl.atomic_or(status, tl.max(bad.to(tl.int32)), mask=tl.max(bad.to(tl.int32)) != 0)


def _validate_indices_once(index, upper_bound):
    if index.numel() == 0:
        return
    status = torch.zeros((1,), dtype=torch.int32, device=index.device)
    grid = (triton.cdiv(index.numel(), _META_BLOCK),)
    with torch_device_fn.device(index.device):
        _index_add_validate_kernel[grid](
            status, index, index.numel(), upper_bound, BLOCK=_META_BLOCK
        )
    if int(status.cpu().item()):
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)


def _ensure_valid_indices(index, upper_bound):
    if index.numel() == 0:
        return
    key = id(index)
    entry = _validation_cache.get(key)
    if entry is not None and entry[0] is index and entry[2] == upper_bound:
        if entry[1] == index._version:
            return
    _validate_indices_once(index, upper_bound)
    if len(_validation_cache) >= _VALIDATION_CACHE_LIMIT:
        _validation_cache.clear()
    _validation_cache[key] = (index, index._version, upper_bound)


def _normalize_dim(inp, dim):
    if inp.ndim == 0:
        raise AssertionError("Expected self to have at least one dimension")
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}], but got {dim}"
        )
    return dim % inp.ndim


def _validate_index_add_args(inp, dim, index, src):
    dim = _normalize_dim(inp, dim)
    if index.ndim != 1:
        raise AssertionError("Index is supposed to be a vector")
    if index.dtype not in (torch.int32, torch.int64):
        raise AssertionError("Expected dtype int32/int64 for index")
    if inp.dtype != src.dtype:
        raise AssertionError("Self and source should have the same dtype")
    if inp.device != src.device:
        raise AssertionError("Self and source should be on the same device")
    if index.device != inp.device:
        raise AssertionError("Index and self should be on the same device")
    if inp.ndim != src.ndim:
        raise AssertionError(
            "Self and source should have the same number of dimensions"
        )
    if not all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim):
        raise AssertionError("src.size(d) == self.size(d) for all dimensions d != dim")
    if index.numel() != src.size(dim):
        raise AssertionError(
            "The dimth dimension of source must have the same size as the length of index"
        )
    return dim


def _volume(shape):
    value = 1
    for item in shape:
        value *= int(item)
    return value


@libentry()
@triton.jit
def _index_add_generic_meta_init(head, status, head_len, BLOCK: tl.constexpr):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(head + offsets, 0, mask=offsets < head_len)
    if ext.program_id(0) == 0:
        tl.store(status, 0)


@libentry()
@triton.jit
def _index_add_generic_meta_build(
    head, next_positions, status, index, index_len, upper_bound, BLOCK: tl.constexpr
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < index_len
    values = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)
    bad = mask & ((values < 0) | (values >= upper_bound))
    any_bad = tl.max(bad.to(tl.int32))
    tl.atomic_or(status, any_bad, mask=any_bad != 0)
    valid = mask & ~bad
    safe = tl.where(valid, values, 0).to(tl.int32)
    old = tl.atomic_xchg(
        head + safe, (offsets + 1).to(tl.int32), mask=valid, sem="relaxed"
    )
    tl.store(next_positions + offsets, old, mask=valid)
    single = valid & (old == 0)
    compare = tl.where(single, (offsets + 1).to(tl.int32), -2147483648)
    value = tl.where(single, -(offsets + 1).to(tl.int32), 0)
    tl.atomic_cas(head + safe, compare, value)


@libentry()
@triton.jit
def _index_add_generic_gather_kernel(
    out_ptr,
    inp_ptr,
    src_ptr,
    head_ptr,
    next_ptr,
    status_ptr,
    alpha,
    dim_size,
    index_len,
    suffix_size,
    suffix_blocks,
    dim_blocks,
    out_pstride,
    out_dstride,
    src_pstride,
    src_dstride,
    BLOCK_R: tl.constexpr,
    BLOCK_S: tl.constexpr,
    INPLACE: tl.constexpr,
):
    pid = ext.program_id(0).to(tl.int64)
    suffix_block = pid % suffix_blocks
    tmp = pid // suffix_blocks
    receiver_block = tmp % dim_blocks
    prefix = tmp // dim_blocks
    if tl.load(status_ptr) == 0:
        receivers = receiver_block * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
        suffix = suffix_block * BLOCK_S + tl.arange(0, BLOCK_S).to(tl.int64)
        receiver_mask = receivers < dim_size
        suffix_mask = suffix < suffix_size
        head = tl.load(head_ptr + receivers, mask=receiver_mask, other=0)
        touched = receiver_mask & (head != 0)
        position = tl.where(touched, tl.abs(head).to(tl.int64) - 1, 0)
        active_mask = touched[:, None] & suffix_mask[None, :]
        source_base = prefix * src_pstride + position[:, None] * src_dstride
        summed = tl.load(
            src_ptr + source_base + suffix[None, :], mask=active_mask, other=0.0
        ).to(tl.float32)
        multiple = touched & (head > 0)
        current = tl.load(next_ptr + position, mask=multiple, other=0).to(tl.int64)
        active = multiple & (current != 0)
        steps = 0
        while (tl.max(active.to(tl.int32)) > 0) & (steps < index_len):
            positive = current > 0
            position2 = tl.where(positive, current - 1, -current - 1)
            load_mask = active[:, None] & suffix_mask[None, :]
            source_base2 = prefix * src_pstride + position2[:, None] * src_dstride
            summed += tl.load(
                src_ptr + source_base2 + suffix[None, :], mask=load_mask, other=0.0
            ).to(tl.float32)
            current_index = tl.where(positive, current - 1, -current - 1)
            current = tl.where(
                active, tl.load(next_ptr + current_index, mask=active, other=0), 0
            ).to(tl.int64)
            active = active & (current != 0)
            steps += 1
        out_base = prefix * out_pstride + receivers[:, None] * out_dstride
        old = tl.load(
            inp_ptr + out_base + suffix[None, :],
            mask=receiver_mask[:, None] & suffix_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        result = old + summed * alpha
        store_mask = (
            touched[:, None] & suffix_mask[None, :]
            if INPLACE
            else receiver_mask[:, None] & suffix_mask[None, :]
        )
        tl.store(
            out_ptr + out_base + suffix[None, :],
            result.to(out_ptr.dtype.element_ty),
            mask=store_mask,
        )


@libentry()
@triton.jit
def _index_add_generic_gather_p_kernel(
    out_ptr,
    inp_ptr,
    src_ptr,
    head_ptr,
    next_ptr,
    status_ptr,
    alpha,
    dim_size,
    index_len,
    prefix_size,
    dim_blocks,
    out_pstride,
    out_dstride,
    src_pstride,
    src_dstride,
    BLOCK_P: tl.constexpr,
    BLOCK_R: tl.constexpr,
    INPLACE: tl.constexpr,
):
    pid = ext.program_id(0).to(tl.int64)
    receiver_block = pid % dim_blocks
    prefix_block = pid // dim_blocks
    receivers = receiver_block * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)
    prefixes = prefix_block * BLOCK_P + tl.arange(0, BLOCK_P).to(tl.int64)
    receiver_mask = receivers < dim_size
    prefix_mask = prefixes < prefix_size
    if tl.load(status_ptr) == 0:
        head = tl.load(head_ptr + receivers, mask=receiver_mask, other=0)
        touched = receiver_mask & (head != 0)
        position = tl.where(touched, tl.abs(head).to(tl.int64) - 1, 0)
        active_mask = prefix_mask[:, None] & touched[None, :]
        source_base = prefixes[:, None] * src_pstride + position[None, :] * src_dstride
        summed = tl.load(src_ptr + source_base, mask=active_mask, other=0.0).to(
            tl.float32
        )
        multiple = touched & (head > 0)
        current = tl.load(next_ptr + position, mask=multiple, other=0).to(tl.int64)
        active = multiple & (current != 0)
        steps = 0
        while (tl.max(active.to(tl.int32)) > 0) & (steps < index_len):
            positive = current > 0
            position2 = tl.where(positive, current - 1, -current - 1)
            load_mask = prefix_mask[:, None] & active[None, :]
            source_base2 = (
                prefixes[:, None] * src_pstride + position2[None, :] * src_dstride
            )
            summed += tl.load(src_ptr + source_base2, mask=load_mask, other=0.0).to(
                tl.float32
            )
            current_index = tl.where(positive, current - 1, -current - 1)
            current = tl.where(
                active, tl.load(next_ptr + current_index, mask=active, other=0), 0
            ).to(tl.int64)
            active = active & (current != 0)
            steps += 1
        out_base = prefixes[:, None] * out_pstride + receivers[None, :] * out_dstride
        old = tl.load(
            inp_ptr + out_base,
            mask=prefix_mask[:, None] & receiver_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        result = old + summed * alpha
        store_mask = (
            prefix_mask[:, None] & touched[None, :]
            if INPLACE
            else prefix_mask[:, None] & receiver_mask[None, :]
        )
        tl.store(
            out_ptr + out_base, result.to(out_ptr.dtype.element_ty), mask=store_mask
        )


def _run_generic_gather(inp, dim, index, src, alpha, inplace):
    dim_size = inp.size(dim)
    index_len = index.numel()
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_size = _volume(src.shape[:dim])
    metadata = torch.empty(
        dim_size + index_len + 1, dtype=torch.int32, device=inp.device
    )
    head = metadata[:dim_size]
    next_positions = metadata[dim_size : dim_size + index_len]
    status = metadata[dim_size + index_len :]
    with torch_device_fn.device(inp.device):
        _index_add_generic_meta_init[
            (triton.cdiv(max(dim_size, 1), _META_INIT_BLOCK),)
        ](head, status, dim_size, BLOCK=_META_INIT_BLOCK)
        _index_add_generic_meta_build[
            (triton.cdiv(index_len, _META_BLOCK),)
        ](
            head,
            next_positions,
            status,
            index,
            index_len,
            dim_size,
            BLOCK=_META_BLOCK,
        )
        out = inp if inplace else torch.empty_like(inp)
        if suffix_size == 1:
            dim_blocks = triton.cdiv(dim_size, 256)
            _index_add_generic_gather_p_kernel[
                (dim_blocks * triton.cdiv(prefix_size, 4),)
            ](
                out,
                inp,
                src,
                head,
                next_positions,
                status,
                alpha,
                dim_size,
                index_len,
                prefix_size,
                dim_blocks,
                dim_size,
                1,
                index_len,
                1,
                BLOCK_P=4,
                BLOCK_R=256,
                INPLACE=inplace,
                num_warps=4,
            )
        else:
            block_s = min(512, triton.next_power_of_2(suffix_size))
            block_r = max(1, min(256, triton.next_power_of_2(dim_size)))
            suffix_blocks = triton.cdiv(suffix_size, block_s)
            dim_blocks = triton.cdiv(dim_size, block_r)
            _index_add_generic_gather_kernel[
                (prefix_size * dim_blocks * suffix_blocks,)
            ](
                out,
                inp,
                src,
                head,
                next_positions,
                status,
                alpha,
                dim_size,
                index_len,
                suffix_size,
                suffix_blocks,
                dim_blocks,
                dim_size * suffix_size,
                suffix_size,
                index_len * suffix_size,
                suffix_size,
                BLOCK_R=block_r,
                BLOCK_S=block_s,
                INPLACE=inplace,
                num_warps=4,
            )
    return out


def _should_use_contiguous_gather(inp, dim, inplace):
    updates = inp.numel()
    threshold = _CONTIGUOUS_GATHER_MIN_UPDATES[inp.dtype]
    if inp.dtype == torch.bfloat16 and inplace:
        threshold *= 2
    return updates >= threshold


@libentry()
@triton.jit
def _index_add_strided_kernel(
    out_ptr,
    index_ptr,
    src_ptr,
    alpha,
    index_len,
    suffix_size,
    dim_size,
    out_pstride,
    out_dstride,
    src_pstride,
    src_dstride,
    BLOCK_N: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    prefix = ext.program_id(0).to(tl.int64)
    ns = (ext.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    ss = (ext.program_id(2) * BLOCK_S + tl.arange(0, BLOCK_S)).to(tl.int64)
    n_mask = ns < index_len
    mask = n_mask[:, None] & (ss < suffix_size)[None, :]

    receivers = tl.load(index_ptr + ns, mask=n_mask, other=0).to(tl.int64)
    receivers = tl.minimum(tl.maximum(receivers, 0), dim_size - 1)

    values = tl.load(
        src_ptr + prefix * src_pstride + ns[:, None] * src_dstride + ss[None, :],
        mask=mask,
        other=0.0,
    )
    tl.atomic_add(
        out_ptr
        + prefix * out_pstride
        + receivers[:, None] * out_dstride
        + ss[None, :],
        alpha * values,
        mask=mask,
        sem="relaxed",
    )


def _plan_strided_blocks(index_len, suffix_size, prefix_size):
    block_s = min(
        triton.next_power_of_2(max(suffix_size, 1)), _STRIDED_MAX_BLOCK_S
    )
    block_n = max(
        1,
        min(
            _STRIDED_LARGE_TILE_ELEMS // block_s,
            triton.next_power_of_2(max(index_len, 1)),
        ),
    )
    blocks = (
        prefix_size
        * triton.cdiv(index_len, block_n)
        * triton.cdiv(max(suffix_size, 1), block_s)
    )
    if blocks < _STRIDED_MIN_BLOCKS:
        block_n = max(1, min(_STRIDED_TILE_ELEMS // block_s, block_n))
    return block_n, block_s


def _run_strided_path(inp, dim, index, src, alpha, inplace):
    dim = _validate_index_add_args(inp, dim, index, src)
    dim_size = inp.size(dim)
    suffix_size = _volume(inp.shape[dim + 1 :])
    prefix_size = _volume(inp.shape[:dim])
    index_len = index.numel()

    if index_len == 0:
        return inp if inplace else inp.clone()

    work_src = src if src.is_contiguous() else src.contiguous()
    work_index = index if index.is_contiguous() else index.contiguous()
    _ensure_valid_indices(work_index, dim_size)

    copy_back = False
    if inplace:
        out = inp if inp.is_contiguous() else inp.contiguous()
        copy_back = out is not inp
    else:
        out = inp.clone() if inp.is_contiguous() else inp.contiguous()

    block_n, block_s = _plan_strided_blocks(index_len, suffix_size, prefix_size)
    grid = (
        prefix_size,
        triton.cdiv(index_len, block_n),
        triton.cdiv(suffix_size, block_s),
    )
    with torch_device_fn.device(inp.device):
        _index_add_strided_kernel[grid](
            out,
            work_index,
            work_src,
            alpha,
            index_len,
            suffix_size,
            dim_size,
            dim_size * suffix_size,
            suffix_size,
            index_len * suffix_size,
            suffix_size,
            BLOCK_N=block_n,
            BLOCK_S=block_s,
            num_warps=4,
        )

    if copy_back:
        inp.copy_(out)
    return inp if inplace else out


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_add"))
@triton.jit
def _index_add_relaxed_kernel(
    out_ptr,
    index_ptr,
    src_ptr,
    M,
    N,
    alpha,
    inp_len,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(axis=0)
    pid_n = ext.program_id(axis=1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask & col_mask
    receivers = tl.load(index_ptr + cols, mask=col_mask, other=0)
    values = tl.load(src_ptr + rows * N + cols, mask=mask, other=0.0)
    tl.atomic_add(
        out_ptr + rows * inp_len + receivers,
        alpha * values,
        mask=mask,
        sem="relaxed",
    )


def _try_run_contiguous_index_add_path(inp, dim, index, src, alpha, inplace):
    if inp.dtype not in _CONTIGUOUS_GATHER_MIN_UPDATES or src.dtype != inp.dtype:
        return None
    if not (inp.is_contiguous() and src.is_contiguous() and index.is_contiguous()):
        return None
    dim = _normalize_dim(inp, dim)
    if dim != inp.ndim - 1:
        return None
    dim = _validate_index_add_args(inp, dim, index, src)
    if index.numel() == 0:
        return inp if inplace else inp.clone()
    if not _metadata_fits_int32(inp.size(dim), index.numel()):
        return None
    _ensure_valid_indices(index, inp.size(dim))
    if _should_use_contiguous_gather(inp, dim, inplace):
        return _run_generic_gather(inp, dim, index, src, alpha, inplace)
    out = inp if inplace else inp.clone()
    rows = src.numel() // index.numel()
    grid = lambda meta: (
        triton.cdiv(rows, meta["BLOCK_M"]),
        triton.cdiv(index.numel(), meta["BLOCK_N"]),
    )
    with torch_device_fn.device(inp.device):
        _index_add_relaxed_kernel[grid](
            out, index, src, rows, index.numel(), alpha, inp.size(dim)
        )
    return out


def index_add(inp, dim, index, src, alpha=1):
    if _needs_native_semantic_fallback(inp, index, src, False):
        return _run_native_semantic_fallback(inp, dim, index, src, alpha, False)
    out = _try_run_contiguous_index_add_path(inp, dim, index, src, alpha, False)
    if out is not None:
        return out
    logger.debug("GEMS_MTHREADS INDEX_ADD")
    return _run_strided_path(inp, dim, index, src, alpha, False)


def index_add_(inp, dim, index, src, alpha=1):
    if _needs_native_semantic_fallback(inp, index, src, True):
        return _run_native_semantic_fallback(inp, dim, index, src, alpha, True)
    out = _try_run_contiguous_index_add_path(inp, dim, index, src, alpha, True)
    if out is not None:
        return out
    logger.debug("GEMS_MTHREADS INDEX_ADD_")
    return _run_strided_path(inp, dim, index, src, alpha, True)
