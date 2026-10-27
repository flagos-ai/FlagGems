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
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_INDEX_OUT_OF_BOUNDS_MESSAGE = "0 <= index < self.size(dim)"
_UNIQUE_DETECTOR_BLOCK = 256
_UNIQUE_DETECTOR_MAX_BITMAP_BYTES = 64 * 1024 * 1024
_UNIQUE_PATH_MIN_SUFFIX = 32
# The detector allocates/clears a receiver bitmap and synchronizes one status
# scalar. Require enough contiguous work to amortize that fixed cost; smaller
# workloads remain on the existing atomic path.
_UNIQUE_PATH_MIN_UPDATES = 1 << 23
_ALL_SAME_PATH_MIN_SUFFIX = 32
_ALL_SAME_PATH_MIN_UPDATES = 1 << 20
_GROUPED_PATH_MIN_SUFFIX = 32
_GROUPED_PATH_MIN_UPDATES = 1 << 20
_GROUPED_REUSE_THRESHOLD = 2
_GROUPED_MAX_RECEIVERS = 8192
_LINKED_GROUP_SENTINEL = -1
_LINKED_GROUP_INIT_BLOCK = 1024
_INT32_MAX = torch.iinfo(torch.int32).max
# S5000 crossover thresholds where receiver-owned gather amortizes metadata
# construction relative to relaxed atomic scatter for contiguous last-dimension
# updates.
_CONTIGUOUS_GATHER_MIN_UPDATES = {
    torch.float16: 4 * (1 << 20),
    torch.bfloat16: 16 * (1 << 20),
    torch.float32: 4 * (1 << 20),
}
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def _resolve_index_for_kernel(index):
    # A contiguous lazy-negative tensor still exposes the un-negated storage
    # to a pointer-based Triton kernel. Materialize only that exceptional case.
    # Calling resolve_neg() from inside use_gems() re-enters FlagGems' Python
    # override and can negate the logical value twice. Toggle the metadata bit
    # off first, then explicitly negate the ordinary physical view.
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


def _native_clone_contiguous(tensor):
    out = torch.empty_strided(
        tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
    )
    torch.ops.aten.copy_.default.redispatch(_FALLBACK_KEYSET, out, tensor, False)
    return out


def _metadata_fits_int32(dim_size, index_len):
    # Source positions use one-based encoding in the linked metadata.
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
    grid = (triton.cdiv(index.numel(), _UNIQUE_DETECTOR_BLOCK),)
    with torch_device_fn.device(index.device):
        _index_add_validate_kernel[grid](
            status, index, index.numel(), upper_bound, BLOCK=_UNIQUE_DETECTOR_BLOCK
        )
    if int(status.cpu().item()):
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)


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
            (triton.cdiv(max(dim_size, 1), _LINKED_GROUP_INIT_BLOCK),)
        ](head, status, dim_size, BLOCK=_LINKED_GROUP_INIT_BLOCK)
        _index_add_generic_meta_build[
            (triton.cdiv(index_len, _UNIQUE_DETECTOR_BLOCK),)
        ](
            head,
            next_positions,
            status,
            index,
            index_len,
            dim_size,
            BLOCK=_UNIQUE_DETECTOR_BLOCK,
        )
        out = inp if inplace else inp.clone()
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
    if int(status.cpu().item()):
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)
    return out


def _should_use_contiguous_gather(inp, dim, inplace):
    updates = inp.numel()
    threshold = _CONTIGUOUS_GATHER_MIN_UPDATES[inp.dtype]
    if inp.dtype == torch.bfloat16 and inplace:
        threshold *= 2
    return updates >= threshold


def _try_run_contiguous_index_add_path(inp, dim, index, src, alpha, inplace):
    if inp.dtype not in _CONTIGUOUS_GATHER_MIN_UPDATES or src.dtype != inp.dtype:
        return None
    if _needs_native_semantic_fallback(inp, index, src, inplace):
        return None
    if not (inp.is_contiguous() and src.is_contiguous() and index.is_contiguous()):
        return None
    dim = _validate_bf16_index_add_args(inp, dim, index, src)
    if index.numel() == 0:
        return inp if inplace else inp.clone()
    if dim != inp.ndim - 1:
        return None
    if not _metadata_fits_int32(inp.size(dim), index.numel()):
        return None
    if _should_use_contiguous_gather(inp, dim, inplace):
        return _run_generic_gather(inp, dim, index, src, alpha, inplace)
    _validate_indices_once(index, inp.size(dim))
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


def _normalize_bf16_dim(inp, dim):
    if inp.ndim == 0:
        raise AssertionError("Expected self to have at least one dimension")
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}], but got {dim}"
        )
    return dim % inp.ndim


def _validate_bf16_index_add_args(inp, dim, index, src):
    dim = _normalize_bf16_dim(inp, dim)
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


def _can_use_bf16_unique_path(inp, dim, index, src):
    if not (
        src.numel() > 0
        and inp.ndim == src.ndim
        and 0 <= dim < inp.ndim
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and inp.dtype == src.dtype == torch.bfloat16
        and index.numel() == src.size(dim)
        and inp.is_contiguous()
        and src.is_contiguous()
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
    ):
        return False
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_size = _volume(src.shape[:dim])
    updates = prefix_size * index.numel() * suffix_size
    return (
        suffix_size >= _UNIQUE_PATH_MIN_SUFFIX
        and updates >= _UNIQUE_PATH_MIN_UPDATES
        and inp.size(dim) * 4 <= _UNIQUE_DETECTOR_MAX_BITMAP_BYTES
    )


def _can_use_bf16_all_same_path(inp, dim, index, src):
    if not (
        src.numel() > 0
        and inp.ndim == src.ndim
        and 0 <= dim < inp.ndim
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and inp.dtype == src.dtype == torch.bfloat16
        and index.numel() == src.size(dim)
        and inp.is_contiguous()
        and src.is_contiguous()
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
    ):
        return False
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_size = _volume(src.shape[:dim])
    updates = prefix_size * index.numel() * suffix_size
    return (
        suffix_size >= _ALL_SAME_PATH_MIN_SUFFIX
        and updates >= _ALL_SAME_PATH_MIN_UPDATES
    )


def _can_use_bf16_grouped_path(inp, dim, index, src):
    if not (
        src.numel() > 0
        and inp.ndim == src.ndim
        and 0 <= dim < inp.ndim
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and inp.dtype == src.dtype == torch.bfloat16
        and index.numel() == src.size(dim)
        and inp.is_contiguous()
        and src.is_contiguous()
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
    ):
        return False
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_size = _volume(src.shape[:dim])
    updates = prefix_size * index.numel() * suffix_size
    return (
        suffix_size >= _GROUPED_PATH_MIN_SUFFIX
        and updates >= _GROUPED_PATH_MIN_UPDATES
        and inp.size(dim) <= _GROUPED_MAX_RECEIVERS
    )


@libentry()
@triton.jit
def _index_add_unique_detector_kernel(
    status,
    bitmap,
    index,
    index_len,
    upper_bound,
    BLOCK: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < index_len
    values = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)
    negative = mask & (values < 0)
    upper = mask & (values >= upper_bound)
    valid = mask & ~negative & ~upper

    has_negative = tl.max(negative.to(tl.int32))
    has_upper = tl.max(upper.to(tl.int32))

    safe_values = tl.where(valid, values, 0)
    previous = tl.atomic_add(bitmap + safe_values, 1, mask=valid)
    duplicate = valid & (previous > 0)
    has_duplicate = tl.max(duplicate.to(tl.int32))
    status_bits = has_negative + has_upper * 2 + has_duplicate * 4
    tl.atomic_or(status, status_bits, mask=status_bits != 0)


@libentry()
@triton.jit
def _index_add_all_same_detector_kernel(
    status,
    index,
    index_len,
    upper_bound,
    BLOCK: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < index_len
    values = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)
    first = tl.load(index).to(tl.int64)
    negative = mask & (values < 0)
    upper = mask & (values >= upper_bound)
    different = mask & (values != first)
    status_bits = tl.max(negative.to(tl.int32))
    status_bits = status_bits + tl.max(upper.to(tl.int32)) * 2
    status_bits = status_bits + tl.max(different.to(tl.int32)) * 4
    tl.atomic_or(status, status_bits, mask=status_bits != 0)


@libentry()
@triton.jit
def _index_add_linked_dense_init_kernel(
    head,
    meta,
    upper_bound,
    SENTINEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(head + offsets, SENTINEL, mask=offsets < upper_bound)
    tl.store(meta + offsets, 0, mask=offsets < 2)


@libentry()
@triton.jit
def _index_add_linked_dense_builder_kernel(
    head,
    next_positions,
    touched_receivers,
    meta,
    index,
    index_len,
    upper_bound,
    SENTINEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < index_len
    values = tl.load(index + offsets, mask=mask, other=0).to(tl.int64)
    negative = mask & (values < 0)
    upper = mask & (values >= upper_bound)
    valid = mask & ~negative & ~upper

    status_bits = tl.max(negative.to(tl.int32))
    status_bits = status_bits + tl.max(upper.to(tl.int32)) * 2
    tl.atomic_or(meta, status_bits, mask=status_bits != 0)

    safe_values = tl.where(valid, values, 0)
    old_head = tl.atomic_xchg(
        head + safe_values,
        offsets.to(tl.int32),
        mask=valid,
        sem="acq_rel",
    )
    tl.store(next_positions + offsets, old_head, mask=valid)

    first_touch = valid & (old_head == SENTINEL)
    counter_ptrs = meta + 1 + offsets * 0
    touched_rank = tl.atomic_add(
        counter_ptrs,
        tl.full((BLOCK,), 1, dtype=tl.int32),
        mask=first_touch,
        sem="acq_rel",
    )
    tl.store(touched_receivers + touched_rank, safe_values, mask=first_touch)


def _validate_and_detect_unique(index, upper_bound):
    """Validate receivers and report uniqueness in one GPU pass.

    The bitmap is intentionally bounded.  Large receiver dimensions use the
    existing validation and atomic scatter path instead of allocating an
    unbounded auxiliary tensor.
    """
    if index.numel() == 0:
        return True
    bitmap = torch.zeros((upper_bound,), dtype=torch.int32, device=index.device)
    status = torch.zeros((1,), dtype=torch.int32, device=index.device)
    grid = (triton.cdiv(index.numel(), _UNIQUE_DETECTOR_BLOCK),)
    with torch_device_fn.device(index.device):
        _index_add_unique_detector_kernel[grid](
            status,
            bitmap,
            index,
            index.numel(),
            upper_bound,
            BLOCK=_UNIQUE_DETECTOR_BLOCK,
        )
    status_bits = int(status.cpu().item())
    if status_bits & 0x3:
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)
    return not (status_bits & 0x4)


def _validate_and_detect_all_same(index, upper_bound):
    if index.numel() == 0:
        return False
    status = torch.zeros((1,), dtype=torch.int32, device=index.device)
    grid = (triton.cdiv(index.numel(), _UNIQUE_DETECTOR_BLOCK),)
    with torch_device_fn.device(index.device):
        _index_add_all_same_detector_kernel[grid](
            status,
            index,
            index.numel(),
            upper_bound,
            BLOCK=_UNIQUE_DETECTOR_BLOCK,
        )
    status_bits = int(status.cpu().item())
    if status_bits & 0x3:
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)
    return not (status_bits & 0x4)


@libentry()
@triton.jit
def _index_add_unique_contiguous_suffix_kernel(
    out,
    index,
    src,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    ALPHA_ONE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    suffix_pid = ext.program_id(0)
    index_pid = ext.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    prefix_pid = ext.program_id(2)
    cols = suffix_pid * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    index_mask = index_pid < index_len
    mask = index_mask & (cols < suffix_size)
    receiver = tl.load(index + index_pid, mask=index_mask, other=0).to(tl.int32)
    src_base = (prefix_pid * index_len + index_pid) * suffix_size
    dst_base = (prefix_pid * out_dim + receiver) * suffix_size
    src_ptrs = src + src_base + cols
    dst_ptrs = out + dst_base + cols
    values = tl.load(src_ptrs, mask=mask, other=0.0)
    current = tl.load(dst_ptrs, mask=mask, other=0.0)
    update = values if ALPHA_ONE else values * alpha
    tl.store(dst_ptrs, current + update, mask=mask)


def _run_bf16_unique_path(out, dim, index, src, alpha):
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_size = _volume(src.shape[:dim])
    block_m = 4
    # S5000's BF16 direct path is resource-bound beyond a 256-wide suffix
    # tile; splitting wider suffixes improves occupancy and measured kernel
    # latency without changing addressing or memory traffic.
    block_n = min(256, triton.next_power_of_2(suffix_size))
    alpha_is_one = alpha == 1
    grid = (
        triton.cdiv(suffix_size, block_n),
        triton.cdiv(index.numel(), block_m),
        prefix_size,
    )
    with torch_device_fn.device(out.device):
        _index_add_unique_contiguous_suffix_kernel[grid](
            out,
            index,
            src,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            ALPHA_ONE=alpha_is_one,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
    return out


@libentry()
@triton.jit
def _index_add_all_same_suffix_kernel(
    out,
    index,
    src,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Reduce an all-same receiver group before updating the destination.

    One program owns one prefix and one contiguous suffix tile.  The complete
    index dimension is reduced in registers, replacing ``index_len`` BF16
    atomic RMW operations per destination element with one load/add/store.
    """
    suffix_pid = ext.program_id(0)
    prefix_pid = ext.program_id(1)
    cols = suffix_pid * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    col_mask = cols < suffix_size
    receiver = tl.load(index).to(tl.int64)
    summed = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_start in tl.range(0, index_len, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)[:, None]
        k_mask = k < index_len
        mask = k_mask & col_mask
        src_base = (prefix_pid * index_len + k) * suffix_size
        src_ptrs = src + src_base + cols
        values = tl.load(src_ptrs, mask=mask, other=0.0).to(tl.float32)
        summed += tl.sum(values, axis=0)
    dst_base = (prefix_pid * out_dim + receiver) * suffix_size
    dst_ptrs = out + dst_base + cols
    current = tl.load(dst_ptrs, mask=col_mask, other=0.0).to(tl.float32)
    tl.store(
        dst_ptrs, (current + summed * alpha).to(out.dtype.element_ty), mask=col_mask
    )


def _run_bf16_all_same_path(out, dim, index, src, alpha):
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_size = _volume(src.shape[:dim])
    # Narrow tiles keep the K reduction's live register footprint bounded.
    block_n = min(32, triton.next_power_of_2(suffix_size))
    block_k = min(128, triton.next_power_of_2(index.numel()))
    grid = (triton.cdiv(suffix_size, block_n), prefix_size)
    with torch_device_fn.device(out.device):
        _index_add_all_same_suffix_kernel[grid](
            out,
            index,
            src,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
        )
    return out


def _build_bf16_linked_dense_groups(index, upper_bound):
    """Build receiver adjacency lists in one index pass for BF16 duplicates."""
    index_len = index.numel()
    head = torch.empty_strided(
        (upper_bound,), (1,), dtype=torch.int32, device=index.device
    )
    next_positions = torch.empty_strided(
        (index_len,), (1,), dtype=torch.int32, device=index.device
    )
    touched_receivers = torch.empty_strided(
        (index_len,), (1,), dtype=torch.int64, device=index.device
    )
    meta = torch.empty_strided((2,), (1,), dtype=torch.int32, device=index.device)
    init_grid = (triton.cdiv(max(upper_bound, 2), _LINKED_GROUP_INIT_BLOCK),)
    build_grid = (triton.cdiv(index_len, _UNIQUE_DETECTOR_BLOCK),)
    with torch_device_fn.device(index.device):
        _index_add_linked_dense_init_kernel[init_grid](
            head,
            meta,
            upper_bound,
            SENTINEL=_LINKED_GROUP_SENTINEL,
            BLOCK=_LINKED_GROUP_INIT_BLOCK,
        )
        _index_add_linked_dense_builder_kernel[build_grid](
            head,
            next_positions,
            touched_receivers,
            meta,
            index,
            index_len,
            upper_bound,
            SENTINEL=_LINKED_GROUP_SENTINEL,
            BLOCK=_UNIQUE_DETECTOR_BLOCK,
        )
    meta_cpu = meta.cpu()
    status_bits = int(meta_cpu[0].item())
    if status_bits & 0x3:
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)
    touched_count = int(meta_cpu[1].item())
    return head, next_positions, touched_receivers, touched_count


@libentry()
@triton.jit
def _index_add_linked_grouped_suffix_kernel(
    out,
    touched_receivers,
    head,
    next_positions,
    src,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    ALPHA_ONE: tl.constexpr,
    SENTINEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Receiver-owned reduction over linked source-position groups."""
    suffix_pid = ext.program_id(0)
    group_pid = ext.program_id(1)
    prefix_pid = ext.program_id(2)
    cols = suffix_pid * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < suffix_size

    receiver = tl.load(touched_receivers + group_pid).to(tl.int64)
    pos = tl.load(head + receiver).to(tl.int32)
    summed = tl.zeros((BLOCK_N,), dtype=tl.float32)
    steps = 0
    while (pos != SENTINEL) & (steps < index_len):
        src_base = (prefix_pid * index_len + pos.to(tl.int64)) * suffix_size
        values = tl.load(src + src_base + cols, mask=col_mask, other=0.0).to(tl.float32)
        summed += values
        pos = tl.load(next_positions + pos).to(tl.int32)
        steps += 1

    dst_base = (prefix_pid * out_dim + receiver) * suffix_size
    dst_ptrs = out + dst_base + cols
    current = tl.load(dst_ptrs, mask=col_mask, other=0.0).to(tl.float32)
    update = summed if ALPHA_ONE else summed * alpha
    tl.store(dst_ptrs, (current + update).to(out.dtype.element_ty), mask=col_mask)


def _run_bf16_linked_grouped_path(
    out,
    dim,
    index,
    src,
    alpha,
    head,
    next_positions,
    touched_receivers,
    touched_count,
):
    suffix_size = _volume(src.shape[dim + 1 :])
    prefix_size = _volume(src.shape[:dim])
    block_n = min(512, triton.next_power_of_2(suffix_size))
    grid = (triton.cdiv(suffix_size, block_n), touched_count, prefix_size)
    with torch_device_fn.device(out.device):
        _index_add_linked_grouped_suffix_kernel[grid](
            out,
            touched_receivers,
            head,
            next_positions,
            src,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            ALPHA_ONE=alpha == 1,
            SENTINEL=_LINKED_GROUP_SENTINEL,
            BLOCK_N=block_n,
        )
    return out


def _try_run_bf16_receiver_owned_path(out, dim, index, src, alpha):
    (
        head,
        next_positions,
        touched_receivers,
        touched_count,
    ) = _build_bf16_linked_dense_groups(index, out.size(dim))
    if touched_count == 1:
        return _run_bf16_all_same_path(out, dim, index, src, alpha), True
    if touched_count == index.numel():
        return _run_bf16_unique_path(out, dim, index, src, alpha), True
    if index.numel() < touched_count * _GROUPED_REUSE_THRESHOLD:
        return out, False
    return (
        _run_bf16_linked_grouped_path(
            out,
            dim,
            index,
            src,
            alpha,
            head,
            next_positions,
            touched_receivers,
            touched_count,
        ),
        True,
    )


@libentry()
@triton.heuristics(runtime.get_heuristic_config("index_add"))
@triton.jit
def index_add_kernel(
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
    """
    Kernel for index_add operation with autotune.

    After dim_compress, tensors are reshaped so that:
    - inp has shape (M, inp_len) where inp_len is the size of target dimension
    - src has shape (M, N) where N is the size of index

    For each row m and each index position n:
        out[m, index[n]] += alpha * src[m, n]
    """
    pid_m = ext.program_id(axis=0)
    pid_n = ext.program_id(axis=1)

    # Calculate row and column offsets
    rows_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    # Create masks
    rows_mask = rows_offset < M
    cols_mask = cols_offset < N
    block_mask = rows_mask & cols_mask

    # Load indices for this block of columns
    cur_indices = tl.load(index_ptr + cols_offset, mask=cols_mask, other=0)

    # Calculate offsets into inp/out (which has shape M x inp_len)
    inp_off = rows_offset * inp_len + cur_indices

    # Calculate offsets into src (which has shape M x N)
    src_off = rows_offset * N + cols_offset

    # Load source values
    cur_src = tl.load(src_ptr + src_off, mask=block_mask, other=0.0)

    # Use atomic_add to correctly handle repeated indices in index,
    # aligned with the common op (src/flag_gems/ops/index_add.py).
    # When multiple source elements map to the same output position (duplicate
    # indices), plain load-store would cause race conditions or lost updates.
    # atomic_add guarantees all contributions are accumulated correctly.
    tl.atomic_add(out_ptr + inp_off, alpha * cur_src, mask=block_mask)


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


def _try_run_bf16_index_add_path(inp, dim, index, src, alpha, inplace):
    """Run a validated BF16 receiver-owned path, or leave the master fallback."""
    if inp.dtype != torch.bfloat16 or src.dtype != torch.bfloat16:
        return None

    dim = _validate_bf16_index_add_args(inp, dim, index, src)
    if _needs_native_semantic_fallback(inp, index, src, inplace):
        return _run_native_semantic_fallback(inp, dim, index, src, alpha, inplace)

    if src.numel() == 0:
        return inp if inplace else _native_clone_contiguous(inp)

    if not _metadata_fits_int32(inp.size(dim), index.numel()):
        return None

    input_src_alias = torch._C._is_alias_of(inp, src)
    input_index_alias = inplace and torch._C._is_alias_of(inp, index)
    if input_src_alias or input_index_alias:
        return None

    use_grouped_path = _can_use_bf16_grouped_path(inp, dim, index, src)
    use_unique_path = _can_use_bf16_unique_path(inp, dim, index, src)
    use_all_same_path = _can_use_bf16_all_same_path(inp, dim, index, src)
    if not (use_grouped_path or use_unique_path or use_all_same_path):
        return None

    # Optimized kernels require the same contiguous logical layout as the
    # validated receiver-owned implementations. Other layouts use the master
    # MTHREADS path below.
    work_inp = inp.contiguous()
    work_index = _resolve_index_for_kernel(index).contiguous()
    work_src = src.contiguous()
    inp_len = work_inp.size(dim)

    if use_grouped_path:
        out = work_inp if inplace else _native_clone_contiguous(work_inp)
        out, handled = _try_run_bf16_receiver_owned_path(
            out, dim, work_index, work_src, alpha
        )
        if handled:
            return out
        return None

    if use_all_same_path and _validate_and_detect_all_same(work_index, inp_len):
        out = work_inp if inplace else _native_clone_contiguous(work_inp)
        return _run_bf16_all_same_path(out, dim, work_index, work_src, alpha)

    if use_unique_path and _validate_and_detect_unique(work_index, inp_len):
        out = work_inp if inplace else _native_clone_contiguous(work_inp)
        return _run_bf16_unique_path(out, dim, work_index, work_src, alpha)

    return None


def index_add(inp, dim, index, src, alpha=1):
    """
    Optimized index_add for mthreads backend.

    self.index_add_(dim, index, source, alpha=1) -> Tensor

    For a 3-D tensor the output is:
        self[index[i], :, :] += alpha * src[i, :, :]  # if dim == 0
        self[:, index[i], :] += alpha * src[:, i, :]  # if dim == 1
        self[:, :, index[i]] += alpha * src[:, :, i]  # if dim == 2
    """
    optimized_out = _try_run_contiguous_index_add_path(
        inp, dim, index, src, alpha, False
    )
    if optimized_out is not None:
        return optimized_out

    bf16_out = _try_run_bf16_index_add_path(inp, dim, index, src, alpha, False)
    if bf16_out is not None:
        return bf16_out

    if _needs_native_semantic_fallback(inp, index, src, False):
        return _run_native_semantic_fallback(inp, dim, index, src, alpha, False)

    logger.debug("GEMS_MTHREADS INDEX_ADD")

    # Keep the original MTHREADS implementation for non-BF16 and unsupported
    # BF16 inputs.
    inp = inp.contiguous()
    if inp.dtype == torch.bfloat16:
        index = _resolve_index_for_kernel(index).contiguous()
    else:
        index = index.contiguous()
    src = src.contiguous()

    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    # Bounds check: the common op (src/flag_gems/ops/index_add.py) performs this
    # inside the Triton kernel. Other backends (kunlunxin, ascend, cambricon) do
    # it in Python instead, which we follow here.
    # Use min/max to avoid allocating full-size boolean tensors.
    idx_min = index.min().item()
    idx_max = index.max().item()
    assert idx_min >= 0 and idx_max < inp_len, "0 <= index < self.size(dim)"

    # Move target dim to last position for coalesced memory access
    final_dim = inp.ndim - 1
    if dim != final_dim:
        inp = dim_compress(inp, dim)
        src = dim_compress(src, dim)

    # Clone input for output
    out = inp.clone()

    # Calculate grid with autotune
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(inp.device):
        index_add_kernel[grid](out, index, src, M, N, alpha, inp_len)

    # Restore original dimension order if needed
    if dim != final_dim:
        order = list(range(out.ndim - 1))
        order.insert(dim, final_dim)
        return out.permute(order).contiguous()
    else:
        return out


def index_add_(inp, dim, index, src, alpha=1):
    """
    In-place version of index_add.
    """
    optimized_out = _try_run_contiguous_index_add_path(
        inp, dim, index, src, alpha, True
    )
    if optimized_out is not None:
        return optimized_out

    bf16_out = _try_run_bf16_index_add_path(inp, dim, index, src, alpha, True)
    if bf16_out is not None:
        return bf16_out

    if _needs_native_semantic_fallback(inp, index, src, True):
        return _run_native_semantic_fallback(inp, dim, index, src, alpha, True)

    logger.debug("GEMS_MTHREADS INDEX_ADD_")

    # Keep the original MTHREADS implementation for non-BF16 and unsupported
    # BF16 inputs.
    if inp.dtype == torch.bfloat16:
        index = _resolve_index_for_kernel(index).contiguous()
    else:
        index = index.contiguous()
    src = src.contiguous()

    dim = dim % inp.ndim
    inp_len = inp.size(dim)
    N = index.numel()
    M = src.numel() // N

    # Bounds check: the common op (src/flag_gems/ops/index_add.py) performs this
    # inside the Triton kernel. Other backends (kunlunxin, ascend, cambricon) do
    # it in Python instead, which we follow here.
    # Use min/max to avoid allocating full-size boolean tensors.
    idx_min = index.min().item()
    idx_max = index.max().item()
    assert idx_min >= 0 and idx_max < inp_len, "0 <= index < self.size(dim)"

    # Move target dim to last position
    final_dim = inp.ndim - 1

    if dim != final_dim:
        # Need to work on a permuted copy
        inp_work = dim_compress(inp.clone().contiguous(), dim)
        src_work = dim_compress(src, dim)

        # Calculate grid with autotune
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](inp_work, index, src_work, M, N, alpha, inp_len)

        # Restore original dimension order and copy back
        order = list(range(inp_work.ndim - 1))
        order.insert(dim, final_dim)
        inp_work = inp_work.permute(order).contiguous()
        inp.copy_(inp_work)
    else:
        # Can work directly on input if already contiguous
        inp_contig = inp.contiguous()

        # Calculate grid with autotune
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        with torch_device_fn.device(inp.device):
            index_add_kernel[grid](inp_contig, index, src, M, N, alpha, inp_len)

        # Copy back if input wasn't contiguous
        if not inp.is_contiguous():
            inp.copy_(inp_contig)

    return inp
