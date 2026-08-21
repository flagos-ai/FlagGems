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
import weakref
from collections import OrderedDict
from threading import RLock

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.index_add import index_add as _common_index_add
from flag_gems.ops.index_add import index_add_ as _common_index_add_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.triton_version_utils import _triton_version_at_least

logger = logging.getLogger(__name__)

_TRITON_SUPPORTS_BF16_ATOMIC_ADD = _triton_version_at_least(3, 4)
_CONTIGUOUS_SUFFIX_TILE_MIN = 80
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

# Keep this cache in the op file: deployment may load this vendor op against an
# older compatible FlagGems package tree. A shared test contract covers both
# vendor-local copies.
_INDEX_BOUNDS_CACHE_MAX_ENTRIES = 128
_INDEX_BOUNDS_CACHE_MAX_BYTES = 256 * 1024 * 1024
_INDEX_OUT_OF_BOUNDS_MESSAGE = "0 <= index < self.size(dim)"


def _read_index_bounds(index):
    # One fused min+max kernel and a single sync instead of two separate
    # min()/max() passes; dispatched at the CompositeExplicitAutograd level
    # to stay clear of FlagGems' own op overrides.
    lower, upper = torch.ops.aten.aminmax.default.redispatch(
        _FALLBACK_KEYSET, index, dim=None, keepdim=False
    )
    return lower.item(), upper.item()


def _resolve_index_for_kernel(index):
    # A contiguous lazy-negative tensor still exposes the un-negated storage
    # to a pointer-based Triton kernel. Materialize only that exceptional case.
    # Calling resolve_neg() from inside use_gems() re-enters FlagGems' Python
    # override and can negate the logical value twice. Toggle the metadata bit
    # off first, then explicitly negate the ordinary physical view.
    if index.is_neg():
        return torch.neg(torch._neg_view(index))
    return index


def _index_version_root(index):
    root = index
    while root._base is not None:
        root = root._base
    return root


def _index_bounds_cache_identity(index, upper_bound):
    try:
        if index.is_neg() or index.is_conj():
            return None
        # Writes through .data or external DMA that do not bump _version are
        # outside this cache's contract; detecting them would require a sync.
        version = index._version
        version_root = _index_version_root(index)
    except Exception:
        return None

    try:
        device = index.device
        key = (
            type(index),
            device.type,
            device.index,
            index.data_ptr(),
            index.numel(),
            tuple(index.shape),
            tuple(index.stride()),
            index.dtype,
            id(version_root),
            version,
            int(upper_bound),
        )
        return key, version_root
    except Exception:
        return None


class _IndexBoundsCache:
    def __init__(
        self,
        max_entries=_INDEX_BOUNDS_CACHE_MAX_ENTRIES,
        max_bytes=_INDEX_BOUNDS_CACHE_MAX_BYTES,
    ):
        self._max_entries = max(0, int(max_entries))
        self._max_bytes = max(0, int(max_bytes))
        self._entries = OrderedDict()
        self._total_bytes = 0
        self._lock = RLock()

    def assert_in_bounds(self, index, upper_bound, cacheable=True):
        if index.numel() == 0:
            return

        cache_enabled = cacheable and self._max_entries > 0 and self._max_bytes > 0
        identity = (
            _index_bounds_cache_identity(index, upper_bound) if cache_enabled else None
        )
        if identity is not None:
            key, version_root = identity
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:
                    confirmed_identity = _index_bounds_cache_identity(
                        index, upper_bound
                    )
                    if confirmed_identity is not None:
                        confirmed_key, confirmed_root = confirmed_identity
                        if (
                            confirmed_key == key
                            and confirmed_root is version_root
                            and entry[0]() is version_root
                        ):
                            self._entries.move_to_end(key)
                            # Concurrent writes after this recheck remain a
                            # data race, as they do after a cold validation.
                            return
                    self._entries.pop(key)
                    self._total_bytes -= entry[1]

        idx_min, idx_max = _read_index_bounds(index)
        assert idx_min >= 0 and idx_max < upper_bound, _INDEX_OUT_OF_BOUNDS_MESSAGE

        if identity is None:
            return
        confirmed_identity = _index_bounds_cache_identity(index, upper_bound)
        entry_bytes = index.numel() * index.element_size()
        if confirmed_identity is None or entry_bytes > self._max_bytes:
            return
        confirmed_key, confirmed_root = confirmed_identity
        if confirmed_key != key or confirmed_root is not version_root:
            return
        try:
            version_root_ref = weakref.ref(version_root)
        except TypeError:
            return

        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._total_bytes -= previous[1]
            # Cache entries retain metadata and a weak root only. max_bytes is
            # a validated logical-footprint gate, not retained storage size.
            self._entries[key] = (version_root_ref, entry_bytes)
            self._total_bytes += entry_bytes
            while (
                len(self._entries) > self._max_entries
                or self._total_bytes > self._max_bytes
            ):
                _, (_, evicted_bytes) = self._entries.popitem(last=False)
                self._total_bytes -= evicted_bytes


_INDEX_BOUNDS_CACHE = _IndexBoundsCache()


def _assert_index_in_bounds(index, upper_bound, cacheable=True):
    _INDEX_BOUNDS_CACHE.assert_in_bounds(index, upper_bound, cacheable=cacheable)


def _volume(shape):
    value = 1
    for item in shape:
        value *= int(item)
    return value


def _normalize_dim(inp, dim):
    if dim < -inp.ndim or dim >= inp.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}], but got {dim})"
        )
    return dim % inp.ndim


def _can_return_empty_index(inp, dim, index, src):
    return (
        inp.ndim > 0
        and inp.ndim == src.ndim
        and inp.dtype == src.dtype
        and inp.device == src.device
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and index.device == inp.device
        and index.numel() == 0
        and src.size(dim) == 0
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
    )


def _can_use_contiguous_suffix_path(inp, dim, index, src):
    return (
        src.numel() > 0
        and inp.ndim == src.ndim
        and 0 <= dim < inp.ndim
        and index.ndim == 1
        and index.dtype in (torch.int32, torch.int64)
        and inp.dtype == src.dtype
        and inp.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and index.numel() == src.size(dim)
        and inp.is_contiguous()
        and src.is_contiguous()
        and all(inp.size(i) == src.size(i) for i in range(inp.ndim) if i != dim)
        and _volume(src.shape[dim + 1 :]) > 1
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("index_add_contiguous_suffix_tile"),
    key=["row_count", "suffix_size"],
    strategy=["log", "log"],
    restore_value=["out"],
    warmup=5,
    rep=10,
)
@triton.jit
def _index_add_contiguous_suffix_tile_kernel(
    out,
    index,
    src,
    row_count,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACCUMULATE_FP32: tl.constexpr,
):
    rows = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = ext.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < row_count
    mask = row_mask & (cols < suffix_size)
    edge = rows % index_len
    prefix = rows // index_len
    receiver = tl.load(index + edge, mask=row_mask, other=0).to(tl.int64)
    src_offsets = rows * suffix_size + cols
    out_offsets = (prefix * out_dim + receiver) * suffix_size + cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)
    if ACCUMULATE_FP32:
        values = values.to(tl.float32)
    tl.atomic_add(out + out_offsets, values * alpha, mask=mask, sem="relaxed")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("index_add_contiguous_suffix_flat"),
    key=["total_count", "suffix_size"],
    strategy=["log", "log"],
    restore_value=["out"],
    warmup=5,
    rep=10,
)
@triton.jit
def _index_add_contiguous_suffix_flat_kernel(
    out,
    index,
    src,
    total_count,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_SIZE: tl.constexpr,
    ACCUMULATE_FP32: tl.constexpr,
):
    # A 1D layout packs valid elements densely regardless of suffix width,
    # avoiding the mostly-masked tiles the 2D kernel produces for narrow
    # suffixes or leading-dim scatter.
    offsets = ext.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_count

    cols = offsets % suffix_size
    rows = offsets // suffix_size
    src_dim_idx = rows % index_len
    prefix_idx = rows // index_len
    dst_dim_idx = tl.load(index + src_dim_idx, mask=mask, other=0).to(tl.int64)
    valid = mask & (dst_dim_idx >= 0) & (dst_dim_idx < out_dim)

    src_offsets = rows * suffix_size + cols
    out_offsets = (prefix_idx * out_dim + dst_dim_idx) * suffix_size + cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)
    if ACCUMULATE_FP32:
        values = values.to(tl.float32)
    tl.atomic_add(out + out_offsets, values * alpha, mask=valid, sem="relaxed")


def _run_contiguous_suffix_flat_path(out, dim, index, src, alpha):
    suffix_size = _volume(src.shape[dim + 1 :])
    row_count = _volume(src.shape[:dim]) * index.numel()
    total_count = row_count * suffix_size
    grid = lambda meta: (triton.cdiv(total_count, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(out.device):
        _index_add_contiguous_suffix_flat_kernel[grid](
            out,
            index,
            src,
            total_count,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            ACCUMULATE_FP32=(
                out.dtype == torch.float32 and src.dtype == torch.bfloat16
            ),
        )
    return out


def _run_contiguous_suffix_tile_path(out, dim, index, src, alpha):
    suffix_size = _volume(src.shape[dim + 1 :])
    row_count = _volume(src.shape[:dim]) * index.numel()
    grid = lambda meta: (
        triton.cdiv(row_count, meta["BLOCK_M"]),
        triton.cdiv(suffix_size, meta["BLOCK_N"]),
    )
    with torch_device_fn.device(out.device):
        _index_add_contiguous_suffix_tile_kernel[grid](
            out,
            index,
            src,
            row_count,
            index.numel(),
            out.size(dim),
            suffix_size,
            alpha,
            ACCUMULATE_FP32=(
                out.dtype == torch.float32 and src.dtype == torch.bfloat16
            ),
        )
    return out


def _run_contiguous_suffix_path(out, dim, index, src, alpha):
    # View contiguous tensors as [prefix, index_len, suffix] and scatter-add
    # dense suffix tiles without generic rank/stride address decomposition.
    # Narrow suffixes waste most lanes of the 2D tile kernel. On C550,
    # suffixes 65 through 79 underfill the second tile and flat wins across
    # fp32, fp16, and bf16; tile wins consistently from 80 onward.
    suffix_size = _volume(src.shape[dim + 1 :])
    if dim == 0 or suffix_size < _CONTIGUOUS_SUFFIX_TILE_MIN:
        return _run_contiguous_suffix_flat_path(out, dim, index, src, alpha)
    return _run_contiguous_suffix_tile_path(out, dim, index, src, alpha)


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_METAX INDEX_ADD")

    normalized_dim = _normalize_dim(inp, dim)
    if _can_return_empty_index(inp, normalized_dim, index, src):
        return inp.clone()

    index = _resolve_index_for_kernel(index)

    use_contiguous_suffix_path = _can_use_contiguous_suffix_path(
        inp, normalized_dim, index, src
    ) and not torch._C._is_alias_of(inp, src)
    if not use_contiguous_suffix_path:
        if not inp.is_contiguous() or not src.is_contiguous():
            return _common_index_add(
                inp.contiguous(), dim, index, src.contiguous(), alpha
            )
        return _common_index_add(inp, dim, index, src, alpha)

    _assert_index_in_bounds(index, inp.size(normalized_dim))
    accumulate_fp32 = (
        inp.dtype == torch.bfloat16 and not _TRITON_SUPPORTS_BF16_ATOMIC_ADD
    )
    out = inp.float() if accumulate_fp32 else inp.clone()
    res = _run_contiguous_suffix_path(
        out, normalized_dim, index.contiguous(), src, alpha
    )
    return res.to(inp.dtype) if accumulate_fp32 else res


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_METAX INDEX_ADD_")

    normalized_dim = _normalize_dim(inp, dim)
    if torch._C._is_alias_of(inp, src):
        raise RuntimeError(
            "input and source overlap; clone source before calling index_add_"
        )
    if _can_return_empty_index(inp, normalized_dim, index, src):
        return inp

    index = _resolve_index_for_kernel(index)

    if not _can_use_contiguous_suffix_path(inp, normalized_dim, index, src):
        if not inp.is_contiguous() or not src.is_contiguous():
            out = _common_index_add(
                inp.contiguous(), dim, index, src.contiguous(), alpha
            )
            inp.copy_(out)
            return inp
        return _common_index_add_(inp, dim, index, src, alpha)

    _assert_index_in_bounds(index, inp.size(normalized_dim))
    accumulate_fp32 = (
        inp.dtype == torch.bfloat16 and not _TRITON_SUPPORTS_BF16_ATOMIC_ADD
    )
    out = inp.float() if accumulate_fp32 else inp
    res = _run_contiguous_suffix_path(
        out, normalized_dim, index.contiguous(), src, alpha
    )
    if accumulate_fp32:
        inp.copy_(res)
    return inp
