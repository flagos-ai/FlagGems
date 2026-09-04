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
from flag_gems.ops.index_add import _validate_index_add_args
from flag_gems.ops.index_add import index_add as _common_index_add
from flag_gems.ops.index_add import index_add_ as _common_index_add_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_CONTIGUOUS_SUFFIX_TILE_MIN = 80
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

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


def _assert_index_in_bounds(index, upper_bound):
    if index.numel() == 0:
        return
    idx_min, idx_max = _read_index_bounds(index)
    if idx_min < 0 or idx_max >= upper_bound:
        raise AssertionError(_INDEX_OUT_OF_BOUNDS_MESSAGE)


def _volume(shape):
    value = 1
    for item in shape:
        value *= int(item)
    return value


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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("index_add_contiguous_suffix_fp16_flat"),
    key=["total_count", "suffix_size"],
    strategy=["log", "log"],
    restore_value=["out"],
    warmup=5,
    rep=10,
)
@triton.jit
def _index_add_contiguous_suffix_fp16_flat_kernel(
    out,
    index,
    src,
    total_count,
    index_len,
    out_dim,
    suffix_size,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    # Wider flat candidates can enter MetaX's failing f16v2 atomic lowering.
    # Keep FP16 on the narrow configuration while preserving the dim-0 flat path.
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
    tl.atomic_add(out + out_offsets, values * alpha, mask=valid, sem="relaxed")


def _run_contiguous_suffix_flat_path(
    out, dim, index, src, alpha, use_fp16_config=False
):
    suffix_size = _volume(src.shape[dim + 1 :])
    row_count = _volume(src.shape[:dim]) * index.numel()
    total_count = row_count * suffix_size
    grid = lambda meta: (triton.cdiv(total_count, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(out.device):
        if use_fp16_config:
            _index_add_contiguous_suffix_fp16_flat_kernel[grid](
                out,
                index,
                src,
                total_count,
                index.numel(),
                out.size(dim),
                suffix_size,
                alpha,
            )
        else:
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
        return _run_contiguous_suffix_flat_path(
            out,
            dim,
            index,
            src,
            alpha,
            use_fp16_config=(src.dtype == torch.float16),
        )
    return _run_contiguous_suffix_tile_path(out, dim, index, src, alpha)


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_METAX INDEX_ADD")

    normalized_dim = _validate_index_add_args(inp, dim, index, src)
    if src.numel() == 0:
        return inp.clone(memory_format=torch.contiguous_format)

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
    # MetaX BF16 atomic support cannot be inferred from the frontend Triton
    # version, so use the portable FP32 accumulation path unconditionally.
    accumulate_fp32 = inp.dtype == torch.bfloat16
    out = inp.float() if accumulate_fp32 else inp.clone()
    res = _run_contiguous_suffix_path(
        out, normalized_dim, index.contiguous(), src, alpha
    )
    return res.to(inp.dtype) if accumulate_fp32 else res


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_METAX INDEX_ADD_")

    normalized_dim = _validate_index_add_args(inp, dim, index, src)
    if src is inp or index is inp:
        raise RuntimeError(
            "input overlaps with source or index; clone the overlapping tensor "
            "before calling index_add_"
        )
    if src.numel() == 0:
        return inp
    if torch._C._is_alias_of(inp, src) or torch._C._is_alias_of(inp, index):
        raise RuntimeError(
            "input overlaps with source or index; clone the overlapping tensor "
            "before calling index_add_"
        )

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
    accumulate_fp32 = inp.dtype == torch.bfloat16
    out = inp.float() if accumulate_fp32 else inp
    res = _run_contiguous_suffix_path(
        out, normalized_dim, index.contiguous(), src, alpha
    )
    if accumulate_fp32:
        inp.copy_(res)
    return inp
