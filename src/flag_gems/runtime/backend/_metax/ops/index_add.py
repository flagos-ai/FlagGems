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
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.index_add import (
    _normalize_dim,
    _normalize_index_add_alpha,
    _resolve_kernel_tensor,
    _validate_index_add_args,
)
from flag_gems.ops.index_add import index_add as _common_index_add
from flag_gems.ops.index_add import index_add_ as _common_index_add_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# C550 sweeps favor flat below the smallest configured 32-column tile.
# Recheck the crossover when changing the MetaX compiler/config pool.
_CONTIGUOUS_SUFFIX_TILE_MIN = 32
_NATIVE_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeExplicitAutograd)


def _prune_flat_configs(configs, named_args, **kwargs):
    # Retain the validated FP16 source layout while changing accumulation
    # precision. Broaden this pool only after a separate compiler/timing sweep.
    if named_args["src"].dtype == torch.float16:
        return [config for config in configs if config.kwargs["BLOCK_SIZE"] == 128]
    return configs


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("index_add_contiguous_suffix_flat"),
    key=["total_count", "suffix_size"],
    strategy=["log", "log"],
    prune_configs_by={"early_config_prune": _prune_flat_configs},
    restore_value=["out"],
    benchmark_mode="event",
    warmup=5,
    rep=10,
)
@triton.jit
def _index_add_suffix_flat_kernel(
    out,
    index,
    src,
    total_count: tl.int64,
    index_len: tl.int64,
    out_dim: tl.int64,
    suffix_size: tl.int64,
    alpha,
    BLOCK_SIZE: tl.constexpr,
    ACCUMULATE_FP32: tl.constexpr,
):
    # Cast before multiplication: MetaX pointer lowering must not wrap at the
    # signed 32-bit byte boundary even when all tensor dimensions fit int32.
    offsets = ext.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_count
    rows = offsets // suffix_size
    edge = rows % index_len
    prefix = rows // index_len
    cols = offsets % suffix_size
    receiver = tl.load(index + edge, mask=mask, other=0).to(tl.int64)
    out_offsets = (prefix * out_dim + receiver) * suffix_size + cols
    values = tl.load(src + offsets, mask=mask, other=0)
    if ACCUMULATE_FP32:
        values = values.to(tl.float32)
    tl.atomic_add(out + out_offsets, values * alpha, mask=mask, sem="relaxed")


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("index_add_contiguous_suffix_tile"),
    key=["row_count", "suffix_size"],
    strategy=["log", "log"],
    restore_value=["out"],
    benchmark_mode="event",
    warmup=5,
    rep=10,
)
@triton.jit
def _index_add_suffix_tile_kernel(
    out,
    index,
    src,
    row_count: tl.int64,
    index_len: tl.int64,
    out_dim: tl.int64,
    suffix_size: tl.int64,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ACCUMULATE_FP32: tl.constexpr,
):
    rows = ext.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = ext.program_id(1).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    row_mask = rows < row_count
    mask = row_mask & (cols < suffix_size)
    edge = rows % index_len
    prefix = rows // index_len
    receiver = tl.load(index + edge, mask=row_mask, other=0).to(tl.int64)
    src_offsets = rows * suffix_size + cols
    out_offsets = (prefix * out_dim + receiver) * suffix_size + cols
    values = tl.load(src + src_offsets, mask=mask, other=0)
    if ACCUMULATE_FP32:
        values = values.to(tl.float32)
    tl.atomic_add(out + out_offsets, values * alpha, mask=mask, sem="relaxed")


def _can_use_suffix_path(inp, dim, index, src):
    # Only dim is normalized here; validate other arguments before launching.
    # Fallbacks perform their own validation in the common implementation.
    return (
        src.numel() > 0
        and inp.ndim > 0
        and index.ndim == 1
        and math.prod(src.shape[dim + 1 :]) > 1
        and inp.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and inp.is_contiguous()
        and src.is_contiguous()
        and not inp.is_neg()
        and not src.is_neg()
        and not torch._C._is_alias_of(inp, src)
        and not torch._C._is_alias_of(inp, index)
    )


def _checked_index(index, upper_bound):
    index = _resolve_kernel_tensor(index)
    # Validate each call before any output write. Tensor version counters do
    # not detect all external/.data mutations and are not a bounds contract.
    lower, upper = torch.ops.aten.aminmax.default.redispatch(
        _NATIVE_KEYSET, index, dim=None, keepdim=False
    )
    if lower.item() < 0 or upper.item() >= upper_bound:
        raise AssertionError("0 <= index < self.size(dim)")
    return index


def _run_suffix_path(out, dim, index, src, alpha):
    suffix = math.prod(src.shape[dim + 1 :])
    rows = math.prod(src.shape[:dim]) * index.numel()
    accumulate_fp32 = src.dtype in (torch.float16, torch.bfloat16)
    with torch_device_fn.device(out.device):
        if dim == 0 or suffix < _CONTIGUOUS_SUFFIX_TILE_MIN:
            total = rows * suffix
            grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
            _index_add_suffix_flat_kernel[grid](
                out,
                index,
                src,
                total,
                index.numel(),
                out.size(dim),
                suffix,
                alpha,
                ACCUMULATE_FP32=accumulate_fp32,
            )
        else:
            grid = lambda meta: (
                triton.cdiv(rows, meta["BLOCK_M"]),
                triton.cdiv(suffix, meta["BLOCK_N"]),
            )
            _index_add_suffix_tile_kernel[grid](
                out,
                index,
                src,
                rows,
                index.numel(),
                out.size(dim),
                suffix,
                alpha,
                ACCUMULATE_FP32=accumulate_fp32,
            )
    return out


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_METAX INDEX_ADD")
    # Last-dimension calls have suffix=1 and cannot enter these kernels. Keep
    # the common-only region cheap, including its own argument validation.
    if type(dim) is int and dim in (-1, inp.ndim - 1):
        return _common_index_add(inp, dim, index, src, alpha)
    dim = _normalize_dim(inp, dim)
    if not _can_use_suffix_path(inp, dim, index, src):
        return _common_index_add(inp, dim, index, src, alpha)
    dim = _validate_index_add_args(inp, dim, index, src)
    alpha = _normalize_index_add_alpha(inp, alpha)
    index = _checked_index(index, inp.size(dim))
    # FP16 atomic writeback loses small updates under contention; BF16
    # atomics are also not reliably supported. Load either source directly
    # into FP32 accumulation and convert the destination only once.
    low_precision = inp.dtype in (torch.float16, torch.bfloat16)
    out = inp.float() if low_precision else inp.clone()
    _run_suffix_path(out, dim, index, src, alpha)
    return out.to(inp.dtype) if low_precision else out


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_METAX INDEX_ADD_")
    if type(dim) is int and dim in (-1, inp.ndim - 1):
        return _common_index_add_(inp, dim, index, src, alpha)
    dim = _normalize_dim(inp, dim)
    if not _can_use_suffix_path(inp, dim, index, src):
        return _common_index_add_(inp, dim, index, src, alpha)
    dim = _validate_index_add_args(inp, dim, index, src)
    alpha = _normalize_index_add_alpha(inp, alpha)
    index = _checked_index(index, inp.size(dim))
    low_precision = inp.dtype in (torch.float16, torch.bfloat16)
    out = inp.float() if low_precision else inp
    _run_suffix_path(out, dim, index, src, alpha)
    if low_precision:
        inp.copy_(out)
    return inp
