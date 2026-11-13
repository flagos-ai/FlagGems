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

# Adapted from vLLM v0.20.2:
# csrc/cache_kernels.cu::indexer_k_quant_and_cache_kernel
#
# GCU (gcu400) adaptation notes:
# - slot_mapping is int64, and a Triton i64 load at the 8-byte dtype stride
#   reads neighbor-joined garbage on torch_gcu: without
#   TORCH_GCU_ENABLE_INT64_AND_UINT64 the physical payload is compact int32
#   words (word j = element j), and with the env gate on the value sits in
#   the low word of each 8-byte slot (word 2*j). Slots are therefore always
#   read through a zero-copy int32 view with the env-aware word stride, so
#   the kernel is correct in both env states (int64_scalar_load_addressing
#   family; same fix as the delivered FlagGems-vllm copy of this op).
# - GCU grid.x hard limit is 65535 (grid_hard_limits): the per-token axis
#   folds into grid.y pages instead of launching num_tokens programs.
# - i32 addptr chains truncate element offsets >= 2**31, and paged KV caches
#   beyond that size are routine, so every scalar cache offset term folds
#   into one i64 expression before the stores.

import os
import sys

import torch
import triton
import triton.language as tl


def _get_fp8_dtype() -> torch.dtype:
    try:
        from vllm.platforms import current_platform

        return current_platform.fp8_dtype()
    except ImportError:
        pass

    if getattr(torch.version, "hip", None) is not None and hasattr(
        torch, "float8_e4m3fnuz"
    ):
        return torch.float8_e4m3fnuz
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    raise RuntimeError("float8_e4m3fn is required for indexer_k_quant_and_cache")


def _is_fp8_fnuz(dtype: torch.dtype) -> bool:
    return hasattr(torch, "float8_e4m3fnuz") and dtype == torch.float8_e4m3fnuz


def _slot_word_stride() -> int:
    # torch_gcu without TORCH_GCU_ENABLE_INT64_AND_UINT64 keeps ATen int64
    # values as compact int32 words (element j lives at word j); with the gate
    # on, values sit in the low word of each 8-byte slot (word 2*j).
    return 2 if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") == "1" else 1


def _slot_words(slot_mapping: torch.Tensor):
    if slot_mapping.dtype == torch.int32:
        return slot_mapping, 1
    if slot_mapping.dtype != torch.int64:
        raise NotImplementedError(
            "indexer_k_quant_and_cache supports int32/int64 slot_mapping, got "
            f"{slot_mapping.dtype}"
        )
    return slot_mapping.contiguous().view(torch.int32), _slot_word_stride()


# Enflame GCU hardware limit for grid.x; extra tokens fold into grid.y pages
# (see grid_hard_limits / scaled_int8_quant for the same adaptation).
GCU_GRID_X_MAX = 65535


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,
    kv_cache_ptr,
    kv_cache_scale_ptr,
    slot_words_ptr,
    slot_word_stride,
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_quant_blocks,
    num_tokens,
    grid_x,
    quant_grids,
    head_dim: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    token_page = pid1 // quant_grids
    quant_page = pid1 % quant_grids
    tid = token_page * grid_x + pid0
    if tid >= num_tokens:
        return
    quant_block_id = quant_page * 4
    quant_block_offsets = tl.arange(0, 4)
    head_offsets = tl.arange(0, QUANT_BLOCK_SIZE)
    offsets = (
        quant_block_id + quant_block_offsets[:, None]
    ) * QUANT_BLOCK_SIZE + head_offsets[None, :]
    mask = offsets < head_dim

    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_words_ptr + tid * slot_word_stride)
    if slot_id < 0:
        return

    block_id = slot_id // block_size
    block_offset = slot_id % block_size

    val = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    amax = tl.max(tl.abs(val).to(tl.float32), axis=1)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = (val.to(tl.float32) / scale[:, None]).to(kv_cache_ptr.type.element_ty)
    # Fold every scalar cache offset term into one i64 expression: i32 addptr
    # chains truncate element offsets >= 2**31 elements.
    cache_base = (
        block_id.to(tl.int64) * kv_cache_value_stride
        + block_offset.to(tl.int64) * head_dim
    )
    dst_ptr = kv_cache_ptr + cache_base
    tl.store(dst_ptr + offsets, fp8_val, mask=mask)

    scale_base = (
        block_id.to(tl.int64) * kv_cache_scale_stride
        + block_offset.to(tl.int64) * num_quant_blocks
        + quant_block_id
    )
    dst_scale_ptr = kv_cache_scale_ptr + scale_base
    scale_mask = quant_block_id + quant_block_offsets < num_quant_blocks
    tl.store(dst_scale_ptr + quant_block_offsets, scale, mask=scale_mask)


def indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    if head_dim % quant_block_size != 0:
        raise ValueError("head_dim must be divisible by quant_block_size")
    num_quant_blocks = head_dim // quant_block_size
    if num_tokens == 0 or num_quant_blocks == 0:
        return

    kv_cache_flat = kv_cache.view(num_blocks, -1)
    fp8_dtype = _get_fp8_dtype()
    kv_cache_value = kv_cache_flat[:, : block_size * head_dim].view(fp8_dtype)
    kv_cache_scale = kv_cache_flat[:, block_size * head_dim :].view(torch.float32)
    slot_words, slot_word_stride = _slot_words(slot_mapping)

    grid_x = min(num_tokens, GCU_GRID_X_MAX)
    quant_grids = triton.cdiv(num_quant_blocks, 4)
    grid = (grid_x, triton.cdiv(num_tokens, grid_x) * quant_grids)
    _indexer_k_quant_and_cache_kernel[grid](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_words,
        slot_word_stride,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_quant_blocks,
        num_tokens,
        grid_x,
        quant_grids,
        head_dim,
        quant_block_size,
        IS_FNUZ=_is_fp8_fnuz(fp8_dtype),
        USE_UE8M0=scale_fmt == "ue8m0",
    )


# ---------------------------------------------------------------------------
# Route the fused package entry point to this backend specialization.
# ``flag_gems.fused.__init__`` imports the generic module before the backend
# registrar loads this file, so the SpecOpRegistrar top-level namespace swap
# cannot reach ``from flag_gems.fused import indexer_k_quant_and_cache``
# (the route the accuracy test and the benchmark bind). Rebind the live
# entry points of the already-imported generic module/package, the same
# mechanism as ``std_mean`` / ``slow_conv_dilated2d`` / bin_topk in this
# tree; the generic module stays the untouched reference source.
# ---------------------------------------------------------------------------
def _rebind_generic_entry_points():
    impl = indexer_k_quant_and_cache
    generic_module = sys.modules.get("flag_gems.fused.indexer_k_quant_and_cache")
    if generic_module is not None and generic_module is not sys.modules.get(__name__):
        setattr(generic_module, "indexer_k_quant_and_cache", impl)
    generic_package = sys.modules.get("flag_gems.fused")
    if generic_package is not None and generic_package is not sys.modules.get(__name__):
        setattr(generic_package, "indexer_k_quant_and_cache", impl)


_rebind_generic_entry_points()
