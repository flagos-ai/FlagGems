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


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,
    kv_cache_ptr,
    kv_cache_scale_ptr,
    slot_mapping_ptr,
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_quant_blocks,
    head_dim: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    tid = tl.program_id(0)
    quant_block_id = tl.program_id(1) * 4
    quant_block_offsets = tl.arange(0, 4)
    head_offsets = tl.arange(0, QUANT_BLOCK_SIZE)
    offsets = (
        quant_block_id + quant_block_offsets[:, None]
    ) * QUANT_BLOCK_SIZE + head_offsets[None, :]
    mask = offsets < head_dim

    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
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
    dst_ptr = kv_cache_ptr + block_id * kv_cache_value_stride + block_offset * head_dim
    tl.store(dst_ptr + offsets, fp8_val, mask=mask)

    dst_scale_ptr = (
        kv_cache_scale_ptr
        + block_id * kv_cache_scale_stride
        + block_offset * num_quant_blocks
        + quant_block_id
    )
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

    kv_cache_flat = kv_cache.view(num_blocks, -1)
    fp8_dtype = _get_fp8_dtype()
    kv_cache_value = kv_cache_flat[:, : block_size * head_dim].view(fp8_dtype)
    kv_cache_scale = kv_cache_flat[:, block_size * head_dim :].view(torch.float32)
    _indexer_k_quant_and_cache_kernel[(num_tokens, triton.cdiv(num_quant_blocks, 4))](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_mapping,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_quant_blocks,
        head_dim,
        quant_block_size,
        IS_FNUZ=_is_fp8_fnuz(fp8_dtype),
        USE_UE8M0=scale_fmt == "ue8m0",
    )
