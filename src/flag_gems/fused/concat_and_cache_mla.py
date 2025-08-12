import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# enum Fp8KVCacheDataType
FP8_KV_CACHE_DATA_TYPE_AUTO = tl.constexpr(0)
FP8_KV_CACHE_DATA_TYPE_FP8E4M3 = tl.constexpr(1)
FP8_KV_CACHE_DATA_TYPE_FP8E5M2 = tl.constexpr(2)


@libentry()
@triton.jit
def concat_and_cache_mla_kernel(
    # pointers
    kv_c_ptr,  # in,  [num_tokens, kv_lora_rank]
    k_pe_ptr,  # in,  [num_tokens, pe_dim]
    kv_cache_ptr,  # out, [num_blocks, block_size, kv_lora_rank + pe_dim]
    slot_mapping_ptr,  # in,  [num_tokens]
    # strides
    block_stride,
    entry_stride,
    kv_c_stride,
    k_pe_stride,
    # dims
    kv_lora_rank,
    pe_dim,
    block_size,  # kv cache block size
    scale_ptr,
    # data type
    kv_dtype: tl.constexpr,  # one of Fp8KVCacheDataType
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    # Skip padded tokens
    if slot_idx < 0:
        return

    # Calculate cache position
    block_id = slot_idx // block_size
    block_offset = slot_idx % block_size
    cache_base = block_id * block_stride + block_offset * entry_stride

    # Preload scale if needed
    if kv_dtype != FP8_KV_CACHE_DATA_TYPE_AUTO:
        scale_val = tl.load(scale_ptr)

    # Process kv_c section
    for i in range(0, kv_lora_rank, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < kv_lora_rank

        src_ptr = kv_c_ptr + token_idx * kv_c_stride + idx
        dst_ptr = kv_cache_ptr + cache_base + idx

        val = tl.load(src_ptr, mask=mask, other=0)

        if kv_dtype != FP8_KV_CACHE_DATA_TYPE_AUTO:
            if kv_dtype == FP8_KV_CACHE_DATA_TYPE_FP8E4M3:
                val = (val / scale_val).to(tl.float8e4nv)
            elif kv_dtype == FP8_KV_CACHE_DATA_TYPE_FP8E5M2:
                val = (val / scale_val).to(tl.float8e5)
            val = val.to(tl.uint8, bitcast=True)
        tl.store(dst_ptr, val, mask=mask)

    # Process k_pe section
    for j in range(0, pe_dim, BLOCK_SIZE):
        idx = j + tl.arange(0, BLOCK_SIZE)
        mask = idx < pe_dim

        src_ptr = k_pe_ptr + token_idx * k_pe_stride + idx
        dst_ptr = kv_cache_ptr + cache_base + kv_lora_rank + idx

        val = tl.load(src_ptr, mask=mask, other=0)

        if kv_dtype != FP8_KV_CACHE_DATA_TYPE_AUTO:
            if kv_dtype == FP8_KV_CACHE_DATA_TYPE_FP8E4M3:
                val = (val / scale_val).to(tl.float8e4nv)
            elif kv_dtype == FP8_KV_CACHE_DATA_TYPE_FP8E5M2:
                val = (val / scale_val).to(tl.float8e5)
            val = val.to(tl.uint8, bitcast=True)
        tl.store(dst_ptr, val, mask=mask)


class ConcatAndCacheMla(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        scale: torch.Tensor,
    ):
        if kv_cache_dtype != "auto" and kv_cache.dtype != torch.uint8:
            raise ValueError("For FP8 kv_cache must be uint8 dtype")
        if kv_cache_dtype == "auto" and kv_cache.dtype != kv_c.dtype:
            raise ValueError("For auto mode kv_cache must match input dtype")

        # Map string dtype to internal constants
        kv_dtype_map = {
            "auto": FP8_KV_CACHE_DATA_TYPE_AUTO,
            "fp8": FP8_KV_CACHE_DATA_TYPE_FP8E4M3,
            "fp8e4m3": FP8_KV_CACHE_DATA_TYPE_FP8E4M3,
            "fp8e5m2": FP8_KV_CACHE_DATA_TYPE_FP8E5M2,
        }
        kv_dtype = kv_dtype_map.get(kv_cache_dtype)
        if kv_dtype is None:
            raise ValueError(f"Unsupported kv_cache_dtype: {kv_cache_dtype}")
        kv_dtype = int(kv_dtype)  # tl.constexpr->int

        kv_lora_rank = kv_c.size(1)
        pe_dim = k_pe.size(1)
        num_tokens = slot_mapping.size(0)

        # make sure `scale` is a scalar tensor
        if scale.numel() != 1:
            scale = scale.view(1)

        # make sure all tensors are on the same device
        device = kv_c.device
        k_pe = k_pe.to(device)
        kv_cache = kv_cache.to(device)
        slot_mapping = slot_mapping.to(device)
        scale = scale.to(device)

        # configure kernel launch
        grid = (num_tokens,)
        BLOCK_SIZE = min(kv_lora_rank, 512)

        assert kv_cache.dim() == 3, "kv_cache must be a 3D tensor"
        assert (
            kv_cache.size(2) == kv_lora_rank + pe_dim
        ), "kv_cache's last dimension must match kv_lora_rank + pe_dim"
        with torch_device_fn.device(device):
            concat_and_cache_mla_kernel[grid](
                kv_c,
                k_pe,
                kv_cache,
                slot_mapping,
                kv_cache.stride(0),  # block_stride
                kv_cache.stride(1),  # entry_stride
                kv_c.stride(0),  # kv_c_stride
                k_pe.stride(0),  # k_pe_stride
                kv_lora_rank,
                pe_dim,
                kv_cache.size(1),  # kv cache block_size
                scale,
                kv_dtype=kv_dtype,
                BLOCK_SIZE=BLOCK_SIZE,
            )
        return kv_cache


def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    logger.debug("GEMS CONCAT_AND_CACHE_MLA")
    return ConcatAndCacheMla.apply(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )
