"""
Tests for flash_mla_with_kvcache Triton implementation.
Uses the CUDA FlashMLA implementation from vLLM as ground truth.
"""

import math

import pytest
import torch

# CUDA reference
try:
    from vllm.third_party.flashmla.flash_mla_interface import (
        flash_mla_with_kvcache as cuda_flash_mla,
    )
    from vllm.third_party.flashmla.flash_mla_interface import (
        get_mla_metadata as cuda_get_mla_metadata,
    )

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# Triton implementation under test
from flag_gems.fused.flash_mla_with_kvcache import (
    flash_mla_with_kvcache as triton_flash_mla,
)
from flag_gems.fused.flash_mla_with_kvcache import (
    get_mla_metadata as triton_get_mla_metadata,
)

DEVICE = "cuda"


def generate_fp8_kv_cache(num_pages, page_block_size, h_k, d_nope=512, d_rope=64):
    """Generate realistic FP8 KV cache data (no NaN byte patterns)."""
    total_tokens = num_pages * page_block_size

    # Generate NoPE data, quantize to FP8 with per-128-element scales
    nope_data = (
        torch.randn(total_tokens, h_k, d_nope, dtype=torch.bfloat16, device=DEVICE)
        * 0.1
    )
    nope_flat = nope_data.reshape(-1, d_nope)
    groups = nope_flat.reshape(-1, 4, 128)
    scales = groups.float().abs().amax(dim=-1) / 448.0
    scales = scales.clamp(min=1e-12)
    quantized = (groups.float() / scales[:, :, None]).clamp(-448, 448)
    fp8_data = quantized.reshape(-1, 512).to(torch.float8_e4m3fn)

    # Generate RoPE data as BF16
    rope_data = (
        torch.randn(total_tokens, h_k, d_rope, dtype=torch.bfloat16, device=DEVICE)
        * 0.1
    )

    # Pack into 656-byte format
    kv_cache = torch.zeros(
        num_pages, page_block_size, h_k, 656, dtype=torch.uint8, device=DEVICE
    )
    fp8_bytes = fp8_data.view(torch.uint8).reshape(num_pages, page_block_size, h_k, 512)
    kv_cache[:, :, :, :512] = fp8_bytes
    scales_reshaped = scales.reshape(num_pages, page_block_size, h_k, 4)
    scales_bytes = (
        scales_reshaped.to(torch.float32)
        .view(torch.uint8)
        .reshape(num_pages, page_block_size, h_k, 16)
    )
    kv_cache[:, :, :, 512:528] = scales_bytes
    rope_bytes = (
        rope_data.reshape(num_pages, page_block_size, h_k, d_rope)
        .view(torch.uint8)
        .reshape(num_pages, page_block_size, h_k, 128)
    )
    kv_cache[:, :, :, 528:656] = rope_bytes

    return kv_cache


def check_close(triton_out, cuda_out, name, rtol=1e-2, atol=1e-2):
    diff = (triton_out.float() - cuda_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        triton_out.float().flatten(), cuda_out.float().flatten(), dim=0
    ).item()
    print(
        f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.8f}"
    )
    assert cos_sim > 0.99, f"{name} cosine similarity too low: {cos_sim}"


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vllm is not available")
def test_sparse_decode_fp8():
    """Test sparse decode with FP8 KV cache."""
    print("\n=== test_sparse_decode_fp8 ===")
    batch, seq_q, h_q, d_qk, topk = 2, 1, 128, 576, 128
    h_k = 1
    head_dim_v = 512
    page_block_size = 64
    num_pages = 100

    torch.manual_seed(42)

    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    kv_cache_raw = generate_fp8_kv_cache(num_pages, page_block_size, h_k)

    total_tokens = num_pages * page_block_size
    indices = torch.randint(
        0, total_tokens, (batch, seq_q, topk), dtype=torch.int32, device=DEVICE
    )

    # CUDA reference
    cuda_meta, _ = cuda_get_mla_metadata()
    cuda_out, cuda_lse = cuda_flash_mla(
        q,
        kv_cache_raw,
        None,
        None,
        head_dim_v,
        cuda_meta,
        is_fp8_kvcache=True,
        indices=indices,
    )

    # Triton implementation
    triton_meta, _ = triton_get_mla_metadata()
    triton_out, triton_lse = triton_flash_mla(
        q,
        kv_cache_raw,
        None,
        None,
        head_dim_v,
        triton_meta,
        is_fp8_kvcache=True,
        indices=indices,
    )

    check_close(triton_out, cuda_out, "out")
    check_close(triton_lse, cuda_lse, "lse")
    print("  PASSED")


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vllm is not available")
def test_sparse_decode_small():
    """Test sparse decode with small topk (FP8)."""
    print("\n=== test_sparse_decode_small ===")
    batch, seq_q, h_q, d_qk, topk = 2, 1, 64, 576, 64
    h_k = 1
    head_dim_v = 512
    page_block_size = 64
    num_pages = 50

    torch.manual_seed(42)

    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    kv_cache_raw = generate_fp8_kv_cache(num_pages, page_block_size, h_k)

    total_tokens = num_pages * page_block_size
    indices = torch.randint(
        0, total_tokens, (batch, seq_q, topk), dtype=torch.int32, device=DEVICE
    )

    # CUDA reference
    cuda_meta, _ = cuda_get_mla_metadata()
    cuda_out, cuda_lse = cuda_flash_mla(
        q,
        kv_cache_raw,
        None,
        None,
        head_dim_v,
        cuda_meta,
        is_fp8_kvcache=True,
        indices=indices,
    )

    # Triton implementation
    triton_meta, _ = triton_get_mla_metadata()
    triton_out, triton_lse = triton_flash_mla(
        q,
        kv_cache_raw,
        None,
        None,
        head_dim_v,
        triton_meta,
        is_fp8_kvcache=True,
        indices=indices,
    )

    check_close(triton_out, cuda_out, "out")
    check_close(triton_lse, cuda_lse, "lse")
    print("  PASSED")


@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vllm is not available")
def test_dense_decode():
    """Test dense decode with paged attention."""
    print("\n=== test_dense_decode ===")
    batch, seq_q, h_q, d_qk = 4, 1, 128, 576
    h_k = 1
    head_dim_v = 512
    page_block_size = 64
    seqlen = 256

    torch.manual_seed(42)

    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)

    # Paged KV cache
    max_pages_per_seq = math.ceil(seqlen / page_block_size) + 4
    total_pages = batch * max_pages_per_seq
    kv_cache = (
        torch.randn(
            total_pages, page_block_size, h_k, d_qk, dtype=torch.bfloat16, device=DEVICE
        )
        * 0.1
    )

    # Block table: [batch, max_pages_per_seq]
    block_table = torch.arange(total_pages, dtype=torch.int32, device=DEVICE).view(
        batch, max_pages_per_seq
    )

    # Cache seqlens
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=DEVICE)
    cache_seqlens[0] = seqlen // 2
    cache_seqlens[-1] = seqlen + 64

    # CUDA reference
    cuda_meta, _ = cuda_get_mla_metadata()
    cuda_out, cuda_lse = cuda_flash_mla(
        q,
        kv_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        cuda_meta,
        causal=True,
    )

    # Triton implementation
    triton_meta, _ = triton_get_mla_metadata()
    triton_out, triton_lse = triton_flash_mla(
        q,
        kv_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        triton_meta,
        causal=True,
    )

    check_close(triton_out, cuda_out, "out")
    check_close(triton_lse, cuda_lse, "lse")
    print("  PASSED")


if __name__ == "__main__":
    test_sparse_decode_small()
    test_sparse_decode_fp8()
    test_dense_decode()
