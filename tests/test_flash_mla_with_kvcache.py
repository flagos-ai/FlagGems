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
from flag_gems.fused.flash_mla_with_kvcache import FlashMLASchedMeta
from flag_gems.fused.flash_mla_with_kvcache import (
    flash_mla_with_kvcache as triton_flash_mla,
)
from flag_gems.fused.flash_mla_with_kvcache import (
    get_mla_metadata as triton_get_mla_metadata,
)

from . import conftest as cfg

DEVICE = "cuda"
CUDA_AVAILABLE = torch.cuda.is_available()

# Shape configs for QUICK_MODE
if cfg.QUICK_MODE:
    V32_FP8_CONFIGS = [(2, 64, 64, 50)]
    EXTRA_KV_CONFIGS = [(64, 512, 12)]
else:
    V32_FP8_CONFIGS = [(2, 64, 64, 50), (2, 128, 128, 100)]
    EXTRA_KV_CONFIGS = [(64, 512, 12), (2, 512, 260)]
FP8_MAX = 448.0


def generate_v32_fp8_kv_cache(
    num_pages, page_block_size, h_k=1, d_nope=512, d_rope=64, device=DEVICE
):
    """Generate V32 FP8 KV cache: 512 FP8 + 4 FP32 scales + 64 BF16."""
    total_tokens = num_pages * page_block_size

    nope_data = (
        torch.randn(total_tokens, h_k, d_nope, dtype=torch.bfloat16, device=device)
        * 0.1
    )
    nope_flat = nope_data.reshape(-1, d_nope)
    groups = nope_flat.reshape(-1, 4, 128)
    scales = groups.float().abs().amax(dim=-1) / FP8_MAX
    scales = scales.clamp(min=1e-12)
    quantized = (groups.float() / scales[:, :, None]).clamp(-FP8_MAX, FP8_MAX)
    fp8_data = quantized.reshape(-1, d_nope).to(torch.float8_e4m3fn)

    rope_data = (
        torch.randn(total_tokens, h_k, d_rope, dtype=torch.bfloat16, device=device)
        * 0.1
    )

    kv_cache = torch.zeros(
        num_pages, page_block_size, h_k, 656, dtype=torch.uint8, device=device
    )
    kv_cache[:, :, :, :512] = fp8_data.view(torch.uint8).reshape(
        num_pages, page_block_size, h_k, 512
    )
    kv_cache[:, :, :, 512:528] = (
        scales.reshape(num_pages, page_block_size, h_k, 4)
        .to(torch.float32)
        .view(torch.uint8)
        .reshape(num_pages, page_block_size, h_k, 16)
    )
    kv_cache[:, :, :, 528:656] = (
        rope_data.reshape(num_pages, page_block_size, h_k, d_rope)
        .view(torch.uint8)
        .reshape(num_pages, page_block_size, h_k, 128)
    )
    return kv_cache


def generate_model1_fp8_kv_cache(
    num_pages, page_block_size, h_k=1, d_nope=448, d_rope=64, device=DEVICE
):
    """Generate MODEL1 page-oriented cache: page data first, then E8M0 scales."""
    assert h_k == 1, "FlashMLA sparse decode currently supports h_k == 1"
    total_tokens = num_pages * page_block_size
    token_data_bytes = d_nope + d_rope * 2
    scale_bytes = 8

    nope_data = (
        torch.randn(total_tokens, d_nope, dtype=torch.bfloat16, device=device) * 0.1
    )
    groups = nope_data.reshape(total_tokens, 7, 64).float()
    block_max = torch.clamp(groups.abs().amax(dim=-1), min=1e-4)
    exponent = torch.ceil(torch.log2(block_max / FP8_MAX))
    scales = torch.exp2(exponent)
    fp8_data = (
        torch.clamp(groups / scales[:, :, None], -FP8_MAX, FP8_MAX)
        .reshape(total_tokens, d_nope)
        .to(torch.float8_e4m3fn)
    )

    rope_data = (
        torch.randn(total_tokens, d_rope, dtype=torch.bfloat16, device=device) * 0.1
    )
    data_bytes = torch.empty(
        total_tokens, token_data_bytes, dtype=torch.uint8, device=device
    )
    data_bytes[:, :d_nope] = fp8_data.view(torch.uint8).reshape(total_tokens, d_nope)
    data_bytes[:, d_nope:] = rope_data.view(torch.uint8).reshape(total_tokens, 128)

    encoded_scales = torch.zeros(
        total_tokens, scale_bytes, dtype=torch.uint8, device=device
    )
    encoded_scales[:, :7] = torch.clamp(exponent + 127.0, 0, 255).to(torch.uint8)

    kv_cache = torch.zeros(
        num_pages, page_block_size, h_k, 584, dtype=torch.uint8, device=device
    )
    page_flat = kv_cache.view(num_pages, -1)
    page_flat[:, : page_block_size * token_data_bytes] = data_bytes.reshape(
        num_pages, page_block_size * token_data_bytes
    )
    page_flat[:, page_block_size * token_data_bytes :] = encoded_scales.reshape(
        num_pages, page_block_size * scale_bytes
    )
    return kv_cache


def check_close(triton_out, cuda_out, name, cos_threshold=0.99):
    diff = (triton_out.float() - cuda_out.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        triton_out.float().flatten(), cuda_out.float().flatten(), dim=0
    ).item()
    print(
        f"  {name}: max_diff={max_diff:.6f}, "
        f"mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.8f}"
    )
    assert cos_sim > cos_threshold, f"{name} cosine similarity too low: {cos_sim}"


def _run_cuda(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    cuda_meta, _ = cuda_get_mla_metadata()
    return cuda_flash_mla(
        q, k_cache, block_table, cache_seqlens, head_dim_v, cuda_meta, **kwargs
    )


def _run_triton(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    triton_meta, _ = triton_get_mla_metadata()
    return triton_flash_mla(
        q, k_cache, block_table, cache_seqlens, head_dim_v, triton_meta, **kwargs
    )


pytestmark = pytest.mark.skipif(
    not (VLLM_AVAILABLE and CUDA_AVAILABLE),
    reason="vLLM FlashMLA and CUDA are required",
)


@pytest.mark.parametrize(
    "batch,h_q,topk,num_pages",
    V32_FP8_CONFIGS,
)
def test_sparse_decode_v32_fp8(batch, h_q, topk, num_pages):
    """V32 sparse decode uses 656-byte FP8 cache and no dynamic topk."""
    print("\n=== test_sparse_decode_v32_fp8 ===")
    seq_q, d_qk = 1, 576
    head_dim_v = 512
    page_block_size = 64

    torch.manual_seed(42)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_v32_fp8_kv_cache(num_pages, page_block_size)
    total_tokens = num_pages * page_block_size
    indices = torch.randint(
        0, total_tokens, (batch, seq_q, topk), dtype=torch.int32, device=DEVICE
    )
    attn_sink = torch.randn(h_q, dtype=torch.float32, device=DEVICE)

    cuda_out_buf = torch.empty(
        batch, seq_q, h_q, head_dim_v, dtype=torch.bfloat16, device=DEVICE
    )
    triton_out_buf = torch.empty_like(cuda_out_buf)
    cuda_out, cuda_lse = _run_cuda(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        out=cuda_out_buf,
    )
    triton_out, triton_lse = _run_triton(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        out=triton_out_buf,
    )

    assert cuda_out.data_ptr() == cuda_out_buf.data_ptr()
    assert triton_out.data_ptr() == triton_out_buf.data_ptr()
    check_close(triton_out, cuda_out, "out")
    check_close(triton_lse, cuda_lse, "lse")


def test_sparse_decode_model1_topk_length_attn_sink_out():
    """MODEL1 log-like path: 584-byte cache, topk_length, attn_sink, out."""
    print("\n=== test_sparse_decode_model1_topk_length_attn_sink_out ===")
    batch, seq_q, h_q, d_qk, topk = 8, 1, 64, 512, 128
    head_dim_v = 512
    page_block_size = 64
    num_pages = 16

    torch.manual_seed(43)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_model1_fp8_kv_cache(num_pages, page_block_size)
    total_tokens = num_pages * page_block_size
    indices = torch.randint(
        0, total_tokens, (batch, seq_q, topk), dtype=torch.int32, device=DEVICE
    )
    topk_length = torch.randint(1, topk + 1, (batch,), dtype=torch.int32, device=DEVICE)
    attn_sink = torch.randn(h_q, dtype=torch.float32, device=DEVICE)
    cuda_out_buf = torch.empty(
        batch, seq_q, h_q, head_dim_v, dtype=torch.bfloat16, device=DEVICE
    )
    triton_out_buf = torch.empty_like(cuda_out_buf)

    cuda_out, cuda_lse = _run_cuda(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        out=cuda_out_buf,
    )
    triton_out, triton_lse = _run_triton(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        out=triton_out_buf,
    )

    check_close(triton_out, cuda_out, "out", cos_threshold=0.98)
    check_close(triton_lse, cuda_lse, "lse", cos_threshold=0.98)


@pytest.mark.parametrize(
    "extra_page_block_size,extra_topk,extra_num_pages",
    EXTRA_KV_CONFIGS,
)
def test_sparse_decode_model1_extra_kv(
    extra_page_block_size, extra_topk, extra_num_pages
):
    """MODEL1 attends original sparse cache plus extra sparse cache."""
    print("\n=== test_sparse_decode_model1_extra_kv ===")
    batch, seq_q, h_q, d_qk, topk = 4, 1, 64, 512, 128
    head_dim_v = 512
    page_block_size = 64
    num_pages = 12

    torch.manual_seed(44 + extra_page_block_size)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_model1_fp8_kv_cache(num_pages, page_block_size)
    extra_k_cache = generate_model1_fp8_kv_cache(extra_num_pages, extra_page_block_size)

    indices = torch.randint(
        0,
        num_pages * page_block_size,
        (batch, seq_q, topk),
        dtype=torch.int32,
        device=DEVICE,
    )
    extra_indices = torch.randint(
        0,
        extra_num_pages * extra_page_block_size,
        (batch, seq_q, extra_topk),
        dtype=torch.int32,
        device=DEVICE,
    )
    topk_length = torch.randint(1, topk + 1, (batch,), dtype=torch.int32, device=DEVICE)
    extra_topk_length = torch.randint(
        1, extra_topk + 1, (batch,), dtype=torch.int32, device=DEVICE
    )
    attn_sink = torch.randn(h_q, dtype=torch.float32, device=DEVICE)

    cuda_out, cuda_lse = _run_cuda(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_topk_length,
    )
    triton_out, triton_lse = _run_triton(
        q,
        kv_cache,
        None,
        None,
        head_dim_v,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_topk_length,
    )

    check_close(triton_out, cuda_out, "out", cos_threshold=0.98)
    check_close(triton_lse, cuda_lse, "lse", cos_threshold=0.98)


def test_dense_decode_seq_q1():
    """Dense decode is covered only for seq_q=1 in the Triton implementation."""
    print("\n=== test_dense_decode_seq_q1 ===")
    batch, seq_q, h_q, d_qk = 4, 1, 128, 576
    h_k = 1
    head_dim_v = 512
    page_block_size = 64
    seqlen = 256

    torch.manual_seed(45)
    q = torch.randn(batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    max_pages_per_seq = math.ceil(seqlen / page_block_size) + 4
    total_pages = batch * max_pages_per_seq
    kv_cache = (
        torch.randn(
            total_pages, page_block_size, h_k, d_qk, dtype=torch.bfloat16, device=DEVICE
        )
        * 0.1
    )
    block_table = torch.arange(total_pages, dtype=torch.int32, device=DEVICE).view(
        batch, max_pages_per_seq
    )
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=DEVICE)
    cache_seqlens[0] = seqlen // 2
    cache_seqlens[-1] = seqlen + page_block_size

    cuda_out, cuda_lse = _run_cuda(
        q, kv_cache, block_table, cache_seqlens, head_dim_v, causal=True
    )
    triton_out, triton_lse = _run_triton(
        q, kv_cache, block_table, cache_seqlens, head_dim_v, causal=True
    )

    check_close(triton_out, cuda_out, "out")
    check_close(triton_lse, cuda_lse, "lse")


def test_error_v32_rejects_topk_length():
    q = torch.randn(1, 1, 64, 576, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_v32_fp8_kv_cache(4, 64)
    indices = torch.randint(0, 256, (1, 1, 64), dtype=torch.int32, device=DEVICE)
    topk_length = torch.ones(1, dtype=torch.int32, device=DEVICE)
    meta, _ = triton_get_mla_metadata()

    with pytest.raises(AssertionError, match="dynamic topk length"):
        triton_flash_mla(
            q,
            kv_cache,
            None,
            None,
            512,
            meta,
            is_fp8_kvcache=True,
            indices=indices,
            topk_length=topk_length,
        )


def test_error_model1_extra_cache_requires_extra_indices():
    q = torch.randn(1, 1, 64, 512, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = generate_model1_fp8_kv_cache(4, 64)
    extra_k_cache = generate_model1_fp8_kv_cache(4, 64)
    indices = torch.randint(0, 256, (1, 1, 64), dtype=torch.int32, device=DEVICE)
    meta, _ = triton_get_mla_metadata()

    with pytest.raises(AssertionError, match="extra_indices_in_kvcache"):
        triton_flash_mla(
            q,
            kv_cache,
            None,
            None,
            512,
            meta,
            is_fp8_kvcache=True,
            indices=indices,
            extra_k_cache=extra_k_cache,
        )


def test_error_dense_rejects_sparse_only_args():
    q = torch.randn(1, 1, 64, 576, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.randn(4, 64, 1, 576, dtype=torch.bfloat16, device=DEVICE)
    block_table = torch.arange(4, dtype=torch.int32, device=DEVICE).view(1, 4)
    cache_seqlens = torch.full((1,), 64, dtype=torch.int32, device=DEVICE)
    attn_sink = torch.randn(64, dtype=torch.float32, device=DEVICE)
    meta, _ = triton_get_mla_metadata()

    with pytest.raises(AssertionError, match="must be None when dense"):
        triton_flash_mla(
            q,
            kv_cache,
            block_table,
            cache_seqlens,
            512,
            meta,
            attn_sink=attn_sink,
        )


def test_error_sched_meta_reuse_mismatch():
    q = torch.empty(1, 1, 64, 512, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.empty(4, 64, 1, 584, dtype=torch.uint8, device=DEVICE)
    meta = FlashMLASchedMeta(
        have_initialized=True,
        config=FlashMLASchedMeta.Config(
            b=2,
            s_q=1,
            h_q=64,
            page_block_size=64,
            h_k=1,
            causal=False,
            is_fp8_kvcache=True,
            topk=64,
            extra_page_block_size=None,
            extra_topk=None,
        ),
    )

    with pytest.raises(AssertionError, match="sched_meta.config.b"):
        triton_flash_mla(q, kv_cache, None, None, 512, meta)


if __name__ == "__main__":
    test_sparse_decode_v32_fp8(2, 64, 64, 50)
    test_sparse_decode_v32_fp8(2, 128, 128, 100)
    test_sparse_decode_model1_topk_length_attn_sink_out()
    test_sparse_decode_model1_extra_kv(64, 512, 12)
    test_dense_decode_seq_q1()
