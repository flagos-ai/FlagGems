"""Accuracy tests for flash_mla_with_kvcache (DeepSeek V4 sparse MLA decode).

Tests the Triton token-parallel split-reduce kernel against the vLLM FlashMLA
CUDA reference. Verifies three scenarios:
    1. SWA-only sparse decode (no extra cache)
    2. SWA + extra compressed KV cache
    3. Attention sink scaling

The KV cache is populated using vLLM's fused_qnorm_rope_kv CUDA kernel to
ensure identical FP8 quantization and SoA layout.

Correctness criterion: output and LSE match within rtol=5e-2, atol=5e-2
(accounting for FP8 dequantization differences between implementations).
"""

import pytest
import torch

import flag_gems
from flag_gems.fused import flash_mla_with_kvcache

device = flag_gems.device

# DeepSeek V4 MLA cache layout constants
_HEAD_DIM = 512
_NOPE_DIM = 448
_ROPE_DIM = 64
_TOKEN_DATA_BYTES = 576
_SCALE_BYTES = 8
_HEAD_BYTES = _TOKEN_DATA_BYTES + _SCALE_BYTES
_H_Q = 128

# --- vLLM CUDA reference ---
try:
    from vllm.third_party.flashmla.flash_mla_interface import \
        flash_mla_with_kvcache as _cuda_flash_mla
    from vllm.third_party.flashmla.flash_mla_interface import get_mla_metadata

    HAS_VLLM = True
except (ImportError, AttributeError):
    HAS_VLLM = False
    _cuda_flash_mla = None
    get_mla_metadata = None

pytestmark = pytest.mark.skipif(
    not HAS_VLLM,
    reason="vLLM FlashMLA CUDA kernel not available",
)


def _make_cos_sin_cache(max_pos, rope_dim, dev):
    base = 10000.0
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=dev) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=dev)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _make_kv_cache(n_tokens, block_size, dev):
    """Create FP8 KV cache using vLLM's fused quantization kernel."""
    from vllm.model_executor.layers import deepseek_v4_attention  # noqa: F401

    cuda_fn = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert

    num_blocks = (n_tokens + block_size - 1) // block_size + 1
    block_stride = block_size * _HEAD_BYTES
    k_cache_2d = torch.zeros(num_blocks, block_stride, device=dev, dtype=torch.uint8)

    kv_data = torch.randn(n_tokens, _HEAD_DIM, device=dev, dtype=torch.bfloat16) * 0.5
    position_ids = torch.arange(n_tokens, device=dev, dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(8192, _ROPE_DIM, dev)
    slot_mapping = torch.arange(n_tokens, device=dev, dtype=torch.int64)
    q_dummy = torch.randn(n_tokens, _H_Q, _HEAD_DIM, device=dev, dtype=torch.bfloat16)

    cuda_fn(
        q_dummy,
        kv_data,
        k_cache_2d,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        1e-6,
        block_size,
    )

    k_cache_4d = k_cache_2d.view(num_blocks, block_size, 1, _HEAD_BYTES)
    return k_cache_2d, k_cache_4d


@pytest.mark.flash_mla_with_kvcache
@pytest.mark.parametrize(
    "batch_size, topk, block_size",
    [
        (1, 64, 64),
        (1, 128, 64),
        (4, 128, 64),
        (32, 128, 64),
    ],
    ids=[
        "B1_topk64",
        "B1_topk128",
        "B4_topk128",
        "B32_topk128",
    ],
)
def test_flash_mla_with_kvcache(batch_size, topk, block_size):
    """Test SWA-only sparse decode against vLLM CUDA kernel."""
    torch.manual_seed(42)
    n_kv_tokens = topk + 32

    k_cache_2d, k_cache_4d = _make_kv_cache(n_kv_tokens, block_size, device)

    q = (
        torch.randn(batch_size, 1, _H_Q, _HEAD_DIM, device=device, dtype=torch.bfloat16)
        * 0.1
    )
    indices = torch.stack(
        [
            torch.randperm(n_kv_tokens, device=device, dtype=torch.int32)[:topk]
            for _ in range(batch_size)
        ]
    ).unsqueeze(1)
    topk_length = torch.full((batch_size,), topk, device=device, dtype=torch.int32)
    attn_sink = torch.full((_H_Q,), float("-inf"), device=device, dtype=torch.float32)
    softmax_scale = _HEAD_DIM**-0.5

    # Reference: vLLM CUDA kernel (expects 4D cache)
    sched_meta_ref, _ = get_mla_metadata()
    out_ref, lse_ref = _cuda_flash_mla(
        q=q,
        k_cache=k_cache_4d,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=sched_meta_ref,
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
    )

    # Triton kernel (accepts both 2D and 4D cache)
    sched_meta_tri, _ = get_mla_metadata()
    out_tri, lse_tri = flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache_2d,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=sched_meta_tri,
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
    )

    torch.testing.assert_close(
        out_tri.float(),
        out_ref.float(),
        rtol=5e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        lse_tri.float(),
        lse_ref.float(),
        rtol=5e-2,
        atol=5e-2,
    )


@pytest.mark.flash_mla_with_kvcache
@pytest.mark.parametrize(
    "batch_size, swa_topk, extra_topk, swa_bs, extra_bs",
    [
        (1, 128, 256, 64, 256),
        (4, 128, 256, 64, 256),
        (4, 128, 512, 64, 256),
    ],
    ids=[
        "B1_swa128_ext256",
        "B4_swa128_ext256",
        "B4_swa128_ext512",
    ],
)
def test_flash_mla_with_kvcache_extra(
    batch_size,
    swa_topk,
    extra_topk,
    swa_bs,
    extra_bs,
):
    """Test SWA + extra (compressed) KV cache against vLLM CUDA kernel."""
    torch.manual_seed(123)
    swa_tokens = swa_topk + 32
    extra_tokens = extra_topk + 64

    swa_cache_2d, swa_cache_4d = _make_kv_cache(swa_tokens, swa_bs, device)
    extra_cache_2d, extra_cache_4d = _make_kv_cache(
        extra_tokens,
        extra_bs,
        device,
    )

    q = (
        torch.randn(batch_size, 1, _H_Q, _HEAD_DIM, device=device, dtype=torch.bfloat16)
        * 0.1
    )
    swa_indices = torch.stack(
        [
            torch.randperm(swa_tokens, device=device, dtype=torch.int32)[:swa_topk]
            for _ in range(batch_size)
        ]
    ).unsqueeze(1)
    extra_indices = torch.stack(
        [
            torch.randperm(extra_tokens, device=device, dtype=torch.int32)[:extra_topk]
            for _ in range(batch_size)
        ]
    ).unsqueeze(1)
    swa_lens = torch.full((batch_size,), swa_topk, device=device, dtype=torch.int32)
    extra_lens = torch.full((batch_size,), extra_topk, device=device, dtype=torch.int32)
    attn_sink = torch.full((_H_Q,), float("-inf"), device=device, dtype=torch.float32)
    softmax_scale = _HEAD_DIM**-0.5

    sched_meta_ref, _ = get_mla_metadata()
    out_ref, lse_ref = _cuda_flash_mla(
        q=q,
        k_cache=swa_cache_4d,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=sched_meta_ref,
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=swa_indices,
        topk_length=swa_lens,
        attn_sink=attn_sink,
        extra_k_cache=extra_cache_4d,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_lens,
    )

    sched_meta_tri, _ = get_mla_metadata()
    out_tri, lse_tri = flash_mla_with_kvcache(
        q=q,
        k_cache=swa_cache_2d,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=sched_meta_tri,
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=swa_indices,
        topk_length=swa_lens,
        attn_sink=attn_sink,
        extra_k_cache=extra_cache_2d,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_lens,
    )

    torch.testing.assert_close(
        out_tri.float(),
        out_ref.float(),
        rtol=5e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        lse_tri.float(),
        lse_ref.float(),
        rtol=5e-2,
        atol=5e-2,
    )


@pytest.mark.flash_mla_with_kvcache
@pytest.mark.parametrize(
    "batch_size, topk",
    [
        (1, 64),
        (1, 128),
        (4, 128),
    ],
    ids=["B1_topk64_sink", "B1_topk128_sink", "B4_topk128_sink"],
)
def test_flash_mla_with_kvcache_attn_sink(batch_size, topk):
    """Test with non-trivial attention sink values."""
    torch.manual_seed(77)
    block_size = 64
    n_kv_tokens = topk + 32

    k_cache_2d, k_cache_4d = _make_kv_cache(n_kv_tokens, block_size, device)

    q = (
        torch.randn(batch_size, 1, _H_Q, _HEAD_DIM, device=device, dtype=torch.bfloat16)
        * 0.1
    )
    indices = torch.stack(
        [
            torch.randperm(n_kv_tokens, device=device, dtype=torch.int32)[:topk]
            for _ in range(batch_size)
        ]
    ).unsqueeze(1)
    topk_length = torch.full((batch_size,), topk, device=device, dtype=torch.int32)
    attn_sink = torch.randn(_H_Q, device=device, dtype=torch.float32) * 2.0
    softmax_scale = _HEAD_DIM**-0.5

    sched_meta_ref, _ = get_mla_metadata()
    out_ref, lse_ref = _cuda_flash_mla(
        q=q,
        k_cache=k_cache_4d,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=sched_meta_ref,
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
    )

    sched_meta_tri, _ = get_mla_metadata()
    out_tri, lse_tri = flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache_2d,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=sched_meta_tri,
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
    )

    torch.testing.assert_close(
        out_tri.float(),
        out_ref.float(),
        rtol=5e-2,
        atol=5e-2,
    )
    torch.testing.assert_close(
        lse_tri.float(),
        lse_ref.float(),
        rtol=5e-2,
        atol=5e-2,
    )
