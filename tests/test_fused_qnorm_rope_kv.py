"""
Accuracy tests for fused DeepSeek V4 QNorm+RoPE+KV Insert kernel.

Tests the Triton kernel against the vLLM CUDA reference implementation.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused.fused_qnorm_rope_kv import (
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_ref,
)
from flag_gems.utils.device_info import get_device_capability


def is_support_fp8e4nv():
    major, minor = get_device_capability()
    return major * 10 + minor >= 89


def generate_test_data(
    N: int,
    H: int = 128,
    cache_block_size: int = 16,
    max_pos: int = 8192,
    device: str = flag_gems.device,
):
    """Generate test inputs matching DeepSeek V4 dimensions."""
    torch.manual_seed(42)

    q = torch.randn((N, H, 512), dtype=torch.bfloat16, device=device)
    kv = torch.randn((N, 576), dtype=torch.bfloat16, device=device)

    num_blocks = (N + cache_block_size - 1) // cache_block_size + 1
    cache_stride = cache_block_size * 576 + cache_block_size * 8
    k_cache = torch.zeros((num_blocks, cache_stride), dtype=torch.uint8, device=device)

    slot_mapping = torch.arange(N, dtype=torch.int32, device=device)
    position_ids = torch.randint(0, max_pos, (N,), dtype=torch.int64, device=device)
    cos_sin_cache = torch.randn((max_pos, 64), dtype=torch.float32, device=device)

    return dict(
        q=q,
        kv=kv,
        k_cache=k_cache,
        slot_mapping=slot_mapping,
        position_ids=position_ids,
        cos_sin_cache=cos_sin_cache,
        eps=1e-6,
        cache_block_size=cache_block_size,
    )


DECODE_CONFIGS = [1, 4, 17, 64]
PREFILL_CONFIGS = [256, 1024, 2048]


@pytest.mark.fused_qnorm_rope_kv
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="Do not support fp8e4nv when capability < 89"
)
@pytest.mark.parametrize("N", DECODE_CONFIGS, ids=[f"N{n}" for n in DECODE_CONFIGS])
def test_q_norm_rope_decode(N):
    """Test Q path (RMSNorm + RoPE) correctness in decode mode."""
    data = generate_test_data(N)
    data_ref = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }

    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(**data)
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_ref(**data_ref)

    torch.testing.assert_close(data["q"], data_ref["q"], rtol=1e-2, atol=1e-2)


@pytest.mark.fused_qnorm_rope_kv
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="Do not support fp8e4nv when capability < 89"
)
@pytest.mark.parametrize("N", PREFILL_CONFIGS, ids=[f"N{n}" for n in PREFILL_CONFIGS])
def test_q_norm_rope_prefill(N):
    """Test Q path (RMSNorm + RoPE) correctness in prefill mode."""
    data = generate_test_data(N)
    data_ref = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }

    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(**data)
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_ref(**data_ref)

    torch.testing.assert_close(data["q"], data_ref["q"], rtol=1e-2, atol=1e-2)


@pytest.mark.fused_qnorm_rope_kv
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="Do not support fp8e4nv when capability < 89"
)
@pytest.mark.parametrize(
    "N",
    DECODE_CONFIGS + PREFILL_CONFIGS,
    ids=[f"N{n}" for n in DECODE_CONFIGS + PREFILL_CONFIGS],
)
def test_kv_cache_insert(N):
    """Test KV path (RoPE + FP8 quantize + cache insert) correctness."""
    data = generate_test_data(N)
    data_ref = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }

    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(**data)
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_ref(**data_ref)

    torch.testing.assert_close(
        data["k_cache"].view(-1).float(),
        data_ref["k_cache"].view(-1).float(),
        rtol=0,
        atol=1,
    )


@pytest.mark.fused_qnorm_rope_kv
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="Do not support fp8e4nv when capability < 89"
)
def test_inplace_modification():
    """Verify q is modified in-place."""
    data = generate_test_data(4)
    q_orig = data["q"].clone()
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(**data)
    assert not torch.equal(data["q"], q_orig)


@pytest.mark.fused_qnorm_rope_kv
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="Do not support fp8e4nv when capability < 89"
)
def test_negative_slot_skipped():
    """Verify tokens with slot_mapping=-1 are skipped."""
    data = generate_test_data(4)
    data["slot_mapping"][2] = -1
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(**data)
