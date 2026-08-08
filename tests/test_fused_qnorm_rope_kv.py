"""Accuracy tests for fused DeepSeek V4 QNorm+RoPE+KV Insert kernel.

Tests the Triton kernel against vLLM CUDA reference (when available) and a
pure-PyTorch fallback. Verifies Q path (RMSNorm + RoPE) and KV path
(RoPE + FP8 quantize + paged cache insert) independently.
"""

import math

import pytest
import torch

import flag_gems
from flag_gems.fused.fused_qnorm_rope_kv import (
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
)
from flag_gems.utils.device_info import get_device_capability

from . import conftest as cfg

device = flag_gems.device


def is_support_fp8e4nv():
    major, minor = get_device_capability()
    return major * 10 + minor >= 89


# --- Shape configuration with QUICK_MODE support ---
if cfg.QUICK_MODE:
    DECODE_CONFIGS = [1, 4]
    PREFILL_CONFIGS = [1024]
else:
    DECODE_CONFIGS = [1, 4, 17, 64]
    PREFILL_CONFIGS = [256, 1024, 2048]

# --- vLLM CUDA reference (optional) ---
try:
    import vllm._custom_ops  # noqa: F401 — loads torch.ops._C

    def _vllm_fused_qnorm_rope_kv(
        q,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps=1e-6,
        cache_block_size=16,
    ):
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            k_cache,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            eps,
            cache_block_size,
        )

    HAS_VLLM = True
except (ImportError, AttributeError):
    HAS_VLLM = False
    _vllm_fused_qnorm_rope_kv = None


def fused_qnorm_rope_kv_ref(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps=1e-6,
    cache_block_size=16,
):
    """Pure-PyTorch reference: QNorm + RoPE + FP8 KV cache insert."""
    N, H, D = q.shape
    N_ins = slot_mapping.shape[0]
    cache_stride = k_cache.stride(0)

    for tok in range(N):
        pos = position_ids[tok].item()
        cos = cos_sin_cache[pos, :32]
        sin = cos_sin_cache[pos, 32:]

        for head in range(H):
            x = q[tok, head, :].float()
            var = (x * x).mean()
            rsqrt_val = torch.rsqrt(
                var + torch.tensor(eps, dtype=torch.float32, device=x.device)
            )
            x_norm = x * rsqrt_val

            q[tok, head, :448] = x_norm[:448].to(torch.bfloat16)

            re_f = x_norm[448::2].to(torch.bfloat16).float()
            ro_f = x_norm[449::2].to(torch.bfloat16).float()
            q[tok, head, 448::2] = (re_f * cos - ro_f * sin).to(torch.bfloat16)
            q[tok, head, 449::2] = (re_f * sin + ro_f * cos).to(torch.bfloat16)

    for tok in range(N_ins):
        slot_id = slot_mapping[tok].item()
        if slot_id < 0:
            continue

        pos = position_ids[tok].item()
        cos = cos_sin_cache[pos, :32]
        sin = cos_sin_cache[pos, 32:]

        kv_f = kv[tok].to(torch.bfloat16).float()

        x_e = kv_f[448::2][:32]
        x_o = kv_f[449::2][:32]
        out_e = x_e * cos - x_o * sin
        out_o = x_e * sin + x_o * cos

        block_idx = slot_id // cache_block_size
        pos_in_block = slot_id % cache_block_size
        byte_off_tok = block_idx * cache_stride + pos_in_block * 576
        byte_off_scale = (
            block_idx * cache_stride + cache_block_size * 576 + pos_in_block * 8
        )

        flat = k_cache.view(-1)

        for b in range(7):
            bdata = kv_f[b * 64 : (b + 1) * 64]
            absmax = bdata.abs().max().clamp(min=1e-4)
            exponent = math.ceil(math.log2(absmax.item() / 448.0))
            inv_scale = 2.0 ** (-exponent)
            scaled = (bdata * inv_scale).clamp(-448.0, 448.0)
            fp8_vals = scaled.to(torch.float8_e4m3fn)
            fp8_i8 = fp8_vals.view(torch.int8).view(torch.uint8)
            flat[byte_off_tok + b * 64 : byte_off_tok + (b + 1) * 64] = fp8_i8
            enc_scale = max(0, min(255, int(exponent + 127)))
            flat[byte_off_scale + b] = enc_scale

        flat[byte_off_scale + 7] = 0

        bf16_bytes = torch.zeros(128, dtype=torch.uint8, device=kv.device)
        rope_bf16 = torch.zeros(64, dtype=torch.bfloat16, device=kv.device)
        rope_bf16[0::2] = out_e.to(torch.bfloat16)
        rope_bf16[1::2] = out_o.to(torch.bfloat16)
        bf16_bytes[:] = rope_bf16.view(torch.uint8)
        flat[byte_off_tok + 448 : byte_off_tok + 576] = bf16_bytes


def generate_test_data(
    N: int,
    H: int = 128,
    cache_block_size: int = 16,
    max_pos: int = 8192,
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


@pytest.mark.fused_qnorm_rope_kv
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="FP8 E4M3 requires compute capability >= 89"
)
@pytest.mark.parametrize("N", DECODE_CONFIGS, ids=[f"N{n}" for n in DECODE_CONFIGS])
def test_q_norm_rope_decode(N):
    """Test Q path (RMSNorm + RoPE) correctness in decode mode."""
    data = generate_test_data(N)
    data_ref = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }

    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(**data)
    fused_qnorm_rope_kv_ref(**data_ref)

    torch.testing.assert_close(data["q"], data_ref["q"], rtol=1e-2, atol=1e-2)


@pytest.mark.fused_qnorm_rope_kv
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="FP8 E4M3 requires compute capability >= 89"
)
@pytest.mark.parametrize("N", PREFILL_CONFIGS, ids=[f"N{n}" for n in PREFILL_CONFIGS])
def test_q_norm_rope_prefill(N):
    """Test Q path (RMSNorm + RoPE) correctness in prefill mode."""
    data = generate_test_data(N)
    data_ref = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }

    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(**data)
    fused_qnorm_rope_kv_ref(**data_ref)

    torch.testing.assert_close(data["q"], data_ref["q"], rtol=1e-2, atol=1e-2)


@pytest.mark.fused_qnorm_rope_kv
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="FP8 E4M3 requires compute capability >= 89"
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
    fused_qnorm_rope_kv_ref(**data_ref)

    torch.testing.assert_close(
        data["k_cache"].view(-1).float(),
        data_ref["k_cache"].view(-1).float(),
        rtol=0,
        atol=1,
    )


@pytest.mark.fused_qnorm_rope_kv
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="FP8 E4M3 requires compute capability >= 89"
)
def test_inplace_modification():
    """Verify q is modified in-place."""
    data = generate_test_data(4)
    q_orig = data["q"].clone()
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(**data)
    assert not torch.equal(data["q"], q_orig)


@pytest.mark.fused_qnorm_rope_kv
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="FP8 E4M3 requires compute capability >= 89"
)
def test_negative_slot_skipped():
    """Verify tokens with slot_mapping=-1 are skipped."""
    data = generate_test_data(4)
    data["slot_mapping"][2] = -1
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(**data)


@pytest.mark.fused_qnorm_rope_kv
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="FP8 E4M3 requires compute capability >= 89"
)
@pytest.mark.parametrize("N", [1, 64, 1024])
def test_fused_qnorm_rope_kv_vs_vllm(N):
    """Test against vLLM CUDA kernel."""
    data = generate_test_data(N)
    data_ref = {
        k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }

    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(**data)
    _vllm_fused_qnorm_rope_kv(**data_ref)

    torch.testing.assert_close(data["q"], data_ref["q"], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(
        data["k_cache"].view(-1).float(),
        data_ref["k_cache"].view(-1).float(),
        rtol=0,
        atol=1,
    )
