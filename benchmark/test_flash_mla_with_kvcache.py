"""Benchmark for flash_mla_with_kvcache (DeepSeek V4 sparse MLA decode).

Shapes match DeepSeek V4 production config on H20 GPU:
    - H_q=128 attention heads, head_dim=512
    - SWA topk=128 with page_block_size=64
    - Optional extra compressed cache topk=256 with block_size=256

The baseline uses vLLM's FlashMLA CUDA kernel when available.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused import flash_mla_with_kvcache

from . import base

device = flag_gems.device

# DeepSeek V4 MLA cache layout constants
_HEAD_DIM = 512
_NOPE_DIM = 448
_ROPE_DIM = 64
_TOKEN_DATA_BYTES = 576
_SCALE_BYTES = 8
_HEAD_BYTES = _TOKEN_DATA_BYTES + _SCALE_BYTES
_H_Q = 128

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required"),
    pytest.mark.skipif(
        not hasattr(torch, "float8_e4m3fn"),
        reason="FP8 support required",
    ),
]

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


def _make_cos_sin_cache(max_pos, rope_dim, dev):
    base_val = 10000.0
    inv_freq = 1.0 / (
        base_val
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
    k_cache = torch.zeros(num_blocks, block_stride, device=dev, dtype=torch.uint8)

    kv_data = torch.randn(n_tokens, _HEAD_DIM, device=dev, dtype=torch.bfloat16) * 0.5
    pos = torch.arange(n_tokens, device=dev, dtype=torch.int64)
    cos_sin = _make_cos_sin_cache(8192, _ROPE_DIM, dev)
    slot_mapping = torch.arange(n_tokens, device=dev, dtype=torch.int64)
    q_dummy = torch.randn(n_tokens, _H_Q, _HEAD_DIM, device=dev, dtype=torch.bfloat16)

    cuda_fn(
        q_dummy,
        kv_data,
        k_cache,
        slot_mapping,
        pos,
        cos_sin,
        1e-6,
        block_size,
    )
    return k_cache


def _cuda_ref_fn(
    q,
    k_cache_4d,
    indices,
    topk_length,
    attn_sink,
    softmax_scale,
    extra_k_cache_4d,
    extra_indices,
    extra_topk_length,
):
    """Wrapper around vLLM CUDA FlashMLA for benchmark comparison."""
    sched_meta, _ = get_mla_metadata()
    return _cuda_flash_mla(
        q=q,
        k_cache=k_cache_4d,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=sched_meta,
        softmax_scale=softmax_scale,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache_4d,
        extra_indices_in_kvcache=extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )


class FlashMLAWithKVCacheBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "batch_size, swa_topk, extra_topk"

    def set_shapes(self, shape_file_path=None):
        # DeepSeek V4 production shapes:
        # (batch_size, swa_topk, extra_topk, swa_block_size, extra_block_size)
        self.shapes = [
            (1, 128, 0, 64, 0),
            (4, 128, 0, 64, 0),
            (32, 128, 0, 64, 0),
            (1, 128, 256, 64, 256),
            (4, 128, 256, 64, 256),
            (32, 128, 256, 64, 256),
        ]

    def get_input_iter(self, dtype):
        for B, swa_topk, extra_topk, swa_bs, extra_bs in self.shapes:
            torch.manual_seed(0)
            swa_tokens = swa_topk + 64
            softmax_scale = _HEAD_DIM**-0.5

            q = (
                torch.randn(B, 1, _H_Q, _HEAD_DIM, device=device, dtype=torch.bfloat16)
                * 0.1
            )
            swa_cache = _make_kv_cache(swa_tokens, swa_bs, device)
            swa_indices = torch.stack(
                [
                    torch.randperm(swa_tokens, device=device, dtype=torch.int32)[
                        :swa_topk
                    ]
                    for _ in range(B)
                ]
            ).unsqueeze(1)
            swa_lens = torch.full((B,), swa_topk, device=device, dtype=torch.int32)
            attn_sink = torch.full(
                (_H_Q,), float("-inf"), device=device, dtype=torch.float32
            )

            extra_cache = None
            extra_cache_4d = None
            extra_indices = None
            extra_lens = None
            if extra_topk > 0:
                extra_tokens = extra_topk + 128
                extra_cache = _make_kv_cache(extra_tokens, extra_bs, device)
                extra_cache_4d = extra_cache.view(-1, extra_bs, 1, _HEAD_BYTES)
                extra_indices = torch.stack(
                    [
                        torch.randperm(extra_tokens, device=device, dtype=torch.int32)[
                            :extra_topk
                        ]
                        for _ in range(B)
                    ]
                ).unsqueeze(1)
                extra_lens = torch.full(
                    (B,), extra_topk, device=device, dtype=torch.int32
                )

            swa_cache_4d = swa_cache.view(-1, swa_bs, 1, _HEAD_BYTES)

            yield (
                q,
                swa_cache,
                swa_cache_4d,
                swa_indices,
                swa_lens,
                attn_sink,
                softmax_scale,
                extra_cache,
                extra_cache_4d,
                extra_indices,
                extra_lens,
            )

    def get_gems_op(self):
        def triton_op(
            q,
            swa_cache,
            swa_cache_4d,
            swa_indices,
            swa_lens,
            attn_sink,
            softmax_scale,
            extra_cache,
            extra_cache_4d,
            extra_indices,
            extra_lens,
        ):
            sched_meta, _ = get_mla_metadata()
            return flash_mla_with_kvcache(
                q=q,
                k_cache=swa_cache,
                block_table=None,
                cache_seqlens=None,
                head_dim_v=512,
                tile_scheduler_metadata=sched_meta,
                softmax_scale=softmax_scale,
                is_fp8_kvcache=True,
                indices=swa_indices,
                topk_length=swa_lens,
                attn_sink=attn_sink,
                extra_k_cache=extra_cache,
                extra_indices_in_kvcache=extra_indices,
                extra_topk_length=extra_lens,
            )

        return triton_op

    def get_torch_op(self):
        if not HAS_VLLM:
            return None

        def cuda_op(
            q,
            swa_cache,
            swa_cache_4d,
            swa_indices,
            swa_lens,
            attn_sink,
            softmax_scale,
            extra_cache,
            extra_cache_4d,
            extra_indices,
            extra_lens,
        ):
            return _cuda_ref_fn(
                q,
                swa_cache_4d,
                swa_indices,
                swa_lens,
                attn_sink,
                softmax_scale,
                extra_cache_4d,
                extra_indices,
                extra_lens,
            )

        return cuda_op


@pytest.mark.flash_mla_with_kvcache
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM FlashMLA not available")
def test_flash_mla_with_kvcache():
    bench = FlashMLAWithKVCacheBenchmark(
        op_name="flash_mla_with_kvcache",
        torch_op=None,
        dtypes=[torch.bfloat16],
    )
    bench.run()
