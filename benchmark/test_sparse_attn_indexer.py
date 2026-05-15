"""Benchmark for sparse_attn_indexer (DeepSeek V4 sparse attention).

Shapes match DeepSeek V4 production config:
    - num_heads=64: number of attention heads in MLA
    - head_dim=128: dimension per head
    - topk=1024: KV positions selected per token in sparse attention
    - num_tokens=1: single-token decode (latency-critical path)
    - num_tokens=128: typical prefill micro-batch
    - num_tokens=2048: max prefill sequence length

The torch reference uses a naive loop-based implementation that computes
logits per token and selects top-K via torch.topk. The Triton kernel uses
FP8 quantization, compact logits, and binary-search threshold for efficiency.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused.DSA.sparse_attn_indexer import sparse_attn_indexer

from . import base

device = flag_gems.device

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)


class SparseAttnIndexerBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_tokens, topk, max_model_len"

    def set_shapes(self, shape_file_path=None):
        # DeepSeek V4 production shapes:
        # - (1, 1024, 4096): decode, single token, latency-critical
        # - (128, 1024, 4096): typical prefill micro-batch
        # - (2048, 1024, 8192): max sequence length prefill
        self.shapes = [
            (1, 1024, 4096),
            (128, 1024, 4096),
            (2048, 1024, 8192),
        ]

    def get_input_iter(self, dtype):
        num_heads = 64
        head_dim = 128
        cache_stride_slot = head_dim + 4  # FP8 data + 4-byte scale

        for num_tokens, topk, max_model_len in self.shapes:
            # Query: [num_tokens, num_heads * head_dim]
            q = torch.randn(
                num_tokens,
                num_heads * head_dim,
                device=device,
                dtype=torch.float16,
            )
            # Key: [num_tokens, head_dim]
            k = torch.randn(
                num_tokens,
                head_dim,
                device=device,
                dtype=torch.float16,
            )
            # Per-head weights: [num_tokens, num_heads]
            weights = torch.rand(
                num_tokens,
                num_heads,
                device=device,
                dtype=torch.float32,
            )
            # KV cache: pre-allocated for max_model_len slots
            kv_cache = torch.zeros(
                max_model_len,
                cache_stride_slot,
                dtype=torch.uint8,
                device=device,
            )

            yield {
                "q": q,
                "k": k,
                "weights": weights,
                "kv_cache": kv_cache,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "topk": topk,
                "num_tokens": num_tokens,
                "total_kv_len": max_model_len,
                "insert_k": True,
            }

    def get_gems_op(self):
        return sparse_attn_indexer

    def get_torch_op(self):
        def torch_ref(
            q,
            k,
            weights,
            kv_cache,
            num_heads,
            head_dim,
            topk,
            num_tokens,
            total_kv_len,
            insert_k,
        ):
            """Naive PyTorch reference: loop over tokens, compute logits, topk."""
            # Quantize K to FP8 in cache
            k_f32 = k.float()
            amax = k_f32.abs().amax(dim=1).clamp(min=1e-4)
            scale = amax / 448.0
            k_fp8 = (k_f32 / scale[:, None]).to(torch.float8_e4m3fn)
            k_uint8 = k_fp8.view(torch.uint8)
            kv_cache[:num_tokens, :head_dim] = k_uint8
            scale_bytes = (
                scale.to(torch.float32).view(torch.uint8).reshape(num_tokens, 4)
            )
            kv_cache[:num_tokens, head_dim : head_dim + 4] = scale_bytes

            # Compute logits and select top-K
            q_reshaped = q.reshape(num_tokens, num_heads, head_dim)
            topk_out = torch.full(
                (num_tokens, topk), -1, dtype=torch.int32, device=q.device
            )

            for i in range(num_tokens):
                kv_end = i + 1
                if kv_end <= topk:
                    topk_out[i, :kv_end] = torch.arange(
                        kv_end, dtype=torch.int32, device=q.device
                    )
                else:
                    # Dequantize keys
                    ki_uint8 = kv_cache[:kv_end, :head_dim]
                    ki_fp8 = ki_uint8.view(torch.float8_e4m3fn)
                    ki_f16 = ki_fp8.to(torch.float16)
                    # Load scales
                    si_bytes = kv_cache[:kv_end, head_dim : head_dim + 4]
                    ki_scale = si_bytes.contiguous().view(torch.float32)
                    # Compute logits
                    qi = q_reshaped[i].to(torch.float16)
                    dots = torch.mm(qi, ki_f16.T).float()
                    dots = torch.relu(dots)
                    w_i = weights[i]
                    acc = (dots * w_i[:, None]).sum(dim=0) * ki_scale
                    _, top_idx = torch.topk(acc, topk, largest=True)
                    topk_out[i] = top_idx.to(torch.int32)

            return topk_out

        return torch_ref


@pytest.mark.sparse_attn_indexer
def test_perf_sparse_attn_indexer():
    bench = SparseAttnIndexerBenchmark(
        op_name="sparse_attn_indexer",
        torch_op=None,
        dtypes=[torch.float16],
    )
    bench.run()
