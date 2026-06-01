import pytest
import torch

import flag_gems

from . import base

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts_impl as vllm_fused_experts_impl,
    )

    HAS_VLLM_FUSED_MOE = True
except ImportError:
    HAS_VLLM_FUSED_MOE = False


class InplaceFusedExpertsBenchmark(base.Benchmark):
    """
    Benchmark for inplace_fused_experts comparing FlagGems Triton kernel vs vLLM.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
        self.shapes = [
            # Mixtral-like shapes
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            (512, 8, 4096, 14336, 2),
            # DeepSeek-V3-like shapes (TP=8 shard)
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._fused_moe_input_fn(config, cur_dtype)

    def _fused_moe_input_fn(self, config, dtype):
        num_tokens, num_experts, hidden_size, intermediate_size, topk = config
        device = flag_gems.device

        hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
        w1 = torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        w2 = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )

        gating = torch.randn(
            num_tokens, num_experts, device=device, dtype=torch.float32
        )
        topk_weights, topk_ids = torch.topk(torch.softmax(gating, dim=-1), topk, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        yield (hidden_states, w1, w2, topk_weights, topk_ids)


def _vllm_inplace_fused_experts_wrapper(hidden_states, w1, w2, topk_weights, topk_ids):
    """Wrapper to call vllm fused_experts_impl in-place."""
    vllm_fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=True,
        activation="silu",
    )
    return hidden_states


def _gems_inplace_fused_experts_wrapper(hidden_states, w1, w2, topk_weights, topk_ids):
    """Wrapper to call FlagGems inplace_fused_experts."""
    flag_gems.inplace_fused_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )
    return hidden_states


@pytest.mark.inplace_fused_experts
@pytest.mark.skipif(not HAS_VLLM_FUSED_MOE, reason="vLLM not installed")
def test_inplace_fused_experts_gems_vs_vllm():
    """
    Benchmark FlagGems inplace_fused_experts vs vLLM inplace fused_experts_impl (bf16/fp16).
    """
    bench = InplaceFusedExpertsBenchmark(
        op_name="inplace_fused_experts",
        torch_op=_vllm_inplace_fused_experts_wrapper,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.set_gems(_gems_inplace_fused_experts_wrapper)
    bench.run()
