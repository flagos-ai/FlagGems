import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import base


class DispatchFusedMoeKernelBenchmark(base.Benchmark):
    """
    Benchmark for dispatch_fused_moe_kernel via fused_experts_impl.

    Measures latency of the fused MoE pipeline which internally calls
    dispatch_fused_moe_kernel for both GEMM1 and GEMM2.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, hidden_size, intermediate_size, topk)
        self.shapes = [
            (1, 8, 4096, 14336, 2),
            (4, 8, 4096, 14336, 2),
            (16, 8, 4096, 14336, 2),
            (64, 8, 4096, 14336, 2),
            (128, 8, 4096, 14336, 2),
            (256, 8, 4096, 14336, 2),
            (512, 8, 4096, 14336, 2),
            (1, 256, 7168, 2048, 8),
            (4, 256, 7168, 2048, 8),
            (16, 256, 7168, 2048, 8),
            (64, 256, 7168, 2048, 8),
            (128, 256, 7168, 2048, 8),
            (256, 256, 7168, 2048, 8),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._dispatch_moe_input_fn(config, cur_dtype)

    def _dispatch_moe_input_fn(self, config, dtype):
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


def _torch_fused_moe_baseline(hidden_states, w1, w2, topk_weights, topk_ids):
    """PyTorch eager baseline for fused MoE (no fusion)."""
    num_experts = w1.shape[0]
    intermediate_size = w1.shape[1] // 2

    output = torch.zeros_like(hidden_states)
    for expert_id in range(num_experts):
        mask = topk_ids == expert_id
        token_indices, slot_indices = mask.nonzero(as_tuple=True)
        if token_indices.numel() == 0:
            continue
        expert_input = hidden_states[token_indices]
        intermediate = expert_input @ w1[expert_id].T
        gate = intermediate[:, :intermediate_size]
        up = intermediate[:, intermediate_size:]
        activated = F.silu(gate) * up
        down = activated @ w2[expert_id].T
        weights = topk_weights[token_indices, slot_indices].unsqueeze(1)
        output.index_add_(0, token_indices, weights * down)
    return output


def _gems_dispatch_fused_moe(hidden_states, w1, w2, topk_weights, topk_ids):
    """Wrapper to call FlagGems fused_experts_impl (exercises dispatch_fused_moe_kernel)."""
    return flag_gems.fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
    )


@pytest.mark.dispatch_fused_moe_kernel
def test_dispatch_fused_moe_kernel_benchmark():
    """
    Benchmark dispatch_fused_moe_kernel via fused_experts_impl (bf16/fp16).
    """
    bench = DispatchFusedMoeKernelBenchmark(
        op_name="dispatch_fused_moe_kernel",
        torch_op=_torch_fused_moe_baseline,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.set_gems(_gems_dispatch_fused_moe)
    bench.run()
