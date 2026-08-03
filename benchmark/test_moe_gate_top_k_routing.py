import pytest
import torch

import flag_gems

from . import base, consts


class MoEGateTopKRoutingBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # (num_tokens, num_experts, topk)
        self.shapes = [
            (1, 8, 2),
            (16, 8, 2),
            (64, 8, 2),
            (128, 8, 2),
            (256, 8, 2),
            (64, 16, 4),
            (128, 16, 4),
            (256, 16, 4),
            (64, 32, 4),
            (128, 32, 8),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self.moe_gate_top_k_routing_input_fn(
                config, cur_dtype, self.device
            )

    def moe_gate_top_k_routing_input_fn(self, config, dtype, device):
        num_tokens, num_experts, topk = config
        gate_logits = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
        yield gate_logits, topk


def moe_gate_top_k_routing_torch_ref(gate_logits, topk):
    """Reference implementation using torch.softmax + torch.topk"""
    probs = torch.softmax(gate_logits, dim=-1)
    return torch.topk(probs, topk, dim=-1)


@pytest.mark.moe_gate_top_k_routing
def test_moe_gate_top_k_routing():
    bench = MoEGateTopKRoutingBenchmark(
        op_name="moe_gate_top_k_routing",
        torch_op=moe_gate_top_k_routing_torch_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.moe_gate_top_k_routing)
    bench.run()
