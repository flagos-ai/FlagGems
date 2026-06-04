import pytest
import torch

import flag_gems

from . import base, consts

# Representative (num_tokens, num_experts) combos covering small to large MoE layers.
MOE_LOAD_BALANCE_LOSS_SHAPES = [
    (8, 4),
    (16, 8),
    (32, 16),
    (64, 32),
    (128, 64),
    (256, 128),
    (512, 256),
]


class MoELoadBalanceLossBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MOE_LOAD_BALANCE_LOSS_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            num_tokens, num_experts = shape
            gate_logits = torch.randn(
                num_tokens, num_experts, dtype=cur_dtype, device=self.device
            )
            # Return as tuple so it gets unpacked as single argument
            yield (gate_logits,)


@pytest.mark.MoELoadBalanceLoss
def test_MoELoadBalanceLoss():
    if flag_gems.vendor_name == "metax":
        pytest.skip("Metax backend CI validates correctness; skip backend benchmark.")

    bench = MoELoadBalanceLossBenchmark(
        op_name="MoELoadBalanceLoss",
        torch_op=flag_gems.MoELoadBalanceLoss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
