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

import pytest
import torch

import flag_gems

from . import base, consts


def _torch_reference(gate_logits, top_k, attention_mask):
    """Same-device eager PyTorch performance baseline."""
    num_tokens, num_experts = gate_logits.shape
    probabilities = torch.softmax(gate_logits.float(), dim=-1)
    selected_experts = torch.topk(gate_logits, top_k, dim=-1).indices

    if attention_mask is None:
        token_weights = torch.ones(
            num_tokens,
            dtype=torch.float32,
            device=gate_logits.device,
        )
        valid = num_tokens
        probability_sum = probabilities.sum(dim=0)
    else:
        flat_mask = attention_mask.reshape(-1).float()
        token_weights = flat_mask
        valid = flat_mask.sum()
        probability_sum = (probabilities * flat_mask.unsqueeze(-1)).sum(dim=0)

    # Count each top-k rank in a separate column. expand_as is a stride-0 view,
    # avoiding the expensive materialization performed by repeat_interleave.
    counts = (
        torch.zeros(
            num_experts,
            top_k,
            dtype=torch.float32,
            device=gate_logits.device,
        )
        .scatter_add_(
            0,
            selected_experts,
            token_weights.unsqueeze(-1).expand_as(selected_experts),
        )
        .sum(dim=1)
    )
    return num_experts * torch.sum(counts * probability_sum) / (valid * valid)


class MoeLoadBalanceLossBenchmark(base.Benchmark):
    def __init__(self, *args, masked, **kwargs):
        super().__init__(*args, **kwargs)
        self.masked = masked

    def get_input_iter(self, dtype):
        for num_tokens, num_experts, top_k in self.shapes:
            logits = torch.randn(
                num_tokens,
                num_experts,
                dtype=dtype,
                device=self.device,
            )
            attention_mask = None
            if self.masked:
                attention_mask = torch.ones(
                    num_tokens,
                    dtype=torch.int64,
                    device=self.device,
                )
                attention_mask[3::4] = 0
            yield logits, top_k, attention_mask


@pytest.mark.moe_load_balance_loss
@pytest.mark.parametrize("masked", (False, True))
def test_moe_load_balance_loss_benchmark(masked):
    op_name = "moe_load_balance_loss_mask" if masked else "moe_load_balance_loss"
    bench = MoeLoadBalanceLossBenchmark(
        op_name=op_name,
        torch_op=_torch_reference,
        gems_op=flag_gems.moe_load_balance_loss,
        dtypes=consts.FLOAT_DTYPES,
        masked=masked,
    )
    bench.run()
