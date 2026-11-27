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
from tests.test_scaled_dot_product_attention import torch_sdpa

from . import base

# (batch, num_q_head, num_kv_head, seq_len, head_size, is_causal)
SDPA_BACKWARD_SHAPES = [
    (2, 8, 8, 1024, 64, False),
    (2, 8, 8, 2048, 128, False),
    (2, 8, 8, 1024, 64, True),
    (2, 8, 2, 1024, 64, True),
]


def sdpa_backward_autograd_reference(
    do,
    query,
    key,
    value,
    o,
    logsumexp,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
):
    # Baseline reuses the autograd reference from
    # tests/test_scaled_dot_product_attention.py: functional SDPA under
    # autograd, then backward. The saved o/logsumexp belong to the FlagGems
    # candidate and are intentionally unused here.
    q = query.detach().requires_grad_(True)
    k = key.detach().requires_grad_(True)
    v = value.detach().requires_grad_(True)
    out = torch_sdpa(q, k, v, scale, is_causal, enable_gqa=enable_gqa)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad_outputs=do)
    return dq, dk, dv


class SDPABackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SDPA_BACKWARD_SHAPES
        self.shape_desc = "batch, q_heads, kv_heads, seq_len, head_size, is_causal"

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for (
            batch,
            q_heads,
            kv_heads,
            seq_len,
            head_size,
            is_causal,
        ) in self.shapes:
            torch.manual_seed(2026)
            q = torch.randn(
                batch, q_heads, seq_len, head_size, dtype=dtype, device=self.device
            )
            k = torch.randn(
                batch, kv_heads, seq_len, head_size, dtype=dtype, device=self.device
            )
            v = torch.randn_like(k)
            do = torch.randn_like(q)
            scale = head_size**-0.5
            enable_gqa = q_heads != kv_heads
            o, logsumexp = flag_gems.scaled_dot_product_attention_forward(
                q, k, v, None, 0.0, is_causal, scale, enable_gqa
            )
            yield (
                do,
                q,
                k,
                v,
                o,
                logsumexp,
                None,
                0.0,
                is_causal,
                scale,
                enable_gqa,
            )


@pytest.mark.skipif(
    torch.__version__ < "2.5", reason="Low Pytorch Version: enable_gqa not supported"
)
@pytest.mark.skipif(flag_gems.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.scaled_dot_product_attention_backward
def test_scaled_dot_product_attention_backward():
    bench = SDPABackwardBenchmark(
        op_name="scaled_dot_product_attention_backward",
        # Baseline reuses the torch autograd reference from tests/: forward
        # through functional SDPA plus backward, so the baseline includes the
        # torch forward pass as well.
        torch_op=sdpa_backward_autograd_reference,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(flag_gems.scaled_dot_product_attention_backward)
    bench.run()
