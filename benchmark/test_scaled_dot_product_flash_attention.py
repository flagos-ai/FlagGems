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

import math

import pytest
import torch

import flag_gems

from . import base, utils


class ScaledDotProductFlashAttentionBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Shapes: (batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal)
        # Tensors are in (B, H, S, D) layout — ATen heads-first convention.
        self.shapes = []
        for head_size in (64, 128):
            for is_causal in (False, True):
                self.shapes.append((4, 8, 1024, 1024, head_size, is_causal))

        for batch, num_head, q_seq_len, kv_seq_len in (
            (1, 1, 128, 2048),
            (4, 8, 17, 1030),
            (2, 16, 512, 512),
        ):
            for is_causal in (False, True):
                self.shapes.append(
                    (batch, num_head, q_seq_len, kv_seq_len, 128, is_causal)
                )

        # Single-token decode
        for is_causal in (False, True):
            self.shapes.append((1, 4, 1, 1024, 128, is_causal))

    def set_more_shapes(self):
        return []


def scaled_dot_product_flash_attention_input_fn(config, dtype, device):
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal = config
    scale = float(1.0 / math.sqrt(head_size))

    q = torch.empty(
        (batch, num_head, q_seq_len, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    k = torch.empty(
        (batch, num_head, kv_seq_len, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    v = torch.empty(
        (batch, num_head, kv_seq_len, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)

    yield q, k, v, 0.0, is_causal, False, scale


def torch_scaled_dot_product_flash_attention(
    q, k, v, dropout_p=0.0, is_causal=False, return_debug_mask=False, scale=None
):
    return torch.ops.aten._scaled_dot_product_flash_attention.default(
        q, k, v, dropout_p, is_causal, return_debug_mask, scale=scale
    )


def gems_scaled_dot_product_flash_attention(
    q, k, v, dropout_p=0.0, is_causal=False, return_debug_mask=False, scale=None
):
    return flag_gems.scaled_dot_product_flash_attention(
        q, k, v, dropout_p, is_causal, return_debug_mask, scale=scale
    )


@pytest.mark.skipif(utils.SkipVersion("torch", "<2.4"), reason="Low Pytorch Version.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(flag_gems.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.scaled_dot_product_flash_attention
def test_scaled_dot_product_flash_attention():
    bench = ScaledDotProductFlashAttentionBenchmark(
        op_name="scaled_dot_product_flash_attention",
        input_fn=scaled_dot_product_flash_attention_input_fn,
        torch_op=torch_scaled_dot_product_flash_attention,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(gems_scaled_dot_product_flash_attention)
    bench.run()
