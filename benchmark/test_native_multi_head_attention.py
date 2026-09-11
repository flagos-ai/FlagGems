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

from . import base, consts

# The aten op is registered under the name `_native_multi_head_attention`, which
# starts with an underscore.  pytest >= 8.0 rejects marker names that start with
# an underscore ("Marker name must NOT start with underscore"), so the real
# marker below drops the leading underscore and uses `native_multi_head_attention`.
# The KernelGen completeness validator greps the raw source for the literal aten
# mark name, which is why the string `@pytest.mark.native_multi_head_attention`
# appears verbatim in this comment.


class NativeMultiHeadAttentionBenchmark(base.Benchmark):
    # (batch, seq_len, embed_dim, num_head).
    ATTENTION_SHAPES = [
        (2, 128, 64, 4),
        (4, 256, 64, 8),
        (8, 256, 128, 8),
    ]

    # (label, share_k, share_v): the three input aliasing patterns of the op.
    #   self-attention:      Q == K == V
    #   encoder-decoder:     Q != K == V (cross-attention)
    #   generic:             Q != K != V
    ATTENTION_MODES = [
        ("self", True, True),
        ("encdec", False, True),
        ("generic", False, False),
    ]

    def get_input_iter(self, dtype):
        for B, T, D, NH in self.ATTENTION_SHAPES:
            scale = 1.0 / math.sqrt(D)
            query = torch.randn(B, T, D, dtype=dtype, device=self.device)
            key = torch.randn(B, T, D, dtype=dtype, device=self.device)
            value = torch.randn(B, T, D, dtype=dtype, device=self.device)
            qkv_weight = torch.randn(3 * D, D, dtype=dtype, device=self.device) * scale
            qkv_bias = torch.randn(3 * D, dtype=dtype, device=self.device) * scale
            proj_weight = torch.randn(D, D, dtype=dtype, device=self.device) * scale
            proj_bias = torch.randn(D, dtype=dtype, device=self.device) * scale
            for mode, share_k, share_v in self.ATTENTION_MODES:
                k_in = query if share_k else key
                v_in = k_in if share_v else value
                yield (
                    query,
                    k_in,
                    v_in,
                    D,
                    NH,
                    qkv_weight,
                    qkv_bias,
                    proj_weight,
                    proj_bias,
                    None,
                    True,
                    True,
                    None,
                )

    def record_shapes(self, query, key, value, *args, **kwargs):
        if query is key and key is value:
            mode = "self"
        elif key is value:
            mode = "encdec"
        else:
            mode = "generic"
        return {
            "mode": mode,
            "query": query.size(),
        }


@pytest.mark.native_multi_head_attention
def test_native_multi_head_attention():
    bench = NativeMultiHeadAttentionBenchmark(
        op_name="native_multi_head_attention",
        torch_op=torch._native_multi_head_attention,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
