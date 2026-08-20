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
import torch.nn.functional as F

import flag_gems
from benchmark.base import Benchmark


class ChunkGatedDeltaRuleFwdBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16, torch.float16]
    DEFAULT_SHAPES = [(64,), (128,), (256,), (512,), (1024,)]
    DEFAULT_SHAPE_DESC = "T"

    def set_more_shapes(self):
        return [
            (64,),
            (128,),
            (256,),
            (512,),
            (1024,),
        ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            T = shape[0]
            yield self._build_inputs(T, cur_dtype)

    def _build_inputs(self, T: int, dtype: torch.dtype):
        device = flag_gems.device
        B, H, K, V = 1, 4, 64, 64

        q = torch.randn(B, T, H, K, device=device, dtype=dtype)
        k = torch.randn(B, T, H, K, device=device, dtype=dtype)
        v = torch.randn(B, T, H, V, device=device, dtype=dtype)
        g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
        beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
        scale = K**-0.5
        initial_state = torch.zeros(B, H, K, V, device=device, dtype=dtype)

        return (
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            True,
            None,
        )


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.xfail(
    reason="Triton 3.6.0 compilation error on Hopper: 'ttng.warp_group_dot' op pipeliner issue"
)
def test_perf_chunk_gated_delta_rule_fwd():
    bench = ChunkGatedDeltaRuleFwdBenchmark(
        op_name="chunk_gated_delta_rule_fwd",
        torch_op=flag_gems.chunk_gated_delta_rule_fwd,
    )
    bench.set_gems(flag_gems.chunk_gated_delta_rule_fwd)
    bench.run()
