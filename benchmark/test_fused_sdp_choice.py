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

from . import base, consts

# (batch, num_heads, seq_len, head_dim) attention shapes exercising the
# dispatch paths that ``_fused_sdp_choice`` discriminates between.
FUSED_SDP_CHOICE_SHAPES = [
    (2, 8, 128, 64),
    (4, 16, 512, 64),
    (2, 8, 1024, 128),
    (1, 8, 4096, 128),
    (2, 8, 128, 256),
]


class FusedSdpChoiceBenchmark(base.Benchmark):
    """Benchmark for ``aten::_fused_sdp_choice``.

    The operator inspects the query/key/value tensors and returns the chosen
    ``SDPBackend`` integer, so the benchmark measures the metadata-dispatch
    overhead rather than a tensor computation.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = FUSED_SDP_CHOICE_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            q = torch.randn(shape, dtype=cur_dtype, device=self.device)
            k = torch.randn(shape, dtype=cur_dtype, device=self.device)
            v = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (q, k, v)


@pytest.mark.fused_sdp_choice
def test_fused_sdp_choice():
    bench = FusedSdpChoiceBenchmark(
        op_name="fused_sdp_choice",
        torch_op=torch.ops.aten._fused_sdp_choice,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
