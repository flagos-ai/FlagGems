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

from typing import Generator

import pytest
import torch

from . import base, consts


class FusedRmsNormBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Transformer-like token counts and hidden sizes.
        self.shapes = [(128, 256), (512, 1024), (1024, 4096)]
        self.shape_desc = "M, N"

    def get_input_iter(self, cur_dtype) -> Generator:
        for m, n in self.shapes:
            inp = torch.randn((m, n), dtype=cur_dtype, device=self.device)
            grad = torch.randn_like(inp)
            weight = torch.randn(n, dtype=cur_dtype, device=self.device)
            rstd = torch.rsqrt(inp.float().square().mean(dim=-1) + 1e-5)
            yield grad, inp, [n], rstd, weight, [True, True]


@pytest.mark.fused_rms_norm_backward
def test_fused_rms_norm_backward():
    bench = FusedRmsNormBackwardBenchmark(
        op_name="fused_rms_norm_backward",
        torch_op=torch.ops.aten._fused_rms_norm_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
