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


class AdaptiveMaxPool2dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Common CNN feature-map shapes paired with representative output sizes.
        self.shapes = [
            (4, 3, 32, 32, 7, 7),
            (8, 64, 56, 56, 7, 7),
            (4, 128, 112, 112, 14, 14),
        ]
        self.shape_desc = "N, C, H, W, OUT_H, OUT_W"

    def get_input_iter(self, cur_dtype) -> Generator:
        for n, c, h, w, out_h, out_w in self.shapes:
            inp = torch.randn((n, c, h, w), dtype=cur_dtype, device=self.device)
            output, indices = torch.ops.aten.adaptive_max_pool2d(inp, (out_h, out_w))
            grad_output = torch.randn_like(output)
            yield grad_output, inp, indices


@pytest.mark.adaptive_max_pool2d_backward
def test_adaptive_max_pool2d_backward():
    bench = AdaptiveMaxPool2dBackwardBenchmark(
        op_name="adaptive_max_pool2d_backward",
        torch_op=torch.ops.aten.adaptive_max_pool2d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
