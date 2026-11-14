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


SHAPES = [(8, 64, 128), (16, 128, 256), (32, 256, 512), (64, 512, 1024)]


class AdaptiveMaxPool1DBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES
        self.shape_desc = "N, C, L"

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            output_size = shape[-1] // 2
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"self": list(shape)},
                    params={"output_size": [output_size]},
                    builder_args=(tuple(shape), output_size),
                ),
            )

    def build_inputs(self, case):
        torch.manual_seed(0)
        plan = case.builder_args[0]
        shape, output_size = plan.builder_args
        inp = torch.randn(shape, dtype=case.dtype, device=self.device)
        return inp, {"output_size": [output_size]}


@pytest.mark.adaptive_max_pool1d
def test_adaptive_max_pool1d():
    AdaptiveMaxPool1DBenchmark(
        op_name="adaptive_max_pool1d",
        torch_op=torch.ops.aten.adaptive_max_pool1d,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
