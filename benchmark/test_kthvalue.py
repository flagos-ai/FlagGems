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

from . import base


class KthvalueBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1024, 256),
            (4096, 64),
            (16384, 128),
            (512, 512),
            (2048, 512),
        ]

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            k = 2 if shape[-1] > 2 else shape[-1]
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"k": k, "dim": -1},
                    builder_args=(tuple(shape), k),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, k = plan.builder_args
        torch.manual_seed(0)
        return torch.randn(shape, device=self.device, dtype=case.dtype), k, -1


@pytest.mark.kthvalue
def test_kthvalue():
    KthvalueBenchmark(
        op_name="kthvalue",
        torch_op=torch.kthvalue,
        dtypes=[torch.float32],
    ).run()
