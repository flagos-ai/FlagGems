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


class OrmqrBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [
        (32, 32),
        (48, 48),
        (64, 64),
        (96, 96),
        (128, 128),
        (256, 256),
        (1024, 1024),
        (4096, 4096),
        (1024, 65536),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = "M, N"

    def set_more_metrics(self):
        return ["tflops"]

    def get_tflops(self, op, *args, **kwargs):
        m, n = args[2].shape[-2:]
        k = args[0].shape[-1]
        return 2 * m * n * k

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        ordinal = 0
        for shape in self.shapes:
            for left in (True, False):
                for transpose in (True, False):
                    yield self._case_from_plan(
                        dtype,
                        ordinal,
                        base.BenchmarkCasePlan(
                            shape={"other": list(shape)},
                            params={"left": left, "transpose": transpose},
                            builder_args=(tuple(shape), left, transpose),
                        ),
                    )
                    ordinal += 1

    def build_inputs(self, case):
        if base.vendor_name == "iluvatar" and case.dtype == torch.float64:
            pytest.skip("Iluvatar native ormqr does not support float64 reference computation")
        plan = case.builder_args[0]
        shape, left, transpose = plan.builder_args
        torch.manual_seed(0)
        m, n = shape
        k = min(m, n)
        rows = m if left else n
        matrix = torch.randn((rows, k), dtype=case.dtype, device=self.device)
        inp, tau = torch.geqrf(matrix)
        other = torch.randn(shape, dtype=case.dtype, device=self.device)
        return inp, tau, other, {"left": left, "transpose": transpose}


@pytest.mark.ormqr
def test_ormqr():
    OrmqrBenchmark(
        op_name="ormqr",
        torch_op=torch.ormqr,
        dtypes=[torch.float32, torch.float64],
    ).run()
