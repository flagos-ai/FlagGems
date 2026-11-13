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

from . import base

# torch geqrf (cuSOLVER/LAPACK) only supports float32/float64 on CUDA; fp64
# is only benchmarked on backends that support it.
GEQRF_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    GEQRF_DTYPES.append(torch.float64)

# Sizes spanning the two implementation paths (single-launch register kernel
# vs blocked column-major panels) plus tall and batched inputs; matrices must
# be square-or-tall-or-wide 2D/3D, matching the KernelGen timing workloads.
GEQRF_SHAPES = [
    (64, 64),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (1024, 128),
    (8, 64, 64),
]


class GeqrfBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = GEQRF_SHAPES
        self.shape_desc = "N, N | M, N | batch, N, N"

    def get_case_iter(self, cur_dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype, ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"diagonal_shift": shape[-2]},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        (shape,) = case.builder_args[0].builder_args
        m, n = shape[-2:]
        A = torch.randn(shape, dtype=case.dtype, device=self.device)
        A = A + m * torch.eye(m, n, dtype=case.dtype, device=self.device)
        return (A,)


@pytest.mark.geqrf
def test_geqrf():
    bench = GeqrfBenchmark(
        op_name="geqrf",
        torch_op=torch.ops.aten.geqrf,
        input_fn=None,
        dtypes=GEQRF_DTYPES,
    )
    bench.run()
