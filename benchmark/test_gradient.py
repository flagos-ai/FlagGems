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

# torch.gradient promotes integral inputs to the default float dtype; only
# floating point dtypes are benchmarked.  fp64 only on backends that
# support it.
GRADIENT_DTYPES = [torch.float16, torch.float32]
if flag_gems.runtime.device.support_fp64:
    GRADIENT_DTYPES.append(torch.float64)

# Sizes matching the KernelGen timing workloads: 1-D sweeps, square 2-D and
# cubic 3-D tensors.
GRADIENT_SHAPES = [
    (65536,),
    (1048576,),
    (4096, 4096),
    (1024, 1024),
    (256, 256, 256),
]


class GradientBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = GRADIENT_SHAPES
        self.shape_desc = "N | N, N | N, N, N"

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (inp,)


@pytest.mark.gradient
def test_gradient():
    bench = GradientBenchmark(
        op_name="gradient",
        torch_op=torch.gradient,
        input_fn=None,
        dtypes=GRADIENT_DTYPES,
    )
    bench.set_gems(flag_gems.gradient)
    bench.run()
