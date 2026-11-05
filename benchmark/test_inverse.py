# Copyright 2026, The FlagOS Contributors.
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
from flag_gems import inverse

from . import base

# torch.inverse only accepts fp32/fp64 on CUDA.
INVERSE_DTYPES = [torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)

# Single matrices up to the size where the register tile spills, plus batched
# stacks of small matrices (the shapes where the fused kernel wins most).
INVERSE_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (1024, 8, 8),
    (512, 16, 16),
    (128, 32, 32),
    (32, 64, 64),
    (8, 128, 128),
]


def _well_conditioned(shape, dtype, device):
    n = shape[-1]
    eye = torch.eye(n, dtype=dtype, device=device)
    if len(shape) > 2:
        eye = eye.expand(shape).contiguous()
    return eye + 0.05 * torch.randn(shape, dtype=dtype, device=device)


class InverseBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = INVERSE_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = _well_conditioned(shape, cur_dtype, self.device)
            yield (A,)


@pytest.mark.inverse
def test_inverse():
    bench = InverseBenchmark(
        op_name="inverse",
        torch_op=torch.inverse,
        dtypes=INVERSE_DTYPES,
    )
    bench.set_gems(inverse)
    bench.run()
