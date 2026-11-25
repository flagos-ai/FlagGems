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

from . import base, consts


class ValueSelectingReductionBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Fixed (grad_shape, input sizes, dim, keepdim) cases covering 2D and
        # 3D inputs reduced along the last dimension, matching typical
        # max/min backward workloads; index 0..5 into _CASES below.
        self.shapes = list(range(len(_VALUE_SELECTING_REDUCTION_BACKWARD_CASES)))
        self.shape_desc = "case"

    def get_input_iter(self, dtype):
        for case in self.shapes:
            yield from self.input_fn(case, dtype, self.device)


# (grad_shape, input sizes, dim, keepdim) benchmark cases
_VALUE_SELECTING_REDUCTION_BACKWARD_CASES = [
    ((1024,), (1024, 1024), 1, False),
    ((4096,), (4096, 4096), 1, False),
    ((8192,), (8192, 8192), 1, False),
    ((32, 64), (32, 64, 128), 2, False),
    ((64, 128), (64, 128, 256), 2, False),
    ((128, 256), (128, 256, 512), 2, False),
]


def _input_fn(shape, dtype, device):
    grad_shape, sizes, dim, keepdim = _VALUE_SELECTING_REDUCTION_BACKWARD_CASES[shape]
    grad = torch.randn(grad_shape, dtype=dtype, device=device)
    # indices must be valid positions along the reduced dimension
    indices = torch.randint(0, sizes[dim], grad_shape, device=device)
    yield grad, dim, indices, list(sizes), keepdim


@pytest.mark.value_selecting_reduction_backward
def test_value_selecting_reduction_backward():
    bench = ValueSelectingReductionBackwardBenchmark(
        op_name="value_selecting_reduction_backward",
        torch_op=torch.ops.aten.value_selecting_reduction_backward,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.value_selecting_reduction_backward)
    bench.run()
