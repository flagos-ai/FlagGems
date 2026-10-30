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
from .generated_operator_utils import OperatorBenchmark
from .test_weight_norm_interface import _AscendWeightBackwardStaticReference

CASES = [((4096, 256), 0), ((4096, 256), 1), ((4, 256, 3), 0), ((4, 256, 3), 2)]


class WeightNormBackwardBenchmark(
    _AscendWeightBackwardStaticReference, OperatorBenchmark
):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = "shape, dim"


def _case_fn(shape, dtype):
    del dtype
    tensor_shape, dim = shape
    tensor_shape = tuple(tensor_shape)
    yield base.BenchmarkCasePlan(
        shape={"grad_w": list(tensor_shape), "saved_v": list(tensor_shape)},
        params={"dim": dim},
        builder_args=(tensor_shape, dim),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, dim = plan.builder_args
    torch.manual_seed(0)
    grad_w = torch.randn(shape, dtype=dtype, device=device)
    saved_v = torch.randn(shape, dtype=dtype, device=device)
    saved_g = torch.randn((shape[dim],), dtype=dtype, device=device)
    if dim == 0:
        reduction_dims = tuple(range(1, len(shape)))
    else:
        reduction_dims = tuple(range(len(shape) - 1))
    saved_norms = torch.sqrt(
        torch.sum(saved_v.float() * saved_v.float(), dim=reduction_dims)
    )
    return grad_w, saved_v, saved_g, saved_norms, dim


@pytest.mark.weight_norm_interface_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_weight_norm_interface_backward():
    WeightNormBackwardBenchmark(
        op_name="weight_norm_interface_backward",
        torch_op=torch.ops.aten._weight_norm_interface_backward.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
