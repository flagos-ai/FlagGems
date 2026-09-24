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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# glu_backward_jvp splits the last axis of these workloads, which aligns them
# with the glu entry in core_shapes.yaml.
GLU_BACKWARD_JVP_SHAPES = [
    (4, 8, 512, 128),
    (4, 8, 1024, 128),
    (4, 8, 2048, 128),
    (4, 8, 3072, 128),
    (4, 8, 4096, 128),
    (1024, 1024),
    (20, 320, 16),
]

# Static capability flag: bfloat16 is dropped on a target that does not support
# it, so listing and execution use the same dtype list.
DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def _case_fn(shape, dtype):
    del dtype
    dim = len(shape) - 1
    grad_glu_shape = list(shape)
    grad_glu_shape[dim] //= 2
    yield base.BenchmarkCasePlan(
        shape={"x": list(shape), "grad_glu": grad_glu_shape},
        params={"dim": dim},
        builder_args=(tuple(shape), dim, tuple(grad_glu_shape)),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, dim, grad_glu_shape = plan.builder_args
    grad_x = utils.generate_tensor_input(shape, dtype, device)
    grad_glu = utils.generate_tensor_input(grad_glu_shape, dtype, device)
    x = utils.generate_tensor_input(shape, dtype, device)
    dgrad_glu = utils.generate_tensor_input(grad_glu_shape, dtype, device)
    dx = utils.generate_tensor_input(shape, dtype, device)
    return grad_x, grad_glu, x, dgrad_glu, dx, {"dim": dim}


class GluBackwardJvpBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=GLU_BACKWARD_JVP_SHAPES)


@pytest.mark.glu_backward_jvp
def test_glu_backward_jvp():
    bench = GluBackwardJvpBenchmark(
        op_name="glu_backward_jvp",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.glu_backward_jvp,
        gems_op=getattr(flag_gems, "glu_backward_jvp", None),
        dtypes=DTYPES,
    )
    bench.run()
