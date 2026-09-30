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

import math

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# (M, K, N): the mkldnn grad_output is (M, N) and the dense float32 weight is
# (N, K), so one case is an M x N x K inner-product backward-data workload. The
# native kernel only runs on CPU, so the extras stay small enough for CPU timing.
LINEAR_BACKWARD_SHAPES = [(2, 16, 8), (16, 64, 32), (64, 128, 64), (96, 96, 96)]


def _native_geometry(shape):
    """Map a configured shape onto this operator's (M, K, N) geometry.

    A shape that already has three positive extents is that triple. Any other
    configured shape keeps its element count: the leading extents become the
    grad_output rows and the last extent its columns, with a single output
    feature (K = 1). A shared generic shape such as (4096, 4096) therefore stays
    one grad_output of the requested size instead of becoming a quadratic
    M x N x N weight allocation.
    """
    dims = tuple(shape)
    if len(dims) == 3:
        return tuple(dims)
    if len(dims) == 1:
        return dims[0], 1, 1
    return math.prod(dims[:-1]), 1, dims[-1]


def _case_fn(shape, dtype):
    # The native op has no dtype argument; only the grad_output takes the case dtype.
    del dtype
    m, k, n = _native_geometry(shape)
    yield base.BenchmarkCasePlan(
        shape={"grad_output": [m, n], "weight": [n, k]},
        params={"input_size": [m, k]},
        builder_args=(m, k, n),
    )


def _build_inputs_fn(plan, dtype, device):
    # An mkldnn operand can only be built from a CPU tensor, so this CPU-only
    # kernel has no accelerator-side path to time.
    del device
    m, k, n = plan.builder_args
    grad_output = torch.empty((m, n), dtype=dtype).to_mkldnn()
    weight = torch.empty((n, k), dtype=torch.float32)
    return (plan.params["input_size"], grad_output, weight)


class MkldnnLinearBackwardInputBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        shapes = [tuple(shape) for shape in list(self.shapes) + LINEAR_BACKWARD_SHAPES]
        # A rank-zero descriptor cannot describe a native backward-data matrix.
        self.shapes = list(dict.fromkeys(shape for shape in shapes if shape))


@pytest.mark.mkldnn_linear_backward_input
def test_mkldnn_linear_backward_input():
    bench = MkldnnLinearBackwardInputBenchmark(
        op_name="mkldnn_linear_backward_input",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_linear_backward_input,
        gems_op=getattr(flag_gems, "mkldnn_linear_backward_input", None),
        # The required weight operand is dense float32; the float dtypes are
        # exactly the mkldnn-carryable set, so each one times a grad_output.
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
