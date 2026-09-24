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

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

# aten::adjoint conjugates and swaps the last two dimensions of a matrix or
# batch of matrices; it is a zero-copy view, so the benchmark measures
# dispatch + view-materialization overhead. 1-D shapes are excluded because
# aten::adjoint rejects rank-1 tensors and 0-D is a deprecated edge case that
# does not exercise the transpose path.
ADJOINT_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (128, 512, 256),
    (64, 512, 512),
    (32, 128, 256),
    (8, 16, 32, 64),
    (4, 8, 16, 32, 64),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {}


class AdjointBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark limited to matrix / batch-of-matrices shapes.

    The default shape set (and GenericBenchmark's additional COMPREHENSIVE
    shapes) contain 1-D tensors, which aten::adjoint rejects at runtime, so the
    benchmark overrides ``set_shapes`` with the rank >= 2 shape list above.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=ADJOINT_SHAPES)


@pytest.mark.adjoint
def test_adjoint():
    bench = AdjointBenchmark(
        op_name="adjoint",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.adjoint,
        gems_op=getattr(flag_gems, "adjoint", None),
        dtypes=consts.FLOAT_DTYPES + consts.COMPLEX_DTYPES,
    )
    bench.run()
